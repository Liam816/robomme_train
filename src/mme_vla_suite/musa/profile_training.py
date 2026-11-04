"""Bounded four-rank profile of the unmodified E2E training loop.

The loop is instrumented in memory. The source, source profile, data ordering,
optimizer, EMA and checkpoint implementation are retained. Optional ablations
are explicitly diagnostic and do not replace the experiment configuration.
"""
from __future__ import annotations
import contextlib
import functools
import inspect
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_musa
import torch.distributed as dist
from mme_vla_suite.musa import train as training
from mme_vla_suite.musa import model as model_module
from mme_vla_suite.musa import e2e

ROOT = Path(os.environ['ROBOMME_PROFILE_DIR'])
ROOT.mkdir(parents=True, exist_ok=True)
RANK = int(os.environ['LOCAL_RANK'])
VARIANT = os.environ.get('ROBOMME_PROFILE_VARIANT', 'e2e')
ENABLED = False
BACKWARD = False
EVENTS = []
HOST = []
HISTORIES = []
ROWS = []
FIRST_STEP = None
PROFILER = None
EMA_SCOPE = None
DELTA_SCOPE = None

@contextlib.contextmanager
def timed(name):
    if not ENABLED:
        yield
        return
    label = ('backward/' if BACKWARD else 'forward/') + name
    begin = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    begin.record()
    start = time.perf_counter()
    with torch.profiler.record_function(label):
        try:
            yield
        finally:
            HOST.append((label, (time.perf_counter() - start) * 1000))
            end.record()
            EVENTS.append((label, begin, end))

def wrapped(function, name):
    @functools.wraps(function)
    def invoke(*args, **kwargs):
        with timed(name):
            return function(*args, **kwargs)
    return invoke

def summarize(events):
    out = {}
    for name, value in events:
        row = out.setdefault(name, {'calls': 0, 'total_ms': 0.0})
        row['calls'] += 1
        row['total_ms'] += value
    for row in out.values():
        row['mean_ms'] = row['total_ms'] / row['calls']
    return out

def begin_step(step):
    global ENABLED, FIRST_STEP, PROFILER
    dist.barrier()
    if FIRST_STEP is None:
        FIRST_STEP = step
    EVENTS.clear(); HOST.clear(); HISTORIES.clear()
    ENABLED = step > FIRST_STEP
    torch.musa.reset_peak_memory_stats()
    if ENABLED and RANK == 0 and step == FIRST_STEP + 2 and os.environ.get('ROBOMME_PROFILE_TRACE', '1') == '1':
        activities = list(torch.profiler.supported_activities())
        PROFILER = torch.profiler.profile(activities=activities, record_shapes=True,
                                           profile_memory=False, with_stack=False, with_flops=True)
        PROFILER.start()

def end_step(step, record):
    global ENABLED, PROFILER
    if ENABLED:
        torch.musa.synchronize()
        row = {'step': step, 'variant': VARIANT, 'rank': RANK, 'train_record': dict(record),
               'inclusive_stream_elapsed': summarize((n, b.elapsed_time(e)) for n, b, e in EVENTS),
               'inclusive_host_elapsed': summarize(HOST), 'microbatch_histories': list(HISTORIES),
               'peak_allocated_gib': torch.musa.max_memory_allocated() / 2**30,
               'peak_reserved_gib': torch.musa.max_memory_reserved() / 2**30,
               'notes': 'Nested timings overlap; stream elapsed includes host dispatch gaps, not just kernel busy time.'}
        ROWS.append(row)
        (ROOT / ('rank%d_step%06d.json' % (RANK, step))).write_text(json.dumps(row, indent=2))
    ENABLED = False
    if PROFILER is not None:
        PROFILER.stop()
        PROFILER.export_chrome_trace(str(ROOT / 'rank0_trace.json'))
        averages = PROFILER.key_averages(group_by_input_shape=True)
        operators = []
        for item in averages:
            operators.append({'key': item.key, 'count': item.count, 'shapes': item.input_shapes,
                              'self_cpu_us': item.self_cpu_time_total, 'total_cpu_us': item.cpu_time_total,
                              'self_device_us': getattr(item, 'self_device_time_total', 0),
                              'total_device_us': getattr(item, 'device_time_total', 0),
                              'flops': item.flops})
        (ROOT / 'rank0_operators.json').write_text(json.dumps(operators, indent=2))
        (ROOT / 'rank0_cpu_table.txt').write_text(averages.table(sort_by='self_cpu_time_total', row_limit=50))
        (ROOT / 'rank0_device_table.txt').write_text(averages.table(sort_by='self_device_time_total', row_limit=50))
        PROFILER = None
    (ROOT / ('rank%d_summary.json' % RANK)).write_text(json.dumps({
        'variant': VARIANT, 'rank': RANK, 'rows': ROWS, 'first_update_is_warmup': FIRST_STEP,
        'source': str(Path(training.__file__)), 'physical_gpus': [4, 5, 6, 7]}, indent=2))

def enter_ema():
    global EMA_SCOPE
    EMA_SCOPE = timed('ema_update'); EMA_SCOPE.__enter__()

def exit_ema():
    global EMA_SCOPE
    EMA_SCOPE.__exit__(None, None, None); EMA_SCOPE = None

def enter_delta():
    global DELTA_SCOPE
    DELTA_SCOPE = timed('parameter_delta_logging'); DELTA_SCOPE.__enter__()

def exit_delta():
    global DELTA_SCOPE
    DELTA_SCOPE.__exit__(None, None, None); DELTA_SCOPE = None

def compatible_profile(saved, current):
    """Profiling only: allow micro/accumulation changes at fixed global64/four ranks."""
    saved = json.loads(json.dumps(saved))
    current = json.loads(json.dumps(current))
    for value in [saved, current]:
        if value['musa']['micro_batch_size'] * value['musa']['gradient_accumulation_steps'] * 4 != 64:
            return False
        for key in ['micro_batch_size', 'gradient_accumulation_steps', 'runtime_overlay', 'runtime_overlay_sha256']:
            value['musa'].pop(key, None)
    return saved == current

def finish_profile(run):
    if RANK == 0:
        path = run / 'status.json'
        state = json.loads(path.read_text())
        state.update(stage='profile_complete', long_training_resumed=False,
                     new_checkpoint_written=False)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(path)
    (ROOT / ('rank%d_complete.json' % RANK)).write_text(json.dumps({
        'stage': 'profile_complete', 'last_step': ROWS[-1]['step'] if ROWS else None,
        'long_training_resumed': False, 'new_checkpoint_written': False,
        'runtime_only_resume_compatibility': os.environ.get('ROBOMME_PROFILE_ALLOW_MICRO_CHANGE') == '1'}, indent=2))

def install():
    # Inject only measurement boundaries into a copy of the existing loop.
    source = Path(training.__file__).read_text()
    substitutions = {
        '        started=time.monotonic();optimizer.zero_grad(set_to_none=True)':
        '        _profile_begin(step+1)\n        started=time.monotonic();optimizer.zero_grad(set_to_none=True)',
        '        optimizer.step()\n        with torch.no_grad():':
        '        optimizer.step()\n        _profile_enter_ema()\n        with torch.no_grad():',
        '        dist.all_reduce(loss_total);loss_total/=4':
        '        _profile_exit_ema()\n        dist.all_reduce(loss_total);loss_total/=4\n        _profile_enter_delta()',
        '        record={"step":step+1,':
        '        _profile_exit_delta()\n        record={"step":step+1,',
        '        records.append(record)':
        '        _profile_end(step+1,record)\n        records.append(record)',
    }
    for before, after in substitutions.items():
        if source.count(before) != 1:
            raise RuntimeError('Instrumentation source mismatch: ' + before)
        source = source.replace(before, after)
    if os.environ.get('ROBOMME_PROFILE_ALLOW_MICRO_CHANGE') == '1':
        before = 'saved_meta["profile"]!=profile'
        if source.count(before) != 1:
            raise RuntimeError('Resume comparison source mismatch')
        source = source.replace(before, 'not _profile_compatible(saved_meta["profile"],profile)')
    if os.environ.get('ROBOMME_PROFILE_ONLY') == '1':
        before = '    if records[-1]["reader_final_parameter_delta_l2"]<=0:'
        if source.count(before) != 1:
            raise RuntimeError('Training end source mismatch')
        source = source.replace(before,
            '    _profile_finish(run)\n    dist.barrier(); del iterator,loader\n    dist.destroy_process_group()\n    return\n' + before)
    namespace = training.__dict__
    namespace.update(_profile_begin=begin_step, _profile_end=end_step,
                     _profile_enter_ema=enter_ema, _profile_exit_ema=exit_ema,
                     _profile_enter_delta=enter_delta, _profile_exit_delta=exit_delta)
    namespace.update(_profile_compatible=compatible_profile, _profile_finish=finish_profile)
    exec(compile(source, training.__file__, 'exec'), namespace)
    training.exact_grad_norm = wrapped(training.exact_grad_norm, 'gradient_norm_and_clip')
    training.global_random = wrapped(training.global_random, 'jax_global_random')
    training.prepare_batch = wrapped(training.prepare_batch, 'cpu_augmentation_and_h2d')
    original_next = torch.utils.data.dataloader._BaseDataLoaderIter.__next__
    torch.utils.data.dataloader._BaseDataLoaderIter.__next__ = wrapped(original_next, 'dataloader_wait')
    torch.optim.AdamW.step = wrapped(torch.optim.AdamW.step, 'adamw_update')
    for cls, method, label in [
        (model_module.PercMemPi05Torch, 'forward', 'model_forward'),
        (model_module.Vision, 'forward', 'current_image_vision'),
        (model_module.GemmaBlock, 'forward', 'gemma_block'),
        (model_module.Memory, 'forward', 'memory_total'),
        (e2e.E2EScorer, 'encode_frames', 'scorer_frame_encoding'),
        (e2e.E2EScorer, 'score_frames', 'scorer_context'),
    ]:
        setattr(cls, method, wrapped(getattr(cls, method), label))
    original_linear = model_module.Tree.linear
    def linear(self, x, path, *a, **kw):
        if path == 'mem_feature_adapter':
            with timed('memory_feature_projection'):
                return original_linear(self, x, path, *a, **kw)
        return original_linear(self, x, path, *a, **kw)
    model_module.Tree.linear = linear
    original_backward = torch.Tensor.backward
    def backward(self, *a, **kw):
        global BACKWARD
        BACKWARD = True
        try:
            with timed('loss_backward'):
                return original_backward(self, *a, **kw)
        finally:
            BACKWARD = False
    torch.Tensor.backward = backward
    for cls in [e2e.SelectMembers, e2e.HardEviction]:
        cls.backward = staticmethod(wrapped(cls.backward, cls.__name__ + '_vjp'))
    original_replay = e2e.replay_memory
    def replay(scorer, features, ticks, valid, now, cfg, **kw):
        if ENABLED:
            counts = valid.sum(1).detach().cpu().tolist()
            HISTORIES.append(counts)
        with timed('differentiable_history_replay'):
            return original_replay(scorer, features, ticks, valid, now, cfg, **kw)
    e2e.replay_memory = replay
    if VARIANT == 'uniform32_reader':
        original_memory = model_module.Memory.forward
        def uniform32(self, native, ticks, valid, now, *a, **kw):
            count = valid.sum(1)
            slots = torch.arange(32, device=native.device)[None]
            indices = torch.div(slots * (count - 1).clamp_min(0)[:, None], 31, rounding_mode='floor')
            indices = torch.where(count[:, None] < 32, slots.clamp_max((count - 1).clamp_min(0)[:, None]), indices)
            native = e2e.gather_frames(native, indices)
            ticks = e2e.gather_frames(ticks, indices)
            valid = slots < count[:, None]
            cfg = self.cfg
            self.cfg = {**cfg, 'stage': 'teacher'}
            try:
                return original_memory(self, native, ticks, valid, now, *a, **kw)
            finally:
                self.cfg = cfg
        model_module.Memory.forward = uniform32
    elif VARIANT == 'trim_padding':
        original_memory = model_module.Memory.forward
        def trimmed(self, native, ticks, valid, now, *a, **kw):
            effective = max(1, int(valid.sum(1).max().item()))
            return original_memory(self, native[:, :effective], ticks[:, :effective], valid[:, :effective], now, *a, **kw)
        model_module.Memory.forward = trimmed
    elif VARIANT != 'e2e':
        raise ValueError(VARIANT)

def main():
    install()
    training.main()

if __name__ == '__main__':
    main()
