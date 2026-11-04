"""Twenty-forward inference latency and real-FM precision diagnostics at step7."""
import argparse
import contextlib
import functools
import inspect
import json
import os
from pathlib import Path
import statistics
import textwrap
import time

import numpy as np
import torch
import torch_musa
from mme_vla_suite.musa import model as mm
from mme_vla_suite.musa import e2e
from mme_vla_suite.musa.checkpoint_tools import load_unsharded
from mme_vla_suite.musa.runtime import load_profile
from mme_vla_suite.musa.data import TrainingDataset, GlobalBatchSampler, collate, prepare_batch, global_random
from mme_vla_suite.training.lerobot_framesamp import VideoReader

def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)

def distribution(values):
    return {'count': len(values), 'mean_ms': float(np.mean(values)),
            'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95)),
            'std_ms': float(np.std(values)), 'min_ms': min(values), 'max_ms': max(values)}

class ModuleTimer:
    def __init__(self, model):
        self.model = model; self.enabled = False; self.events = []; self.undo = []
        self.reader_start = None; self.camera = 0

    @contextlib.contextmanager
    def scope(self, name):
        if not self.enabled:
            yield; return
        start = torch.musa.Event(enable_timing=True); end = torch.musa.Event(enable_timing=True)
        start.record()
        try: yield
        finally:
            end.record(); self.events.append((name, start, end))

    def replace(self, owner, name, value):
        self.undo.append((owner, name, getattr(owner, name))); setattr(owner, name, value)

    def wrap(self, function, name):
        @functools.wraps(function)
        def call(*a, **kw):
            with self.scope(name): return function(*a, **kw)
        return call

    def install(self):
        for owner, name, label in [
            (mm.Memory, 'forward', 'memory_total'),
            (mm.GemmaBlock, 'forward', 'joint_vlm_action_transformer_18_layers'),
            (e2e.E2EScorer, 'encode_frames', 'scorer_frame_encoding'),
            (e2e.E2EScorer, 'score_frames', 'scorer_context_total'),
            (e2e, 'replay_memory', 'history_replay_including_scorer'),
        ]:
            self.replace(owner, name, self.wrap(getattr(owner, name), label))
        original_vision = mm.Vision.forward
        def vision(module, *a, **kw):
            label = 'front_vision' if self.camera % 2 == 0 else 'wrist_vision'; self.camera += 1
            with self.scope(label): return original_vision(module, *a, **kw)
        self.replace(mm.Vision, 'forward', vision)
        original_pool = mm.pool_tokens
        def pool(*a, **kw):
            if self.enabled:
                self.reader_start = torch.musa.Event(enable_timing=True); self.reader_start.record()
            return original_pool(*a, **kw)
        self.replace(mm, 'pool_tokens', pool)
        original_linear = mm.Tree.linear
        def linear(module, x, path, *a, **kw):
            label = {'mem_feature_adapter': 'history_feature_projection',
                     'mem_cond_adapter': 'memory_condition_adapter',
                     'action_in_proj': 'action_input_projection',
                     'time_mlp_in': 'time_mlp', 'time_mlp_out': 'time_mlp',
                     'action_out_proj': 'action_output_projection'}.get(path)
            if path == 'mem_cond_adapter' and self.enabled and self.reader_start is not None:
                end = torch.musa.Event(enable_timing=True); end.record()
                self.events.append(('reader_including_pool_age_position', self.reader_start, end)); self.reader_start = None
            with self.scope(label) if label else contextlib.nullcontext():
                return original_linear(module, x, path, *a, **kw)
        self.replace(mm.Tree, 'linear', linear)
        original_embedding = mm.F.embedding
        def embedding(inputs, weight, *a, **kw):
            with self.scope('text_embedding') if weight is self.model.embedding else contextlib.nullcontext():
                return original_embedding(inputs, weight, *a, **kw)
        self.replace(mm.F, 'embedding', embedding)
        original_rms = mm.GemmaBlock.rms
        def rms(module, *a, **kw):
            with self.scope('final_action_norm') if module is self.model.final else contextlib.nullcontext():
                return original_rms(module, *a, **kw)
        self.replace(mm.GemmaBlock, 'rms', rms)

    def reset(self):
        self.events = []; self.reader_start = None; self.camera = 0

    def values(self):
        sums, counts = {}, {}
        for name, start, end in self.events:
            sums[name] = sums.get(name, 0.) + start.elapsed_time(end)
            counts[name] = counts.get(name, 0) + 1
        return sums, counts

    def close(self):
        for owner, name, original in reversed(self.undo): setattr(owner, name, original)
        self.undo.clear()

def measure(function, timer, count=20):
    timer.enabled = False
    for _ in range(5): function()
    torch.musa.synchronize()
    plain = []
    for _ in range(count):
        torch.musa.synchronize(); started = time.perf_counter(); value = function()
        torch.musa.synchronize(); plain.append((time.perf_counter() - started) * 1000)
    reference = value.detach().clone()
    timer.enabled = True; instrumented = []; stages = {}; calls = {}
    torch.musa.reset_peak_memory_stats()
    for _ in range(count):
        timer.reset(); torch.musa.synchronize(); started = time.perf_counter(); value = function()
        torch.musa.synchronize(); instrumented.append((time.perf_counter() - started) * 1000)
        values, current_calls = timer.values()
        for key, value_ms in values.items(): stages.setdefault(key, []).append(value_ms)
        for key, n in current_calls.items(): calls.setdefault(key, []).append(n)
    assert torch.equal(reference, value), 'Instrumentation changed inference output'
    timer.enabled = False
    return {'unmodified_total_wall': distribution(plain), 'instrumented_total_wall': distribution(instrumented),
            'module_inclusive_stream_elapsed': {key: {**distribution(values),
                'calls_per_forward': float(np.mean(calls[key]))} for key, values in stages.items()},
            'peak_allocated_gib': torch.musa.max_memory_allocated() / 2**30,
            'instrumentation_output_identical': True,
            'scope': 'resident device inputs, eval/inference_mode, unsharded single GPU; nested times overlap'}

def policy_batch(adapter, count, device):
    dataset = adapter.dataset; source = dataset.metadata; cache = dataset.root
    episode = max(source['episodes'], key=lambda e: e['length'])
    name = 'episode_%06d' % episode['episode_index']
    with np.load(cache / ('keyframes/' + name + '.npz')) as data:
        plan = {key: data[key] for key in data.files}
    if len(plan['row']) < count: raise ValueError('Requested history unavailable')
    decision = int(plan['row'][count - 1]); row = episode['dataset_from_index'] + decision
    native = np.load(cache / ('native/' + name + '.npy'), mmap_mode='r')
    features = np.zeros((320, 64, 2048), np.float16); features[:count] = native[:count]
    ticks = np.zeros(320, np.int32); ticks[:count] = plan['tick'][:count]
    mask = np.zeros(320, bool); mask[:count] = True
    reader = VideoReader(source)
    front = next(reader.frames(episode, 'image', decision, 1)); wrist = next(reader.frames(episode, 'wrist_image', decision, 1)); reader.close()
    arrays = dataset.arrays; actions = np.minimum(row + np.arange(20), episode['dataset_to_index'] - 1)
    raw = {'image': front, 'wrist_image': wrist, 'state': np.array(arrays['state'][row]),
           'actions': np.array(arrays['actions'][actions]),
           'prompt': source['tasks'][str(int(arrays['task_index'][row]))].lower(),
           'simple_subgoal': '', 'grounded_subgoal': '', 'perc_mem_feats': features,
           'perc_mem_frame_tick': ticks, 'perc_mem_frame_mask': mask,
           'perc_mem_now_tick': np.int32(plan['tick'][count - 1])}
    keys, noise, times = global_random(42, 7, 64)
    batch = prepare_batch(collate([adapter.transform_raw(raw)]), keys[:1], device, augment=False)
    return batch, torch.from_numpy(noise[:1].copy()).to(device), torch.from_numpy(times[:1].copy()).to(device), {
        'episode': episode['episode_index'], 'decision': decision, 'history_frames': count, 'padding': 320}

def real_training_batch(adapter, device):
    rows = next(iter(GlobalBatchSampler(len(adapter), 0, 4, 64, 16, 42, start_step=7)))
    raw = collate([adapter[index] for index in rows]); keys, noise, times = global_random(42, 7, 64)
    batch = prepare_batch(raw, keys[:16], device)
    return batch, torch.from_numpy(noise[:16].copy()).to(device), torch.from_numpy(times[:16].copy()).to(device)

def run_inference(model, adapter, device, output):
    model.eval(); timer = ModuleTimer(model); timer.install()
    report = {'iterations': 20, 'warmup': 5, 'physical_gpu': 4,
              'model_parameters': sum(p.numel() for p in model.parameters()), 'cases': {}}
    try:
        with torch.inference_mode():
            for frames in [32, 41, 80, 142]:
                batch, noise, times, fixture = policy_batch(adapter, frames, device)
                noisy = times[:, None, None] * noise + (1 - times[:, None, None]) * batch['actions'].float()
                value = measure(lambda: model.predict_velocity(batch, noisy, times), timer)
                value['fixture'] = fixture; report['cases']['velocity_b1_n%d' % frames] = value
                atomic(output / 'inference20.json', report)
                print('INFERENCE_CASE', frames, value['unmodified_total_wall']['mean_ms'], flush=True)
                if frames == 41:
                    result = measure(lambda: model.sample_actions(batch, num_steps=10, noise=noise), timer)
                    result['fixture'] = fixture; result['denoising_steps'] = 10
                    report['cases']['sample_actions10_b1_n41'] = result
                    atomic(output / 'inference20.json', report)
                del batch, noise, times, noisy
            batch, noise, times = real_training_batch(adapter, device)
            noisy = times[:, None, None] * noise + (1 - times[:, None, None]) * batch['actions'].float()
            result = measure(lambda: model.predict_velocity(batch, noisy, times), timer)
            result['history_counts'] = batch['perc_mem_frame_mask'].sum(1).cpu().tolist()
            report['cases']['velocity_b16_real_batch_step8_rank0'] = result
        report['passed'] = True; atomic(output / 'inference20.json', report)
        print('INFERENCE20_COMPLETE', flush=True)
    finally:
        timer.close()

def gradient_stat(tensor):
    if tensor is None: return {'none': True}
    value = tensor.detach().float()
    return {'dtype': str(tensor.dtype), 'elements': tensor.numel(),
            'nonzero': int(torch.count_nonzero(value).cpu()),
            'max_abs': float(value.abs().max().cpu()), 'l2': float(value.norm().cpu())}

class QuantizedForwardFP32Gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value): return value.half().float()
    @staticmethod
    def backward(ctx, gradient): return gradient

def run_gradients(model, adapter, device, output):
    model.train(); batch, noise, times = real_training_batch(adapter, device)
    parameters = [(n, p) for n, p in model.named_parameters() if n.startswith('memory.') and p.requires_grad]
    scorer = [(n, p) for n, p in parameters if n.startswith('memory.scorer.')]
    selected_hooks = []; hooks = {}; traces = {}
    original_linear = mm.Tree.linear; original_replay = e2e.replay_memory; original_memory = mm.Memory.forward
    def capture(name, tensor):
        if tensor.requires_grad:
            tensor.register_hook(lambda gradient: traces.setdefault(name, []).append(gradient_stat(gradient)))
        return tensor
    def linear(module, x, path, *args, **kwargs):
        value = original_linear(module, x, path, *args, **kwargs)
        if path in ['mem_feature_adapter', 'mem_reader/output_final', 'mem_cond_adapter']:
            capture(path + '_output', value)
        return value
    def replay(scorer_module, features, *args, **kwargs):
        capture('projected_storage_input', features)
        result = original_replay(scorer_module, features, *args, **kwargs)
        capture('selected_storage_output', result[0]); return result
    mm.Tree.linear = linear; e2e.replay_memory = replay
    report = {'scope': 'real step8/rank0 batch16 FM objective, step7 weights unchanged; memory partial gradients',
              'history_counts': batch['perc_mem_frame_mask'].sum(1).cpu().tolist(), 'cases': {}}
    vectors = {}; forward_reference = None
    try:
        for label, scale in [('unscaled_fp16', 1.), ('scaled_65536', 65536.), ('scaled_1048576', 1048576.), ('oracle_quantized_forward_fp32_gradient', 1.)]:
            model.zero_grad(set_to_none=True); traces.clear()
            if label.startswith('oracle'):
                source = textwrap.dedent(inspect.getsource(original_memory))
                before = 'self.linear(native.float(),"mem_feature_adapter").to(torch.float16)'
                after = '_oracle_quantize.apply(self.linear(native.float(),"mem_feature_adapter"))'
                assert source.count(before) == 1
                namespace = {**mm.__dict__, '_oracle_quantize': QuantizedForwardFP32Gradient}
                exec(compile(source.replace(before, after), '<diagnostic_fp32_gradient_oracle>', 'exec'), namespace)
                mm.Memory.forward = namespace['forward']
            else: mm.Memory.forward = original_memory
            loss = model(batch, noise, times)
            raw = torch.autograd.grad(loss * scale, [p for _, p in parameters], allow_unused=True)
            grads = {name: (torch.zeros_like(parameter) if gradient is None else gradient.detach() / scale)
                     for (name, parameter), gradient in zip(parameters, raw)}
            vector = torch.cat([grads[name].flatten() for name, _ in scorer])
            assert torch.isfinite(vector).all()
            vectors[label] = vector.detach().cpu()
            row = {'loss': float(loss.detach().cpu()), 'scale': scale,
                   'scorer_grad_norm': float(vector.norm().cpu()), 'scorer_nonzero': int(torch.count_nonzero(vector).cpu()),
                   'scorer_max_abs': float(vector.abs().max().cpu()),
                   'feature_adapter_grad_norm': float(torch.cat([g.flatten() for n, g in grads.items() if 'mem_feature_adapter__' in n]).norm().cpu()),
                   'boundary_gradients_before_unscale': dict(traces)}
            if forward_reference is None: forward_reference = loss.detach().clone()
            row['forward_loss_bitwise_equal'] = bool(torch.equal(loss.detach(), forward_reference))
            report['cases'][label] = row; atomic(output / 'gradient_diagnostic.json', report)
            print('GRADIENT_CASE', label, row['scorer_grad_norm'], row['scorer_nonzero'], flush=True)
            del loss, raw, grads, vector
        reference = vectors['oracle_quantized_forward_fp32_gradient'].double()
        for label, vector in vectors.items():
            value = vector.double(); denominator = reference.norm().clamp_min(1e-100)
            report['cases'][label]['relative_l2_vs_fp32_gradient_oracle'] = float((value - reference).norm() / denominator)
            report['cases'][label]['cosine_vs_fp32_gradient_oracle'] = float(torch.dot(value, reference) / (value.norm() * denominator).clamp_min(1e-100))
        chosen = report['cases']['scaled_1048576']
        report['passed'] = (report['cases']['unscaled_fp16']['scorer_nonzero'] == 0 and
            chosen['scorer_nonzero'] > 0 and chosen['relative_l2_vs_fp32_gradient_oracle'] < .03 and
            chosen['cosine_vs_fp32_gradient_oracle'] > .999 and
            all(v['forward_loss_bitwise_equal'] for v in report['cases'].values()))
        report['recommended_loss_scale'] = 1048576.
        atomic(output / 'gradient_diagnostic.json', report)
        if not report['passed']: raise AssertionError('Real FM gradient diagnostic did not meet the repair gate')
        print('GRADIENT_DIAGNOSIS_COMPLETE', flush=True)
    finally:
        mm.Tree.linear = original_linear; e2e.replay_memory = original_replay; mm.Memory.forward = original_memory
        model.zero_grad(set_to_none=True)

def main():
    cli = argparse.ArgumentParser(); cli.add_argument('--checkpoint', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True); cli.add_argument('--mode', choices=['inference', 'gradients', 'both'], default='both')
    args = cli.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    assert os.environ['MUSA_VISIBLE_DEVICES'] == os.environ['CUDA_VISIBLE_DEVICES'] == '4'
    os.sched_setaffinity(0, set(range(64, 72))); torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest'); torch.backends.mudnn.allow_tf32 = False
    torch.backends.musa.matmul.allow_tf32 = False; torch.musa.set_device(0); device = torch.device('musa', 0)
    profile = load_profile(); model, restored = load_unsharded(profile, args.checkpoint, device)
    atomic(args.output / 'checkpoint_consolidation.json', restored)
    print('CHECKPOINT7_CONSOLIDATED', json.dumps(restored), flush=True)
    adapter = TrainingDataset(profile)
    if args.mode in ['inference', 'both']: run_inference(model, adapter, device, args.output)
    if args.mode in ['gradients', 'both']: run_gradients(model, adapter, device, args.output)
    atomic(args.output / 'complete.json', {'mode': args.mode, 'passed': True, 'checkpoint_step': restored['step']})

if __name__ == '__main__': main()
