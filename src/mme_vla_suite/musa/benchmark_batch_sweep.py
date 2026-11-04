"""Bounded compute benchmark; original four-rank batch64, fixed real inputs.

Restores the verified E2E checkpoint. No experiment is resumed in the background.
Each trial measures forward/backward, clipping, AdamW and EMA. Inputs are resident
so decoding/augmentation/H2D are excluded and reported separately by full profile.
"""
import contextlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch_musa
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from mme_vla_suite.musa.model import PercMemPi05Torch, Memory
from mme_vla_suite.musa.prepare_assets import load_tree
from mme_vla_suite.musa.runtime import load_profile
from mme_vla_suite.musa.data import TrainingDataset, GlobalBatchSampler, collate, global_random, prepare_batch
from mme_vla_suite.musa.train import restore_local, exact_grad_norm, lr_at
from mme_vla_suite.musa import e2e

def main():
    rank = int(os.environ['LOCAL_RANK'])
    assert os.environ['MUSA_VISIBLE_DEVICES'] == '4,5,6,7'
    os.sched_setaffinity(0, set(range(64 + rank * 8, 72 + rank * 8)))
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.mudnn.allow_tf32 = False
    torch.backends.musa.matmul.allow_tf32 = False
    torch.musa.set_device(rank)
    device = torch.device('musa', rank)
    dist.init_process_group('mccl')
    profile = load_profile()
    root = Path(profile['musa']['root'])
    output = root / 'profiles/batch_sweep_20261104'
    output.mkdir(exist_ok=True, parents=True)
    meta = json.loads((root / 'assets/export_complete.json').read_text())
    base = load_tree(profile['musa']['base_export'])
    memory = load_tree(profile['musa']['memory_initial'])
    model = PercMemPi05Torch(base, memory, meta['memory_config']).to(device)
    del base, memory
    for i, block in enumerate(model.blocks):
        model.blocks[i] = FSDP(block, device_id=device, use_orig_params=True)
    model = FSDP(model, device_id=device, use_orig_params=True, ignored_modules=[model.vision])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=lr_at(profile, 3), betas=(.9, .95), eps=1e-8, weight_decay=0., foreach=False)
    ema = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    checkpoint = root / 'runs/ckpts/mme_vla_suite/percmem-v1-e2e-lerobot-s42/e2e_80k_20261104_preflight/step_000003'
    assert restore_local(model, optimizer, ema, checkpoint) == 3
    dataset = TrainingDataset(profile)
    rows = next(iter(GlobalBatchSampler(len(dataset), rank, 4, 64, 16, 42, start_step=3)))
    raw = collate([dataset[i] for i in rows])
    keys, noise, times = global_random(42, 3, 64)
    begin = rank * 16
    batch = prepare_batch(raw, keys[begin:begin + 16], device)
    del raw
    noise = torch.from_numpy(noise[begin:begin + 16].copy()).to(device)
    times = torch.from_numpy(times[begin:begin + 16].copy()).to(device)
    counts = batch['perc_mem_frame_mask'].sum(1).cpu().tolist()
    print('FIXTURE', json.dumps({'rank': rank, 'histories': counts}), flush=True)
    baseline_memory = Memory.forward
    def uniform_memory(self, native, ticks, valid, now):
        count = valid.sum(1)
        slots = torch.arange(32, device=native.device)[None]
        ids = torch.div(slots * (count - 1).clamp_min(0)[:, None], 31, rounding_mode='floor')
        ids = torch.where(count[:, None] < 32, slots.clamp_max((count - 1).clamp_min(0)[:, None]), ids)
        cfg = self.cfg
        self.cfg = {**cfg, 'stage': 'teacher'}
        try:
            return baseline_memory(self, e2e.gather_frames(native, ids), e2e.gather_frames(ticks, ids), slots < count[:, None], now)
        finally:
            self.cfg = cfg
    def trim_memory(self, native, ticks, valid, now):
        end = max(1, int(valid.sum(1).max().item()))
        return baseline_memory(self, native[:, :end], ticks[:, :end], valid[:, :end], now)
    cases = [('e2e', n) for n in [2, 4, 8, 16]] + [('trim_padding', 16), ('uniform32_same_reader', 16)]
    results = []
    for variant, micro in cases:
        Memory.forward = baseline_memory if variant == 'e2e' else trim_memory if variant == 'trim_padding' else uniform_memory
        accumulation = 16 // micro
        trials = []
        for trial in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.musa.empty_cache()
            torch.musa.reset_peak_memory_stats()
            dist.barrier(); torch.musa.synchronize()
            started = time.perf_counter()
            total_loss = torch.zeros((), device=device)
            for j in range(accumulation):
                start, end = j * micro, (j + 1) * micro
                current = {k: v[start:end] for k, v in batch.items()}
                sync = contextlib.nullcontext() if j == accumulation - 1 else model.no_sync()
                with sync:
                    loss = model(current, noise[start:end], times[start:end])
                    (loss / accumulation).backward()
                total_loss += loss.detach() / accumulation
            norms = exact_grad_norm(model, device, 1.)
            optimizer.step()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        ema[name].mul_(.999).add_(parameter.detach(), alpha=.001)
            dist.all_reduce(total_loss); total_loss /= 4
            torch.musa.synchronize()
            seconds = time.perf_counter() - started
            elapsed = torch.tensor(seconds, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            row = {'variant': variant, 'micro_batch': micro, 'accumulation': accumulation,
                   'global_batch': 64, 'trial': trial, 'seconds_slowest_rank': float(elapsed.cpu()),
                   'loss': float(total_loss.cpu()), 'grad_norm': norms[0],
                   'peak_allocated_gib': torch.musa.max_memory_allocated() / 2**30,
                   'rank': rank, 'timing_scope': 'resident input compute incl optimizer/EMA; no data/H2D'}
            trials.append(row)
            if rank == 0:
                print('TRIAL', json.dumps(row), flush=True)
            (output / ('rank%d_trials.json' % rank)).write_text(json.dumps(results + trials, indent=2))
        results += trials
    if rank == 0:
        summary = []
        for variant, micro in cases:
            measured = [r for r in results if r['variant'] == variant and r['micro_batch'] == micro and r['trial'] > 0]
            summary.append({'variant': variant, 'micro_batch': micro, 'accumulation': 16 // micro,
                            'seconds': statistics.mean(r['seconds_slowest_rank'] for r in measured),
                            'peak_allocated_gib_rank0': max(r['peak_allocated_gib'] for r in measured)})
        report = {'passed': True, 'scope': 'fixed real global64 input, same MUSA/FSDP/backend; diagnostic trials only',
                  'results': summary, 'physical_gpus': [4, 5, 6, 7],
                  'ablations': 'trim drops trailing padding only; uniform32 changes policy and is not FrameSamp+Modul'}
        (output / 'summary.json').write_text(json.dumps(report, indent=2)); print('SUMMARY', json.dumps(report), flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__':
    main()
