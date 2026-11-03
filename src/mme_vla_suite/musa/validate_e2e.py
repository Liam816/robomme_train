"""Compare the b4432b1 replay and its complete gradient against JAX."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx, traverse_util

from mme_vla_suite.models.integration.percmem_pi05 import _identity_pad, _COND_KERNEL_INIT
from mme_vla_suite.models.representation.perc_memory_e2e import (
    hard_eviction as jax_hard, select_members as jax_select, replay_memory as jax_replay,
)
from mme_vla_suite.models.representation.perc_memory_v1 import EvictScorer, MemReader, PercMemConfig
from mme_vla_suite.musa.e2e import E2EScorer, HardEviction, SelectMembers, replay_memory
from mme_vla_suite.musa.model import Memory, flatten

ROOT = Path(os.environ["ROBOMME_MUSA_ROOT"])
report = {}
device = torch.device("cpu")
output = ROOT / "e2e_validation_cpu.json"


def array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def check(name, actual, expected, atol=3e-5, rtol=2e-3, exact=False):
    a, b = array(actual), array(expected)
    good = bool(np.array_equal(a, b) if exact else np.allclose(a, b, atol=atol, rtol=rtol))
    value = {"passed": good, "max_abs_error": float(np.max(np.abs(a - b))), "exact": exact,
             "atol": atol, "rtol": rtol}
    report[name] = value
    output.write_text(json.dumps(report, indent=2))
    if not good:
        raise AssertionError(name + ": " + str(value))


def tensor(x, *, grad=False):
    value = torch.from_numpy(np.asarray(x).copy()).to(device)
    return value.requires_grad_(grad)


def parameters(module):
    return flatten(nnx.state(module, nnx.Param).to_pure_dict())


class ReferenceMemory(nnx.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        rngs = nnx.Rngs(42)
        self.mem_feature_adapter = nnx.Linear(cfg.native_dim, cfg.feature_dim, kernel_init=_identity_pad, rngs=rngs)
        self.mem_reader = MemReader(cfg, rngs=rngs)
        self.mem_cond_adapter = nnx.Linear(cfg.output_dim, 1024, use_bias=False, kernel_init=_COND_KERNEL_INIT, rngs=rngs)
        self.mem_scorer = EvictScorer(cfg, rngs=rngs)

    def __call__(self, x, ticks, valid, now):
        valid = valid & (ticks <= now[:, None])
        x = jnp.where(valid[..., None, None], x, 0)
        features = self.mem_feature_adapter(x.astype(jnp.float32)).astype(jnp.float16)
        selected, ticks, valid, _ = jax_replay(self.mem_scorer, features, ticks, valid, now, self.cfg)
        return self.mem_cond_adapter(self.mem_reader(selected.astype(jnp.float32), ticks, valid, now))


def main():
    global device, output
    cli = argparse.ArgumentParser()
    cli.add_argument("--device", choices=["cpu", "musa"], default="cpu")
    args = cli.parse_args()
    if args.device == "musa":
        import torch_musa
        assert os.environ["MUSA_VISIBLE_DEVICES"] in {"4", "4,5,6,7"}
        torch.musa.set_device(0)
        device = torch.device("musa", 0)
        torch.backends.mudnn.allow_tf32 = False
        torch.backends.musa.matmul.allow_tf32 = False
        output = ROOT / "e2e_validation_musa.json"
    report["device"] = str(device)
    report["source_commit"] = "b4432b1ef2ab746b13eec32f25bcc5862c582164"
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    rng = np.random.default_rng(42)
    cfg = dataclasses.replace(PercMemConfig(), native_dim=32, feature_dim=40, output_dim=32,
        store_grid=4, read_grid=2, score_grid=2, age_dim=8, scorer_dim=16, scorer_ffn=64,
        reader_dim=16, reader_ffn=64, reader_queries=2, max_frames=12, recent_frames=2,
        queue_frames=3, stage="e2e")

    scores = rng.normal(size=(2, 4)).astype(np.float32)
    coefficient = rng.normal(size=scores.shape).astype(np.float32)
    ts = tensor(scores, grad=True)
    hard = HardEviction.apply(ts)
    check("hard_forward", hard, jax_hard(jnp.asarray(scores)), exact=True)
    (hard * tensor(coefficient)).sum().backward()
    check("hard_gradient", ts.grad,
          jax.grad(lambda s: (jax_hard(s) * jnp.asarray(coefficient)).sum())(jnp.asarray(scores)))

    features = rng.normal(size=(2, 7, 2, 3)).astype(np.float32)
    ids = np.array([[0, 2, 4], [1, 3, 0]], np.int32)
    valid = np.array([[True, True, True], [True, True, False]])
    membership = np.eye(7, dtype=np.float32)[ids]
    coefficient = rng.normal(size=(2, 3, 2, 3)).astype(np.float32)
    tx, tm = tensor(features, grad=True), tensor(membership, grad=True)
    result = SelectMembers.apply(tx, tm, tensor(ids), tensor(valid))
    expected = jax_select(jnp.asarray(features), jnp.asarray(membership), jnp.asarray(ids), jnp.asarray(valid))
    check("select_forward", result, expected, exact=True)
    (result * tensor(coefficient)).sum().backward()
    jf, jm = jax.grad(lambda f, m: (jax_select(f, m, jnp.asarray(ids), jnp.asarray(valid)) *
        jnp.asarray(coefficient)).sum(), argnums=(0, 1))(jnp.asarray(features), jnp.asarray(membership))
    check("select_feature_gradient", tx.grad, jf)
    check("select_membership_gradient", tm.grad, jm)

    scorer = EvictScorer(cfg, rngs=nnx.Rngs(8))
    port = E2EScorer(parameters(scorer), dataclasses.asdict(cfg)).to(device)
    feats = rng.normal(size=(3, 12, 16, 40)).astype(np.float16)
    ticks = np.broadcast_to(np.arange(12, dtype=np.int32)[None] * 10, (3, 12)).copy()
    valid = np.array([[True] * 12,
        [True, False, True, True, True, False, True, True, True, True, True, True], [False] * 12])
    now = np.array([115, 85, 115], np.int32)
    jf = scorer.encode_frames(jnp.asarray(feats, dtype=jnp.float32))
    tf = port.encode_frames(tensor(feats).float())
    check("scorer_frame_encoder", tf, jf)
    full_mask = np.ones((3, 12), bool)
    is_new = np.broadcast_to(np.arange(12)[None] == 11, (3, 12)).copy()
    js = scorer.score_frames(jf, jnp.asarray(ticks), jnp.asarray(full_mask), jnp.asarray(now), jnp.asarray(is_new))
    ts = port.score_frames(tf, tensor(ticks), tensor(full_mask), tensor(now), tensor(is_new))
    check("scorer_context", ts, js)
    target = rng.normal(size=(3, 12)).astype(np.float32)
    def scorer_loss(module):
        values = module(jnp.asarray(feats, dtype=jnp.float32), jnp.asarray(ticks),
                        jnp.asarray(full_mask), jnp.asarray(now), jnp.asarray(is_new))
        return jnp.mean((values - jnp.asarray(target)) ** 2)
    jl, jg = nnx.value_and_grad(scorer_loss)(scorer)
    tl = (ts - tensor(target)).square().mean()
    tl.backward()
    check("scorer_loss", tl, jl)
    for key, gradient in flatten(jg.to_pure_dict()).items():
        check("scorer_gradient/" + key, port.p(key).grad, gradient, atol=8e-5, rtol=4e-3)

    port.zero_grad(set_to_none=True)
    inputs = [tensor(v) for v in (feats, ticks, valid, now)]
    actual = replay_memory(port, *inputs, dataclasses.asdict(cfg))
    reference = jax.jit(lambda x, t, v, z: jax_replay(scorer, x, t, v, z, cfg))(
        *[jnp.asarray(v) for v in (feats, ticks, valid, now)])
    for name, a, b in zip(["tokens", "ticks", "mask", "ids"], actual, reference):
        check("replay_" + name, a, b, exact=True)
    coefficient = rng.normal(size=actual[0].shape).astype(np.float32)
    def replay_loss(module):
        selected = jax_replay(module, jnp.asarray(feats), jnp.asarray(ticks), jnp.asarray(valid), jnp.asarray(now), cfg)[0]
        return (selected.astype(jnp.float32) * jnp.asarray(coefficient)).mean()
    jl, jg = nnx.value_and_grad(replay_loss)(scorer)
    tl = (actual[0].float() * tensor(coefficient)).mean()
    tl.backward()
    check("replay_loss", tl, jl)
    for key, gradient in flatten(jg.to_pure_dict()).items():
        check("replay_gradient/" + key, port.p(key).grad, gradient, atol=1e-4, rtol=5e-3)

    class Table(torch.nn.Module):
        def __init__(self):
            super().__init__()
            values = torch.zeros(12, 4, device=device)
            values[:, -1] = -.25
            self.table = torch.nn.Parameter(values)
        def encode_frames(self, x):
            return x[:, :, 0, :1]
        def score_frames(self, x, ticks, valid, now, is_new):
            return self.table[now.long() // 10]
    table = Table()
    deterministic = np.broadcast_to(np.arange(12)[None, :, None, None], (1, 12, 16, 40)).astype(np.float16)
    selected = replay_memory(table, tensor(deterministic), tensor(ticks[:1]), tensor(full_mask[:1]),
                             tensor(now[:1]), dataclasses.asdict(cfg))[0]
    selected.float().sum().backward()
    check("candidate_rejection", selected, tensor(deterministic[:, [0, 1, 2, 10, 11]]), exact=True)
    assert float(table.table.grad[:5].abs().sum().cpu()) == 0
    assert float(table.table.grad[5].norm().cpu()) > 0 and float(table.table.grad[11].norm().cpu()) > 0
    report["early_write_credit"] = {"passed": True, "first_overflow_gradient_norm": float(table.table.grad[5].norm().cpu())}

    memory = ReferenceMemory(cfg)
    native = rng.normal(size=(3, 12, 16, 32)).astype(np.float16)
    pm = Memory(parameters(memory), dataclasses.asdict(cfg)).to(device)
    mi = [tensor(v) for v in (native, ticks, valid, now)]
    value = pm(*mi)
    check("zero_reader_initialization", value, np.zeros((3, 1024), np.float32), exact=True)
    zero_target = tensor(rng.normal(size=(3, 1024)).astype(np.float32))
    (value-zero_target).square().mean().backward()
    assert float(pm.p('mem_reader/output_final/kernel').grad.norm().cpu())>0
    assert all(p.grad is None or not torch.count_nonzero(p.grad) for p in pm.scorer.parameters())
    perturb = rng.normal(0, .03, size=(32, 32)).astype(np.float32)
    memory.mem_reader.output_final.kernel.value = jnp.asarray(perturb)
    pm = Memory(parameters(memory), dataclasses.asdict(cfg)).to(device)
    target = rng.normal(size=(3, 1024)).astype(np.float32)
    def memory_loss(module):
        out = module(*[jnp.asarray(v) for v in (native, ticks, valid, now)])
        return jnp.mean((out - jnp.asarray(target)) ** 2)
    jl, jg = nnx.value_and_grad(memory_loss)(memory)
    expected = memory(*[jnp.asarray(v) for v in (native, ticks, valid, now)])
    actual = pm(*mi)
    check("memory_condition", actual, expected, atol=1e-4)
    tl = (actual - tensor(target)).square().mean()
    tl.backward()
    check("memory_loss", tl, jl)
    gradients = flatten(jg.to_pure_dict())
    for key in ["mem_feature_adapter/kernel", "mem_reader/output_final/kernel", "mem_cond_adapter/kernel"]:
        check("memory_gradient/" + key, pm.p(key).grad, gradients[key], atol=1e-4, rtol=5e-3)
    for key in pm.scorer.names:
        check("memory_gradient/mem_scorer/" + key, pm.scorer.p(key).grad, gradients["mem_scorer/" + key], atol=1e-4, rtol=5e-3)
    assert sum(float(p.grad.float().square().sum().cpu()) for p in pm.scorer.parameters()) > 0

    pm = Memory(parameters(ReferenceMemory(cfg)), dataclasses.asdict(cfg)).to(device)
    initial = {n: p.detach().clone() for n, p in pm.scorer.named_parameters()}
    optimizer = torch.optim.AdamW(pm.parameters(), lr=1e-3, betas=(.9, .95), weight_decay=0, foreach=False)
    losses = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (pm(*mi) - tensor(target)).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    delta = sum(float((p.detach() - initial[n]).square().sum().cpu()) for n, p in pm.scorer.named_parameters()) ** .5
    assert delta > 0 and all(np.isfinite(losses))
    report["controlled_optimizer"] = {"passed": True, "scope": "memory objective; diagnostic lr1e-3", "losses": losses,
                                      "scorer_parameter_delta_l2": delta}
    report["passed"] = True
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
