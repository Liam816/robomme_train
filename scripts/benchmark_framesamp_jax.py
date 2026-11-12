"""Measure the released JAX training step and real input pipeline without saving checkpoints."""

import argparse
import functools
import json
import os
from pathlib import Path
import statistics
import sys
import time

import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/train/robomme_framesamp_modul_repro.yaml")
    parser.add_argument("--gpus", type=int, nargs=4)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error("A measured window and at least one compilation/warmup step are required")
    profile = yaml.safe_load(args.config.resolve().read_text())
    cache = args.cache.resolve()
    os.chdir(root)
    sys.path[:0] = [str(root / "src"), str(root / "packages/openpi-client/src")]
    from run_training_profile import build_config
    from run_training_profile import configure_environment

    if args.gpus:
        profile["runtime"]["cuda_visible_devices"] = args.gpus
    configure_environment(profile["runtime"], True)

    import jax
    import numpy as np
    import train

    from mme_vla_suite.training.dataloader import create_data_loader
    from openpi.training import sharding

    cfg, _ = build_config(profile)
    if len(jax.devices("gpu")) != cfg.fsdp_devices:
        raise ValueError("JAX device count does not match the reference profile")
    jax.config.update("jax_compilation_cache_dir", str(cache))
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    batch_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    loader = iter(
        create_data_loader(
            cfg.dataset_path,
            data,
            cfg.model.history_config,
            cfg.model.action_horizon,
            cfg.batch_size,
            sharding=batch_spec,
            shuffle=True,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )
    )
    rng, init_key = jax.random.split(jax.random.key(cfg.seed))
    state, spec = train.init_train_state(cfg, init_key, mesh, resume=False)
    jax.block_until_ready(state)
    update = jax.jit(
        functools.partial(train.train_step, cfg),
        in_shardings=(replicated, spec, batch_spec),
        out_shardings=(spec, replicated, replicated),
        donate_argnums=(1,),
    )
    times, losses = [], []
    for step in range(args.warmup + args.steps):
        begin = time.perf_counter()
        batch = next(loader)
        with sharding.set_mesh(mesh):
            state, info, stats = update(rng, state, batch)
        jax.block_until_ready((state, info, stats))
        elapsed = time.perf_counter() - begin
        if step >= args.warmup:
            times.append(elapsed)
        losses.append(float(info["loss"]))
        if step % cfg.log_interval == 0 or step == args.warmup + args.steps - 1:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": statistics.mean(losses),
                        "last_batch_loss": losses[-1],
                        "grad_norm": float(info["grad_norm"]),
                        "elapsed_seconds": elapsed,
                    }
                ),
                flush=True,
            )
            losses.clear()
    print(
        json.dumps(
            {
                "backend": "reference_jax",
                "global_batch": cfg.batch_size,
                "measured_steps": len(times),
                "warmup_excluded": args.warmup,
                "seconds_per_step": statistics.mean(times),
                "median_seconds": statistics.median(times),
                "p90_seconds": float(np.percentile(times, 90)),
                "samples_per_second": cfg.batch_size * len(times) / sum(times),
                "jax_version": jax.__version__,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
