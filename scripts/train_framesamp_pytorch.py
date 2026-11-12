"""Train RoboMME FrameSamp+Modul with OpenPI PyTorch and the released JAX profile."""

import argparse
from contextlib import nullcontext
import dataclasses
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs/train/robomme_framesamp_modul_pytorch.yaml"


def merge(config, changes):
    for key, value in changes.items():
        if key not in config:
            raise ValueError(f"Unknown setting {key}")
        if isinstance(value, dict):
            merge(config[key], value)
        else:
            config[key] = value


def resolve(path, overrides):
    overlay = yaml.safe_load(path.read_text())
    if "reference_profile" in overlay:
        base = yaml.safe_load((ROOT / overlay.pop("reference_profile")).read_text())
    elif "pytorch" in overlay:
        # A fully resolved PyTorch profile is portable and preserves its explicit settings.
        base, overlay = overlay, {}
    else:
        # A JAX profile can be passed directly. Migrate baseline defaults, preserving
        # explicit changes to e.g. the GPU list, seed grid and checkpoint directory.
        base, defaults = overlay, yaml.safe_load(DEFAULT.read_text())
        reference = yaml.safe_load((ROOT / defaults.pop("reference_profile")).read_text())
        overlay = {"pytorch": defaults.pop("pytorch")}
        for section, values in defaults.items():
            overlay[section] = {k: v for k, v in values.items() if base[section].get(k) == reference[section].get(k)}
    base["pytorch"] = {
        "activation_checkpointing": True,
        "memory_init": None,
        "benchmark_warmup": 16,
        "benchmark_steps": 100,
        **base.get("pytorch", {}),
    }
    base["train"].setdefault("checkpoint_max_to_keep", 1)
    merge(base, overlay)
    for item in overrides:
        name, sep, raw = item.partition("=")
        if not sep:
            raise ValueError("--set requires section.key=value")
        node = base
        for key in name.split(".")[:-1]:
            node = node[key]
        merge(node, {name.split(".")[-1]: yaml.safe_load(raw)})
    t, p, rt = base["train"], base["pytorch"], base["runtime"]
    devices = rt["cuda_visible_devices"]
    if not devices or len(set(devices)) != len(devices) or any(type(i) is not int or i < 0 for i in devices):
        raise ValueError("cuda_visible_devices must be distinct nonnegative GPU indices")
    if t["fsdp_devices"] != len(devices) or t["batch_size"] % len(devices):
        raise ValueError("Use one FSDP group and a global batch divisible by its size")
    if any(
        type(t[k]) is not int or t[k] <= 0
        for k in ("batch_size", "num_train_steps", "save_interval", "log_interval", "checkpoint_max_to_keep")
    ):
        raise ValueError("Batch, steps, logging, saving and retention must be positive integers")
    if t["keep_period"] is not None and (type(t["keep_period"]) is not int or t["keep_period"] < 1):
        raise ValueError("keep_period must be positive or null")
    if t["num_workers"] < 0 or not 0 <= t["ema_decay"] < 1 or p["benchmark_warmup"] < 0 or p["benchmark_steps"] < 1:
        raise ValueError("Invalid worker count, EMA decay or timing window")
    if t["resume"] or t["resum_ckpt_id"] is not None:
        raise ValueError(
            "This profile saves inference EMA/normalization only; optimizer resume requires full training state"
        )
    if t["wandb_enabled"]:
        raise ValueError("This runner logs JSON to stdout; keep wandb_enabled=false")
    if base["freeze"] != "vision_only" or base["learning_rate"]["schedule"] != "warmup_then_constant":
        raise ValueError("Unsupported freeze filter or schedule")
    if base["learning_rate"]["warmup_steps"] < 0 or base["learning_rate"]["value"] <= 0:
        raise ValueError("Invalid learning rate or warmup")
    return base


def build(profile):
    from omegaconf import OmegaConf
    from run_training_profile import assert_subset

    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.training.config import AssetsConfig
    from mme_vla_suite.training.config import DataConfig
    from mme_vla_suite.training.config import RoboMMEDataConfig
    from mme_vla_suite.training.config import get_config

    base = get_config(profile["base_config"])
    model = dataclasses.replace(base.model, **profile["model"])
    if not (
        model.pi05
        and model.use_history
        and not model.discrete_state_input
        and model.action_dim == 32
        and model.action_horizon == 20
        and model.dtype in ("bfloat16", "float32")
        and model.paligemma_variant == "gemma_2b"
        and model.action_expert_variant == "gemma_300m"
        and model.history_config == "perceptual-framesamp-modul.yaml"
    ):
        raise ValueError("Expected the released π0.5 FrameSamp+Modul model and physical action representation")
    memory = get_history_config(model.history_config)
    assert_subset(profile["expected_memory"], OmegaConf.to_container(memory, resolve=True))
    data = profile["data"]
    factory = RoboMMEDataConfig(
        repo_id=data["repo_id"],
        assets=AssetsConfig(assets_dir=data["norm_assets_dir"], asset_id=data["norm_asset_id"]),
        base_config=DataConfig(prompt_from_task=True),
    )
    return model, memory, factory.create(Path(profile["train"]["assets_base_dir"]), model)


def lr_at(profile, step):
    lr = profile["learning_rate"]
    warmup, peak = lr["warmup_steps"], lr["value"]
    if step >= warmup:
        return peak
    start = peak / (warmup + 1)
    return start + (peak - start) * step / warmup


def ckpt_root(profile):
    t = profile["train"]
    return Path(t["checkpoint_base_dir"]) / profile["base_config"] / t["exp_name"]


def should_save(step, train):
    return (step > 0 and step % train["save_interval"] == 0) or step + 1 == train["num_train_steps"]


def log(**items):
    print(json.dumps(items, ensure_ascii=False), flush=True)


def run(profile, mode, smoke_dir):
    import numpy as np
    import torch
    from torch import distributed as dist
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp.wrap import CustomPolicy

    from mme_vla_suite.models.framesamp_pytorch import FrameSampPolicy
    from mme_vla_suite.models.framesamp_pytorch import JointBlock
    from mme_vla_suite.training.framesamp_torch_data import FrameSampDataset
    from mme_vla_suite.training.framesamp_torch_data import make_loader
    from mme_vla_suite.training.framesamp_weights import load_orbax
    from mme_vla_suite.training.framesamp_weights import orbax_tree
    from openpi.shared import normalize
    from openpi.training import checkpoints

    t, pt = profile["train"], profile["pytorch"]
    model_config, memory, dc = build(profile)
    dataset = FrameSampDataset(profile, model_config, memory, dc)
    weights = Path(profile["weights"]["params_path"])
    if not (weights / "_METADATA").is_file() or not Path(pt["memory_init"]).is_file():
        raise FileNotFoundError("Base Orbax parameters or seeded FrameSamp memory initialization missing")
    if mode == "check":
        sample, _ = dataset[0, 0]
        with np.load(pt["memory_init"]) as init:
            if int(init["seed"]) != t["seed"] or len(init.files) != 11:
                raise ValueError("Invalid seeded memory initialization")
        log(
            samples=len(dataset),
            shapes={k: list(v.shape) for k, v in sample.items()},
            checkpoint_dir=str(ckpt_root(profile)),
            global_batch=t["batch_size"],
            samples_per_gpu=t["batch_size"] // t["fsdp_devices"],
            checkpoint_ids=[s for s in range(t["num_train_steps"]) if should_save(s, t)],
        )
        return
    cpu = mode == "cpu_smoke"
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not cpu and (world != t["fsdp_devices"] or torch.cuda.device_count() != world):
        raise ValueError("Visible CUDA device count/world size differs from the YAML")
    device = torch.device("cpu" if cpu else f"cuda:{local_rank}")
    if not cpu:
        torch.cuda.set_device(device)
    multi = world > 1
    if multi:
        dist.init_process_group(
            "gloo" if cpu else "nccl", timeout=timedelta(minutes=30), device_id=None if cpu else device
        )
    torch.manual_seed(t["seed"])
    manager = None
    try:
        root = smoke_dir.resolve() if cpu else ckpt_root(profile)
        if rank == 0 and mode != "benchmark":
            manager, _ = checkpoints.initialize_checkpoint_dir(
                root,
                keep_period=t["keep_period"],
                max_to_keep=t["checkpoint_max_to_keep"],
                overwrite=False if cpu else t["overwrite"],
                resume=False,
            )
            (root / "history_config.txt").write_text(model_config.history_config)
            (root / "pytorch_profile.json").write_text(
                json.dumps(
                    {
                        **profile,
                        "execution": {
                            "mode": mode,
                            "world_size": world,
                            "global_batch": world if cpu else t["batch_size"],
                            "num_train_steps": 1 if cpu else t["num_train_steps"],
                        },
                    },
                    indent=2,
                )
            )
        if multi:
            dist.barrier()
        model = FrameSampPolicy(model_config, memory, activation_checkpointing=pt["activation_checkpointing"])
        frozen = [model.vision, model.projector]
        for module in frozen:
            module.to(dtype=torch.bfloat16)
        if rank == 0 or cpu:
            load_orbax(model, weights, memory_init=pt["memory_init"], seed=t["seed"])
        if multi:
            # Sync ignored (replicated, frozen) vision tensors explicitly; FSDP syncs the other weights.
            for module in frozen:
                module.to(device)
                for param in module.parameters():
                    dist.broadcast(param.data, 0)
            mp = MixedPrecision(param_dtype=model.compute_dtype, reduce_dtype=torch.float32)

            def wrap_policy(module):
                if isinstance(module, JointBlock):
                    return {"mixed_precision": mp}
                return isinstance(module, torch.nn.Embedding)

            model = FSDP(
                model,
                device_id=device,
                auto_wrap_policy=CustomPolicy(wrap_policy),
                use_orig_params=True,
                sync_module_states=not cpu,
                ignored_modules=frozen,
                limit_all_gathers=True,
            )
        else:
            model.to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        ema = [p.detach().clone() for p in trainable]
        opt = profile["optimizer"]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=lr_at(profile, 0),
            betas=(opt["b1"], opt["b2"]),
            eps=opt["eps"],
            weight_decay=opt["weight_decay"],
            fused=not cpu,
            foreach=False,
        )
        benchmark = mode == "benchmark"
        steps = pt["benchmark_warmup"] + pt["benchmark_steps"] if benchmark else t["num_train_steps"]
        if cpu:
            steps = 1
        data_profile = {**profile, "train": {**t, "batch_size": world, "num_workers": 0}} if cpu else profile
        loader = iter(make_loader(dataset, data_profile, rank, world, steps))
        elapsed_steps, losses = [], []
        last = time.perf_counter()
        for step in range(steps):
            if benchmark and not cpu:
                torch.cuda.synchronize()
                start = time.perf_counter()
            batch = {k: v.to(device, non_blocking=True) for k, v in next(loader).items()}
            optimizer.param_groups[0]["lr"] = lr_at(profile, step)
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch, batch["noise"], batch["time"]).mean()
            loss.backward()
            if multi:
                norm = model.clip_grad_norm_(opt["clip_gradient_norm"])
            else:
                norm = torch.nn.utils.clip_grad_norm_(trainable, opt["clip_gradient_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Non-finite gradient norm at step {step}")
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(ema, trainable, 1 - t["ema_decay"])
            losses.append(loss.detach())
            if benchmark:
                torch.cuda.synchronize()
                duration = torch.tensor(time.perf_counter() - start, device=device)
                if multi:
                    dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                if step >= pt["benchmark_warmup"]:
                    elapsed_steps.append(duration.item())
            if step % t["log_interval"] == 0 or step + 1 == steps:
                average = torch.stack(losses).mean()
                if multi:
                    dist.all_reduce(average, op=dist.ReduceOp.AVG)
                if rank == 0:
                    log(
                        step=step,
                        loss=average.item(),
                        grad_norm=norm.item(),
                        lr=lr_at(profile, step),
                        interval_seconds=time.perf_counter() - last,
                        updates_in_interval=len(losses),
                    )
                last, losses = time.perf_counter(), []
            if not benchmark and (cpu or should_save(step, t)):
                with torch.no_grad():
                    live = [p.detach().to("cpu", copy=True) for p in trainable]
                    for p, avg in zip(trainable, ema, strict=True):
                        p.copy_(avg)
                    ctx = (
                        FSDP.state_dict_type(
                            model,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(offload_to_cpu=not cpu, rank0_only=True),
                        )
                        if multi
                        else nullcontext()
                    )
                    try:
                        with ctx:
                            state = model.state_dict()
                        if rank == 0:
                            tree = orbax_tree(state)

                            def assets(path):
                                normalize.save(path / dc.asset_id, dc.norm_stats)

                            manager.save(step, {"params": tree, "assets": assets})
                            manager.wait_until_finished()
                            log(saved=str(root / str(step)))
                            del tree
                        del state
                    finally:
                        for p, value in zip(trainable, live, strict=True):
                            p.copy_(value)
                        del live
                if multi:
                    dist.barrier()
        if benchmark and rank == 0:
            log(
                backend="openpi_pytorch",
                global_batch=t["batch_size"],
                measured_steps=len(elapsed_steps),
                warmup_excluded=pt["benchmark_warmup"],
                seconds_per_step=statistics.mean(elapsed_steps),
                median_seconds=statistics.median(elapsed_steps),
                p90_seconds=float(np.percentile(elapsed_steps, 90)),
                samples_per_second=t["batch_size"] * len(elapsed_steps) / sum(elapsed_steps),
                device=torch.cuda.get_device_name(device),
                torch_version=torch.__version__,
            )
    finally:
        if manager is not None:
            manager.close()
        if multi:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--set", action="append", default=[])
    modes = parser.add_mutually_exclusive_group()
    for flag in ("dry-run", "check", "benchmark", "execute"):
        modes.add_argument("--" + flag, dest="mode", action="store_const", const=flag.replace("-", "_"))
    modes.add_argument(
        "--cpu-smoke",
        type=Path,
        metavar="NEW_CHECKPOINT_DIR",
        help="One batch-1 CPU update, including EMA and Orbax checkpoint",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    mode = "cpu_smoke" if args.cpu_smoke else args.mode or "dry_run"
    profile = resolve(args.config.resolve(), args.set)
    smoke = args.cpu_smoke.resolve() if args.cpu_smoke else None
    rt = profile["runtime"]
    os.environ.update(
        JAX_PLATFORMS="cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        OPENPI_DATA_HOME=rt["openpi_data_home"],
        OPENBLAS_NUM_THREADS=str(rt["openblas_num_threads"]),
        OMP_NUM_THREADS=str(rt["omp_num_threads"]),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES=",".join(map(str, rt["cuda_visible_devices"])) if mode in ("execute", "benchmark") else "",
    )
    os.environ["PYTHONPATH"] = os.pathsep.join(str(ROOT / p) for p in ("scripts", "src", "packages/openpi-client/src"))
    sys.path[:0] = os.environ["PYTHONPATH"].split(os.pathsep)
    if mode in ("execute", "benchmark") and not args.worker and len(rt["cuda_visible_devices"]) > 1:
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={len(rt['cuda_visible_devices'])}",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--worker",
            ],
        )
    os.chdir(ROOT)
    if mode == "dry_run":
        print(yaml.safe_dump(profile, sort_keys=False))
        return
    run(profile, mode, smoke)


if __name__ == "__main__":
    main()
