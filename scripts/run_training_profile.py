"""Resolve a reviewable MME-VLA YAML profile; train only with --execute."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

import yaml


PROJECT = Path(__file__).resolve().parents[1]


def configure_environment(runtime, execute):
    if runtime["cuda_visible_devices"] != [0, 1, 2, 3]:
        raise ValueError("This reviewed profile must use physical GPUs 0,1,2,3")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["JAX_PLATFORMS"] = "cuda" if execute else "cpu"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(runtime["xla_python_client_mem_fraction"])
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(runtime["xla_python_client_preallocate"]).lower()
    os.environ["OMP_NUM_THREADS"] = str(runtime["omp_num_threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(runtime["openblas_num_threads"])
    os.environ["OPENPI_DATA_HOME"] = runtime["openpi_data_home"]


def assert_subset(expected, actual, prefix="memory"):
    for key, value in expected.items():
        if key not in actual:
            raise ValueError(f"Missing {prefix}.{key}")
        if isinstance(value, dict):
            assert_subset(value, actual[key], f"{prefix}.{key}")
        elif actual[key] != value:
            raise ValueError(f"{prefix}.{key}: expected {value!r}, found {actual[key]!r}")


def build_config(profile):
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.training.config import AssetsConfig, DataConfig, RoboMMEDataConfig, get_config
    from openpi.training.optimizer import AdamW, CosineDecaySchedule
    from openpi.training.weight_loaders import CheckpointWeightLoader
    from omegaconf import OmegaConf

    base = get_config(profile["base_config"])
    model = dataclasses.replace(base.model, **profile["model"])
    if profile["freeze"] != "vision_only" or "lora" in model.paligemma_variant or "lora" in model.action_expert_variant:
        raise ValueError("Expected full fine-tuning with only the vision encoder frozen")
    memory = get_history_config(model.history_config)
    assert_subset(profile["expected_memory"], OmegaConf.to_container(memory, resolve=True))
    lr = profile["learning_rate"]
    if lr["schedule"] != "warmup_then_constant":
        raise ValueError("Only the reviewed warmup-then-constant schedule is supported")
    data = profile["data"]
    config = dataclasses.replace(
        base,
        **profile["train"],
        dataset_path=data["dataset_path"],
        model=model,
        data=RoboMMEDataConfig(
            repo_id=data["repo_id"],
            assets=AssetsConfig(assets_dir=data["norm_assets_dir"], asset_id=data["norm_asset_id"]),
            base_config=DataConfig(prompt_from_task=True),
        ),
        freeze_filter=model.get_freeze_filter(),
        weight_loader=CheckpointWeightLoader(profile["weights"]["params_path"]),
        lr_schedule=CosineDecaySchedule(
            warmup_steps=lr["warmup_steps"], peak_lr=lr["value"],
            decay_steps=profile["train"]["num_train_steps"], decay_lr=lr["value"],
        ),
        optimizer=AdamW(**profile["optimizer"]),
    )
    if config.batch_size % 4 or config.fsdp_devices != 4:
        raise ValueError("Expected four-way FSDP and a global batch divisible by four")
    if config.resume or config.overwrite or config.resum_ckpt_id is not None:
        raise ValueError("This profile is a fresh run; upstream does not save full optimizer state")
    if config.wandb_enabled:
        raise ValueError("Upstream W&B entity is author-specific; configure your entity before enabling it")
    return config, memory


def review_config(profile, config, memory):
    from omegaconf import OmegaConf

    cache = Path(config.dataset_path)
    checks, blockers = {}, []
    for name, path in {
        "jax_base_params": Path(profile["weights"]["params_path"]),
        "complete_feature_cache": cache / "meta/features_complete.json",
        "normalization": Path(profile["data"]["norm_assets_dir"]) / profile["data"]["norm_asset_id"] / "norm_stats.json",
    }.items():
        present = path.is_dir() if name == "jax_base_params" else path.is_file()
        checks[name] = {"path": str(path), "present": present}
        if not present:
            blockers.append(f"Missing {name}: {path}")
    metadata_path = cache / "meta/lerobot_framesamp.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        data = profile["data"]
        for key, expected in [("total_frames", data["expected_frames"]),
                              ("execution_samples", data["expected_execution_samples"])]:
            if metadata[key] != expected:
                raise ValueError(f"Dataset {key} mismatch")
        if len(metadata["episodes"]) != data["expected_episodes"]:
            raise ValueError("Dataset episode count mismatch")
        checks["dataset"] = {"source": metadata["source"], "episodes": len(metadata["episodes"]),
                             "frames": metadata["total_frames"], "execution_samples": metadata["execution_samples"]}
    else:
        blockers.append(f"Missing LeRobot cache index: {metadata_path}")
    if config.checkpoint_dir.exists():
        blockers.append(f"Fresh-run checkpoint directory already exists: {config.checkpoint_dir}")
    cache_link = Path.home() / ".cache" / f"jax_{config.exp_name}"
    cache_target = Path(profile["runtime"]["compilation_cache_dir"])
    if cache_link.is_symlink():
        if cache_link.resolve() != cache_target.resolve():
            blockers.append(f"Compilation-cache symlink has a different target: {cache_link}")
    elif cache_link.exists():
        blockers.append(f"Compilation-cache path already exists and is not the planned symlink: {cache_link}")
    schedule = config.lr_schedule.create()
    saved_steps = [step for step in range(config.num_train_steps)
                   if (step > 0 and step % config.save_interval == 0) or step == config.num_train_steps - 1]
    return {
        "profile_name": profile["profile_name"], "mode": "configuration_review_only",
        "training_started": False, "runtime": profile["runtime"],
        "base_config_name": config.name, "model": dataclasses.asdict(config.model),
        "memory": OmegaConf.to_container(memory, resolve=True),
        "train": profile["train"], "dataset_path": config.dataset_path,
        "global_batch_size": config.batch_size, "samples_per_device": config.batch_size // 4,
        "freeze": "vision encoder only; VLM, action expert and memory parameters trainable",
        "optimizer": dataclasses.asdict(config.optimizer),
        "learning_rate": dataclasses.asdict(config.lr_schedule),
        "learning_rate_samples": {str(step): float(schedule(step)) for step in [0, 9999, 10000, 40000, 79999]},
        "checkpoint_dir": str(config.checkpoint_dir), "actual_checkpoint_ids": saved_steps,
        "evaluation_checkpoint_ids": saved_steps[-3:], "evaluation_plan": profile["evaluation"],
        "compilation_cache_binding": {"upstream_path": str(cache_link), "symlink_target": str(cache_target),
                                      "created_only_on_execute": True},
        "checks": checks, "execution_blockers": blockers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Start training; default only prints the resolved configuration")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    profile = yaml.safe_load(args.config.read_text())
    configure_environment(profile["runtime"], args.execute)
    sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "packages/openpi-client/src")]
    os.chdir(PROJECT)
    config, memory = build_config(profile)
    report = review_config(profile, config, memory)
    report["execution_requested"] = args.execute
    if args.execute:
        report["mode"] = "training_preflight"
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text, flush=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n")
    if not args.execute:
        return
    if report["execution_blockers"]:
        raise SystemExit("Execution blocked:\n" + "\n".join(report["execution_blockers"]))
    import jax
    if len(jax.devices("gpu")) != 4:
        raise RuntimeError("Expected exactly four visible CUDA devices mapped from physical GPUs 0-3")
    binding = report["compilation_cache_binding"]
    target, link = Path(binding["symlink_target"]), Path(binding["upstream_path"])
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink():
        link.symlink_to(target, target_is_directory=True)
    import train
    # Bypass the upstream __main__ path that runs and discards a tentative training pass.
    train.main(config, tentative_run=False)


if __name__ == "__main__":
    main()
