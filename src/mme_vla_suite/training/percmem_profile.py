"""Resolve the v1 profiles without model construction or training during review."""

import dataclasses
import json
import logging
from pathlib import Path

from flax import traverse_util
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.representation.perc_memory_v1 import PercMemConfig
from mme_vla_suite.shared.percmem_io import tree_hash
from mme_vla_suite.training.config import AssetsConfig, DataConfig, RoboMMEDataConfig, get_config
from openpi.models.model import restore_params
from openpi.training.optimizer import AdamW, CosineDecaySchedule


@dataclasses.dataclass(frozen=True)
class PercMemWeightLoader:
    params_path: str
    stage: str
    scorer_path: str | None = None
    native_cache: str | None = None
    memory_config: PercMemConfig | None = None

    def load(self, reference):
        loaded = restore_params(self.params_path, restore_type=np.ndarray)
        if self.stage == "teacher" and self.native_cache:
            from mme_vla_suite.shared.percmem_io import canonical_vision_params
            cached = json.loads((Path(self.native_cache) / "meta/native_complete.json").read_text())
            if tree_hash(canonical_vision_params(loaded["PaliGemma"]["img"])) != cached["vision_id"]:
                raise ValueError("Initial backbone image weights differ from the native feature cache")
        original_id = tree_hash(loaded) if self.stage == "student" else None
        ref = traverse_util.flatten_dict(reference)
        flat = traverse_util.flatten_dict(loaded)
        missing, extra = ref.keys() - flat.keys(), flat.keys() - ref.keys()
        allowed = {"mem_feature_adapter", "mem_reader", "mem_cond_adapter", "mem_scorer"}
        illegal = [key for key in missing if self.stage != "teacher" or key[0] not in allowed]
        if illegal or extra:
            raise ValueError(f"Checkpoint mismatch: missing={illegal}, unexpected={sorted(extra)}")
        for key in flat:
            if np.shape(flat[key]) != np.shape(ref[key]):
                raise ValueError(f"Checkpoint shape mismatch at {key}")
        logging.info("P0 explicitly initialized new memory leaves: %s", sorted(missing))
        merged = {key: flat[key].astype(ref[key].dtype) if key in flat else ref[key] for key in ref}
        if self.stage == "student":
            if not self.scorer_path:
                raise ValueError("Student initialization requires a trained scorer artifact")
            with np.load(self.scorer_path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["__metadata__"]))
                params = traverse_util.unflatten_dict(
                    {k: archive[k] for k in archive.files if k != "__metadata__"}, sep="/")
            from mme_vla_suite.shared.percmem_io import validate_scorer_config
            validate_scorer_config(self.memory_config, metadata["config"])
            if metadata["teacher_id"] != original_id or metadata["params_id"] != tree_hash(params):
                raise ValueError("Trained scorer does not match the P0 teacher or has an invalid hash")
            scorer_ref = {key[1:]: value for key, value in ref.items() if key[0] == "mem_scorer"}
            scorer = traverse_util.flatten_dict(params)
            if scorer_ref.keys() != scorer.keys():
                raise ValueError("Scorer parameter structure mismatch")
            for key, value in scorer.items():
                if value.shape != np.shape(scorer_ref[key]):
                    raise ValueError(f"Scorer shape mismatch: {key}")
                merged[("mem_scorer", *key)] = value
        return traverse_util.unflatten_dict(merged)


def build_config(profile):
    base = get_config(profile["base_config"])
    model = dataclasses.replace(base.model, **profile["model"])
    memory = get_history_config(model.history_config)
    if not model.pi05 or memory.integration_type != "perc_adarms" or not model.use_history:
        raise ValueError("Expected the perceptual v1 pi0.5 integration")
    cfg = PercMemConfig.from_history(memory)
    expected_freeze = "vision_only" if cfg.stage == "teacher" else "percmem_student"
    if profile["freeze"] != expected_freeze or "lora" in model.paligemma_variant or "lora" in model.action_expert_variant:
        raise ValueError("Profile stage and freeze policy disagree")
    if cfg.stage == "student" and (cfg.input_space != "projected" or not cfg.teacher_params):
        raise ValueError("Student requires the frozen projected feature space and P0 teacher")
    lr, data = profile["learning_rate"], profile["data"]
    if lr["schedule"] != "warmup_then_constant":
        raise ValueError("Expected the reference warmup-then-constant schedule")
    config = dataclasses.replace(
        base, **profile["train"], model=model, dataset_path=data["dataset_path"],
        data=RoboMMEDataConfig(repo_id=data["repo_id"],
            assets=AssetsConfig(assets_dir=data["norm_assets_dir"], asset_id=data["norm_asset_id"]),
            base_config=DataConfig(prompt_from_task=True)),
        freeze_filter=model.get_freeze_filter(),
        weight_loader=PercMemWeightLoader(profile["weights"]["params_path"], cfg.stage, cfg.scorer_path, data["dataset_path"], cfg),
        lr_schedule=CosineDecaySchedule(warmup_steps=lr["warmup_steps"], peak_lr=lr["value"],
            decay_steps=profile["train"]["num_train_steps"], decay_lr=lr["value"]),
        optimizer=AdamW(**profile["optimizer"]),
        policy_metadata={"percmem_profile": profile},
    )
    if config.fsdp_devices != 4 or config.batch_size % 4 or config.resume or config.overwrite or config.resum_ckpt_id is not None:
        raise ValueError("Reviewed profiles require four-way FSDP, divisible batch and a fresh output directory")
    if config.wandb_enabled:
        raise ValueError("Configure your W&B entity before enabling it")
    return config, memory


def review_config(profile, config, memory):
    from mme_vla_suite.training.lerobot_percmem import MANIFEST, validate_cache

    cfg, root = PercMemConfig.from_history(memory), Path(config.dataset_path)
    checks, blockers = {}, []
    paths = {
        "initial_params": Path(profile["weights"]["params_path"]),
        "native_cache_manifest": root / MANIFEST,
        "native_cache_complete": root / "meta/native_complete.json",
        "normalization": Path(profile["data"]["norm_assets_dir"]) / profile["data"]["norm_asset_id"] / "norm_stats.json",
    }
    if cfg.stage == "student":
        paths.update(teacher_params=Path(cfg.teacher_params), scorer=Path(cfg.scorer_path),
                     projected_complete=Path(cfg.projected_dir) / "manifest.json", banks_complete=Path(cfg.banks_dir) / "manifest.json")
    for name, path in paths.items():
        present = path.exists()
        checks[name] = {"path": str(path), "present": present}
        if not present:
            blockers.append(f"Missing {name}: {path}")
    if (root / MANIFEST).exists():
        meta = validate_cache(root, require_complete=False)
        for name, expected in [("episodes", profile["data"]["expected_episodes"]),
                               ("total_frames", profile["data"]["expected_frames"]),
                               ("execution_samples", profile["data"]["expected_execution_samples"])]:
            if meta[name] != expected:
                raise ValueError(f"Dataset {name} mismatch")
        if meta["max_keyframes"] > cfg.max_frames:
            blockers.append("max_frames would truncate full-history teacher memory")
    if config.checkpoint_dir.exists():
        blockers.append(f"Fresh-run output directory already exists: {config.checkpoint_dir}")
    link = Path.home() / ".cache" / f"jax_{config.exp_name}"
    target = Path(profile["runtime"]["compilation_cache_dir"])
    if (link.is_symlink() and link.resolve() != target.resolve()) or (link.exists() and not link.is_symlink()):
        blockers.append(f"Unexpected compilation-cache binding: {link}")
    n = cfg.max_frames if cfg.stage == "teacher" else cfg.capacity
    dim = cfg.native_dim if cfg.input_space == "native" else cfg.feature_dim
    data_bytes = config.batch_size * n * cfg.store_grid ** 2 * dim * 2
    if cfg.stage == "student":
        data_bytes += config.batch_size * cfg.max_frames * cfg.store_grid ** 2 * cfg.feature_dim * 2
    schedule = config.lr_schedule.create()
    steps = [i for i in range(config.num_train_steps) if (i > 0 and i % config.save_interval == 0) or i == config.num_train_steps - 1]
    return dict(profile_name=profile["profile_name"], mode="configuration_review_only", training_started=False,
        model=dataclasses.asdict(config.model), memory=OmegaConf.to_container(memory, resolve=True),
        train=profile["train"], optimizer=dataclasses.asdict(config.optimizer),
        learning_rate=dataclasses.asdict(config.lr_schedule),
        learning_rate_samples={str(i): float(schedule(i)) for i in [0, profile["learning_rate"]["warmup_steps"], config.num_train_steps - 1]},
        checkpoint_dir=str(config.checkpoint_dir), checkpoint_ids=steps, checks=checks, execution_blockers=blockers,
        runtime=profile["runtime"], global_batch_size=config.batch_size, samples_per_device=config.batch_size // 4,
        input_memory_bytes_per_batch=data_bytes, capacity_tested=False,
        freeze=("vision and unused scorer; VLM/action/reader/adapters trainable" if cfg.stage == "teacher" else
                "only reader, output adapter and action expert/time/action projections trainable"),
        compilation_cache_binding={"upstream_path": str(link), "symlink_target": str(target)},
        evaluation_plan=profile["evaluation"])


def save_run_metadata(config):
    memory = get_history_config(config.model.history_config)
    if memory is None or memory.integration_type != "perc_adarms":
        return
    root = Path(config.checkpoint_dir)
    (root / "percmem_history.json").write_text(json.dumps(OmegaConf.to_container(memory, resolve=True), indent=2))
    (root / "percmem_model.json").write_text(json.dumps(dataclasses.asdict(config.model), indent=2))
    profile = config.policy_metadata["percmem_profile"]
    (root / "percmem_profile.json").write_text(json.dumps(profile, indent=2))
    (root / "percmem_protocol.json").write_text(json.dumps({
        "history_compute": "float32", "vision_parameter_storage": "bfloat16", "cache_storage": "float16",
        "seed": config.seed, "rng_protocol": "fold_in(train_rng, optimizer_step)",
        "full_train_state_saved": True, "exact_dataloader_resume_implemented": False,
    }, indent=2))


def load_frozen_teacher(config, train_state, sharding):
    from flax import nnx
    import jax
    cfg = PercMemConfig.from_history(get_history_config(config.model.history_config))
    if cfg.stage != "student":
        return None
    if Path(cfg.teacher_params).resolve() != Path(config.weight_loader.params_path).resolve():
        raise ValueError("P3 and its frozen VLM prefix must initialize from the same teacher")
    params = restore_params(cfg.teacher_params, restore_type=np.ndarray)
    identity = tree_hash(params)
    projected = json.loads((Path(cfg.projected_dir) / "manifest.json").read_text())
    if identity != projected["teacher_id"]:
        raise ValueError("Full teacher parameters do not match projected features/banks")
    flat, reference = traverse_util.flatten_dict(params), traverse_util.flatten_dict(train_state.params.to_pure_dict())
    if flat.keys() != reference.keys() or any(np.shape(flat[k]) != np.shape(reference[k]) for k in flat):
        raise ValueError("Frozen teacher parameter structure differs from the student backbone")
    model = nnx.merge(train_state.model_def, train_state.params)
    state = nnx.state(model)
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, params))
    return jax.device_put(state, sharding)
