import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import dataclasses

import openpi.models.model as _model
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as transforms

import mme_vla_suite.policies.policy as _policy
import mme_vla_suite.training.config as _config



def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    seed: int = 42,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.MME_VLA_Policy:
    
    repack_transforms = repack_transforms or transforms.Group()
    
    logging.info(f"Checking history config")
    history_config = None
    history_config_path = checkpoint_dir.parent / "history_config.txt"
    if history_config_path.exists():
        with open(history_config_path, "r") as f:
            history_config = f.read()
    
    if train_config.model.history_config != history_config:
        print(f" == You are using {train_config.model.history_config}, changing to {history_config} ==")
        train_config = dataclasses.replace(
            train_config, 
            model=dataclasses.replace(train_config.model, history_config=history_config, use_history=history_config is not None)
        )
    

    resolved = checkpoint_dir.parent / "percmem_history.json"
    is_percmem = resolved.is_file()
    if is_percmem:
        import json
        from omegaconf import OmegaConf
        history = OmegaConf.create(json.loads(resolved.read_text()))
        if history.perc_mem.stage == "e2e":
            # Training replays native history to differentiate writes. Deployment
            # uses the same hard writer online and stores only projected FP16 W/Q.
            history.perc_mem.input_space = "projected"
        train_config = dataclasses.replace(train_config, model=dataclasses.replace(
            train_config.model, history_config=history, use_history=True))
    logging.info("Loading model...")
    # Preserve FP32 frozen A_feat/scorer parameters for identical offline/online decisions.
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=None if is_percmem else jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    print("Training config: ", train_config)
    print("Data config: ", data_config)

    policy_class = _policy.MME_VLA_Policy
    if is_percmem:
        from mme_vla_suite.policies.percmem_policy import PercMemPolicy
        policy_class = PercMemPolicy
    return policy_class(
        model,
        seed=seed,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        norm_stats=norm_stats,
        use_quantiles=data_config.use_quantile_norm
    )
