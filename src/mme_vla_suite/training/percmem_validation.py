"""Fixed-noise deletion checks and the explicit counterfactual P1 fallback."""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mme_vla_suite.models.integration.history_observation import HistAugObservation, preprocess_observation
from mme_vla_suite.training.lerobot_framesamp import write_json
from mme_vla_suite.training.lerobot_percmem import LeRobotPercMemDataset
from mme_vla_suite.training.percmem_stages import episode_data, future_values, lookup_utility
from openpi.models.model import restore_params
from openpi.shared.nnx_utils import module_jit
from openpi.training.data_loader import transform_dataset


def setup(config, history, params_path, cache):
    model = config.model.load(restore_params(params_path))
    model.eval()
    data_config = config.data.create(config.assets_dirs, config.model)
    raw = LeRobotPercMemDataset(cache, data_config, history, config.model.action_horizon)
    dataset = transform_dataset(raw, data_config)
    return model, raw, dataset


def sample_inputs(sample, seed):
    batched = jax.tree.map(lambda x: None if x is None else jnp.asarray(x)[None], sample)
    obs = preprocess_observation(None, HistAugObservation.from_dict(batched), train=False)
    actions = batched["actions"]
    noise_key, time_key = jax.random.split(jax.random.key(seed))
    noise = jax.random.normal(noise_key, actions.shape)
    time = jax.random.beta(time_key, 1.5, 1, (1,)) * .999 + .001
    noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    return obs, noisy, time, noise - actions


def validate_attention_labels(config, history, params_path, cache, labels_dir, *, samples=200):
    model, raw, dataset = setup(config, history, params_path, cache)
    labels_dir = Path(labels_dir)
    predict, prefix_fn = module_jit(model.predict_velocity), module_jit(model.encode_prefix)
    rng = np.random.default_rng(42)
    differences, records = [], []
    for i in rng.permutation(len(dataset)):
        row = int(raw.samples[i])
        episode = int(raw.arrays["episode_index"][row])
        with np.load(labels_dir / f"episode_{episode:06d}.npz") as archive:
            labels = {k: archive[k] for k in archive.files}
        obs, noisy, time, target = sample_inputs(dataset[int(i)], int(rng.integers(2**31)))
        n = int(np.asarray(obs.perc_mem_frame_mask).sum())
        if n < 8:
            continue
        utility = lookup_utility(labels, int(obs.perc_mem_now_tick[0]), np.arange(n))
        if utility is None:
            continue
        cache_value = prefix_fn(obs)
        baseline = float(jnp.mean((predict(obs, noisy, time, prefix_cache=cache_value) - target) ** 2))
        losses = []
        order = np.argsort(utility, kind="stable")
        for removed in [order[-4:], order[:4]]:
            mask = obs.perc_mem_frame_mask.at[0, removed].set(False)
            modified = dataclasses.replace(obs, perc_mem_frame_mask=mask)
            losses.append(float(jnp.mean((predict(modified, noisy, time, prefix_cache=cache_value) - target) ** 2)))
        differences.append(losses[0] - losses[1])
        records.append({"episode": episode, "row": row, "delta_high": losses[0] - baseline,
                        "delta_low": losses[1] - baseline})
        if len(records) == samples:
            break
    if len(differences) < 2:
        lower, mean = None, None
    else:
        values = np.asarray(differences)
        bootstrap = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
        lower, mean = float(np.quantile(bootstrap, .025)), float(values.mean())
    passed = len(records) == samples and lower is not None and lower > 0
    report = {"passed": passed, "requested_samples": samples, "actual_samples": len(records),
              "paired_mean_high_minus_low": mean, "paired_bootstrap95_lower": lower,
              "fixed_noise_and_flow_time_per_anchor": True, "records": records}
    write_json(labels_dir / "deletion_validation.json", report)
    manifest = json.loads((labels_dir / "manifest.json").read_text())
    manifest["trusted"] = passed
    write_json(labels_dir / "manifest.json", manifest)
    if not passed:
        raise ValueError("Attention labels failed deletion validation; P2 is blocked. Use the explicit counterfactual fallback.")
    return report


def counterfactual_labels(config, history, params_path, cache, projected, output, max_episodes, scale=1.0):
    model, raw, dataset = setup(config, history, params_path, cache)
    predict, prefix_fn = module_jit(model.predict_velocity), module_jit(model.encode_prefix)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((Path(cache) / "meta/source_index.json").read_text())
    selected = metadata["episodes"][:max_episodes]
    random = np.random.default_rng(42)
    for episode in selected:
        name, plan, feature = episode_data(cache, episode, projected)
        rows = list(range(episode["exec_start_idx"], episode["length"], 10))
        if not rows or rows[-1] != episode["length"] - 1:
            rows.append(episode["length"] - 1)
        values, query_ticks = [], []
        for row in rows:
            global_row = episode["dataset_from_index"] + row
            location = int(np.searchsorted(raw.samples, global_row))
            if location == len(raw.samples) or raw.samples[location] != global_row:
                raise ValueError("Counterfactual query is not an execution sample")
            obs, noisy, time, target = sample_inputs(dataset[location], int(random.integers(2**31)))
            prefix = prefix_fn(obs)
            baseline = jnp.mean((predict(obs, noisy, time, prefix_cache=prefix) - target) ** 2)
            n = int(np.asarray(obs.perc_mem_frame_mask).sum())
            importance = np.zeros(len(feature), np.float32)
            for key in range(n):
                mask = obs.perc_mem_frame_mask.at[0, key].set(False)
                modified = dataclasses.replace(obs, perc_mem_frame_mask=mask)
                deleted = jnp.mean((predict(modified, noisy, time, prefix_cache=prefix) - target) ** 2)
                importance[key] = float(deleted - baseline) * scale
            values.append(importance)
            query_ticks.append(int(obs.perc_mem_now_tick[0]))
        valid_labels = np.array(query_ticks)[:, None] >= plan["tick"][None]
        utilities = np.where(valid_labels, future_values(np.asarray(values)), 0).astype(np.float16)
        np.savez(output / f"{name}.npz", query_row=np.array(rows, np.int32),
                 query_tick=np.array(query_ticks, np.int32), key_tick=plan["tick"],
                 utility=utilities, valid=valid_labels)
    identity = json.loads((Path(projected) / "manifest.json").read_text())
    write_json(output / "manifest.json", dict(identity, source="counterfactual_fm_delta_suffix_max",
        trusted=True, validation_required=False, cf_scale=scale, query_stride=10,
        episode_ids=[e["episode_index"] for e in selected]))
