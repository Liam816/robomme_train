"""Validate real cached samples through the unchanged MME-VLA input transforms."""

import argparse
import dataclasses
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.training.config import get_config
from mme_vla_suite.training.lerobot_framesamp import LeRobotFrameSampDataset
from openpi.training.data_loader import transform_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = get_config("mme_vla_suite")
    model = dataclasses.replace(config.model, history_config="perceptual-framesamp-modul.yaml")
    data = config.data.create(config.assets_dirs, model)
    dataset = LeRobotFrameSampDataset(args.cache, data, get_history_config(model.history_config),
                                    model.action_horizon, require_complete=not args.allow_partial)
    transformed = transform_dataset(dataset, data)
    ready = []
    for episode in dataset.metadata["episodes"]:
        if not (Path(args.cache) / f"features/episode_{episode['episode_index']:06d}.json").exists():
            continue
        begin = episode["dataset_from_index"] + episode["exec_start_idx"]
        end = episode["dataset_to_index"] - 1
        ready.extend([int(np.searchsorted(dataset.samples, begin)), int(np.searchsorted(dataset.samples, end))])
        if len(ready) >= 8:
            break
    if not ready:
        raise RuntimeError("No completed episodes to validate")
    started = time.monotonic()
    samples = [transformed[index] for index in ready]
    batch = jax.tree.map(lambda *items: np.stack(items), *samples)
    batch = jax.tree.map(jnp.asarray, batch)
    observation = HistAugObservation.from_dict(batch)
    count = len(ready)
    assert batch["actions"].shape == (count, 20, 32)
    assert observation.static_image_emb.shape == (count, 512, 2048)
    assert observation.static_pos_emb.shape == (count, 512, 768)
    assert observation.static_mask.shape == (count, 512)
    assert observation.tokenized_prompt.shape == (count, 64)
    for value in jax.tree.leaves((observation, batch["actions"])):
        assert np.isfinite(np.asarray(value)).all()
    report = {"samples_checked": count, "dataset_execution_samples": len(dataset),
              "actions_shape": list(batch["actions"].shape),
              "memory_shape": list(observation.static_image_emb.shape),
              "position_shape": list(observation.static_pos_emb.shape),
              "image_shapes": {key: list(value.shape) for key, value in observation.images.items()},
              "tokenized_prompt_shape": list(observation.tokenized_prompt.shape),
              "seconds": time.monotonic() - started, "all_finite": True,
              "partial_cache_check": args.allow_partial}
    if not args.allow_partial:
        from mme_vla_suite.training.dataloader import create_data_loader
        loader = create_data_loader(str(args.cache), data, model.history_config, model.action_horizon,
                                    batch_size=4, num_workers=2, num_batches=1, shuffle=True, seed=42)
        worker_observation, worker_actions = next(iter(loader))
        assert worker_actions.shape == (4, 20, 32)
        assert worker_observation.static_image_emb.shape == (4, 512, 2048)
        for value in jax.tree.leaves((worker_observation, worker_actions)):
            assert np.isfinite(np.asarray(value)).all()
        report["spawn_workers_checked"] = 2
        report["worker_batch_actions_shape"] = list(worker_actions.shape)
        del loader
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
    if dataset.reader is not None:
        dataset.reader.close()


if __name__ == "__main__":
    main()
