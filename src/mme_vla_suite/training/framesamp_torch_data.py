"""Use the released execution index, transforms and global shuffle for Torch FSDP."""

from functools import partial
import os

import augmax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import default_collate

from mme_vla_suite.training.lerobot_framesamp import LeRobotFrameSampDataset
from openpi.training.data_loader import transform_dataset


class FrameSampDataset:
    def __init__(self, profile, model, memory, data_config):
        data = profile["data"]
        cache = LeRobotFrameSampDataset(data["dataset_path"], data_config, memory, model.action_horizon)
        actual = (len(cache.episodes), cache.metadata["total_frames"], len(cache))
        expected = tuple(data[k] for k in ("expected_episodes", "expected_frames", "expected_execution_samples"))
        if actual != expected:
            raise ValueError(f"FrameSamp index mismatch: found {actual}, expected {expected}")
        self.dataset = transform_dataset(cache, data_config)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        index, step = item
        row = self.dataset[index]
        images = np.stack([row["image"][camera] for camera in ("base_0_rgb", "left_wrist_0_rgb")])
        return {
            "images": images,
            "state": np.asarray(row["state"], dtype=np.float32),
            "actions": np.asarray(row["actions"], dtype=np.float32),
            "tokenized_prompt": row["tokenized_prompt"],
            "tokenized_prompt_mask": row["tokenized_prompt_mask"],
            "static_image_emb": torch.from_numpy(row["static_image_emb"]).bfloat16(),
            "static_pos_emb": torch.from_numpy(row["static_pos_emb"]).float(),
            "static_mask": row["static_mask"],
        }, step


class GlobalBatches:
    """Replay the reference DataLoader generator, then split each global batch by rank."""

    def __init__(self, size, global_batch, world, rank, seed, steps, persistent):
        self.size, self.batch, self.world, self.rank = size, global_batch, world, rank
        self.seed, self.steps, self.persistent = seed, steps, persistent

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed)
        step = 0
        local = self.batch // self.world
        while step < self.steps:
            if not self.persistent or step == 0:
                # Base seed drawn by the original DataLoader iterator before RandomSampler.
                torch.empty((), dtype=torch.int64).random_(generator=rng)
            sampler = RandomSampler(range(self.size), generator=rng)
            for batch in BatchSampler(sampler, self.batch, drop_last=True):
                yield [(i, step) for i in batch[self.rank * local : (self.rank + 1) * local]]
                step += 1
                if step == self.steps:
                    return


@partial(jax.jit, static_argnames=("global_batch", "horizon", "action_dim"))
def stochastic_inputs(images, step, seed, rank, *, global_batch, horizon, action_dim):
    """Generate identical per-global-slot augmentation, flow noise and Beta time on CPU."""
    local = images.shape[0]
    train_rng, _ = jax.random.split(jax.random.key(seed))
    aug, noise, t = jax.random.split(jax.random.fold_in(train_rng, step), 3)
    select = lambda array: jax.lax.dynamic_slice_in_dim(array, rank * local, local)
    keys = select(jax.random.split(aug, global_batch))
    images = images.astype(jnp.float32) / 255.0 * 2.0 - 1.0
    images = images / 2.0 + 0.5
    color = augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
    front = jax.vmap(augmax.Chain(augmax.RandomCrop(212, 212), augmax.Resize(224, 224), augmax.Rotate((-5, 5)), color))(
        keys, images[:, 0]
    )
    wrist = jax.vmap(augmax.Chain(color))(keys, images[:, 1])
    images = jnp.stack((front, wrist), 1) * 2.0 - 1.0
    return (
        images.transpose(0, 1, 4, 2, 3),
        select(jax.random.normal(noise, (global_batch, horizon, action_dim))),
        select(jax.random.beta(t, 1.5, 1, (global_batch,)) * 0.999 + 0.001),
    )


def collate(items, *, seed, rank, global_batch, horizon, action_dim):
    rows, steps = zip(*items, strict=True)
    batch = default_collate(rows)
    images, noise, time = stochastic_inputs(
        batch["images"].numpy(), steps[0], seed, rank, global_batch=global_batch, horizon=horizon, action_dim=action_dim
    )
    for key, array in (("images", images), ("noise", noise), ("time", time)):
        batch[key] = torch.from_numpy(np.array(array))
    return batch


def init_worker(i):
    # XLA sizes its CPU thread pools from affinity; limit each input worker on large hosts.
    cpus = sorted(os.sched_getaffinity(0))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    start = (rank + i * int(os.environ.get("WORLD_SIZE", 1))) * 4
    os.sched_setaffinity(0, [cpus[(start + j) % len(cpus)] for j in range(min(4, len(cpus)))])


def make_loader(dataset, config, rank, world, steps):
    train = config["train"]
    batch, workers = train["batch_size"], train["num_workers"]
    if len(dataset) < batch:
        raise ValueError("Global batch exceeds the dataset")
    # num_workers is global, as in the JAX single-process trainer.
    local_workers = workers // world + int(rank < workers % world)
    sampler = GlobalBatches(len(dataset), batch, world, rank, train["seed"], steps, workers > 0)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=local_workers,
        multiprocessing_context="spawn" if local_workers else None,
        persistent_workers=bool(local_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=init_worker,
        collate_fn=partial(
            collate,
            seed=train["seed"],
            rank=rank,
            global_batch=batch,
            horizon=config["model"]["action_horizon"],
            action_dim=config["model"]["action_dim"],
        ),
    )
