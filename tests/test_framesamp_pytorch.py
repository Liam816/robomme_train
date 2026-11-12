"""Cross-backend contracts that a single-rank PyTorch smoke run cannot exercise."""

import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mme_vla_suite.training.framesamp_torch_data import GlobalBatches
from mme_vla_suite.training.framesamp_torch_data import stochastic_inputs


def test_profile_round_trip_and_user_overrides(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_framesamp_pytorch import DEFAULT
    from train_framesamp_pytorch import ROOT
    from train_framesamp_pytorch import resolve

    reference = ROOT / yaml.safe_load(DEFAULT.read_text())["reference_profile"]
    resolved = resolve(DEFAULT, ["train.exp_name=example", "pytorch.benchmark_steps=5"])
    saved = tmp_path / "profile.yaml"
    saved.write_text(yaml.safe_dump(resolved))
    assert resolve(saved, []) == resolved
    assert resolve(reference, []) == resolve(DEFAULT, [])

    raw = yaml.safe_load(reference.read_text())
    raw["runtime"]["cuda_visible_devices"] = [7, 6, 5, 4]
    raw["train"]["checkpoint_base_dir"] = str(tmp_path)
    saved.write_text(yaml.safe_dump(raw))
    migrated = resolve(saved, [])
    assert migrated["runtime"]["cuda_visible_devices"] == [7, 6, 5, 4]
    assert migrated["train"]["checkpoint_base_dir"] == str(tmp_path)


@pytest.mark.parametrize("workers", [0, 1])
def test_ranks_reproduce_reference_shuffle_across_epochs(workers):
    size, batch, world, steps = 11, 4, 2, 7
    reference = torch.utils.data.DataLoader(
        range(size),
        batch_size=batch,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=workers,
        persistent_workers=bool(workers),
    )

    def epochs():
        while True:
            yield from reference

    ranks = [iter(GlobalBatches(size, batch, world, i, 42, steps, bool(workers))) for i in range(world)]
    for step, expected in enumerate(itertools.islice(epochs(), steps)):
        items = next(ranks[0]) + next(ranks[1])
        assert items == [(i, step) for i in expected.tolist()]


def test_rank_slice_matches_reference_augmentation_and_noise():
    import jax
    import jax.numpy as jnp

    from openpi.models.model import Observation
    from openpi.models.model import preprocess_observation

    images = np.random.default_rng(7).integers(0, 256, (4, 2, 224, 224, 3), dtype=np.uint8)
    train_key, _ = jax.random.split(jax.random.key(42))
    image_key, noise_key, time_key = jax.random.split(jax.random.fold_in(train_key, 17), 3)
    obs = Observation.from_dict(
        {
            "image": {"base_0_rgb": jnp.asarray(images[:, 0]), "left_wrist_0_rgb": jnp.asarray(images[:, 1])},
            "image_mask": {},
            "state": jnp.zeros((4, 32)),
        }
    )
    expected = preprocess_observation(image_key, obs, train=True)
    pixels = jnp.stack(list(expected.images.values()), axis=1).transpose(0, 1, 4, 2, 3)
    image, noise, t = stochastic_inputs(images[2:], 17, 42, 1, global_batch=4, horizon=20, action_dim=32)
    np.testing.assert_allclose(image, pixels[2:], atol=4e-5)
    np.testing.assert_array_equal(noise, jax.random.normal(noise_key, (4, 20, 32))[2:])
    np.testing.assert_allclose(t, (jax.random.beta(time_key, 1.5, 1, (4,)) * 0.999 + 0.001)[2:], atol=1e-6)


def test_weight_mapping_is_one_to_one():
    from mme_vla_suite.training.framesamp_weights import weight_rules

    rules = weight_rules()
    assert len({r.path for r in rules}) == len(rules) == 61
    names = [n for r in rules for n in r.targets]
    assert len(names) == len(set(names))
    assert sum("mem" in r.path for r in rules) == 10
