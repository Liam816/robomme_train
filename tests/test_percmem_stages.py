"""P1 normalization, actual P2 optimizer update and causal P3 bank replay."""

import dataclasses
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from flax import nnx
import jax.numpy as jnp
import numpy as np

from mme_vla_suite.models.representation.perc_memory_v1 import EvictScorer
from mme_vla_suite.shared.percmem_io import load_module
from mme_vla_suite.training.percmem_stages import (
    attention_utility, future_values, lookup_utility, rollout_banks, train_scorer,
)
from test_percmem_v1 import tiny_config


class StagesTest(unittest.TestCase):
    def test_attention_normalization_and_future_lookup(self):
        weights = jnp.ones((1, 2, 8, 8, 3, 16)) / 48
        result = attention_utility(weights, jnp.ones((1, 3), bool))
        np.testing.assert_allclose(result, 1, atol=1e-6)
        u = future_values(np.array([[1, 2], [3, 1], [0, 4]], np.float32))
        np.testing.assert_array_equal(u, [[3, 4], [3, 4], [0, 4]])
        labels = {"query_tick": np.array([0, 10, 20]), "utility": u}
        np.testing.assert_array_equal(lookup_utility(labels, 11, [0, 1]), [0, 4])
        self.assertIsNone(lookup_utility(labels, 21, [0]))

    def test_real_p2_step_and_p3_replay(self):
        cfg = dataclasses.replace(tiny_config(), recent_frames=1, queue_frames=1)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, projected, labels = root / "cache", root / "projected", root / "labels"
            for directory in [cache / "meta", cache / "keyframes", projected, labels]:
                directory.mkdir(parents=True)
            identity = {"cache_id": "cache", "teacher_id": "teacher", "adapter_id": "adapter"}
            (cache / "meta/source_index.json").write_text(json.dumps({"fps": 10, "episodes": [
                {"episode_index": 0, "length": 25, "exec_start_idx": 5}]}))
            rows, ticks = np.arange(0, 25, 5, dtype=np.int32), np.arange(0, 50, 10, dtype=np.int32)
            np.savez(cache / "keyframes/episode_000000.npz", row=rows, tick=ticks)
            np.save(projected / "episode_000000.npy", np.random.default_rng(4).normal(size=(5, 16, 40)).astype(np.float16))
            (projected / "manifest.json").write_text(json.dumps(identity))
            (labels / "manifest.json").write_text(json.dumps(dict(identity, trusted=True)))
            utility = np.tile(np.array([3, 2, 1, 0, .5], np.float16), (2, 1))
            np.savez(labels / "episode_000000.npz", query_tick=np.array([0, 100]), utility=utility)
            path = root / "scorer.npz"
            metadata = train_scorer(cache, projected, labels, path, cfg, steps=1, batch_trajectories=1)
            self.assertEqual(metadata["steps"], 1)
            scorer = EvictScorer(cfg, rngs=nnx.Rngs(12))
            loaded = load_module(scorer, path)
            for name in ["a", "b"]:
                rollout_banks(cache, projected, root / name, cfg, scorer, loaded)
            with np.load(root / "a/episode_000000.npz") as a, np.load(root / "b/episode_000000.npz") as b:
                np.testing.assert_array_equal(a["members"], b["members"])
                self.assertEqual(a["members"].shape, (20, 2))
                for row, members in zip(a["row"], a["members"], strict=True):
                    selected = members[members >= 0]
                    self.assertTrue((rows[selected] <= row).all())
                    self.assertLessEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
