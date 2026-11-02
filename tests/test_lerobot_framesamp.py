"""Behavior checks for demonstrations, episode boundaries and FrameSamp histories."""

import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ml_dtypes
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mme_vla_suite.training.lerobot_framesamp import LeRobotFrameSampDataset, META_FILE, prepare_index


class FrameSampTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "source"
        self.cache = Path(self.temporary.name) / "cache"
        for sub in ["meta/episodes/chunk-000", "data/chunk-000", "videos/image/chunk-000", "videos/wrist_image/chunk-000"]:
            (self.source / sub).mkdir(parents=True)
        info = {"codebase_version": "v3.0", "total_frames": 10, "total_episodes": 2, "fps": 10,
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
        (self.source / "meta/info.json").write_text(json.dumps(info))
        rows, episodes = [], []
        for episode_id, length, start in [(0, 6, 2), (1, 4, 1)]:
            begin = len(rows)
            for step in range(length):
                rows.append({"index": len(rows), "state": [float(episode_id * 100 + step)] * 8,
                    "actions": [float(episode_id * 100 + step)] * 8, "episode_index": episode_id,
                    "step_idx": step, "exec_start_idx": start, "task_index": episode_id, "is_demo": step < start})
            episode = {"episode_index": episode_id, "length": length, "dataset_from_index": begin,
                       "dataset_to_index": len(rows)}
            for key in ["image", "wrist_image"]:
                episode.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                                f"videos/{key}/from_timestamp": begin / 10})
                (self.source / f"videos/{key}/chunk-000/file-000.mp4").touch()
            episodes.append(episode)
        pq.write_table(pa.Table.from_pylist(rows), self.source / "data/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist(episodes), self.source / "meta/episodes/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist([{"task_index": 0, "task": "First Task"}, {"task_index": 1, "task": "Second Task"}]),
                       self.source / "meta/tasks.parquet")
        metadata = prepare_index(self.source, self.cache)
        np.save(self.cache / "meta/position_spatial.npy", np.arange(16 * 512, dtype=np.float32).reshape(16, 512))
        np.save(self.cache / "meta/position_temporal.npy", np.arange(6 * 256, dtype=np.float32).reshape(6, 256))
        (self.cache / "features").mkdir()
        for episode in metadata["episodes"]:
            array = np.empty((episode["length"], 16, 2048), dtype=ml_dtypes.bfloat16)
            for step in range(episode["length"]):
                array[step] = -step - 0.5
            np.save(self.cache / f"features/episode_{episode['episode_index']:06d}.npy", array.view(np.uint16))
        self.cfg = SimpleNamespace(representation_type="perceptual", perceptual_memory=SimpleNamespace(type="frame_sampling"),
                                   num_views=1, token_per_image=16, budget=64)
        stats = SimpleNamespace(mean=np.zeros(8), std=np.ones(8))
        self.data_cfg = SimpleNamespace(norm_stats={"state": stats}, use_quantile_norm=False)

    def dataset(self):
        return LeRobotFrameSampDataset(self.cache, self.data_cfg, self.cfg, 4, require_complete=False)

    @staticmethod
    def frames(reader, episode, key="image", start=0, count=None):
        yield np.full((16, 16, 3), start, dtype=np.uint8)

    def test_demonstrations_are_history_but_not_training_samples(self):
        dataset = self.dataset()
        self.assertEqual(len(dataset), 7)
        with patch("mme_vla_suite.training.lerobot_framesamp.VideoReader.frames", self.frames):
            sample = dataset[0]
        self.assertEqual(sample["image"][0, 0, 0], 2)
        self.assertEqual(sample["static_mask"].sum(), 48)
        np.testing.assert_array_equal(sample["static_image_emb"].reshape(4, 16, 2048)[:3, 0, 0], [-0.5, -1.5, -2.5])
        self.assertFalse(sample["static_mask"][48:].any())
        self.assertFalse(sample["static_image_emb"][48:].any())

    def test_uniform_sampling_uses_episode_relative_timeline(self):
        with patch("mme_vla_suite.training.lerobot_framesamp.VideoReader.frames", self.frames):
            sample = self.dataset()[3]
        np.testing.assert_array_equal(sample["static_image_emb"].reshape(4, 16, 2048)[:, 0, 0], [-0.5, -1.5, -3.5, -5.5])
        self.assertTrue(sample["static_mask"].all())
        temporal = np.load(self.cache / "meta/position_temporal.npy")
        np.testing.assert_array_equal(sample["static_pos_emb"].reshape(4, 16, 768)[:, 0, :256], temporal[[0, 1, 3, 5]])

    def test_action_chunk_repeats_last_action_without_crossing_episode(self):
        with patch("mme_vla_suite.training.lerobot_framesamp.VideoReader.frames", self.frames):
            first_end = self.dataset()[3]
            second_start = self.dataset()[4]
        np.testing.assert_array_equal(first_end["actions"][:, 0], [5, 5, 5, 5])
        np.testing.assert_array_equal(second_start["actions"][:, 0], [101, 102, 103, 103])
        self.assertEqual(second_start["prompt"], "second task")

    def test_spawn_reopens_memmaps_instead_of_copying_entire_index(self):
        dataset = self.dataset()
        payload = pickle.dumps(dataset)
        self.assertLess(len(payload), 100_000)
        restored = pickle.loads(payload)
        self.assertIsInstance(restored.arrays["state"], np.memmap)
        self.assertIsNone(restored.reader)

    def test_training_rejects_partial_cache(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            LeRobotFrameSampDataset(self.cache, self.data_cfg, self.cfg, 4)

    def test_index_rejects_duplicate_rows(self):
        table = pq.read_table(self.source / "data/chunk-000/file-000.parquet")
        pq.write_table(table.slice(0, 1), self.source / "data/chunk-000/file-001.parquet")
        (self.cache / META_FILE).unlink()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_index(self.source, self.cache)


if __name__ == "__main__":
    unittest.main()
