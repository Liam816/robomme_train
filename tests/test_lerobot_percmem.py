"""Numeric/video data boundaries, schema rejection and spawned-worker loading."""

import dataclasses
import json
from pathlib import Path
import unittest

import av
import numpy as np
from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mme_vla_suite.training.lerobot_percmem import LeRobotPercMemDataset, prepare_cache
from test_percmem_v1 import tiny_config


class DataTest(unittest.TestCase):
    def setUp(self):
        from test_lerobot_framesamp import FrameSampTest
        self.fixture = FrameSampTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = dataclasses.replace(tiny_config(), recent_frames=1, queue_frames=1, evict="fifo")
        source = self.fixture.source
        path = source / "data/chunk-000/file-000.parquet"
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["timestamp"] = row["step_idx"] / 10
        pq.write_table(pa.Table.from_pylist(rows), path)
        for key in ["image", "wrist_image"]:
            with av.open(str(source / f"videos/{key}/chunk-000/file-000.mp4"), "w") as container:
                stream = container.add_stream("mpeg4", rate=10)
                stream.width = stream.height = 16
                stream.pix_fmt = "yuv420p"
                for i in range(10):
                    frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i * 10, np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        self.root = self.fixture.cache.parent / "percmem"
        self.meta = prepare_cache(self.fixture.cache, self.root, self.cfg)
        self.source_meta = json.loads((self.root / "meta/source_index.json").read_text())
        for episode in self.source_meta["episodes"]:
            name = f"episode_{episode['episode_index']:06d}"
            with np.load(self.root / f"keyframes/{name}.npz") as plan:
                n = len(plan["row"])
            np.save(self.root / f"native/{name}.npy", np.full((n, 16, 32), episode["episode_index"], np.float16))
            (self.root / f"native/{name}.json").write_text(json.dumps({"cache_id": self.meta["cache_id"], "vision_id": "test"}))
        (self.root / "meta/native_complete.json").write_text(json.dumps({"cache_id": self.meta["cache_id"], "vision_id": "test"}))

    def dataset(self, cfg=None):
        history = OmegaConf.create({"perc_mem": dataclasses.asdict(cfg or self.cfg)})
        return LeRobotPercMemDataset(self.root, self.fixture.data_cfg, history, 4)

    def test_demo_history_actions_and_nonkeyframe_age(self):
        data = self.dataset()
        self.assertEqual(len(data), 7)
        item = data[0]
        self.assertEqual(int(item["perc_mem_frame_mask"].sum()), 1)
        self.assertEqual(item["perc_mem_now_tick"], 4)  # row 2 at 10Hz, most recent keyframe tick=0
        self.assertEqual(item["perc_mem_frame_tick"][0], 0)
        np.testing.assert_array_equal(data[3]["actions"], np.full((4, 8), 5))
        self.assertEqual(item["perc_mem_feats"].dtype, np.float16)
        self.assertEqual(item["image"].shape, (16, 16, 3))

    def test_e2e_replays_native_history_without_teacher_or_offline_banks(self):
        cfg = dataclasses.replace(self.cfg, stage="e2e", evict="learned")
        item = self.dataset(cfg)[0]
        self.assertEqual(item["perc_mem_feats"].shape[0], cfg.max_frames)
        self.assertEqual(item["perc_mem_feats"].shape[-1], cfg.native_dim)
        self.assertNotIn("perc_teacher_feats", item)
        self.assertEqual(int(item["perc_mem_frame_mask"].sum()), 1)

    def test_spawned_workers_reopen_mmaps_and_video(self):
        loader = torch.utils.data.DataLoader(self.dataset(), batch_size=None, num_workers=2,
                                             multiprocessing_context="spawn")
        items = list(loader)
        self.assertEqual(len(items), 7)
        self.assertEqual(tuple(items[0]["perc_mem_feats"].shape), (32, 16, 32))

    def test_old_schema_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            LeRobotPercMemDataset(self.fixture.cache, self.fixture.data_cfg,
                                 OmegaConf.create({"perc_mem": dataclasses.asdict(self.cfg)}), 4)

    def test_projected_cache_and_p1_attention_artifact(self):
        from flax import nnx
        from mme_vla_suite.shared.percmem_io import MemoryBundle
        from mme_vla_suite.training.percmem_stages import build_attention_labels, project_cache
        bundle = MemoryBundle(self.cfg, rngs=nnx.Rngs(3))
        bundle.vision_id = "test"
        projected, labels = self.root / "projected", self.root / "labels"
        project_cache(self.root, projected, bundle, "teacher")
        build_attention_labels(self.root, projected, labels, bundle, "teacher")
        self.assertFalse(json.loads((labels / "manifest.json").read_text())["trusted"])
        with np.load(labels / "episode_000000.npz") as artifact:
            self.assertEqual(artifact["utility"].shape, (2, 2))
            self.assertEqual(artifact["utility"].dtype, np.float16)
            self.assertFalse(artifact["valid"][0, 1])
            self.assertEqual(artifact["utility"][0, 1], 0)

    def test_student_banks_and_future_rejection(self):
        projected, banks = self.root / "projected", self.root / "banks"
        projected.mkdir()
        banks.mkdir()
        identity = {"cache_id": self.meta["cache_id"], "adapter_id": "adapter", "teacher_id": "teacher", "vision_id": "test"}
        (projected / "manifest.json").write_text(json.dumps(identity))
        (banks / "manifest.json").write_text(json.dumps(dict(identity, evict="fifo", capacity=2, scorer_id="scorer")))
        scorer = self.root / "scorer.npz"
        np.savez(scorer, __metadata__=np.array(json.dumps({"params_id": "scorer"})))
        for episode in self.source_meta["episodes"]:
            name = f"episode_{episode['episode_index']:06d}"
            with np.load(self.root / f"keyframes/{name}.npz") as plan:
                key_rows = plan["row"]
            np.save(projected / f"{name}.npy", np.zeros((len(key_rows), 16, 40), np.float16))
            rows = np.arange(episode["exec_start_idx"], episode["length"], dtype=np.int32)
            members = np.full((len(rows), 2), -1, np.int16)
            for i, row in enumerate(rows):
                n = np.searchsorted(key_rows, row, side="right")
                members[i, :n] = np.arange(n)
            np.savez(banks / f"{name}.npz", row=rows, members=members)
        cfg = dataclasses.replace(self.cfg, stage="student", input_space="projected", projected_dir=str(projected),
                                  banks_dir=str(banks), scorer_path=str(scorer))
        item = self.dataset(cfg)[0]
        self.assertEqual(item["perc_mem_feats"].shape, (2, 16, 40))
        self.assertEqual(item["perc_teacher_feats"].shape, (32, 16, 40))
        with np.load(banks / "episode_000000.npz") as values:
            rows, members = values["row"], values["members"]
        members[0, 0] = 1  # key at row 5 is in the future of the first sample at row 2
        np.savez(banks / "episode_000000.npz", row=rows, members=members)
        with self.assertRaisesRegex(ValueError, "future"):
            self.dataset(cfg)[0]


if __name__ == "__main__":
    unittest.main()
