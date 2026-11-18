"""Ordered, task-balanced frame pairs from the RoboMME LeRobot v3 training release."""

from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from mme_vla_suite.training.deltatok import IMAGE_SIZE
from mme_vla_suite.training.lerobot_framesamp import VideoReader

# pepijn223/robomme_data_lerobot_video preserves the official v2.1 episode order.
# Each task occupies 100 contiguous episodes. task_index is a prompt ID, and
# VideoUnmask and VideoUnmaskSwap share prompts.
TASKS = (
    "PatternLock", "ButtonUnmaskSwap", "ButtonUnmask", "VideoPlaceButton",
    "VideoUnmaskSwap", "PickXtimes", "StopCube", "SwingXtimes",
    "PickHighlight", "MoveCube", "InsertPeg", "RouteStick",
    "BinFill", "VideoPlaceOrder", "VideoRepick", "VideoUnmask",
)
EPISODES_PER_TASK = 100
CAMERAS = ("image", "wrist_image")


class RoboMMEDeltaPairs(Dataset):
    def __init__(self, root: Path, frame_gap: int, cameras: tuple[str, ...]):
        root = root.resolve()
        if frame_gap < 1:
            raise ValueError("frame_gap must be positive")
        if not cameras or len(set(cameras)) != len(cameras) or not set(cameras).issubset(CAMERAS):
            raise ValueError(f"Choose distinct cameras from {CAMERAS}")
        raw_info = (root / "meta/info.json").read_bytes()
        info = json.loads(raw_info)
        num_episodes = len(TASKS) * EPISODES_PER_TASK
        if info["codebase_version"] != "v3.0" or info["total_episodes"] != num_episodes:
            raise ValueError("Expected the 1,600-episode RoboMME LeRobot v3 video release")
        for camera in cameras:
            feature = info["features"][camera]
            if feature["dtype"] != "video" or feature["shape"] != [IMAGE_SIZE, IMAGE_SIZE, 3]:
                raise ValueError(f"{camera} must contain {IMAGE_SIZE}×{IMAGE_SIZE} RGB video")

        columns = ["episode_index", "length"]
        for camera in cameras:
            columns.extend(f"videos/{camera}/{field}" for field in ("chunk_index", "file_index", "from_timestamp"))
        episodes = []
        for path in sorted((root / "meta/episodes").glob("*/*.parquet")):
            episodes.extend(pq.read_table(path, columns=columns).to_pylist())
        episodes.sort(key=lambda episode: episode["episode_index"])
        if [row["episode_index"] for row in episodes] != list(range(num_episodes)):
            raise ValueError("RoboMME episode IDs are incomplete or duplicated")
        if sum(row["length"] for row in episodes) != info["total_frames"]:
            raise ValueError("Episode lengths disagree with dataset total_frames")
        for camera in cameras:
            files = {(row[f"videos/{camera}/chunk_index"], row[f"videos/{camera}/file_index"]) for row in episodes}
            for chunk, file in files:
                path = root / info["video_path"].format(video_key=camera, chunk_index=chunk, file_index=file)
                if not path.is_file():
                    raise FileNotFoundError(path)

        self.cameras = cameras
        self.frame_gap = frame_gap
        self.metadata = {"source": str(root), "video_path": info["video_path"], "fps": info["fps"]}
        self.fingerprint = hashlib.sha256(raw_info + json.dumps(episodes, sort_keys=True).encode()).hexdigest()
        self.episodes = episodes
        self.pair_offsets = [0]
        for row in episodes:
            self.pair_offsets.append(self.pair_offsets[-1] + max(0, row["length"] - frame_gap) * len(cameras))
        self.task_offsets = np.array(self.pair_offsets[::EPISODES_PER_TASK], dtype=np.int64)
        if np.any(np.diff(self.task_offsets) == 0):
            raise ValueError(f"At least one RoboMME task has no valid pairs for frame_gap={frame_gap}")
        self.reader = None

    def __len__(self) -> int:
        return self.pair_offsets[-1]

    def __getstate__(self):
        return {**self.__dict__, "reader": None}

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_id = bisect_right(self.pair_offsets, index) - 1
        start, camera_id = divmod(index - self.pair_offsets[episode_id], len(self.cameras))
        if self.reader is None:
            self.reader = VideoReader(self.metadata)
        frames = self.reader.frames(
            self.episodes[episode_id], self.cameras[camera_id], start, 2, stride=self.frame_gap
        )
        return torch.from_numpy(np.stack(list(frames))).permute(0, 3, 1, 2).contiguous()


class TaskBalancedBatches(Sampler[list[int]]):
    """Draw a global task-balanced batch deterministically, then take this rank's slice."""

    def __init__(
        self, task_offsets: np.ndarray, batch_size: int, rank: int, world_size: int,
        seed: int, start_step: int, total_steps: int,
    ):
        self.task_offsets = task_offsets
        self.rank_slice = slice(rank * batch_size, (rank + 1) * batch_size)
        self.global_batch_size = batch_size * world_size
        self.steps = range(start_step, total_steps)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        num_tasks = len(self.task_offsets) - 1
        for step in self.steps:
            rng = np.random.default_rng([self.seed, step])
            tasks = np.resize(rng.permutation(num_tasks), self.global_batch_size)
            rng.shuffle(tasks)
            pairs = rng.integers(self.task_offsets[tasks], self.task_offsets[tasks + 1])
            yield pairs[self.rank_slice].tolist()
