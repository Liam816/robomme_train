"""Read RoboMME LeRobot v3 videos with a compact, front-camera FrameSamp cache."""

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import av
import ml_dtypes
import numpy as np
import pyarrow.parquet as pq


FORMAT = "robomme_lerobot_framesamp_v1"
META_FILE = "meta/lerobot_framesamp.json"
COLUMNS = {
    "state": (np.float32, (8,)),
    "actions": (np.float32, (8,)),
    "episode_index": (np.int32, ()),
    "step_idx": (np.int32, ()),
    "exec_start_idx": (np.int32, ()),
    "task_index": (np.int32, ()),
    "is_demo": (np.bool_, ()),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def prepare_index(source, cache):
    """Index numeric columns once; keep all video files in the original dataset."""
    source, cache = Path(source).resolve(), Path(cache).resolve()
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("This adapter requires LeRobot v3.0")
    signature = hashlib.sha256(info_path.read_bytes()).hexdigest()
    marker = cache / META_FILE
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["source"] != str(source) or previous["source_info_sha256"] != signature:
            raise ValueError("Cache belongs to a different source dataset")
        return previous

    count = info["total_frames"]
    index_dir = cache / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.lib.format.open_memmap(index_dir / f"{name}.npy", mode="w+", dtype=dtype,
                                             shape=(count, *shape))
              for name, (dtype, shape) in COLUMNS.items()}
    seen = np.zeros(count, dtype=np.bool_)
    for path in sorted((source / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=["index", *COLUMNS]).to_pydict()
        indices = np.asarray(table.pop("index"), dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= count) or np.any(seen[indices]) or len(np.unique(indices)) != len(indices):
            raise ValueError(f"Invalid or duplicate global row indices in {path}")
        for name, (dtype, _) in COLUMNS.items():
            arrays[name][indices] = np.asarray(table[name], dtype=dtype)
        seen[indices] = True
    if not np.all(seen):
        raise ValueError(f"Missing {count - int(seen.sum())} rows in source dataset")

    episodes = []
    for path in sorted((source / "meta/episodes").glob("*/*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda row: row["episode_index"])
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode metadata is incomplete")
    for episode in episodes:
        begin, end = episode["dataset_from_index"], episode["dataset_to_index"]
        length = episode["length"]
        if end - begin != length or not np.all(arrays["episode_index"][begin:end] == episode["episode_index"]):
            raise ValueError(f"Episode row boundaries disagree: {episode['episode_index']}")
        if not np.array_equal(arrays["step_idx"][begin:end], np.arange(length)):
            raise ValueError(f"Non-contiguous steps in episode {episode['episode_index']}")
        execution_start = int(arrays["exec_start_idx"][begin])
        if not np.all(arrays["exec_start_idx"][begin:end] == execution_start):
            raise ValueError("Execution start changes within an episode")
        if not np.array_equal(arrays["is_demo"][begin:end], np.arange(length) < execution_start):
            raise ValueError("Demonstration mask disagrees with execution start")
        episode["exec_start_idx"] = execution_start
        for key in ("image", "wrist_image"):
            video = source / info["video_path"].format(video_key=key,
                chunk_index=episode[f"videos/{key}/chunk_index"], file_index=episode[f"videos/{key}/file_index"])
            if not video.is_file():
                raise FileNotFoundError(video)
    for value in arrays.values():
        value.flush()
    samples = np.flatnonzero(~arrays["is_demo"])
    np.save(index_dir / "sample_indices.npy", samples)
    tasks = pq.read_table(source / "meta/tasks.parquet").to_pylist()
    metadata = {"format": FORMAT, "source": str(source), "source_info_sha256": signature,
                "total_frames": count, "execution_samples": len(samples), "fps": info["fps"],
                "video_path": info["video_path"], "episodes": episodes,
                "tasks": {str(row["task_index"]): row["task"] for row in tasks},
                "feature_shape_per_frame": [16, 2048], "feature_storage": "uint16_bfloat16_bits"}
    write_json(marker, metadata)
    return metadata


class VideoReader:
    """Bound open decoders and align video frames using LeRobot timestamps."""

    def __init__(self, metadata, max_open=4):
        self.metadata = metadata
        self.max_open = max_open
        self.containers = OrderedDict()

    def _container(self, episode, key):
        path = Path(self.metadata["source"]) / self.metadata["video_path"].format(video_key=key,
            chunk_index=episode[f"videos/{key}/chunk_index"], file_index=episode[f"videos/{key}/file_index"])
        path = str(path)
        if path not in self.containers:
            container = av.open(path)
            container.streams.video[0].codec_context.thread_count = 2
            self.containers[path] = container
            if len(self.containers) > self.max_open:
                _, previous = self.containers.popitem(last=False)
                previous.close()
        self.containers.move_to_end(path)
        return self.containers[path]

    def frames(self, episode, key="image", start=0, count=None):
        count = episode["length"] - start if count is None else count
        if start < 0 or count < 1 or start + count > episode["length"]:
            raise ValueError("Requested frames cross episode boundaries")
        container = self._container(episode, key)
        stream = container.streams.video[0]
        fps = self.metadata["fps"]
        target = episode[f"videos/{key}/from_timestamp"] + start / fps
        container.seek(max(0, int(target / stream.time_base)), stream=stream, backward=True)
        emitted = 0
        tolerance = 0.45 / fps
        for frame in container.decode(stream):
            timestamp = float(frame.pts * stream.time_base)
            expected = target + emitted / fps
            if timestamp < expected - tolerance:
                continue
            if abs(timestamp - expected) > tolerance:
                raise ValueError(f"Video timestamp mismatch: episode={episode['episode_index']} {timestamp=} {expected=}")
            yield frame.to_ndarray(format="rgb24")
            emitted += 1
            if emitted == count:
                return
        raise ValueError(f"Truncated video for episode {episode['episode_index']}: {emitted}/{count}")

    def close(self):
        for container in self.containers.values():
            container.close()
        self.containers.clear()


class LeRobotFrameSampDataset:
    """Provide the original MME-VLA sample interface without expanded pickle files."""

    def __init__(self, dataset_path, data_config, history_config, action_horizon, require_complete=True):
        self.root = Path(dataset_path)
        self.metadata = json.loads((self.root / META_FILE).read_text())
        if self.metadata["format"] != FORMAT:
            raise ValueError("Unknown FrameSamp cache format")
        source_info = Path(self.metadata["source"]) / "meta/info.json"
        if hashlib.sha256(source_info.read_bytes()).hexdigest() != self.metadata["source_info_sha256"]:
            raise ValueError("Source dataset metadata has changed since cache preparation")
        if history_config is None or history_config.representation_type != "perceptual" or history_config.perceptual_memory.type != "frame_sampling":
            raise ValueError("This compact cache supports perceptual FrameSamp policies only")
        if history_config.num_views != 1 or history_config.token_per_image != 16:
            raise ValueError("This cache contains one front camera and 16 tokens per frame")
        if history_config.budget < 16 or history_config.budget % 16:
            raise ValueError("FrameSamp token budget must be a positive multiple of 16")
        if require_complete and not (self.root / "meta/features_complete.json").exists():
            raise RuntimeError("Feature cache is incomplete; finish scripts/build_lerobot_framesamp.py first")
        self.cfg, self.action_horizon = history_config, action_horizon
        self.arrays = {name: np.load(self.root / f"index/{name}.npy", mmap_mode="r") for name in COLUMNS}
        self.samples = np.load(self.root / "index/sample_indices.npy", mmap_mode="r")
        self.episodes = {row["episode_index"]: row for row in self.metadata["episodes"]}
        self.spatial = np.load(self.root / "meta/position_spatial.npy")
        self.temporal = np.load(self.root / "meta/position_temporal.npy", mmap_mode="r")
        self.stats, self.quantiles = data_config.norm_stats["state"], data_config.use_quantile_norm
        self.reader = None
        self.features = OrderedDict()

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["reader"] = None
        state["features"] = OrderedDict()
        for key in ("arrays", "samples", "temporal"):
            state[key] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.arrays = {name: np.load(self.root / f"index/{name}.npy", mmap_mode="r") for name in COLUMNS}
        self.samples = np.load(self.root / "index/sample_indices.npy", mmap_mode="r")
        self.temporal = np.load(self.root / "meta/position_temporal.npy", mmap_mode="r")

    def __getitem__(self, index):
        row = int(self.samples[index])
        episode_id = int(self.arrays["episode_index"][row])
        episode = self.episodes[episode_id]
        step = int(self.arrays["step_idx"][row])
        if self.reader is None:
            self.reader = VideoReader(self.metadata)
        if episode_id not in self.features:
            path = self.root / f"features/episode_{episode_id:06d}.npy"
            feature = np.load(path, mmap_mode="r")
            if feature.shape != (episode["length"], 16, 2048) or feature.dtype != np.uint16:
                raise ValueError(f"Unexpected feature cache shape or dtype: {path}")
            self.features[episode_id] = feature
            if len(self.features) > 8:
                self.features.popitem(last=False)
        self.features.move_to_end(episode_id)
        max_frames = self.cfg.budget // 16
        indices = np.arange(step + 1) if step < max_frames else np.linspace(0, step, max_frames, dtype=np.int32)
        valid = len(indices)
        images = np.zeros((max_frames, 16, 2048), dtype=np.float32)
        images[:valid] = self.features[episode_id][indices].view(ml_dtypes.bfloat16).astype(np.float32)
        positions = np.zeros((max_frames, 16, 768), dtype=np.float32)
        positions[:valid, :, :256] = self.temporal[indices, None, :]
        positions[:valid, :, 256:] = self.spatial[None, :, :]
        states = np.zeros((max_frames, 8), dtype=np.float32)
        states[:valid] = self.arrays["state"][episode["dataset_from_index"] + indices]
        if self.quantiles:
            states = (states - self.stats.q01) / (self.stats.q99 - self.stats.q01 + 1e-6) * 2.0 - 1.0
        else:
            states = (states - self.stats.mean) / (self.stats.std + 1e-6)
        action_rows = np.minimum(row + np.arange(self.action_horizon), episode["dataset_to_index"] - 1)
        mask = np.arange(max_frames) < valid
        return {"image": next(self.reader.frames(episode, "image", step, 1)),
                "wrist_image": next(self.reader.frames(episode, "wrist_image", step, 1)),
                "state": np.array(self.arrays["state"][row]),
                "actions": np.array(self.arrays["actions"][action_rows]),
                "prompt": self.metadata["tasks"][str(int(self.arrays["task_index"][row]))].lower(),
                "simple_subgoal": "", "grounded_subgoal": "",
                "static_image_emb": images.reshape(-1, 2048),
                "static_pos_emb": positions.reshape(-1, 768),
                "static_state_emb": np.repeat(states, 16, axis=0),
                "static_mask": np.repeat(mask, 16),
                "recur_image_emb": None, "recur_pos_emb": None,
                "recur_state_emb": None, "recur_mask": None}
