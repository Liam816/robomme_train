"""A separate FP16 8x8 keyframe cache; never reinterpret the old 4x4 cache."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from mme_vla_suite.models.representation.perc_memory_v1 import PercMemConfig
from mme_vla_suite.shared.perc_mem_bank import MemoryClock
from mme_vla_suite.training.lerobot_framesamp import COLUMNS, META_FILE, VideoReader, write_json


FORMAT = "robomme_percmem_v1_fp16_8x8"
MANIFEST = "meta/percmem_v1.json"


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_cache(index_cache, destination, cfg=PercMemConfig()):
    """Copy only numeric indexes from the validated FrameSamp index, not its features."""
    index_cache, destination = Path(index_cache), Path(destination)
    source = json.loads((index_cache / META_FILE).read_text())
    info = Path(source["source"]) / "meta/info.json"
    if hashlib.sha256(info.read_bytes()).hexdigest() != source["source_info_sha256"]:
        raise ValueError("Source metadata no longer matches the original index")
    index_signature = hashlib.sha256((index_cache / META_FILE).read_bytes()).hexdigest()
    marker = destination / MANIFEST
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["index_signature"] != index_signature or old["clock_hz"] != cfg.clock_hz or old["key_stride"] != cfg.key_stride:
            raise ValueError("Existing cache uses a different source or clock protocol")
        return old
    for directory in ["meta", "index", "keyframes", "native"]:
        (destination / directory).mkdir(parents=True, exist_ok=True)
    for name in [*COLUMNS, "sample_indices"]:
        target = destination / f"index/{name}.npy"
        if target.exists():
            raise FileExistsError(f"Uncommitted cache already has {target}; use a fresh directory")
        shutil.copyfile(index_cache / f"index/{name}.npy", target)
    times = np.full(source["total_frames"], np.nan, np.float64)
    for path in sorted((Path(source["source"]) / "data").glob("*/*.parquet")):
        rows = pq.read_table(path, columns=["index", "timestamp"]).to_pydict()
        indices = np.asarray(rows["index"], np.int64)
        if np.isfinite(times[indices]).any():
            raise ValueError("Duplicate timestamp row")
        times[indices] = np.asarray(rows["timestamp"], np.float64)
    if not np.isfinite(times).all():
        raise ValueError("Missing or non-finite source timestamps")
    # VideoReader uses the declared uniform time grid; refuse inconsistent source times.
    count, maximum = 0, 0
    for episode in source["episodes"]:
        begin, end = episode["dataset_from_index"], episode["dataset_to_index"]
        seconds = times[begin:end] - times[begin]
        if not np.allclose(seconds, np.arange(episode["length"]) / source["fps"], atol=2e-5, rtol=0):
            raise ValueError("Source timestamps disagree with video timing; an explicit timing adapter is required")
        clock, selected, ticks = MemoryClock(cfg.clock_hz, cfg.key_stride), [], []
        for row, timestamp in enumerate(seconds):
            keyframe, tick = clock.observe(timestamp)
            if keyframe:
                selected.append(row)
                ticks.append(tick)
        np.savez(destination / f"keyframes/episode_{episode['episode_index']:06d}.npz",
                 row=np.array(selected, np.int32), tick=np.array(ticks, np.int32),
                 timestamp=seconds[selected].astype(np.float64))
        count += len(selected)
        maximum = max(maximum, len(selected))
    np.save(destination / "index/timestamp.npy", times)
    write_json(destination / "meta/source_index.json", source)
    value = {"format": FORMAT, "index_signature": index_signature,
             "source_info_sha256": source["source_info_sha256"], "source": source["source"],
             "fps": source["fps"], "clock_hz": cfg.clock_hz, "key_stride": cfg.key_stride,
             "grid": cfg.store_grid, "native_dim": cfg.native_dim, "storage": "float16",
             "keyframes": count, "max_keyframes": maximum, "episodes": len(source["episodes"]),
             "total_frames": source["total_frames"], "execution_samples": source["execution_samples"]}
    value["cache_id"] = json_hash(value)
    write_json(marker, value)
    return value


def validate_cache(root, *, require_complete=True):
    root = Path(root)
    meta = json.loads((root / MANIFEST).read_text())
    if meta["format"] != FORMAT or meta["storage"] != "float16":
        raise ValueError("Expected a separate FP16 v1 cache, not uint16 BF16 FrameSamp features")
    if require_complete:
        complete = json.loads((root / "meta/native_complete.json").read_text())
        if complete["cache_id"] != meta["cache_id"]:
            raise ValueError("Feature completion marker belongs to a different cache")
        for episode in json.loads((root / "meta/source_index.json").read_text())["episodes"]:
            name = f"episode_{episode['episode_index']:06d}"
            with np.load(root / f"keyframes/{name}.npz") as plan:
                n = len(plan["row"])
            feature = np.load(root / f"native/{name}.npy", mmap_mode="r")
            record = json.loads((root / f"native/{name}.json").read_text())
            if feature.dtype != np.float16 or feature.shape != (n, meta["grid"] ** 2, meta["native_dim"]):
                raise ValueError(f"Invalid native cache {name}")
            if record["cache_id"] != meta["cache_id"] or record["vision_id"] != complete["vision_id"]:
                raise ValueError(f"Incompatible native cache identity {name}")
    return meta


class LeRobotPercMemDataset:
    def __init__(self, dataset_path, data_config, history_config, action_horizon, require_complete=True):
        self.root = Path(dataset_path)
        self.meta = validate_cache(self.root, require_complete=require_complete)
        self.cfg = PercMemConfig.from_history(history_config)
        if self.meta["clock_hz"] != self.cfg.clock_hz or self.meta["key_stride"] != self.cfg.key_stride:
            raise ValueError("Dataset and model have different memory clocks")
        if self.meta["native_dim"] != self.cfg.native_dim or self.meta["grid"] != self.cfg.store_grid:
            raise ValueError("Dataset and model have different stored grids/dimensions")
        if self.meta["max_keyframes"] > self.cfg.max_frames:
            raise ValueError("max_frames would truncate the full-history teacher")
        self.metadata = json.loads((self.root / "meta/source_index.json").read_text())
        info = Path(self.meta["source"]) / "meta/info.json"
        if hashlib.sha256(info.read_bytes()).hexdigest() != self.meta["source_info_sha256"]:
            raise ValueError("Raw source metadata changed")
        self.episodes = {e["episode_index"]: e for e in self.metadata["episodes"]}
        self.action_horizon = action_horizon
        self.reader, self.features, self.plans = None, OrderedDict(), OrderedDict()
        self._open_indices()
        if self.cfg.input_space == "projected":
            projected = json.loads((Path(self.cfg.projected_dir) / "manifest.json").read_text())
            native_id = json.loads((self.root / "meta/native_complete.json").read_text())["vision_id"]
            if projected.get("vision_id") != native_id:
                raise ValueError("Projected/native vision encoder mismatch")
            if projected["cache_id"] != self.meta["cache_id"]:
                raise ValueError("Projected features belong to a different native cache")
            if self.cfg.stage == "student":
                banks = json.loads((Path(self.cfg.banks_dir) / "manifest.json").read_text())
                for key in ["cache_id", "adapter_id", "teacher_id"]:
                    if banks[key] != projected[key]:
                        raise ValueError(f"Bank/projected identity mismatch: {key}")
                with np.load(self.cfg.scorer_path, allow_pickle=False) as checkpoint:
                    scorer = json.loads(str(checkpoint["__metadata__"]))
                if banks["scorer_id"] != scorer["params_id"]:
                    raise ValueError("Offline banks and deployed scorer have different weights")
                if banks["evict"] != self.cfg.evict or banks["capacity"] != self.cfg.capacity:
                    raise ValueError("Bank selection differs from requested policy/budget")
        elif self.cfg.stage == "student":
            raise ValueError("P3 requires frozen projected features and a full teacher bank")

    def _open_indices(self):
        self.arrays = {name: np.load(self.root / f"index/{name}.npy", mmap_mode="r") for name in COLUMNS}
        self.times = np.load(self.root / "index/timestamp.npy", mmap_mode="r")
        self.samples = np.load(self.root / "index/sample_indices.npy", mmap_mode="r")

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(reader=None, features=OrderedDict(), plans=OrderedDict(), arrays=None, times=None, samples=None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_indices()

    def __len__(self):
        return len(self.samples)

    def _episode_features(self, episode_id):
        if episode_id not in self.features:
            directory = self.root / "native" if self.cfg.input_space == "native" else Path(self.cfg.projected_dir)
            feature = np.load(directory / f"episode_{episode_id:06d}.npy", mmap_mode="r")
            with np.load(self.root / f"keyframes/episode_{episode_id:06d}.npz") as plan:
                values = {k: plan[k] for k in plan.files}
            dim = self.cfg.native_dim if self.cfg.input_space == "native" else self.cfg.feature_dim
            if feature.dtype != np.float16 or feature.shape != (len(values["row"]), self.cfg.store_grid ** 2, dim):
                raise ValueError("Cached keyframe feature schema mismatch")
            if self.cfg.stage == "student":
                with np.load(Path(self.cfg.banks_dir) / f"episode_{episode_id:06d}.npz") as bank:
                    values["bank_rows"], values["members"] = bank["row"], bank["members"]
                episode = self.episodes[episode_id]
                expected_rows = np.arange(episode["exec_start_idx"], episode["length"])
                if not np.array_equal(values["bank_rows"], expected_rows):
                    raise ValueError("Banks must cover every execution anchor exactly once")
                if values["members"].shape != (len(expected_rows), self.cfg.capacity) or np.any(values["members"] < -1):
                    raise ValueError("Invalid bank member array or padding value")
            self.features[episode_id], self.plans[episode_id] = feature, values
            if len(self.features) > 8:
                old, _ = self.features.popitem(last=False)
                del self.plans[old]
        self.features.move_to_end(episode_id)
        return self.features[episode_id], self.plans[episode_id]

    def _pack(self, feature, plan, indices, capacity):
        if len(indices) > capacity:
            raise ValueError("Memory truncation is forbidden")
        feats = np.zeros((capacity, *feature.shape[1:]), np.float16)
        ticks, mask = np.zeros(capacity, np.int32), np.zeros(capacity, bool)
        feats[:len(indices)], ticks[:len(indices)], mask[:len(indices)] = feature[indices], plan["tick"][indices], True
        return feats, ticks, mask

    def __getitem__(self, index):
        row = int(self.samples[index])
        episode_id = int(self.arrays["episode_index"][row])
        episode = self.episodes[episode_id]
        step = int(self.arrays["step_idx"][row])
        feature, plan = self._episode_features(episode_id)
        available = np.arange(np.searchsorted(plan["row"], step, side="right"))
        if self.cfg.stage == "student":
            location = np.searchsorted(plan["bank_rows"], step)
            if location == len(plan["bank_rows"]) or plan["bank_rows"][location] != step:
                raise ValueError("Missing bank for an execution training anchor")
            members = plan["members"][location]
            selected = members[members >= 0]
            if len(np.unique(selected)) != len(selected) or np.any(selected >= len(available)):
                raise ValueError("Bank has duplicate or future keyframes")
            packed = self._pack(feature, plan, selected, self.cfg.capacity)
        else:
            packed = self._pack(feature, plan, available, self.cfg.max_frames)
        if self.reader is None:
            self.reader = VideoReader(self.metadata)
        action_rows = np.minimum(row + np.arange(self.action_horizon), episode["dataset_to_index"] - 1)
        seconds = self.times[row] - self.times[episode["dataset_from_index"]]
        result = {"image": next(self.reader.frames(episode, "image", step, 1)),
                  "wrist_image": next(self.reader.frames(episode, "wrist_image", step, 1)),
                  "state": np.array(self.arrays["state"][row]), "actions": np.array(self.arrays["actions"][action_rows]),
                  "prompt": self.metadata["tasks"][str(int(self.arrays["task_index"][row]))].lower(),
                  "simple_subgoal": "", "grounded_subgoal": "",
                  "perc_mem_feats": packed[0], "perc_mem_frame_tick": packed[1], "perc_mem_frame_mask": packed[2],
                  "perc_mem_now_tick": np.int32(np.rint(seconds * self.cfg.clock_hz))}
        for key in ["static_image_emb", "static_pos_emb", "static_state_emb", "static_mask",
                    "recur_image_emb", "recur_pos_emb", "recur_state_emb", "recur_mask"]:
            result[key] = None
        if self.cfg.stage == "student":
            full = self._pack(feature, plan, available, self.cfg.max_frames)
            result.update(perc_teacher_feats=full[0], perc_teacher_tick=full[1], perc_teacher_mask=full[2])
        return result
