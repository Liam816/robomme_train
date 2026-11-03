"""CPU streaming bank and clock, shared by offline replay and serving."""

from dataclasses import dataclass
import numpy as np


class MemoryClock:
    def __init__(self, clock_hz=20, stride=10):
        self.hz, self.stride = clock_hz, stride
        self.reset()

    def reset(self):
        self.origin = None
        self.last_time = None
        self.last_bucket = -1
        self.now_tick = 0

    def observe(self, timestamp):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("Observation timestamp must be finite")
        if self.last_time is not None and timestamp < self.last_time - 1e-8:
            raise ValueError("Observations must arrive in chronological order")
        if self.origin is None:
            self.origin = timestamp
        self.last_time = timestamp
        self.now_tick = int(np.rint((timestamp - self.origin) * self.hz))
        bucket = self.now_tick // self.stride
        if bucket <= self.last_bucket:
            return False, self.now_tick
        self.last_bucket = bucket
        return True, self.now_tick


@dataclass
class Frame:
    index: int
    tick: int
    tokens: np.ndarray | None = None


def eviction_index(ticks, mode, *, scores=None, utility=None, rng=None):
    ticks = np.asarray(ticks)
    if mode == "learned":
        if scores is None or np.shape(scores) != ticks.shape or not np.isfinite(scores).all():
            raise ValueError("Learned eviction needs finite scores for every candidate")
        return int(np.argmin(scores))
    if mode == "oracle":
        if utility is None or np.shape(utility) != ticks.shape or not np.isfinite(utility).all():
            raise ValueError("Oracle is offline-only and requires future utility")
        return int(np.argmin(utility))
    if mode == "fifo":
        return int(np.argmin(ticks))
    if mode == "random":
        if rng is None:
            raise ValueError("Random eviction requires an explicit RNG")
        return int(rng.integers(len(ticks)))
    if mode == "uniform":
        order = np.argsort(ticks, kind="stable")
        if len(order) < 3:
            return int(order[0])
        span = ticks[order[2:]] - ticks[order[:-2]]
        return int(order[1:-1][np.argmin(span)])
    raise ValueError(f"Unknown eviction mode {mode}")


class PercMemBank:
    def __init__(self, recent=4, queue=28, mode="learned", *, score_fn=None, seed=42):
        if recent < 1 or queue < 1:
            raise ValueError("Bank capacity must be positive")
        self.recent, self.queue, self.mode = recent, queue, mode
        self.score_fn, self.seed = score_fn, seed
        self.reset()

    def reset(self):
        self.W, self.Q = [], []
        self.rng = np.random.default_rng(self.seed)
        self.last_tick = -1

    def write(self, index, tick, tokens=None, *, utility_fn=None, mode=None):
        if tick <= self.last_tick:
            raise ValueError("Keyframes must be unique and strictly chronological")
        self.last_tick = int(tick)
        frame = Frame(int(index), int(tick), None if tokens is None else np.array(tokens, np.float16, copy=True))
        self.W.append(frame)
        if len(self.W) <= self.recent:
            return None
        candidate = self.W.pop(0)
        if len(self.Q) < self.queue:
            self.Q.append(candidate)
            return None
        candidates = self.Q + [candidate]
        selected_mode = mode or self.mode
        scores = self.score_fn(candidates, tick) if selected_mode == "learned" and self.score_fn else None
        utility = utility_fn(candidates, tick) if selected_mode == "oracle" and utility_fn else None
        j = eviction_index([f.tick for f in candidates], selected_mode, scores=scores, utility=utility, rng=self.rng)
        if j != self.queue:
            self.Q[j] = candidate
        self.Q.sort(key=lambda f: f.tick)
        return candidates, j

    def frames(self):
        return sorted(self.Q + self.W, key=lambda f: f.tick)

    def arrays(self, capacity, feature_dim=2560, store_tokens=64):
        frames = self.frames()
        if len(frames) > capacity:
            raise ValueError("Bank exceeds configured observation capacity")
        feats = np.zeros((capacity, store_tokens, feature_dim), np.float16)
        ticks, valid = np.zeros(capacity, np.int32), np.zeros(capacity, bool)
        indices = np.full(capacity, -1, np.int32)
        for i, frame in enumerate(frames):
            if frame.tokens is not None:
                if frame.tokens.shape != feats.shape[1:]:
                    raise ValueError("Stored feature shape does not match memory configuration")
                feats[i] = frame.tokens
            ticks[i], valid[i], indices[i] = frame.tick, True, frame.index
        return feats, ticks, valid, indices
