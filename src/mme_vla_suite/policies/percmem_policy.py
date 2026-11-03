"""Bounded per-policy memory; explicit observation timestamps or control_hz."""

import jax
import jax.numpy as jnp
import numpy as np

from mme_vla_suite.models.representation.perc_memory_v1 import PercMemConfig
from mme_vla_suite.policies.policy import MME_VLA_Policy
from mme_vla_suite.shared.perc_mem_bank import MemoryClock, PercMemBank
from mme_vla_suite.shared.percmem_io import FrozenHistoryEncoder, module_params
from openpi.shared.nnx_utils import module_jit


class PercMemPolicy(MME_VLA_Policy):
    def _prepare_mem_buffer(self):
        self.perc_cfg = cfg = PercMemConfig.from_history(self.config)
        if cfg.evict == "oracle":
            raise ValueError("Oracle future labels are not available during real serving")
        if cfg.scorer_use_instr:
            raise ValueError("Instruction-conditioned scorer is an offline ablation; serving requires explicit embedding input")
        if not hasattr(self, "_history_encode"):
            self._history_encoder = FrozenHistoryEncoder(module_params(self._model.PaliGemma.img))
            self._history_encode = module_jit(self._history_encoder.__call__)
            self._project = module_jit(self._model.project_memory)
            self._score = module_jit(self._model.mem_scorer.__call__, static_argnames=("deterministic",))
        self.clock = MemoryClock(cfg.clock_hz, cfg.key_stride)
        self.mem_buffer = PercMemBank(cfg.recent_frames, cfg.queue_frames, cfg.evict,
                                     score_fn=self._score_candidates, seed=self._seed)
        self.full_frames = [] if cfg.stage == "teacher" else None
        self.next_key = 0

    def _score_candidates(self, candidates, now):
        tokens = jnp.asarray(np.stack([f.tokens for f in candidates]))[None]
        ticks = jnp.array([[f.tick for f in candidates]], jnp.int32)
        valid = jnp.ones(ticks.shape, bool)
        is_new = jnp.arange(len(candidates))[None] == len(candidates) - 1
        return np.asarray(self._score(tokens, ticks, valid, jnp.array([now]), is_new))[0]

    def add_buffer(self, obs):
        cfg = self.perc_cfg
        images = np.asarray(obs["images"])
        if images.ndim == 5:
            if images.shape[1] != 1:
                raise ValueError("Perceptual v1 stores only the front view")
            images = images[:, 0]
        if images.ndim != 4 or images.dtype != np.uint8:
            raise ValueError("Expected uint8 front-camera RGB observations")
        timestamps = obs.get("timestamps")
        if timestamps is None:
            hz = obs.get("control_hz", cfg.control_hz)
            if hz is None or not np.isfinite(hz) or hz <= 0:
                raise ValueError("Supply timestamps or a verified positive control_hz; video playback FPS is insufficient")
            timestamps = (self.step_idx + 1 + np.arange(len(images))) / float(hz)
        if len(timestamps) != len(images):
            raise ValueError("One timestamp is required per observation")
        pending, times = [], []

        def flush():
            if not pending:
                return
            native = self._history_encode(jnp.asarray(np.stack(pending)))
            encoded = native if cfg.input_space == "native" else self._project(native)
            for value, tick in zip(np.asarray(encoded), times, strict=True):
                if cfg.stage == "teacher":
                    if len(self.full_frames) >= cfg.max_frames:
                        raise ValueError("Full-history teacher exceeds max_frames; truncation is forbidden")
                    from mme_vla_suite.shared.perc_mem_bank import Frame
                    self.full_frames.append(Frame(self.next_key, tick, value.copy()))
                else:
                    self.mem_buffer.write(self.next_key, tick, value)
                self.next_key += 1
            pending.clear()
            times.clear()

        for image, timestamp in zip(images, timestamps, strict=True):
            selected, tick = self.clock.observe(timestamp)
            if selected:
                pending.append(image)
                times.append(tick)
                if len(pending) == 16:
                    flush()
        flush()
        self.step_idx += len(images)

    def _prepare_history(self, inputs):
        cfg = self.perc_cfg
        n = cfg.max_frames if cfg.stage == "teacher" else cfg.capacity
        dim = cfg.native_dim if cfg.input_space == "native" else cfg.feature_dim
        if cfg.stage == "teacher":
            # Reuse the same packing contract without retaining a second full-history dictionary.
            bank = PercMemBank()
            bank.Q = self.full_frames
        else:
            bank = self.mem_buffer
        feats, ticks, mask, _ = bank.arrays(n, dim, cfg.store_grid ** 2)
        inputs.update(perc_mem_feats=feats, perc_mem_frame_tick=ticks, perc_mem_frame_mask=mask,
                      perc_mem_now_tick=np.int32(self.clock.now_tick))
        return inputs
