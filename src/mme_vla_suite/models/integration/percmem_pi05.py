"""pi0.5 action conditioning via the original v1 global memory reader."""

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp

from mme_vla_suite.models.integration.history_observation import HistAugObservation, preprocess_observation
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0, make_attn_mask
from mme_vla_suite.models.representation.perc_memory_v1 import EvictScorer, MemReader, PercMemConfig


def _identity_pad(key, shape, dtype=jnp.float32):
    del key
    return jnp.eye(shape[0], shape[1], dtype=dtype)


# NNX stores initializer callables in static graph metadata. Reuse their identity
# across eval_shape and the subsequent jitted train-state initialization.
_COND_KERNEL_INIT = nnx.initializers.xavier_uniform()


def inputs_spec(base, action, history, batch_size):
    cfg = PercMemConfig.from_history(history)
    n = cfg.max_frames if cfg.stage == "teacher" else cfg.capacity
    dim = cfg.native_dim if cfg.input_space == "native" else cfg.feature_dim
    values = dict(
        perc_mem_feats=jax.ShapeDtypeStruct((batch_size, n, cfg.store_grid ** 2, dim), jnp.float16),
        perc_mem_frame_tick=jax.ShapeDtypeStruct((batch_size, n), jnp.int32),
        perc_mem_frame_mask=jax.ShapeDtypeStruct((batch_size, n), jnp.bool_),
        perc_mem_now_tick=jax.ShapeDtypeStruct((batch_size,), jnp.int32),
    )
    if cfg.stage == "student":
        values.update(
            perc_teacher_feats=jax.ShapeDtypeStruct(
                (batch_size, cfg.max_frames, cfg.store_grid ** 2, cfg.feature_dim), jnp.float16),
            perc_teacher_tick=jax.ShapeDtypeStruct((batch_size, cfg.max_frames), jnp.int32),
            perc_teacher_mask=jax.ShapeDtypeStruct((batch_size, cfg.max_frames), jnp.bool_),
        )
    return HistAugObservation.from_base_obs(base, **values), action


class PercMemPi05(HistoryPi0):
    def __init__(self, config, rngs):
        if not config.pi05:
            raise ValueError("perc_adarms requires pi05=True")
        cfg = PercMemConfig.from_history(config.history_config)
        # Build exactly the current-frame pi0.5 backbone: no old Modul parameters.
        super().__init__(dataclasses.replace(config, use_history=False, history_config=None), rngs)
        if self.action_in_proj.out_features != 1024 and config.action_expert_variant != "dummy":
            raise ValueError("Production perceptual v1 requires the pi0.5 1024-wide action expert")
        self.config, self.history_config = config, config.history_config
        self.use_history, self.integration_type, self.representation_type = True, "perc_adarms", "perceptual"
        self.memory_cfg = cfg

        self.mem_feature_adapter = nnx.Linear(cfg.native_dim, cfg.feature_dim, kernel_init=_identity_pad, rngs=rngs)
        self.mem_reader = MemReader(cfg, rngs=rngs)
        self.mem_cond_adapter = nnx.Linear(cfg.output_dim, self.action_in_proj.out_features,
                                           use_bias=False, kernel_init=_COND_KERNEL_INIT, rngs=rngs)
        self.mem_scorer = EvictScorer(cfg, rngs=rngs)

    def project_memory(self, native):
        return self.mem_feature_adapter(native.astype(jnp.float32)).astype(jnp.float16)

    def memory_condition(self, observation, *, return_attention=False):
        cfg = self.memory_cfg
        if observation.perc_mem_feats is None:
            zeros = jnp.zeros((observation.state.shape[0], self.action_in_proj.out_features), jnp.float32)
            if return_attention:
                raise ValueError("Attention labels require a nonempty memory input structure")
            return zeros
        feats, ticks, mask = observation.perc_mem_feats, observation.perc_mem_frame_tick, observation.perc_mem_frame_mask
        if cfg.memory_mode == "shuffle":
            # Caller must construct a batch from distinct episodes for this diagnostic.
            feats, ticks, mask = (jnp.roll(x, 1, axis=0) for x in (feats, ticks, mask))
            # Keep donor ages while expressing them on the recipient clock: no future ticks.
            ticks = ticks + observation.perc_mem_now_tick[:, None] - jnp.roll(observation.perc_mem_now_tick, 1)[:, None]
        if cfg.input_space == "native":
            feats = self.project_memory(feats)
        elif feats.shape[-1] != cfg.feature_dim:
            raise ValueError("Projected memory has the wrong feature dimension")
        result = self.mem_reader(feats.astype(jnp.float32), ticks, mask, observation.perc_mem_now_tick,
                                 return_attention=return_attention)
        c_mem, attention = result if return_attention else (result, None)
        cond = self.mem_cond_adapter(c_mem)
        cond = jnp.where(mask.any(axis=1)[:, None], cond, 0.0)
        if cfg.memory_mode == "zero":
            cond = jnp.zeros_like(cond)
        return (cond, attention) if return_attention else cond

    def encode_prefix(self, observation):
        prefix, mask, ar, _, _ = self.embed_prefix(observation)
        _, cache = self.PaliGemma.llm(
            [prefix, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, axis=1) - 1)
        return mask, cache

    def predict_velocity(self, observation, noisy_actions, flow_t, *, memory_cond=None, prefix_cache=None):
        if memory_cond is None:
            memory_cond = self.memory_condition(observation)
        suffix, smask, sar, _, time_cond = self.embed_suffix(observation, noisy_actions, flow_t)
        cond = time_cond + memory_cond.astype(time_cond.dtype)
        if prefix_cache is None:
            prefix, pmask, par, _, _ = self.embed_prefix(observation)
            mask = jnp.concatenate([pmask, smask], axis=1)
            ar = jnp.concatenate([par, sar])
            (_, suffix_out), _ = self.PaliGemma.llm(
                [prefix, suffix], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, axis=1) - 1,
                adarms_cond=[None, cond])
        else:
            pmask, cache = prefix_cache
            prefix_mask = jnp.broadcast_to(pmask[:, None, :], (smask.shape[0], smask.shape[1], pmask.shape[1]))
            mask = jnp.concatenate([prefix_mask, make_attn_mask(smask, sar)], axis=-1)
            positions = jnp.sum(pmask, axis=-1)[:, None] + jnp.cumsum(smask, axis=1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix], mask=mask, positions=positions, kv_cache=cache, adarms_cond=[None, cond])
        return self.action_out_proj(suffix_out[:, -self.action_horizon:])

    def compute_loss(self, rng, observation, actions, *, train=False):
        prep_key, noise_key, time_key = jax.random.split(rng, 3)
        observation = preprocess_observation(prep_key, observation, train=train)
        noise = jax.random.normal(noise_key, actions.shape)
        time = jax.random.beta(time_key, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        noisy = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        velocity = self.predict_velocity(observation, noisy, time)
        return jnp.mean(jnp.square(velocity - (noise - actions)), axis=-1), None

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        observation = preprocess_observation(None, observation, train=False)
        delta = self.memory_condition(observation)  # Once per decision, outside the flow loop.
        prefix = self.encode_prefix(observation)
        b = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (b, self.action_horizon, self.action_dim))
        dt = -1.0 / num_steps

        def step(carry):
            x, time = carry
            velocity = self.predict_velocity(observation, x, jnp.broadcast_to(time, (b,)),
                                              memory_cond=delta, prefix_cache=prefix)
            return x + dt * velocity, time + dt

        result, _ = jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, step, (noise, 1.0))
        return result


def distillation_loss(student, teacher, rng, observation, actions, *, train=True):
    """One augmentation/noise/time draw; one frozen current-prefix computation."""
    prep_key, noise_key, time_key = jax.random.split(rng, 3)
    observation = preprocess_observation(prep_key, observation, train=train)
    if observation.perc_teacher_feats is None:
        raise ValueError("P3 requires a separate full-history teacher bank")
    teacher_obs = dataclasses.replace(
        observation, perc_mem_feats=observation.perc_teacher_feats,
        perc_mem_frame_tick=observation.perc_teacher_tick, perc_mem_frame_mask=observation.perc_teacher_mask)
    noise = jax.random.normal(noise_key, actions.shape)
    time = jax.random.beta(time_key, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    noisy = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
    prefix = jax.tree.map(jax.lax.stop_gradient, teacher.encode_prefix(observation))
    target = jax.lax.stop_gradient(teacher.predict_velocity(teacher_obs, noisy, time, prefix_cache=prefix))
    velocity = student.predict_velocity(observation, noisy, time, prefix_cache=prefix)
    fm = jnp.mean(jnp.square(velocity - (noise - actions)))
    distill = jnp.mean(jnp.square(velocity - target))
    return fm + student.memory_cfg.distill_weight * distill, {"flow_loss": fm, "distill_loss": distill}
