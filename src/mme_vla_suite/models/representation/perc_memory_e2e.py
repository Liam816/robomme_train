"""Hard W/Q streaming forward with a softmax straight-through eviction derivative.

Replay stores frame indices and a small membership matrix, not an entire token
bank per scan step. The scorer frame encoder is evaluated once per source frame;
its original contextual layers still see Q + the frame leaving W at every write.
No history gradient is truncated and no offline bank membership is reused.
"""

import jax
import jax.numpy as jnp


@jax.custom_jvp
def hard_eviction(scores):
    """Exactly one-hot argmin forward; derivative of softmax(-scores)."""
    return jax.nn.one_hot(jnp.argmin(scores, axis=-1), scores.shape[-1], dtype=scores.dtype)


@hard_eviction.defjvp
def _hard_eviction_jvp(primals, tangents):
    (scores,), (delta,) = primals, tangents
    probability = jax.nn.softmax(-scores, axis=-1)
    derivative = -probability * (delta - jnp.sum(probability * delta, axis=-1, keepdims=True))
    return hard_eviction(scores), derivative


def _gather(features, indices):
    return features[jnp.arange(features.shape[0])[:, None], indices]


@jax.custom_jvp
def select_members(features, membership, indices, valid):
    """Exact hard gather forward, derivative of membership @ features.

    Forward membership is always zero/one. Gathering avoids both a large dense
    token matmul and rounding the scorer's FP32 embeddings through tensor cores.
    """
    del membership
    shape = (*valid.shape, *((1,) * (features.ndim - 2)))
    return _gather(features, indices) * valid.reshape(shape)


@select_members.defjvp
def _select_members_jvp(primals, tangents):
    features, membership, indices, valid = primals
    d_features, d_membership, _, _ = tangents
    soft = jnp.einsum("bkn,bnd->bkd", d_membership, features.reshape(*features.shape[:2], -1),
                      precision=jax.lax.Precision.HIGHEST)
    shape = (*valid.shape, *((1,) * (features.ndim - 2)))
    derivative = (_gather(d_features, indices) + soft.reshape(valid.shape + features.shape[2:])) * valid.reshape(shape)
    return select_members(features, membership, indices, valid), derivative


def replay_memory(scorer, features, ticks, valid, now, cfg):
    """Rebuild each anchor's bank with current weights; return tokens/ticks/mask/ids.

    Inputs are chronological keyframes. Padding holes and future-masked frames do
    not advance W or Q. Write-time ages use that keyframe's tick; reader ages use
    the caller's decision time. Queue scoring disables dropout, as in deployment.
    """
    b, n = features.shape[:2]
    w, k = cfg.recent_frames, cfg.queue_frames
    valid = valid & (ticks <= now[:, None])
    features = jnp.where(valid[..., None, None], features, 0)
    frame_features = scorer.encode_frames(features.astype(jnp.float32))
    batch = jnp.arange(b)
    # q ids, q membership, q ticks, q count, w ids, w ticks, w count
    initial = (jnp.zeros((b, k), jnp.int32), jnp.zeros((b, k, n), jnp.float32),
               jnp.zeros((b, k), ticks.dtype), jnp.zeros((b,), jnp.int32),
               jnp.zeros((b, w), jnp.int32), jnp.zeros((b, w), ticks.dtype),
               jnp.zeros((b,), jnp.int32))

    def write(carry, index):
        q_ids, members, q_ticks, q_count, w_ids, w_ticks, w_count = carry
        active, event_tick = valid[:, index], ticks[:, index]
        emit = active & (w_count == w)
        candidate, candidate_tick = w_ids[:, 0], w_ticks[:, 0]
        candidate_members = jax.nn.one_hot(candidate, n, dtype=jnp.float32)
        overflow = emit & (q_count == k)

        def score(_):
            q_valid = jnp.arange(k)[None] < q_count[:, None]
            encoded = select_members(frame_features, members, q_ids, q_valid)
            encoded = jnp.concatenate([encoded, _gather(frame_features, candidate[:, None])], axis=1)
            candidate_ticks = jnp.concatenate([q_ticks, candidate_tick[:, None]], axis=1)
            all_valid = jnp.ones((b, k + 1), bool)
            is_new = jnp.broadcast_to(jnp.arange(k + 1)[None] == k, all_valid.shape)
            logits = scorer.score_frames(encoded, candidate_ticks, all_valid, event_tick, is_new,
                                          deterministic=True)
            return hard_eviction(logits / cfg.e2e_temperature)

        # Inactive batch elements get finite dummy logits; their updates are masked.
        dropped = jax.lax.cond(jnp.any(overflow), score,
                               lambda _: jnp.zeros((b, k + 1), jnp.float32), operand=None)
        replacement = dropped[:, :k]
        updated = members * (1 - replacement[..., None]) + candidate_members[:, None] * replacement[..., None]
        members = jnp.where(overflow[:, None, None], updated, members)
        replace = overflow[:, None] & replacement.astype(bool)
        q_ids = jnp.where(replace, candidate[:, None], q_ids)
        q_ticks = jnp.where(replace, candidate_tick[:, None], q_ticks)

        append = emit & (q_count < k)
        slot = jnp.minimum(q_count, k - 1)
        members = members.at[batch, slot].set(jnp.where(append[:, None], candidate_members, members[batch, slot]))
        q_ids = q_ids.at[batch, slot].set(jnp.where(append, candidate, q_ids[batch, slot]))
        q_ticks = q_ticks.at[batch, slot].set(jnp.where(append, candidate_tick, q_ticks[batch, slot]))
        q_count = q_count + append.astype(jnp.int32)
        q_valid = jnp.arange(k)[None] < q_count[:, None]
        order = jnp.argsort(jnp.where(q_valid, q_ticks, jnp.iinfo(ticks.dtype).max), axis=1, stable=True)
        q_ids, q_ticks = (jnp.take_along_axis(x, order, axis=1) for x in (q_ids, q_ticks))
        members = jnp.take_along_axis(members, order[..., None], axis=1)

        slot = jnp.minimum(w_count, w - 1)
        append_ids = w_ids.at[batch, slot].set(index)
        append_ticks = w_ticks.at[batch, slot].set(event_tick)
        full_ids = jnp.concatenate([w_ids[:, 1:], jnp.full((b, 1), index, jnp.int32)], axis=1)
        full_ticks = jnp.concatenate([w_ticks[:, 1:], event_tick[:, None]], axis=1)
        next_ids = jnp.where((w_count == w)[:, None], full_ids, append_ids)
        next_ticks = jnp.where((w_count == w)[:, None], full_ticks, append_ticks)
        w_ids = jnp.where(active[:, None], next_ids, w_ids)
        w_ticks = jnp.where(active[:, None], next_ticks, w_ticks)
        w_count = jnp.minimum(w, w_count + active.astype(jnp.int32))
        return (q_ids, members, q_ticks, q_count, w_ids, w_ticks, w_count), None

    # Rematerialize contextual scoring in backward; carry contains only small
    # membership/metadata arrays rather than 64*2560 token tensors at every step.
    final, _ = jax.lax.scan(jax.checkpoint(write, prevent_cse=False), initial, jnp.arange(n))
    q_ids, members, q_ticks, q_count, w_ids, w_ticks, w_count = final
    output_valid = jnp.concatenate([jnp.arange(k)[None] < q_count[:, None],
                                    jnp.arange(w)[None] < w_count[:, None]], axis=1)
    output_ids = jnp.concatenate([q_ids, w_ids], axis=1)
    output_ticks = jnp.concatenate([q_ticks, w_ticks], axis=1)
    recent_members = jax.nn.one_hot(w_ids, n, dtype=jnp.float32)
    members = jnp.concatenate([members, recent_members], axis=1)
    order = jnp.argsort(jnp.where(output_valid, output_ticks, jnp.iinfo(ticks.dtype).max), axis=1, stable=True)
    output_ids, output_ticks, output_valid = (jnp.take_along_axis(x, order, axis=1)
                                            for x in (output_ids, output_ticks, output_valid))
    members = jnp.take_along_axis(members, order[..., None], axis=1)
    selected = select_members(features.astype(jnp.float32), members, output_ids, output_valid).astype(features.dtype)
    return selected, jnp.where(output_valid, output_ticks, 0), output_valid, jnp.where(output_valid, output_ids, -1)
