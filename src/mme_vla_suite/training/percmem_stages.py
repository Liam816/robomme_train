"""Shared P1/P2/P3 artifact operations, including chronological queue replay."""

import json
import math
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mme_vla_suite.models.representation.perc_memory_v1 import EvictScorer, pairwise_ranking_loss
from mme_vla_suite.shared.perc_mem_bank import PercMemBank
from mme_vla_suite.shared.percmem_io import load_module, module_params, save_module, tree_hash
from mme_vla_suite.training.lerobot_framesamp import write_json
from mme_vla_suite.training.lerobot_percmem import validate_cache
from openpi.shared.nnx_utils import module_jit


def attention_utility(attention, valid):
    # B,L,H,Q,N,P -> B,N; padded frames contribute exactly zero.
    per_frame = attention.sum(axis=-1).mean(axis=(1, 2, 3))
    return per_frame * valid.sum(axis=1)[:, None]


def future_values(attention_values):
    return np.maximum.accumulate(attention_values[::-1], axis=0)[::-1].copy()


def lookup_utility(labels, tick, indices):
    if "key_tick" in labels and np.any(labels["key_tick"][indices] > tick):
        raise ValueError("Requested label for an unborn future keyframe")
    location = np.searchsorted(labels["query_tick"], tick, side="left")
    if location == len(labels["query_tick"]):
        return None
    return labels["utility"][location, indices].astype(np.float32)


def episode_data(root, episode, projected):
    name = f"episode_{episode['episode_index']:06d}"
    with np.load(Path(root) / f"keyframes/{name}.npz") as plan:
        values = {k: plan[k] for k in plan.files}
    feature = np.load(Path(projected) / f"{name}.npy", mmap_mode="r")
    return name, values, feature


def project_cache(root, destination, bundle, teacher_id):
    root, destination = Path(root), Path(destination)
    meta = validate_cache(root)
    vision_id = json.loads((root / "meta/native_complete.json").read_text())["vision_id"]
    if getattr(bundle, "vision_id", None) != vision_id:
        raise ValueError("Teacher image weights differ from the frozen native-cache encoder")
    adapter_id = tree_hash(module_params(bundle.mem_feature_adapter))
    identity = {"cache_id": meta["cache_id"], "adapter_id": adapter_id, "teacher_id": teacher_id,
                "storage": "float16", "feature_dim": bundle.mem_reader.cfg.feature_dim, "vision_id": vision_id}
    manifest = destination / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("Projected cache already belongs to a different teacher/adapter")
        return identity
    destination.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "meta/source_index.json").read_text())
    project = module_jit(bundle.mem_feature_adapter.__call__)
    for episode in metadata["episodes"]:
        name = f"episode_{episode['episode_index']:06d}"
        native = np.load(root / f"native/{name}.npy", mmap_mode="r")
        output = np.empty((*native.shape[:-1], identity["feature_dim"]), np.float16)
        for begin in range(0, len(native), 16):
            batch = np.asarray(native[begin:begin+16], np.float32)
            valid = len(batch)
            batch = np.pad(batch, ((0, 16-valid), (0, 0), (0, 0)))
            output[begin:begin+valid] = np.asarray(project(jnp.asarray(batch)))[:valid].astype(np.float16)
        temporary = destination / f"{name}.tmp.npy"
        np.save(temporary, output)
        temporary.replace(destination / f"{name}.npy")
    write_json(manifest, identity)
    return identity


def build_attention_labels(root, projected, output, bundle, teacher_id):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = bundle.mem_reader.cfg
    identity = json.loads((Path(projected) / "manifest.json").read_text())
    if identity["teacher_id"] != teacher_id:
        raise ValueError("Projected cache and reader teacher differ")
    metadata = json.loads((root / "meta/source_index.json").read_text())
    reader = module_jit(bundle.mem_reader.__call__, static_argnames=("return_attention",))
    for episode in metadata["episodes"]:
        name, plan, feature = episode_data(root, episode, projected)
        rows = list(range(episode["exec_start_idx"], episode["length"], 10))
        if not rows or rows[-1] != episode["length"] - 1:
            rows.append(episode["length"] - 1)
        values, query_ticks = [], []
        for row in rows:
            now = int(round(row / metadata["fps"] * cfg.clock_hz))
            n = np.searchsorted(plan["row"], row, side="right")
            if len(feature) > cfg.max_frames:
                raise ValueError("P1 would truncate full-history memory")
            x = np.zeros((1, cfg.max_frames, cfg.store_grid ** 2, cfg.feature_dim), np.float32)
            ticks, valid = np.zeros((1, cfg.max_frames), np.int32), np.zeros((1, cfg.max_frames), bool)
            x[0, :n], ticks[0, :n], valid[0, :n] = feature[:n], plan["tick"][:n], True
            _, attention = reader(jnp.asarray(x), jnp.asarray(ticks), jnp.asarray(valid), jnp.array([now]), return_attention=True)
            values.append(np.asarray(attention_utility(attention, jnp.asarray(valid)))[0, :len(feature)])
            query_ticks.append(now)
        valid_labels = np.array(query_ticks)[:, None] >= plan["tick"][None]
        utilities = np.where(valid_labels, future_values(np.array(values, np.float32)), 0).astype(np.float16)
        np.savez(output / f"{name}.npz", query_row=np.array(rows, np.int32),
                 query_tick=np.array(query_ticks, np.int32), key_tick=plan["tick"],
                 utility=utilities, valid=valid_labels)
    write_json(output / "manifest.json", dict(identity, source="reader_attention_suffix_max", trusted=False,
                                               validation_required=True, query_stride=10))


def score_candidates(scorer, feature, plan, candidates, now, instruction=None):
    indices = [frame.index for frame in candidates]
    x = jnp.asarray(feature[indices], jnp.float32)[None]
    ticks = jnp.asarray(plan["tick"][indices])[None]
    valid = jnp.ones(ticks.shape, bool)
    new = jnp.arange(len(indices))[None] == len(indices) - 1
    return scorer(x, ticks, valid, jnp.array([now]), new, instruction=instruction, deterministic=True)


def rollout_banks(root, projected, output, cfg, scorer, scorer_metadata, *, labels_dir=None, seed=42):
    root, output = Path(root), Path(output)
    identity = json.loads((Path(projected) / "manifest.json").read_text())
    if any(scorer_metadata[k] != identity[k] for k in ["cache_id", "adapter_id", "teacher_id"]):
        raise ValueError("Scorer and projected feature identities disagree")
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((root / "meta/source_index.json").read_text())
    evaluate = module_jit(scorer.__call__, static_argnames=("deterministic",))
    for episode in metadata["episodes"]:
        name, plan, feature = episode_data(root, episode, projected)
        instruction = None
        if cfg.scorer_use_instr:
            instruction = jnp.asarray(np.load(Path(projected) / f"instruction/{name}.npy"))[None]
        labels = None
        if cfg.evict == "oracle":
            if labels_dir is None:
                raise ValueError("Offline oracle requires labels_dir")
            with np.load(Path(labels_dir) / f"{name}.npz") as archive:
                labels = {k: archive[k] for k in archive.files}
        def utility(candidates, now):
            result = lookup_utility(labels, now, [f.index for f in candidates])
            if result is None:
                raise ValueError("Oracle replay exceeds available future labels")
            return result
        bank = PercMemBank(cfg.recent_frames, cfg.queue_frames, cfg.evict, seed=seed,
            score_fn=lambda c, t: np.asarray(score_candidates(evaluate, feature, plan, c, t, instruction))[0])
        rows = np.arange(episode["exec_start_idx"], episode["length"], dtype=np.int32)
        members = np.full((len(rows), cfg.capacity), -1, np.int16)
        key = 0
        for i, row in enumerate(rows):
            while key < len(feature) and plan["row"][key] <= row:
                bank.write(key, int(plan["tick"][key]), utility_fn=utility)
                key += 1
            indices = [f.index for f in bank.frames()]
            if indices and max(indices) > np.iinfo(np.int16).max:
                raise ValueError("Keyframe indices exceed int16 bank storage")
            members[i, :len(indices)] = indices
        np.savez(output / f"{name}.npz", row=rows, members=members)
    write_json(output / "manifest.json", dict(identity, evict=cfg.evict, capacity=cfg.capacity,
                                               scorer_id=scorer_metadata["params_id"], seed=seed))


def train_scorer(root, projected, labels_dir, output, cfg, *, steps, seed=42, batch_trajectories=64):
    if steps < 1 or batch_trajectories < 1:
        raise ValueError("Explicit positive optimizer steps and trajectory batch are required")
    root, labels_dir = Path(root), Path(labels_dir)
    identity = json.loads((Path(projected) / "manifest.json").read_text())
    labels_meta = json.loads((labels_dir / "manifest.json").read_text())
    if not labels_meta.get("trusted"):
        raise ValueError("P1 labels have not passed the fixed-noise high/low deletion validation")
    if any(identity[k] != labels_meta[k] for k in ["cache_id", "adapter_id", "teacher_id"]):
        raise ValueError("P1 labels and projected features disagree")
    metadata = json.loads((root / "meta/source_index.json").read_text())
    allowed = set(labels_meta.get("episode_ids", [e["episode_index"] for e in metadata["episodes"]]))
    episodes = [e for e in metadata["episodes"] if e["episode_index"] in allowed]
    if not episodes:
        raise ValueError("No labeled episodes are available")
    scorer = EvictScorer(cfg, rngs=nnx.Rngs(seed))
    tx = optax.adamw(optax.cosine_decay_schedule(3e-4, steps), weight_decay=0.01)
    optimizer_state = tx.init(nnx.state(scorer, nnx.Param))
    @nnx.jit
    def evaluate_model(model, x, tick, valid, now, new, instruction):
        return model(x, tick, valid, now, new, instruction=instruction, deterministic=True)

    def evaluate(x, tick, valid, now, new, *, instruction=None, deterministic=True):
        if not deterministic:
            raise ValueError("Queue decisions must disable dropout")
        return evaluate_model(scorer, x, tick, valid, now, new, instruction)

    @nnx.jit
    @nnx.value_and_grad
    def event_loss(model, x, tick, now, utility, key, instruction):
        valid = jnp.ones(tick.shape, bool)
        new = jnp.arange(tick.shape[1])[None] == tick.shape[1] - 1
        drop_key, pair_key = jax.random.split(key)
        scores = model(x, tick, valid, now, new, instruction=instruction, deterministic=False, rng=drop_key)
        return pairwise_ranking_loss(scores[0], utility, valid[0], pair_key)

    random = np.random.default_rng(seed)
    for step in range(steps):
        total = jax.tree.map(jnp.zeros_like, nnx.state(scorer, nnx.Param))
        loss_sum, events, consistent = 0., 0, 0
        p_oracle = max(0.0, 1.0 - step / max(0.3 * steps, 1))
        for episode_index in random.integers(len(episodes), size=batch_trajectories):
            episode = episodes[int(episode_index)]
            name, plan, feature = episode_data(root, episode, projected)
            with np.load(labels_dir / f"{name}.npz") as archive:
                labels = {k: archive[k] for k in archive.files}
            instruction = None
            if cfg.scorer_use_instr:
                instruction = jnp.asarray(np.load(Path(projected) / f"instruction/{name}.npy"))[None]
            utility_fn = lambda candidates, now: lookup_utility(labels, now, [f.index for f in candidates])
            bank = PercMemBank(cfg.recent_frames, cfg.queue_frames, "learned", seed=seed,
                score_fn=lambda c, t: np.asarray(score_candidates(evaluate, feature, plan, c, t, instruction))[0])
            for key_index, tick in enumerate(plan["tick"]):
                mode = "oracle" if random.random() < p_oracle else "learned"
                event = bank.write(key_index, int(tick), utility_fn=utility_fn, mode=mode)
                if event is None:
                    continue
                candidates, evicted = event
                indices = [f.index for f in candidates]
                utility = lookup_utility(labels, int(tick), indices)
                if utility is None or not np.any(utility[:, None] - utility[None, :] > 0.1):
                    continue
                loss, grads = event_loss(scorer, jnp.asarray(feature[indices], jnp.float32)[None],
                    jnp.asarray(plan["tick"][indices])[None], jnp.array([tick]), jnp.asarray(utility),
                    jax.random.key(int(random.integers(2**31))), instruction)
                total = jax.tree.map(lambda a, b: a + b, total, grads)
                loss_sum += float(loss)
                events += 1
                consistent += int(evicted in np.argsort(utility, kind="stable")[:math.ceil(len(utility) * 0.2)])
        if events == 0:
            raise ValueError("No eligible ranking pairs in this batch; inspect label scale, keyframe budget and dataset")
        params = nnx.state(scorer, nnx.Param)
        gradients = jax.tree.map(lambda x: x / events, total)
        updates, optimizer_state = tx.update(gradients, optimizer_state, params)
        nnx.update(scorer, optax.apply_updates(params, updates))
        if step % 100 == 0 or step == steps - 1:
            print(json.dumps({"step": step, "loss": loss_sum / events, "eligible_events": events,
                              "mixed_policy_low20_rate": consistent / events, "p_oracle": p_oracle}), flush=True)
    # Evaluate the final learned policy alone, never mixing oracle decisions into this metric.
    total_events, low20 = 0, 0
    for episode in episodes:
        name, plan, feature = episode_data(root, episode, projected)
        with np.load(labels_dir / f"{name}.npz") as archive:
            labels = {k: archive[k] for k in archive.files}
        instruction = None
        if cfg.scorer_use_instr:
            instruction = jnp.asarray(np.load(Path(projected) / f"instruction/{name}.npy"))[None]
        bank = PercMemBank(cfg.recent_frames, cfg.queue_frames, "learned", seed=seed,
            score_fn=lambda c, t: np.asarray(score_candidates(evaluate, feature, plan, c, t, instruction))[0])
        for index, tick in enumerate(plan["tick"]):
            event = bank.write(index, int(tick))
            if event is None:
                continue
            candidates, evicted = event
            utility = lookup_utility(labels, int(tick), [f.index for f in candidates])
            if utility is not None and np.any(utility[:, None] - utility[None, :] > .1):
                total_events += 1
                low20 += int(evicted in np.argsort(utility, kind="stable")[:math.ceil(len(utility) * .2)])
    metric = {"learned_only_training_replay_low20_rate": low20 / max(total_events, 1),
              "eligible_events": total_events, "held_out": False}
    print(json.dumps(metric), flush=True)
    return save_module(output, scorer, dict(identity, steps=steps, batch_trajectories=batch_trajectories,
                                             labels_manifest=labels_meta, seed=seed, evaluation=metric))
