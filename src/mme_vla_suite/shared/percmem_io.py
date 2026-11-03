"""Versioned memory weights and a common deterministic historical image encoder."""

import dataclasses
import hashlib
import json
from pathlib import Path

from flax import nnx
from flax.nnx import bridge
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes

from mme_vla_suite.models.representation.perc_memory_v1 import EvictScorer, MemReader, PercMemConfig, pool_tokens
from openpi.models import siglip
from openpi.models.model import restore_params
from openpi.shared import image_tools


def tree_hash(tree):
    digest = hashlib.sha256()
    for path, value in sorted(traverse_util.flatten_dict(tree, sep="/").items()):
        array = np.asarray(value)
        digest.update(path.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def module_params(module):
    return nnx.state(module, nnx.Param).to_pure_dict()


def save_module(path, module, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = module_params(module)
    meta = dict(metadata, format="percmem_module_v1", params_id=tree_hash(params),
                config=dataclasses.asdict(module.cfg))
    arrays = {name: np.asarray(value) for name, value in traverse_util.flatten_dict(params, sep="/").items()}
    arrays["__metadata__"] = np.array(json.dumps(meta, sort_keys=True))
    temp = path.with_name(path.name + ".tmp.npz")
    np.savez(temp, **arrays)
    temp.replace(path)
    return meta


SCORER_SEMANTICS = ("feature_dim", "score_grid", "scorer_dim", "scorer_heads", "scorer_layers", "scorer_ffn",
                    "scorer_dropout", "scorer_use_instr", "use_redund", "age_dim", "clock_hz", "norm_eps",
                    "sinusoid_base", "attention_bias")


def validate_scorer_config(cfg, saved):
    if any(getattr(cfg, key) != saved[key] for key in SCORER_SEMANTICS):
        raise ValueError("Scorer configuration differs from its trained checkpoint")


def load_module(module, path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["__metadata__"]))
        flat = {k: archive[k] for k in archive.files if k != "__metadata__"}
    validate_scorer_config(module.cfg, meta["config"])
    if meta["format"] != "percmem_module_v1":
        raise ValueError("Unknown scorer checkpoint format")
    reference = traverse_util.flatten_dict(module_params(module), sep="/")
    if reference.keys() != flat.keys() or any(np.shape(reference[k]) != flat[k].shape for k in flat):
        raise ValueError("Scorer checkpoint structure or shapes differ")
    params = traverse_util.unflatten_dict(flat, sep="/")
    if tree_hash(params) != meta["params_id"]:
        raise ValueError("Scorer checkpoint hash mismatch")
    state = nnx.state(module, nnx.Param)
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, params))
    nnx.update(module, state)
    return meta


class MemoryBundle(nnx.Module):
    """Offline P1/P2 utilities can restore only the small memory modules after I/O."""
    def __init__(self, cfg, *, rngs):
        self.mem_feature_adapter = nnx.Linear(cfg.native_dim, cfg.feature_dim, rngs=rngs)
        self.mem_reader = MemReader(cfg, rngs=rngs)
        self.mem_cond_adapter = nnx.Linear(cfg.output_dim, 1024, use_bias=False, rngs=rngs)
        self.mem_scorer = EvictScorer(cfg, rngs=rngs)


def load_bundle(params_path, cfg):
    # Existing Orbax helper restores all arrays once. No VLM forward is performed.
    params = restore_params(params_path, restore_type=np.ndarray)
    teacher_id = tree_hash(params)
    bundle = MemoryBundle(cfg, rngs=nnx.Rngs(0))
    bundle.vision_id = tree_hash(canonical_vision_params(params["PaliGemma"]["img"]))
    state = nnx.state(bundle)
    subset = {name: params[name] for name in state.to_pure_dict()}
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, subset))
    nnx.update(bundle, state)
    return bundle, teacher_id


def canonical_vision_params(params):
    """Match the frozen backbone's BF16 storage, with deterministic FP32 computation."""
    return jax.tree.map(lambda x: np.asarray(x).astype(ml_dtypes.bfloat16).astype(np.float32), params)


class FrozenHistoryEncoder(nnx.Module):
    def __init__(self, params):
        def create():
            image = bridge.ToNNX(siglip.Module(num_classes=2048, variant="So400m/14", pool_type="none",
                                               scan=True, dtype_mm="float32"))
            image.lazy_init(jnp.zeros((1, 224, 224, 3)), train=False, rngs=nnx.Rngs(0))
            return image
        image = nnx.eval_shape(create)
        graph, state = nnx.split(image)
        state.replace_by_pure_dict(jax.tree.map(jnp.asarray, canonical_vision_params(params)))
        self.image = nnx.merge(graph, state)

    def __call__(self, rgb):
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError("History encoder expects a uint8 RGB image batch")
        normalized = rgb.astype(jnp.float32) / 255.0 * 2 - 1
        resized = image_tools.resize_with_pad(normalized, 224, 224)
        tokens, _ = self.image(resized, train=False)
        return pool_tokens(tokens, 8).astype(jnp.float16)


def restore_history_encoder(params_path):
    params = restore_params(params_path, restore_type=np.ndarray)
    canonical = canonical_vision_params(params["PaliGemma"]["img"])
    vision_id = tree_hash(canonical)
    del params
    return FrozenHistoryEncoder(canonical), vision_id
