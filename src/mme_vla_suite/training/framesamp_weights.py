"""Bidirectional layout mapping between RoboMME Orbax and OpenPI PyTorch parameters."""

from dataclasses import dataclass

import einops
from flax import traverse_util
import ml_dtypes
import numpy as np
import torch

from openpi.models.model import restore_params


@dataclass
class Rule:
    path: str
    targets: tuple
    layout: str
    axes: dict


def weight_rules():
    rules = []

    def add(path, targets, layout, **axes):
        if isinstance(targets, str):
            targets = (targets,)
        rules.append(Rule(path, tuple(targets), layout, axes))

    def dense(jax, pt):
        add(jax + "/kernel", pt + ".weight", "i o -> 1 o i")
        add(jax + "/bias", pt + ".bias", "o -> 1 o")

    img = "PaliGemma/img/"
    add(img + "embedding/kernel", "vision.embeddings.patch_embedding.weight", "h w i o -> 1 o i h w")
    add(img + "embedding/bias", "vision.embeddings.patch_embedding.bias", "o -> 1 o")
    add(img + "pos_embedding", "vision.embeddings.position_embedding.weight", "1 s d -> 1 s d")
    for suffix, torch_suffix in (("scale", "weight"), ("bias", "bias")):
        add(img + "Transformer/encoder_norm/" + suffix, "vision.post_layernorm." + torch_suffix, "d -> 1 d")
    dense(img + "head", "projector")
    enc = img + "Transformer/encoderblock/"
    for suffix, pt in (
        ("LayerNorm_0", "layer_norm1"),
        ("LayerNorm_1", "layer_norm2"),
        ("MlpBlock_0/Dense_0", "mlp.fc1"),
        ("MlpBlock_0/Dense_1", "mlp.fc2"),
    ):
        is_norm = suffix.startswith("LayerNorm")
        names = [f"vision.encoder.layers.{i}.{pt}" for i in range(27)]
        add(
            enc + suffix + ("/scale" if is_norm else "/kernel"),
            (n + ".weight" for n in names),
            "l d -> l d" if is_norm else "l i o -> l o i",
        )
        add(enc + suffix + "/bias", (n + ".bias" for n in names), "l o -> l o")
    for key, pt in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj"), ("out", "out_proj")):
        path = enc + "MultiHeadDotProductAttention_0/" + key
        names = [f"vision.encoder.layers.{i}.self_attn.{pt}" for i in range(27)]
        output = key == "out"
        add(
            path + "/kernel",
            (n + ".weight" for n in names),
            "l h d o -> l o (h d)" if output else "l i h d -> l (h d) i",
            h=16,
        )
        if output:
            add(path + "/bias", (n + ".bias" for n in names), "l o -> l o")
        else:
            add(path + "/bias", (n + ".bias" for n in names), "l h d -> l (h d)", h=16)

    llm = "PaliGemma/llm/"
    add(llm + "embedder/input_embedding", "tokens.weight", "v d -> 1 v d")
    add(llm + "final_norm/scale", "vlm_norm.weight", "d -> 1 d")
    dense(llm + "final_norm_1/Dense_0", "action_norm.dense")
    layers = llm + "layers/"
    for kind, suffix in (("vlm", ""), ("action", "_1"), ("memory", "_mem")):
        mem = kind == "memory"
        names = [f"blocks.{i}.{kind}" for i in range(18)]
        att = layers + ("mem_attn/" if mem else "attn/")
        names_att = names if mem else [n + ".self_attn" for n in names]
        head_count = 4 if mem else 8
        add(
            att + "q_einsum" + suffix + "/w",
            (n + (".q.weight" if mem else ".q_proj.weight") for n in names_att),
            "l h i d -> l (h d) i",
            h=head_count,
        )
        add(
            att + "kv_einsum" + suffix + "/w",
            (n + "." + p + ".weight" for n in names_att for p in (("k", "v") if mem else ("k_proj", "v_proj"))),
            "l k 1 i d -> (l k) d i",
            k=2,
        )
        add(
            att + ("out_einsum" if mem else "attn_vec_einsum") + suffix + "/w",
            (n + (".out.weight" if mem else ".o_proj.weight") for n in names_att),
            "l h d o -> l o (h d)",
            h=head_count,
        )
        if mem:
            add(att + "mem_rms_norm/scale", (n + ".scale" for n in names), "l d -> l d")
            add(layers + "mem_rms_norm_ffn/Dense_0/kernel", (n + ".condition.weight" for n in names), "l i o -> l o i")
            add(layers + "mem_rms_norm_ffn/Dense_0/bias", (n + ".condition.bias" for n in names), "l o -> l o")
            continue
        add(
            layers + "mlp" + suffix + "/gating_einsum",
            (n + ".mlp." + p + ".weight" for n in names for p in ("gate_proj", "up_proj")),
            "l k i o -> (l k) o i",
            k=2,
        )
        add(layers + "mlp" + suffix + "/linear", (n + ".mlp.down_proj.weight" for n in names), "l i o -> l o i")
        for jax_norm, pt_norm in (
            ("pre_attention_norm", "input_layernorm"),
            ("pre_ffw_norm", "post_attention_layernorm"),
        ):
            path = layers + jax_norm + suffix
            if kind == "vlm":
                add(path + "/scale", (f"{n}.{pt_norm}.weight" for n in names), "l d -> l d")
            else:
                add(path + "/Dense_0/kernel", (f"{n}.{pt_norm}.dense.weight" for n in names), "l i o -> l o i")
                add(path + "/Dense_0/bias", (f"{n}.{pt_norm}.dense.bias" for n in names), "l o -> l o")
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        dense(name, name)
    for name in ("encoder_static", "pos_proj"):
        dense("mem_encoder/feature_encoder/" + name, name)
    return rules


def numpy_tensor(a):
    if a.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(np.asarray(a).view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(np.asarray(a))


@torch.no_grad()
def load_orbax(model, params_path, *, memory_init=None, seed=None):
    flat = traverse_util.flatten_dict(restore_params(params_path, restore_type=np.ndarray), sep="/")
    if memory_init is not None:
        with np.load(memory_init, allow_pickle=False) as extra:
            if int(extra["seed"]) != seed:
                raise ValueError("Memory initialization seed differs from the training seed")
            for key in extra.files:
                if key == "seed":
                    continue
                if "mem" not in key or key in flat:
                    raise ValueError(f"Unexpected initialization array {key}")
                flat[key] = extra[key]
    state = dict(model.named_parameters())
    for rule in weight_rules():
        values = einops.rearrange(numpy_tensor(flat.pop(rule.path)), rule.layout, **rule.axes)
        for name, value in zip(rule.targets, values.unbind(), strict=True):
            param = state.pop(name)
            if param.shape != value.shape:
                raise ValueError(f"Shape mismatch for {name}: {param.shape} / {value.shape}")
            param.copy_(value)
    if flat or state:
        raise ValueError(f"Unmapped weights: Orbax {list(flat)}, PyTorch {list(state)}")


def orbax_tree(state):
    """Materialize the original scanned JAX arrays, including NNX variable wrappers."""
    state = dict(state)
    result = {}
    for rule in weight_rules():
        source, dest = rule.layout.split(" -> ")
        parts = [state.pop(name).detach().cpu() for name in rule.targets]
        stacked = torch.stack(parts) if len(parts) > 1 else parts[0].unsqueeze(0)
        tensor = einops.rearrange(stacked, dest + " -> " + source, **rule.axes).contiguous()
        if tensor.dtype == torch.bfloat16:
            array = tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
        else:
            array = tensor.numpy()
        result[rule.path + "/value"] = array
    if state:
        raise ValueError(f"Unmapped PyTorch checkpoint entries: {list(state)}")
    return {"params": traverse_util.unflatten_dict(result, sep="/")}
