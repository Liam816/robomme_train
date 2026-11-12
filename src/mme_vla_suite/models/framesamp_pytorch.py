"""OpenPI π0.5 with the RoboMME perceptual FrameSamp+Modul action expert."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import no_init_weights
from transformers.models.gemma.modeling_gemma import _gated_residual

from openpi.models import gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


def rotary(x, positions):
    # Match Gemma's split-half RoPE, including the FP32 rotation before the BF16 cast.
    dim = x.shape[-1]
    freq = 10000 ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angle = positions[:, None, :, None] * freq
    left, right = x.float().chunk(2, dim=-1)
    return torch.cat((left * angle.cos() - right * angle.sin(), right * angle.cos() + left * angle.sin()), -1).to(
        x.dtype
    )


def heads(proj, x, head_dim=256):
    return proj(x).unflatten(-1, (-1, head_dim)).transpose(1, 2)


def attend(q, k, v, mask):
    # Expand the one KV head as a view. SDPA selects an available backend on this device.
    k, v = (t.expand(-1, q.shape[1], -1, -1) for t in (k, v))
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    return y.transpose(1, 2).flatten(2)


def rms(x):
    return x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)


class MemoryModulation(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(width))
        self.q = nn.Linear(width, 4 * 256, bias=False)
        self.k = nn.Linear(width, 256, bias=False)
        self.v = nn.Linear(width, 256, bias=False)
        self.out = nn.Linear(4 * 256, width, bias=False)
        self.condition = nn.Linear(width, 2 * width)

    def forward(self, x, memory, mask):
        norm_x = (rms(x) * (1 + self.scale.float())).to(x.dtype)
        mem = (rms(memory) * (1 + self.scale.float())).to(memory.dtype)
        memory_pos = torch.arange(mem.shape[1], device=x.device)[None]
        action_pos = memory_pos.size(1) + torch.arange(x.shape[1], device=x.device)[None]
        q = rotary(heads(self.q, norm_x), action_pos)
        k = rotary(heads(self.k, mem), memory_pos)
        vector = self.out(attend(q, k, heads(self.v, mem), mask[:, None, None, :]))
        scale, shift = self.condition(vector).chunk(2, dim=-1)
        return (rms(x) * (1 + scale) + shift).to(x.dtype)


class JointBlock(nn.Module):
    """One FSDP unit owns both experts and the corresponding modulation weights."""

    def __init__(self, vlm, action, width):
        super().__init__()
        self.vlm, self.action = vlm, action
        self.memory = MemoryModulation(width)

    def forward(self, prefix, suffix, positions, mask, cond, memory, memory_mask):
        inputs, layers = (prefix, suffix), (self.vlm, self.action)
        queries, keys, values, gates = [], [], [], []
        for x, layer, conditioning in zip(inputs, layers, (None, cond), strict=True):
            norm, gate = layer.input_layernorm(x, cond=conditioning)
            queries.append(heads(layer.self_attn.q_proj, norm))
            keys.append(heads(layer.self_attn.k_proj, norm))
            values.append(heads(layer.self_attn.v_proj, norm))
            gates.append(gate)
        q = rotary(torch.cat(queries, dim=2), positions)
        k = rotary(torch.cat(keys, dim=2), positions)
        outputs = attend(q, k, torch.cat(values, dim=2), mask)
        outputs = outputs.split((prefix.shape[1], suffix.shape[1]), dim=1)
        result = []
        for i, layer in enumerate(layers):
            residual = _gated_residual(inputs[i], layer.self_attn.o_proj(outputs[i]), gates[i])
            before_ffn = self.memory(residual, memory, memory_mask) if i == 1 else residual
            hidden, gate = layer.post_attention_layernorm(before_ffn, cond=cond if i == 1 else None)
            result.append(_gated_residual(residual, layer.mlp(hidden), gate))
        return tuple(result)


class FrameSampPolicy(nn.Module):
    def __init__(self, config, memory_config, *, activation_checkpointing=True):
        super().__init__()
        vlm = gemma.get_config(config.paligemma_variant)
        expert = gemma.get_config(config.action_expert_variant)
        self.action_horizon, self.action_dim = config.action_horizon, config.action_dim
        self.history_config = memory_config
        self.compute_dtype = getattr(torch, config.dtype)
        self.activation_checkpointing = activation_checkpointing
        # Every parameter is loaded strictly before use. Avoid initializing discarded LM heads.
        with no_init_weights():
            backbone = PaliGemmaWithExpertModel(vlm, expert, use_adarms=[False, True], precision="float32")
        language = backbone.paligemma.language_model
        action = backbone.gemma_expert.model
        self.vision = backbone.paligemma.vision_tower.vision_model
        if self.vision.use_head:
            self.vision.use_head = False
            del self.vision.head
        self.projector = backbone.paligemma.multi_modal_projector.linear
        self.vision.requires_grad_(False)
        self.projector.requires_grad_(False)
        self.tokens = language.embed_tokens
        self.blocks = nn.ModuleList(
            JointBlock(v, a, expert.width) for v, a in zip(language.layers, action.layers, strict=True)
        )
        self.vlm_norm, self.action_norm = language.norm, action.norm
        self.pos_proj = nn.Linear(768, 768)
        self.encoder_static = nn.Linear(2048 + 768, expert.width)
        self.action_in_proj = nn.Linear(config.action_dim, expert.width)
        self.action_out_proj = nn.Linear(expert.width, config.action_dim)
        self.time_mlp_in = nn.Linear(expert.width, expert.width)
        self.time_mlp_out = nn.Linear(expert.width, expert.width)

    def amp(self, device):
        return torch.autocast(device.type, dtype=self.compute_dtype, enabled=self.compute_dtype == torch.bfloat16)

    def encode_images(self, images):
        with self.amp(images.device):
            return self.projector(self.vision(pixel_values=images).last_hidden_state)

    def context(self, batch):
        device = batch["state"].device
        with self.amp(device):
            b, v, c, h, w = batch["images"].shape
            image_tokens = self.encode_images(batch["images"].reshape(b * v, c, h, w)).reshape(b, -1, 2048)
            words = (self.tokens(batch["tokenized_prompt"]) * math.sqrt(2048)).to(self.compute_dtype)
            prefix = torch.cat((image_tokens, words), 1).to(self.compute_dtype)
            image_mask = torch.ones(image_tokens.shape[:2], dtype=torch.bool, device=device)
            mask = torch.cat((image_mask, batch["tokenized_prompt_mask"]), dim=1)
            pos_emb = F.silu(self.pos_proj(batch["static_pos_emb"]))
            mem = self.encoder_static(torch.cat((batch["static_image_emb"].to(pos_emb.dtype), pos_emb), -1))
        return prefix, mask, mem

    def velocity(self, context, memory_mask, noisy_actions, time):
        prefix, prefix_mask, memory = context
        device = prefix.device
        # The reference's NNX action/time projections promote to FP32 with FP32 inputs.
        time_emb = create_sinusoidal_pos_embedding(time, self.time_mlp_in.in_features, 4e-3, 4.0, device).float()
        cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
        suffix = self.action_in_proj(noisy_actions.float()).to(self.compute_dtype)
        action_mask = torch.ones(suffix.shape[:2], device=device, dtype=torch.bool)
        valid = torch.cat((prefix_mask, action_mask), dim=1)
        boundary = torch.zeros_like(valid)
        boundary[:, prefix.shape[1]] = True
        mask = make_att_2d_masks(valid, boundary)[:, None, :, :]
        positions = valid.long().cumsum(1) - 1
        with self.amp(device):
            for block in self.blocks:
                args = (prefix, suffix, positions, mask, cond, memory, memory_mask)
                if self.training and self.activation_checkpointing:
                    prefix, suffix = checkpoint(block, *args, use_reentrant=False, preserve_rng_state=False)
                else:
                    prefix, suffix = block(*args)
            suffix, _ = self.action_norm(suffix, cond=cond)
        return self.action_out_proj(suffix.float())

    def forward(self, batch, noise, time):
        actions = batch["actions"]
        velocity = self.velocity(
            self.context(batch),
            batch["static_mask"],
            time[:, None, None] * noise + (1 - time[:, None, None]) * actions,
            time,
        )
        return (velocity - (noise - actions)).square().mean(dim=-1)

    @torch.inference_mode()
    def sample(self, batch, noise, steps=10):
        context = self.context(batch)
        x = noise
        time = torch.ones(x.shape[0], device=x.device)
        for _ in range(steps):
            x = x - self.velocity(context, batch["static_mask"], x, time) / steps
            time = time - 1 / steps
        return x
