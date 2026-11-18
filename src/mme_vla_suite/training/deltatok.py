"""Kinetics DeltaTok with the frozen DINOv3 features used by EvoMem-VLA.

Tokenizer adapted from amazon-far/deltatok at 8cfafc97 (Apache-2.0).
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
The DINOv3 position math follows Transformers' Apache-2.0 implementation.
Copyright 2025 Meta AI and The HuggingFace Inc. team. All rights reserved.
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

IMAGE_SIZE = 256
PATCH_SIZE = 16
EMBED_DIM = 768
NUM_HEADS = 12
HEAD_DIM = EMBED_DIM // NUM_HEADS
DEPTH = 12
MLP_WIDTH = 2048
NORM_EPS = 1e-5
DINO_MODEL = "vit_base_patch16_dinov3.lvd1689m"


class FrozenDINOv3(nn.Module):
    """Extract the final normalized spatial patches from local timm ViT-B/16 weights."""

    def __init__(self, checkpoint: Path):
        super().__init__()
        import timm

        self.backbone = timm.create_model(DINO_MODEL, pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.backbone.load_state_dict(state, strict=True)
        self.requires_grad_(False).eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Map uint8 RGB [B, 2, 3, 256, 256] to patch features [B, 2, 256, 768]."""
        pixels = frames.flatten(0, 1).float().div_(255.0)
        features = self.backbone.forward_features((pixels - self.mean) / self.std)
        patches = features[:, self.backbone.num_prefix_tokens :]
        return patches.unflatten(0, frames.shape[:2])


class _PatchRotaryEmbedding(nn.Module):
    def __init__(self, grid_size: int):
        super().__init__()
        centers = (torch.arange(grid_size, dtype=torch.float32) + 0.5) * (2 / grid_size) - 1
        positions = torch.stack(torch.meshgrid(centers, centers, indexing="ij"), dim=-1).flatten(0, 1)
        self.register_buffer("positions", positions, persistent=False)
        frequencies = 100.0 ** -torch.arange(0, 1, 4 / HEAD_DIM, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.positions
        if self.training:
            # Same isotropic log-uniform coordinate rescaling as upstream DeltaTok.
            log_scale = positions.new_empty(()).uniform_(-math.log(2), math.log(2), generator=generator)
            positions = positions * log_scale.exp()
        angles = (2 * math.pi * positions[..., None] * self.frequencies).flatten(1).repeat(1, 2)
        return angles.cos(), angles.sin()


def _rotate_patches(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    query, patches = tensor[:, :, :1], tensor[:, :, 1:]
    left, right = patches.chunk(2, dim=-1)
    rotated = patches * cos + torch.cat((-right, left), dim=-1) * sin
    return torch.cat((query, rotated), dim=2)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.v_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.o_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.q_norm_prerope = nn.LayerNorm(HEAD_DIM, eps=NORM_EPS, bias=False)
        self.k_norm_prerope = nn.LayerNorm(HEAD_DIM, eps=NORM_EPS, bias=False)
        self.attn_gate = nn.Linear(EMBED_DIM, NUM_HEADS)

    def forward(self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(tokens).unflatten(-1, (NUM_HEADS, HEAD_DIM)).transpose(1, 2)
        k = self.k_proj(tokens).unflatten(-1, (NUM_HEADS, HEAD_DIM)).transpose(1, 2)
        v = self.v_proj(tokens).unflatten(-1, (NUM_HEADS, HEAD_DIM)).transpose(1, 2)
        q = _rotate_patches(self.q_norm_prerope(q), cos, sin).to(v.dtype)
        k = _rotate_patches(self.k_norm_prerope(k), cos, sin).to(v.dtype)
        attention = F.scaled_dot_product_attention(q, k, v)
        gate = self.attn_gate(tokens).transpose(1, 2).unsqueeze(-1).sigmoid()
        return self.o_proj((attention * gate).transpose(1, 2).flatten(2))


class _SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(EMBED_DIM, MLP_WIDTH)
        self.up_proj = nn.Linear(EMBED_DIM, MLP_WIDTH)
        self.down_proj = nn.Linear(MLP_WIDTH, EMBED_DIM)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(tokens)) * self.up_proj(tokens))


class _LayerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.full((EMBED_DIM,), 1e-5))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens * self.lambda1


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM, eps=NORM_EPS)
        self.attention = _Attention()
        self.layer_scale1 = _LayerScale()
        self.norm2 = nn.LayerNorm(EMBED_DIM, eps=NORM_EPS)
        self.mlp = _SwiGLU()
        self.layer_scale2 = _LayerScale()

    def forward(self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.layer_scale1(self.attention(self.norm1(tokens), cos, sin))
        return tokens + self.layer_scale2(self.mlp(self.norm2(tokens)))


class DeltaEncoder(nn.Module):
    """Encode ordered DINOv3 patch pairs as one continuous [B, 1, 768] delta token."""

    def __init__(self, image_size: int = IMAGE_SIZE):
        super().__init__()
        if image_size <= 0 or image_size % PATCH_SIZE:
            raise ValueError(f"image_size must be a positive multiple of {PATCH_SIZE}")
        self.grid_size = image_size // PATCH_SIZE
        self.rope = _PatchRotaryEmbedding(self.grid_size)
        # Checkpoint names and shapes follow the official Kinetics tokenizer.
        self.z_embed = nn.Embedding(1, EMBED_DIM)
        self.xy_embed = nn.Embedding(2, EMBED_DIM)
        self.encoder_blocks = nn.ModuleList(_Block() for _ in range(DEPTH))
        self.norm = nn.LayerNorm(EMBED_DIM, eps=NORM_EPS)

    def _encode(
        self, source: torch.Tensor, target: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        patch_shape = (self.grid_size**2, EMBED_DIM)
        if source.ndim != 3 or source.shape[1:] != patch_shape or target.shape != source.shape:
            raise ValueError(f"Expected matching source and target patch tensors of shape [B, {patch_shape[0]}, 768]")
        query = self.z_embed.weight.unsqueeze(0).expand(source.size(0), -1, -1)
        tokens = torch.cat((query, source + self.xy_embed.weight[0], target + self.xy_embed.weight[1]), dim=1)
        # Identical spatial coordinates for the source and target grids.
        cos, sin = cos.repeat(2, 1), sin.repeat(2, 1)
        for block in self.encoder_blocks:
            tokens = block(tokens, cos, sin)
        return self.norm(tokens[:, :1])

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._encode(source, target, *self.rope())


class DeltaTokenizer(DeltaEncoder):
    """Train the encoder through a decoder conditioned only on source patches and the delta."""

    def __init__(self, image_size: int = IMAGE_SIZE):
        super().__init__(image_size)
        self.decoder_blocks = nn.ModuleList(_Block() for _ in range(DEPTH))

    def forward(
        self, source: torch.Tensor, target: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        cos, sin = self.rope(generator)
        delta = self._encode(source, target, cos, sin)
        tokens = torch.cat((delta, source), dim=1)
        for block in self.decoder_blocks:
            tokens = block(tokens, cos, sin)
        return tokens[:, 1:]


def reconstruction_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise log-cosh averaged over batch, patches, and channels in FP32."""
    error = (reconstructed.float() - target.float()).abs()
    return (error + F.softplus(-2.0 * error) - math.log(2.0)).mean()
