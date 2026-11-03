"""Torch version of b4432b1's differentiable hard W4/Q28 replay.

The forward pass gathers exactly the selected frames. Its custom backwards are
the VJPs of the original JAX custom JVPs; gradients cross every eviction event.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mme_vla_suite.musa.model import Tree, pool_tokens


def gather_frames(features, indices):
    shape = (*indices.shape, *features.shape[2:])
    index = indices.reshape(*indices.shape, *((1,) * (features.ndim - 2))).expand(shape)
    return torch.gather(features, 1, index.long())


class HardEviction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        ctx.save_for_backward(scores)
        return F.one_hot(scores.argmin(-1), scores.shape[-1]).to(scores.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (scores,) = ctx.saved_tensors
        probability = (-scores).softmax(-1)
        return -probability * (gradient - (probability * gradient).sum(-1, keepdim=True))


class SelectMembers(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, membership, indices, valid):
        ctx.save_for_backward(features, indices, valid)
        ctx.membership_dtype = membership.dtype
        mask = valid.reshape(*valid.shape, *((1,) * (features.ndim - 2)))
        return gather_frames(features, indices) * mask

    @staticmethod
    def backward(ctx, gradient):
        features, indices, valid = ctx.saved_tensors
        b, n = features.shape[:2]
        k = indices.shape[1]
        mask = valid.reshape(*valid.shape, *((1,) * (features.ndim - 2)))
        used = gradient * mask
        feature_gradient = None
        membership_gradient = None
        if ctx.needs_input_grad[0]:
            flat = features.new_zeros((b * n, math.prod(features.shape[2:])))
            rows = (indices.long() + torch.arange(b, device=features.device)[:, None] * n).reshape(-1)
            flat.index_add_(0, rows, used.reshape(b * k, -1))
            feature_gradient = flat.reshape_as(features)
        if ctx.needs_input_grad[1]:
            membership_gradient = torch.bmm(
                used.reshape(b, k, -1), features.reshape(b, n, -1).transpose(1, 2)
            ).to(ctx.membership_dtype)
        return feature_gradient, membership_gradient, None, None


class E2EScorer(Tree):
    def __init__(self, arrays, cfg, *, trainable=True):
        super().__init__(arrays, trainable=trainable)
        self.cfg = cfg
        if cfg["scorer_use_instr"]:
            raise ValueError("The requested E2E profile has scorer_use_instr=False")

    def attention(self, query, memory, valid, path):
        heads = self.cfg["scorer_heads"]
        b, nq, dim = query.shape
        dh = dim // heads
        q, k, v = [
            self.linear(x, path + "/" + name).reshape(b, -1, heads, dh)
            for x, name in [(query, "q"), (memory, "k"), (memory, "v")]
        ]
        logits = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(dh)
        weights = torch.where(valid[:, None, None], logits, -1e30).softmax(-1)
        weights = weights * valid[:, None, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        result = torch.einsum("bhqk,bkhd->bqhd", weights, v).reshape(b, nq, dim)
        return self.linear(result, path + "/out")

    def encode_frames(self, features):
        cfg = self.cfg
        b, n = features.shape[:2]
        x = self.linear(self.ln(pool_tokens(features, cfg["score_grid"]), "norm_input"), "proj")
        x = x.reshape(b * n, cfg["score_grid"] ** 2, cfg["scorer_dim"])
        query = self.p("query").expand(b * n, -1, -1)
        encoded = self.attention(query, x, torch.ones(x.shape[:2], device=x.device, dtype=torch.bool), "pool_attn")
        return encoded.reshape(b, n, cfg["scorer_dim"])

    def score_frames(self, frames, ticks, valid, now, is_new):
        cfg = self.cfg
        n = frames.shape[1]
        age = (now[:, None] - ticks).float() / cfg["clock_hz"]
        frequencies = torch.exp(
            -math.log(cfg["sinusoid_base"])
            * torch.arange(cfg["age_dim"] // 2, device=frames.device).float()
            / (cfg["age_dim"] // 2)
        )
        phase = age[..., None] * frequencies
        age_embedding = self.linear(torch.cat((phase.sin(), phase.cos()), -1), "age")
        x = frames + age_embedding + F.embedding(is_new.long(), self.p("is_new/embedding"))
        if cfg["use_redund"]:
            with torch.no_grad():
                normalized = frames / torch.linalg.vector_norm(frames, dim=-1, keepdim=True).clamp_min(1e-8)
                similarity = torch.einsum("bid,bjd->bij", normalized, normalized)
                pairs = valid[:, :, None] & valid[:, None, :] & ~torch.eye(n, device=x.device, dtype=torch.bool)[None]
                redundancy = torch.where(pairs, similarity, -1e30).max(-1).values
                redundancy = torch.where(pairs.any(-1), redundancy, 0)
            x = x + self.linear(redundancy[..., None], "redund")
        for i in range(cfg["scorer_layers"]):
            path = f"layers/layer_{i}/"
            normalized = self.ln(x, path + "norm_attn")
            x = x + self.attention(normalized, normalized, valid, path + "attn")
            x = x + self.linear(
                F.gelu(self.linear(self.ln(x, path + "norm_ffn"), path + "ffn/up"), approximate="none"),
                path + "ffn/down",
            )
        logits = self.linear(self.ln(x, "norm_output"), "output")[..., 0]
        return torch.where(valid, logits, torch.inf)


def replay_memory(scorer, features, ticks, valid, now, cfg, *, rematerialize=True):
    b, n = features.shape[:2]
    w, k = cfg["recent_frames"], cfg["queue_frames"]
    valid = valid & (ticks <= now[:, None])
    features = torch.where(valid[..., None, None], features, 0)
    frame_features = scorer.encode_frames(features.float())
    device = features.device
    q_ids = torch.zeros((b, k), device=device, dtype=torch.long)
    members = torch.zeros((b, k, n), device=device, dtype=torch.float32)
    q_ticks = ticks.new_zeros((b, k))
    q_count = torch.zeros((b,), device=device, dtype=torch.long)
    w_ids = torch.zeros((b, w), device=device, dtype=torch.long)
    w_ticks = ticks.new_zeros((b, w))
    w_count = torch.zeros((b,), device=device, dtype=torch.long)
    batch = torch.arange(b, device=device)
    slots = torch.arange(k, device=device)[None]
    # Writes after the last valid event are exact identities, including their
    # derivatives. Skip trailing padding; retain every real chronological write.
    effective = int(torch.where(valid, torch.arange(n, device=device)[None], -1).max().item()) + 1

    for index in range(effective):
        active, event_tick = valid[:, index], ticks[:, index]
        emit = active & (w_count == w)
        candidate, candidate_tick = w_ids[:, 0], w_ticks[:, 0]
        candidate_members = F.one_hot(candidate, n).float()
        overflow = emit & (q_count == k)
        if bool(overflow.any()):
            encoded = SelectMembers.apply(frame_features, members, q_ids, slots < q_count[:, None])
            encoded = torch.cat((encoded, gather_frames(frame_features, candidate[:, None])), 1)
            candidate_ticks = torch.cat((q_ticks, candidate_tick[:, None]), 1)
            all_valid = torch.ones((b, k + 1), device=device, dtype=torch.bool)
            is_new = (torch.arange(k + 1, device=device)[None] == k).expand(b, -1)
            args = (encoded, candidate_ticks, all_valid, event_tick, is_new)
            if rematerialize and torch.is_grad_enabled():
                logits = checkpoint(scorer.score_frames, *args, use_reentrant=False)
            else:
                logits = scorer.score_frames(*args)
            dropped = HardEviction.apply(logits / cfg["e2e_temperature"])
        else:
            dropped = features.new_zeros((b, k + 1), dtype=torch.float32)
        replacement = dropped[:, :k]
        updated = members * (1 - replacement[..., None]) + candidate_members[:, None] * replacement[..., None]
        members = torch.where(overflow[:, None, None], updated, members)
        replace = overflow[:, None] & replacement.bool()
        q_ids = torch.where(replace, candidate[:, None], q_ids)
        q_ticks = torch.where(replace, candidate_tick[:, None], q_ticks)

        append = emit & (q_count < k)
        slot = q_count.clamp_max(k - 1)
        values = torch.where(append[:, None], candidate_members, members[batch, slot])
        members = members.scatter(1, slot[:, None, None].expand(-1, 1, n), values[:, None])
        q_ids = q_ids.scatter(1, slot[:, None], torch.where(append, candidate, q_ids[batch, slot])[:, None])
        q_ticks = q_ticks.scatter(1, slot[:, None], torch.where(append, candidate_tick, q_ticks[batch, slot])[:, None])
        q_count = q_count + append.long()
        q_valid = slots < q_count[:, None]
        order = torch.where(q_valid, q_ticks, torch.iinfo(ticks.dtype).max).argsort(dim=1, stable=True)
        q_ids, q_ticks, members = [gather_frames(x, order) for x in (q_ids, q_ticks, members)]

        slot = w_count.clamp_max(w - 1)
        append_ids = w_ids.scatter(1, slot[:, None], torch.full((b, 1), index, device=device, dtype=torch.long))
        append_ticks = w_ticks.scatter(1, slot[:, None], event_tick[:, None])
        full_ids = torch.cat((w_ids[:, 1:], torch.full((b, 1), index, device=device, dtype=torch.long)), 1)
        full_ticks = torch.cat((w_ticks[:, 1:], event_tick[:, None]), 1)
        w_ids = torch.where(active[:, None], torch.where((w_count == w)[:, None], full_ids, append_ids), w_ids)
        w_ticks = torch.where(active[:, None], torch.where((w_count == w)[:, None], full_ticks, append_ticks), w_ticks)
        w_count = (w_count + active.long()).clamp_max(w)

    output_valid = torch.cat((slots < q_count[:, None], torch.arange(w, device=device)[None] < w_count[:, None]), 1)
    output_ids = torch.cat((q_ids, w_ids), 1)
    output_ticks = torch.cat((q_ticks, w_ticks), 1)
    members = torch.cat((members, F.one_hot(w_ids, n).float()), 1)
    order = torch.where(output_valid, output_ticks, torch.iinfo(ticks.dtype).max).argsort(dim=1, stable=True)
    output_ids, output_ticks, output_valid, members = [
        gather_frames(x, order) for x in (output_ids, output_ticks, output_valid, members)
    ]
    selected = SelectMembers.apply(features.float(), members, output_ids, output_valid).to(features.dtype)
    return selected, torch.where(output_valid, output_ticks, 0), output_valid, torch.where(output_valid, output_ids, -1)
