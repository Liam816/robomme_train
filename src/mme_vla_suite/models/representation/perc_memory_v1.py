"""Perceptual memory v1. All backbone adaptation lives outside scorer/reader."""

from dataclasses import dataclass, fields
import math

from flax import nnx
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PercMemConfig:
    native_dim: int = 2048
    feature_dim: int = 2560
    output_dim: int = 2048
    store_grid: int = 8
    read_grid: int = 4
    score_grid: int = 4
    clock_hz: int = 20
    key_stride: int = 10
    recent_frames: int = 4
    queue_frames: int = 28
    scorer_dim: int = 256
    scorer_heads: int = 4
    scorer_layers: int = 2
    scorer_ffn: int = 1024
    scorer_dropout: float = 0.1
    scorer_use_instr: bool = False
    use_redund: bool = True
    reader_dim: int = 512
    reader_queries: int = 8
    reader_heads: int = 8
    reader_layers: int = 2
    reader_ffn: int = 2048
    age_dim: int = 256
    norm_eps: float = 1e-5
    sinusoid_base: float = 10000.0
    attention_bias: bool = True
    stage: str = "teacher"
    evict: str = "learned"
    max_frames: int = 320
    input_space: str = "native"
    memory_mode: str = "normal"
    banks_dir: str | None = None
    projected_dir: str | None = None
    scorer_path: str | None = None
    teacher_params: str | None = None
    distill_weight: float = 1.0
    control_hz: float | None = None
    e2e_temperature: float = 1.0

    def __post_init__(self):
        if self.stage not in {"teacher", "student", "e2e"}:
            raise ValueError("stage must be teacher, student or e2e")
        if not math.isfinite(self.e2e_temperature) or self.e2e_temperature <= 0:
            raise ValueError("e2e_temperature must be finite and positive")
        if self.stage == "e2e" and (self.evict != "learned" or self.scorer_use_instr):
            raise ValueError("E2E uses learned visual-only eviction")
        if self.input_space not in {"native", "projected"}:
            raise ValueError("input_space must be native or projected")
        if self.memory_mode not in {"normal", "zero", "shuffle"}:
            raise ValueError("Unknown memory diagnostic mode")
        if self.evict not in {"learned", "fifo", "uniform", "random", "oracle"}:
            raise ValueError("Unknown eviction policy")
        if self.recent_frames < 1 or self.queue_frames < 1 or self.clock_hz < 1 or self.key_stride < 1:
            raise ValueError("Memory capacities and clock parameters must be positive")
        if self.max_frames < self.recent_frames + self.queue_frames:
            raise ValueError("max_frames cannot be smaller than the student bank")
        if self.store_grid % self.read_grid or self.store_grid % self.score_grid:
            raise ValueError("Read/score grids must divide the storage grid")
        if self.scorer_dim % self.scorer_heads or self.reader_dim % self.reader_heads:
            raise ValueError("Attention width must divide into heads")
        if self.age_dim % 2 or self.norm_eps <= 0 or self.sinusoid_base <= 1:
            raise ValueError("Invalid normalization or age embedding configuration")

    @classmethod
    def from_history(cls, history):
        values = dict(history.perc_mem)
        unknown = values.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown perceptual memory settings: {sorted(unknown)}")
        return cls(**values)

    @property
    def capacity(self):
        return self.recent_frames + self.queue_frames


def sinusoid(age, dim=256, base=10000.0):
    frequencies = jnp.exp(-math.log(base) * jnp.arange(dim // 2, dtype=jnp.float32) / (dim // 2))
    phase = jnp.asarray(age, jnp.float32)[..., None] * frequencies
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


def pool_tokens(tokens, grid=4):
    """Row-major square token grid → average-pooled row-major square grid."""
    source = math.isqrt(tokens.shape[-2])
    if source * source != tokens.shape[-2] or source % grid:
        raise ValueError(f"Cannot average-pool {tokens.shape[-2]} tokens to {grid}x{grid}")
    stride = source // grid
    shape = (*tokens.shape[:-2], grid, stride, grid, stride, tokens.shape[-1])
    return tokens.astype(jnp.float32).reshape(shape).mean(axis=(-4, -2)).reshape(
        *tokens.shape[:-2], grid * grid, tokens.shape[-1])


def dropout(x, rate, key, deterministic):
    if deterministic or rate == 0:
        return x
    if key is None:
        raise ValueError("Training dropout requires an explicit PRNG key")
    keep = 1.0 - rate
    return jnp.where(jax.random.bernoulli(key, keep, x.shape), x / keep, 0)


class Attention(nnx.Module):
    def __init__(self, dim, heads, *, rngs, bias=True, dropout=0.0):
        self.heads = heads
        self.q = nnx.Linear(dim, dim, use_bias=bias, rngs=rngs)
        self.k = nnx.Linear(dim, dim, use_bias=bias, rngs=rngs)
        self.v = nnx.Linear(dim, dim, use_bias=bias, rngs=rngs)
        self.out = nnx.Linear(dim, dim, use_bias=bias, rngs=rngs)
        self.dropout = dropout

    def __call__(self, query, memory, mask, *, deterministic=True, rng=None):
        b, nq, dim = query.shape
        head_dim = dim // self.heads
        q = self.q(query).reshape(b, nq, self.heads, head_dim)
        k = self.k(memory).reshape(b, memory.shape[1], self.heads, head_dim)
        v = self.v(memory).reshape(b, memory.shape[1], self.heads, head_dim)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(head_dim)
        valid = mask[:, None, None, :]
        weights = jax.nn.softmax(jnp.where(valid, logits.astype(jnp.float32), -1e30), axis=-1)
        weights = jnp.where(valid, weights, 0.0)
        weights = weights / jnp.maximum(weights.sum(-1, keepdims=True), 1e-20)
        used = dropout(weights, self.dropout, rng, deterministic)
        result = jnp.einsum("bhqk,bkhd->bqhd", used, v).reshape(b, nq, dim)
        return self.out(result), weights


class FFN(nnx.Module):
    def __init__(self, dim, hidden, *, rngs, dropout=0.0):
        self.up = nnx.Linear(dim, hidden, rngs=rngs)
        self.down = nnx.Linear(hidden, dim, rngs=rngs)
        self.dropout = dropout

    def __call__(self, x, *, deterministic=True, rng=None):
        return self.down(dropout(jax.nn.gelu(self.up(x), approximate=False), self.dropout, rng, deterministic))


class ScorerLayer(nnx.Module):
    def __init__(self, cfg, *, rngs):
        self.norm_attn = nnx.LayerNorm(cfg.scorer_dim, epsilon=cfg.norm_eps, rngs=rngs)
        self.norm_ffn = nnx.LayerNorm(cfg.scorer_dim, epsilon=cfg.norm_eps, rngs=rngs)
        self.attn = Attention(cfg.scorer_dim, cfg.scorer_heads, rngs=rngs,
                              bias=cfg.attention_bias, dropout=cfg.scorer_dropout)
        self.ffn = FFN(cfg.scorer_dim, cfg.scorer_ffn, rngs=rngs, dropout=cfg.scorer_dropout)
        self.dropout = cfg.scorer_dropout

    def __call__(self, x, valid, deterministic, rng=None):
        keys = [None] * 4 if rng is None else jax.random.split(rng, 4)
        normalized = self.norm_attn(x)
        update, _ = self.attn(normalized, normalized, valid, deterministic=deterministic, rng=keys[0])
        x = x + dropout(update, self.dropout, keys[1], deterministic)
        update = self.ffn(self.norm_ffn(x), deterministic=deterministic, rng=keys[2])
        return x + dropout(update, self.dropout, keys[3], deterministic)


class EvictScorer(nnx.Module):
    def __init__(self, cfg=PercMemConfig(), *, rngs):
        self.cfg = cfg
        self.norm_input = nnx.LayerNorm(cfg.feature_dim, epsilon=cfg.norm_eps, rngs=rngs)
        self.proj = nnx.Linear(cfg.feature_dim, cfg.scorer_dim, rngs=rngs)
        self.query = nnx.Param(jax.random.normal(rngs.params(), (1, 1, cfg.scorer_dim)) * 0.02)
        self.pool_attn = Attention(cfg.scorer_dim, cfg.scorer_heads, rngs=rngs, bias=cfg.attention_bias)
        self.age = nnx.Linear(cfg.age_dim, cfg.scorer_dim, rngs=rngs)
        self.is_new = nnx.Embed(2, cfg.scorer_dim, rngs=rngs)
        self.redund = nnx.Linear(1, cfg.scorer_dim, rngs=rngs)
        self.layers = {f"layer_{i}": ScorerLayer(cfg, rngs=rngs) for i in range(cfg.scorer_layers)}
        self.norm_output = nnx.LayerNorm(cfg.scorer_dim, epsilon=cfg.norm_eps, rngs=rngs)
        self.output = nnx.Linear(cfg.scorer_dim, 1, rngs=rngs)
        if cfg.scorer_use_instr:
            self.instruction = nnx.Linear(cfg.feature_dim, cfg.scorer_dim, rngs=rngs)

    def __call__(self, feats, ticks, valid, now, is_new, *, instruction=None, deterministic=True, rng=None):
        return self.score_frames(self.encode_frames(feats), ticks, valid, now, is_new,
                                 instruction=instruction, deterministic=deterministic, rng=rng)

    def encode_frames(self, feats):
        """The unchanged frame-local encoder, shared across chronological writes."""
        cfg = self.cfg
        b, n = feats.shape[:2]
        x = self.proj(self.norm_input(pool_tokens(feats, cfg.score_grid)))
        x = x.reshape(b * n, cfg.score_grid ** 2, cfg.scorer_dim)
        query = jnp.broadcast_to(self.query.value, (b * n, 1, cfg.scorer_dim))
        f, _ = self.pool_attn(query, x, jnp.ones(x.shape[:2], bool))
        return f.reshape(b, n, cfg.scorer_dim)

    def score_frames(self, f, ticks, valid, now, is_new, *, instruction=None, deterministic=True, rng=None):
        cfg = self.cfg
        b, n = f.shape[:2]
        age = (now[:, None] - ticks) / cfg.clock_hz
        x = f + self.age(sinusoid(age, cfg.age_dim, cfg.sinusoid_base)) + self.is_new(is_new.astype(jnp.int32))
        if cfg.use_redund:
            normalized = f / jnp.maximum(jnp.linalg.norm(f, axis=-1, keepdims=True), 1e-8)
            similarity = jnp.einsum("bid,bjd->bij", normalized, normalized)
            pair_mask = valid[:, :, None] & valid[:, None, :] & ~jnp.eye(n, dtype=bool)[None]
            redundancy = jnp.max(jnp.where(pair_mask, similarity, -1e30), axis=-1)
            redundancy = jnp.where(pair_mask.any(-1), redundancy, 0.0)
            x = x + self.redund(jax.lax.stop_gradient(redundancy)[..., None])
        if cfg.scorer_use_instr:
            if instruction is None:
                raise ValueError("Instruction ablation requires frozen input-embedding means")
            x = jnp.concatenate([self.instruction(instruction)[:, None], x], axis=1)
            encoder_mask = jnp.concatenate([jnp.ones((b, 1), bool), valid], axis=1)
        else:
            encoder_mask = valid
        for i, layer in enumerate(self.layers.values()):
            key = None if rng is None else jax.random.fold_in(rng, i)
            x = layer(x, encoder_mask, deterministic, key)
        if cfg.scorer_use_instr:
            x = x[:, 1:]
        logits = self.output(self.norm_output(x))[..., 0]
        return jnp.where(valid, logits, jnp.inf)


class ReaderLayer(nnx.Module):
    def __init__(self, cfg, *, rngs):
        self.cross_norm = nnx.LayerNorm(cfg.reader_dim, epsilon=cfg.norm_eps, rngs=rngs)
        self.self_norm = nnx.LayerNorm(cfg.reader_dim, epsilon=cfg.norm_eps, rngs=rngs)
        self.ffn_norm = nnx.LayerNorm(cfg.reader_dim, epsilon=cfg.norm_eps, rngs=rngs)
        self.cross = Attention(cfg.reader_dim, cfg.reader_heads, rngs=rngs, bias=cfg.attention_bias)
        self.self_attn = Attention(cfg.reader_dim, cfg.reader_heads, rngs=rngs, bias=cfg.attention_bias)
        self.ffn = FFN(cfg.reader_dim, cfg.reader_ffn, rngs=rngs)

    def __call__(self, z, m, valid):
        update, weights = self.cross(self.cross_norm(z), m, valid)
        z = z + update
        normed = self.self_norm(z)
        update, _ = self.self_attn(normed, normed, jnp.ones(z.shape[:2], bool))
        z = z + update
        return z + self.ffn(self.ffn_norm(z)), weights


class MemReader(nnx.Module):
    def __init__(self, cfg=PercMemConfig(), *, rngs):
        self.cfg = cfg
        self.input_norm = nnx.LayerNorm(cfg.feature_dim, epsilon=cfg.norm_eps, rngs=rngs)
        self.proj = nnx.Linear(cfg.feature_dim, cfg.reader_dim, rngs=rngs)
        self.proj_norm = nnx.LayerNorm(cfg.reader_dim, epsilon=cfg.norm_eps, rngs=rngs)
        self.position = nnx.Embed(cfg.read_grid ** 2, cfg.reader_dim, rngs=rngs)
        self.age_in = nnx.Linear(cfg.age_dim, cfg.reader_dim, rngs=rngs)
        self.age_out = nnx.Linear(cfg.reader_dim, cfg.reader_dim, rngs=rngs)
        self.query = nnx.Param(jax.random.normal(rngs.params(), (1, cfg.reader_queries, cfg.reader_dim)) * 0.02)
        self.layers = {f"layer_{i}": ReaderLayer(cfg, rngs=rngs) for i in range(cfg.reader_layers)}
        self.output_norm = nnx.LayerNorm(cfg.reader_dim, epsilon=cfg.norm_eps, rngs=rngs)
        self.output_in = nnx.Linear(cfg.reader_queries * cfg.reader_dim, cfg.output_dim, rngs=rngs)
        self.output_final = nnx.Linear(cfg.output_dim, cfg.output_dim, kernel_init=nnx.initializers.zeros,
                                       bias_init=nnx.initializers.zeros, rngs=rngs)

    def __call__(self, feats, ticks, valid, now, *, return_attention=False):
        cfg = self.cfg
        b, n = feats.shape[:2]
        m = self.proj_norm(self.proj(self.input_norm(pool_tokens(feats, cfg.read_grid))))
        age = (now[:, None] - ticks) / cfg.clock_hz
        age = self.age_out(jax.nn.gelu(self.age_in(sinusoid(age, cfg.age_dim, cfg.sinusoid_base)), approximate=False))
        m = m + self.position(jnp.arange(cfg.read_grid ** 2))[None, None] + age[:, :, None]
        m = m.reshape(b, n * cfg.read_grid ** 2, cfg.reader_dim)
        mask = jnp.repeat(valid, cfg.read_grid ** 2, axis=1)
        z = jnp.broadcast_to(self.query.value, (b, cfg.reader_queries, cfg.reader_dim))
        attention = []
        for layer in self.layers.values():
            z, weights = layer(z, m, mask)
            if return_attention:
                attention.append(weights.reshape(b, cfg.reader_heads, cfg.reader_queries, n, cfg.read_grid ** 2))
        result = self.output_final(jax.nn.silu(self.output_in(self.output_norm(z).reshape(b, -1))))
        result = jnp.where(valid.any(axis=1)[:, None], result, 0.0)
        if return_attention:
            return result, jnp.stack(attention, axis=1)  # B, layer, head, query, frame, patch
        return result


def pairwise_ranking_loss(scores, utility, valid, key, max_pairs=64, margin=0.1):
    """Sample at most max_pairs eligible ordered pairs, without replacement."""
    eligible = valid[:, None] & valid[None, :] & ((utility[:, None] - utility[None, :]) > margin)
    priority = jax.random.uniform(key, eligible.shape)
    flat = jnp.where(eligible, priority, -1.0).reshape(-1)
    count = min(max_pairs, flat.size)
    _, indices = jax.lax.top_k(flat, count)
    mask = eligible.reshape(-1)[indices]
    safe_scores = jnp.where(valid, scores, 0.0)
    differences = (safe_scores[:, None] - safe_scores[None, :]).reshape(-1)[indices]
    return jnp.where(mask, jax.nn.softplus(-differences), 0).sum() / jnp.maximum(mask.sum(), 1)
