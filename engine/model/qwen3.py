"""Qwen3 forward pass (the dense variants, e.g. Qwen3-0.6B).

Operation order follows transformers' modeling_qwen3.py so fp32 outputs match it closely: RMSNorm
computed in fp32 then cast back, per-head RMSNorm on Q and K before RoPE, rotate-half RoPE from an
fp32 cos/sin table. Two deliberate differences, both standard in serving engines: q/k/v and
gate/up projections are fused into single matmuls, and the model takes a flat token batch
([T] ids and positions) instead of padded [batch, seq] tensors.

Attention is delegated to a backend object (engine/model/attention.py) which owns the KV cache;
the model never sees block tables.

Tensor parallelism splits each layer across ranks, Megatron-style: q/k/v and gate/up are split by
output (heads, intermediate columns), o_proj and down_proj by input, and an all-reduce after each of
those two sums the partial results, two per layer. The embedding and lm_head are split by vocabulary.
Pipeline parallelism gives each stage a contiguous block of layers: the first stage embeds, the last
normalizes and computes logits. With the default single-process ParallelState every all-reduce is a
no-op and every size is the full one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from engine.distributed.parallel import SINGLE, ParallelState, layer_range, tp_all_reduce


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]

    @classmethod
    def from_hf_dict(cls, d: dict) -> Qwen3Config:
        """Build from a HF config.json, refusing any setting that would change the math below."""
        unsupported = {
            "model_type": d.get("model_type") != "qwen3",
            "hidden_act": d.get("hidden_act", "silu") != "silu",
            "attention_bias": bool(d.get("attention_bias", False)),
            "rope_scaling": bool(d.get("rope_scaling")),
            "use_sliding_window": bool(d.get("use_sliding_window", False)),
        }
        bad = [k for k, is_bad in unsupported.items() if is_bad]
        if bad:
            raise ValueError(f"unsupported Qwen3 config settings: {bad}")
        eos = d.get("eos_token_id")
        eos_ids = tuple(eos) if isinstance(eos, list) else ((eos,) if eos is not None else ())
        return cls(
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            # Trap 1: head_dim is its own field (128 for 0.6B), not hidden_size // num_heads (64).
            head_dim=d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"],
            vocab_size=d["vocab_size"],
            rope_theta=float(d["rope_theta"]),
            rms_norm_eps=d["rms_norm_eps"],
            max_position_embeddings=d["max_position_embeddings"],
            tie_word_embeddings=bool(d.get("tie_word_embeddings", False)),
            eos_token_ids=eos_ids,
        )


class AttentionFn(Protocol):
    def __call__(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Write this step's k/v ([T, kv_heads, head_dim]) into the cache, then return attention
        output [T, num_heads, head_dim] for q ([T, num_heads, head_dim])."""
        ...


class VocabParallelEmbedding(nn.Module):
    """An embedding table split by rows across tensor-parallel ranks. Each rank looks up the ids that
    fall in its slice and contributes zeros for the rest; an all-reduce assembles every row."""

    def __init__(self, vocab_size: int, dim: int, parallel: ParallelState) -> None:
        super().__init__()
        self.parallel = parallel
        self.per_rank = vocab_size // parallel.tp_size
        self.start = parallel.tp_rank * self.per_rank
        self.weight = nn.Parameter(torch.empty(self.per_rank, dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if self.parallel.tp_size == 1:
            return F.embedding(ids, self.weight)
        local = ids - self.start
        inside = (local >= 0) & (local < self.per_rank)
        out = F.embedding(local.clamp(0, self.per_rank - 1), self.weight) * inside.unsqueeze(-1).to(self.weight.dtype)
        return tp_all_reduce(self.parallel, out)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotate-half RoPE. The fp32 cos/sin table is indexed by each token's position *within its own
    sequence*: never its index in the batch, never its slot in the cache."""

    def __init__(self, head_dim: int, theta: float, max_positions: int, device: torch.device | str | None = None) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim))
        freqs = torch.outer(torch.arange(max_positions, dtype=torch.float, device=device), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[positions].to(q.dtype).unsqueeze(1)  # [T, 1, head_dim], broadcast over heads
        sin = self.sin[positions].to(q.dtype).unsqueeze(1)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config, layer_idx: int, parallel: ParallelState = SINGLE) -> None:
        super().__init__()
        self.layer_idx = layer_idx  # index within this stage's layers, which is how the KV cache is indexed
        self.parallel = parallel
        self.num_heads = cfg.num_attention_heads // parallel.tp_size  # this rank's heads
        self.num_kv_heads = cfg.num_key_value_heads // parallel.tp_size
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim  # 2048 for 0.6B, twice hidden_size
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(cfg.hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor, rotary: RotaryEmbedding, attn: AttentionFn) -> torch.Tensor:
        t = hidden.shape[0]
        q, k, v = self.qkv_proj(hidden).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Trap 2: QK-norm is an RMSNorm over head_dim, per head, applied before RoPE.
        q = self.q_norm(q.view(t, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(t, self.num_kv_heads, self.head_dim))
        v = v.view(t, self.num_kv_heads, self.head_dim)
        q, k = rotary(positions, q, k)
        # Trap 3: 8 KV heads serve 16 query heads; the backend handles the GQA expansion.
        out = attn(self.layer_idx, q, k, v)
        return tp_all_reduce(self.parallel, self.o_proj(out.reshape(t, self.q_size)))


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config, parallel: ParallelState = SINGLE) -> None:
        super().__init__()
        self.parallel = parallel
        self.intermediate_size = cfg.intermediate_size // parallel.tp_size  # this rank's columns
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).split(self.intermediate_size, dim=-1)
        return tp_all_reduce(self.parallel, self.down_proj(F.silu(gate) * up))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config, layer_idx: int, parallel: ParallelState = SINGLE) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(cfg, layer_idx, parallel)
        self.mlp = Qwen3MLP(cfg, parallel)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor, rotary: RotaryEmbedding, attn: AttentionFn) -> torch.Tensor:
        hidden = hidden + self.self_attn(positions, self.input_layernorm(hidden), rotary, attn)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config, parallel: ParallelState = SINGLE) -> None:
        super().__init__()
        self.config = cfg
        self.parallel = parallel
        self.first_layer, last_layer = layer_range(cfg.num_hidden_layers, parallel.pp_size, parallel.pp_rank)
        self.num_local_layers = last_layer - self.first_layer
        self.num_local_heads = cfg.num_attention_heads // parallel.tp_size
        self.num_local_kv_heads = cfg.num_key_value_heads // parallel.tp_size
        first, last = parallel.is_first_stage, parallel.is_last_stage
        # The first stage embeds; with tied weights the last stage needs the same table for logits.
        needs_embedding = first or (last and cfg.tie_word_embeddings)
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size, parallel) if needs_embedding else None
        self.layers = nn.ModuleList(Qwen3DecoderLayer(cfg, i, parallel) for i in range(self.num_local_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps) if last else None
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.lm_head = None
        if last and not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size // parallel.tp_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, attn: AttentionFn, hidden: torch.Tensor | None = None
    ) -> torch.Tensor:
        """input_ids, positions: [T] for every token of every sequence in the batch, flattened.
        Later pipeline stages pass the previous stage's `hidden` instead of embedding. Returns
        hidden states [T, hidden_size], normalized on the last stage."""
        if hidden is None:
            hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(positions, hidden, self.rotary, attn)
        return self.norm(hidden) if self.norm is not None else hidden

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Logits over this rank's slice of the vocabulary (the whole vocabulary without tensor
        parallelism)."""
        weight = self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(hidden, weight)
