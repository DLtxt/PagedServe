"""KV-cache storage and attention backends.

The cache is a single tensor [num_layers, 2, num_blocks + 1, block_size, num_kv_heads, head_dim].
Per-layer K and V views, [num_blocks + 1, block_size, kv_heads, head_dim], are contiguous, which is
FlashInfer's NHD paged layout, and one index_select along the block dimension moves every layer at
once when swapping. Block `num_blocks` is a scratch page the block manager never hands out: padded
rows of a CUDA-graph batch write there.

Backends are called from inside the model as `backend(layer_idx, q, k, v) -> out`. Each first writes
this step's k/v into its storage, then attends:

  ContiguousAttention   one sequence, per-layer tensors grown by concatenation; no paging at all.
                        The week-10 reference that gates 1 and 2 compare against.
  NaivePagedAttention   gathers each sequence's pages into a contiguous tensor and calls SDPA.
                        Slow and obviously correct: the oracle every faster path is checked against.
  FlashInferAttention   FlashInfer's paged kernels, CUDA only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


class KVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.scratch_block = num_blocks
        self.data = torch.zeros(
            num_layers, 2, num_blocks + 1, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )

    @staticmethod
    def bytes_per_block(num_layers: int, block_size: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
        return num_layers * 2 * block_size * num_kv_heads * head_dim * dtype.itemsize

    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """[num_blocks + 1, block_size, kv_heads, head_dim] K and V pages of one layer."""
        return self.data[layer_idx, 0], self.data[layer_idx, 1]

    def flat(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """K and V of one layer viewed as [(num_blocks + 1) * block_size, kv_heads, head_dim], so a
        token's slot index (block * block_size + offset) addresses it directly."""
        k, v = self.layer(layer_idx)
        return k.flatten(0, 1), v.flatten(0, 1)


@dataclass
class AttentionMetadata:
    """Per-step batch layout, built on the CPU by the model runner.

    Sequence i contributes query_lens[i] new tokens, at flattened rows
    query_start[i]:query_start[i + 1]. After this step's writes it has kv_lens[i] tokens in the
    cache; its query tokens are the last query_lens[i] of them.
    """

    query_lens: list[int]
    kv_lens: list[int]
    query_start: list[int]
    slot_mapping: torch.Tensor  # [T] int64: destination slot of each new token's K/V
    # Naive backend only: every cached slot of every sequence, concatenated; sequence i owns
    # kv_slots[kv_start[i]:kv_start[i + 1]].
    kv_slots: torch.Tensor | None = None
    kv_start: list[int] | None = None
    # FlashInfer backend only: each sequence's block table, trimmed to its kv_len (numpy int64).
    block_tables: list | None = None

    @property
    def num_seqs(self) -> int:
        return len(self.query_lens)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """q: [Tq, H, D]; k, v: [Tk, KVH, D] with Tk >= Tq. Returns [Tq, H, D].

    The queries are the *last* Tq positions of the sequence, so the causal mask is aligned
    bottom-right: query i may attend to keys j <= i + (Tk - Tq). With Tk == Tq this is the usual
    triangle. With Tk > Tq (a decode step, a later prefill chunk, or a prefill behind a cached
    prefix) the offset is what keeps earlier context visible.
    """
    num_heads, num_kv_heads = q.shape[1], k.shape[1]
    if num_heads != num_kv_heads:
        k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
    tq, tk = q.shape[0], k.shape[0]
    mask = None
    if tq > 1:
        rows = torch.arange(tq, device=q.device).unsqueeze(1)
        cols = torch.arange(tk, device=q.device).unsqueeze(0)
        mask = cols <= rows + (tk - tq)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask, scale=scale
    )
    return out.transpose(0, 1)


class ContiguousAttention:
    def __init__(self, num_layers: int, head_dim: int) -> None:
        self.k: list[torch.Tensor | None] = [None] * num_layers
        self.v: list[torch.Tensor | None] = [None] * num_layers
        self.scale = head_dim**-0.5

    def __call__(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.k[layer_idx] is None:
            self.k[layer_idx], self.v[layer_idx] = k.contiguous(), v.contiguous()
        else:
            self.k[layer_idx] = torch.cat((self.k[layer_idx], k))
            self.v[layer_idx] = torch.cat((self.v[layer_idx], v))
        return causal_attention(q, self.k[layer_idx], self.v[layer_idx], self.scale)


class NaivePagedAttention:
    def __init__(self, cache: KVCache, head_dim: int) -> None:
        self.cache = cache
        self.scale = head_dim**-0.5
        self.meta: AttentionMetadata | None = None

    def begin_step(self, meta: AttentionMetadata) -> None:
        self.meta = meta

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        # The slot-mapping trick: one scatter writes the whole batch's new K/V into their pages.
        k_flat, v_flat = self.cache.flat(layer_idx)
        k_flat.index_copy_(0, self.meta.slot_mapping, k)
        v_flat.index_copy_(0, self.meta.slot_mapping, v)

    def gather(self, layer_idx: int, seq_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequence seq_idx's cached K and V, [kv_len, kv_heads, head_dim], in position order."""
        meta = self.meta
        slots = meta.kv_slots[meta.kv_start[seq_idx] : meta.kv_start[seq_idx + 1]]
        k_flat, v_flat = self.cache.flat(layer_idx)
        return k_flat.index_select(0, slots), v_flat.index_select(0, slots)

    def __call__(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self.write(layer_idx, k, v)
        meta = self.meta
        out = torch.empty_like(q)
        for i in range(meta.num_seqs):
            start, end = meta.query_start[i], meta.query_start[i + 1]
            keys, values = self.gather(layer_idx, i)
            out[start:end] = causal_attention(q[start:end], keys, values, self.scale)
        return out


@torch.inference_mode()
def greedy_generate_contiguous(
    model, prompt_ids: list[int], max_new_tokens: int, stop_ids: tuple[int, ...] = ()
) -> tuple[list[int], list[torch.Tensor]]:
    """Single-sequence greedy decoding with a plain contiguous KV cache (the week-10 build).
    Returns the generated tokens and each step's fp32 logits."""
    cfg = model.config
    device = model.embed_tokens.weight.device
    attn = ContiguousAttention(cfg.num_hidden_layers, cfg.head_dim)
    ids = torch.tensor(prompt_ids, device=device)
    positions = torch.arange(len(prompt_ids), device=device)
    tokens: list[int] = []
    step_logits: list[torch.Tensor] = []
    for step in range(max_new_tokens):
        hidden = model(ids, positions, attn)
        logits = model.compute_logits(hidden[-1:])[0].float()
        token = int(logits.argmax())
        tokens.append(token)
        step_logits.append(logits)
        if token in stop_ids:
            break
        ids = torch.tensor([token], device=device)
        positions = torch.tensor([len(prompt_ids) + step], device=device)
    return tokens, step_logits


def paged_kv_index(block_tables: list, kv_lens: list[int], block_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block tables in FlashInfer's CSR form: (indptr [B+1], indices [pages], last_page_len [B]),
    int32 on the CPU. `block_tables[i]` covers exactly ceil(kv_lens[i] / block_size) pages."""
    import numpy as np

    pages = [len(t) for t in block_tables]
    indptr = np.zeros(len(pages) + 1, dtype=np.int32)
    np.cumsum(pages, out=indptr[1:])
    indices = np.concatenate(block_tables).astype(np.int32) if block_tables else np.zeros(0, np.int32)
    last = np.array([n - (p - 1) * block_size for n, p in zip(kv_lens, pages)], dtype=np.int32)
    return torch.from_numpy(indptr), torch.from_numpy(indices), torch.from_numpy(last)


class FlashInferAttention:
    """FlashInfer's paged kernels (CUDA only): one plan() per step, one run() per layer.

    Batches containing any prefill tokens use BatchPrefillWithPagedKVCacheWrapper; its causal mask
    is aligned bottom-right (query i of a sequence sees keys up to kv_len - q_len + i), exactly the
    semantics chunked prefill and prefix-cache hits need. Pure decode batches use the faster
    BatchDecodeWithPagedKVCacheWrapper. GQA is handled inside the kernels. K/V writes use the same
    slot-mapping scatter as the naive backend, so the two paths differ only in how attention reads
    the cache, which is what gate 9 compares.
    """

    WORKSPACE_BYTES = 256 * 1024 * 1024

    def __init__(self, cache: KVCache, num_heads: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> None:
        import flashinfer

        self.cache = cache
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.workspace = torch.empty(self.WORKSPACE_BYTES, dtype=torch.uint8, device=cache.data.device)
        self.prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "NHD")
        self.decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace, "NHD")
        self.meta: AttentionMetadata | None = None
        self.decode_only = False

    def begin_step(self, meta: AttentionMetadata) -> None:
        self.meta = meta
        self.decode_only = all(n == 1 for n in meta.query_lens)
        indptr, indices, last = (t.pin_memory() for t in paged_kv_index(meta.block_tables, meta.kv_lens, self.cache.block_size))
        if self.decode_only:
            self.decode.plan(
                indptr, indices, last, self.num_heads, self.num_kv_heads, self.head_dim, self.cache.block_size,
                q_data_type=self.dtype, kv_data_type=self.dtype,
            )
        else:
            qo_indptr = torch.tensor(meta.query_start, dtype=torch.int32).pin_memory()
            self.prefill.plan(
                qo_indptr, indptr, indices, last, self.num_heads, self.num_kv_heads, self.head_dim, self.cache.block_size,
                causal=True, q_data_type=self.dtype, kv_data_type=self.dtype,
            )

    def __call__(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        k_flat, v_flat = self.cache.flat(layer_idx)
        k_flat.index_copy_(0, self.meta.slot_mapping, k)
        v_flat.index_copy_(0, self.meta.slot_mapping, v)
        wrapper = self.decode if self.decode_only else self.prefill
        return wrapper.run(q, self.cache.layer(layer_idx))
