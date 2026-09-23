"""Scheduled batch → flat tensors → forward pass → logits.

Flatten, don't pad: every token of every scheduled sequence goes into one 1-D batch, with offsets
marking sequence boundaries (the varlen convention FlashAttention and FlashInfer use).

    3 sequences computing 4, 2 and 7 tokens, with c0, c1, c2 tokens already cached:
      input_ids    [t0..t3, t0..t1, t0..t6]                 shape [13]
      positions    [c0..c0+3, c1..c1+1, c2..c2+6]           position within each sequence
      slot_mapping block_table[p // bs] * bs + p % bs        where each token's K/V is written
      query_start  [0, 4, 6, 13]

Positions are indices within each token's own sequence: not its row in the batch, not its cache
slot. All index tensors are built on the CPU with numpy and moved to the device in a single copy per
step. An .item(), .tolist() or .cpu() in here would force a GPU sync every step.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from engine.config import EngineConfig
from engine.core.sampler import Sampler
from engine.core.scheduler import ScheduledBatch
from engine.core.sequence import SamplingParams
from engine.model.attention import AttentionMetadata, KVCache, NaivePagedAttention

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(self, model, config: EngineConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.torch_dtype
        mcfg = model.config
        self.num_layers = mcfg.num_hidden_layers
        self.num_heads = mcfg.num_attention_heads
        self.num_kv_heads = mcfg.num_key_value_heads
        self.head_dim = mcfg.head_dim
        self.vocab_size = mcfg.vocab_size
        self.block_size = config.block_size
        self.pin_memory = self.device.type == "cuda"
        self.kv_cache: KVCache | None = None
        self.cpu_cache: torch.Tensor | None = None
        self.backend = None
        self.graph_runner = None

    @property
    def bytes_per_block(self) -> int:
        return KVCache.bytes_per_block(self.num_layers, self.block_size, self.num_kv_heads, self.head_dim, self.dtype)

    def allocate_kv_cache(self, num_blocks: int, num_cpu_blocks: int) -> None:
        self.kv_cache = KVCache(
            self.num_layers, num_blocks, self.block_size, self.num_kv_heads, self.head_dim, self.dtype, self.device
        )
        if num_cpu_blocks > 0:
            self.cpu_cache = torch.zeros(
                (self.num_layers, 2, num_cpu_blocks, self.block_size, self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
                pin_memory=self.pin_memory,
            )
        if self.config.attention_backend == "flashinfer":
            from engine.model.attention import FlashInferAttention

            self.backend = FlashInferAttention(self.kv_cache, self.num_heads, self.num_kv_heads, self.head_dim, self.dtype)
        else:
            self.backend = NaivePagedAttention(self.kv_cache, self.head_dim)
        if self.config.cuda_graphs:
            from engine.core.cuda_graphs import CudaGraphRunner

            self.graph_runner = CudaGraphRunner(self)
            self.graph_runner.capture()

    # --- one step --------------------------------------------------------------------------------

    @torch.inference_mode()
    def execute(self, batch: ScheduledBatch) -> torch.Tensor | None:
        """Run the batch's swaps and forward pass. Returns logits [num_sampling_seqs, vocab] in batch
        order, or None when no sequence samples this step (only intermediate prefill chunks)."""
        if batch.swap_out:
            self._swap_out(batch.swap_out)
        if batch.swap_in:
            self._swap_in(batch.swap_in)
        if not batch.seqs:
            return None
        if self.graph_runner is not None and self.graph_runner.can_run(batch):
            return self.graph_runner.run(batch)
        input_ids, positions, logits_indices, meta = self.prepare(batch)
        self.backend.begin_step(meta)
        hidden = self.model(input_ids, positions, self.backend)
        if not any(batch.do_sample):
            return None
        return self.model.compute_logits(hidden.index_select(0, logits_indices))

    def prepare(self, batch: ScheduledBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, AttentionMetadata]:
        bs = self.block_size
        query_lens = batch.num_new_tokens
        starts = [seq.num_computed_tokens for seq in batch.seqs]
        kv_lens = [start + n for start, n in zip(starts, query_lens)]
        query_start = [0]
        for n in query_lens:
            query_start.append(query_start[-1] + n)
        total = query_start[-1]

        input_ids = np.empty(total, dtype=np.int64)
        positions = np.empty(total, dtype=np.int64)
        slot_mapping = np.empty(total, dtype=np.int64)
        tables = []
        for i, seq in enumerate(batch.seqs):
            lo, hi, start = query_start[i], query_start[i + 1], starts[i]
            input_ids[lo:hi] = seq.token_ids[start : start + query_lens[i]]
            pos = np.arange(start, start + query_lens[i], dtype=np.int64)
            positions[lo:hi] = pos
            table = np.asarray(seq.block_table[: -(-kv_lens[i] // bs)], dtype=np.int64)
            tables.append(table)
            slot_mapping[lo:hi] = table[pos // bs] * bs + pos % bs
        logits_indices = np.asarray(
            [query_start[i + 1] - 1 for i, sample in enumerate(batch.do_sample) if sample], dtype=np.int64
        )

        naive = isinstance(self.backend, NaivePagedAttention)
        kv_start = None
        kv_slots = np.empty(0, dtype=np.int64)
        if naive:
            kv_start = np.cumsum([0, *kv_lens]).tolist()
            kv_slots = np.concatenate(
                [(table[:, None] * bs + np.arange(bs)).reshape(-1)[:kv_len] for table, kv_len in zip(tables, kv_lens)]
            )
        parts = [input_ids, positions, slot_mapping, logits_indices, kv_slots]
        packed = torch.from_numpy(np.concatenate(parts))
        if self.pin_memory:
            packed = packed.pin_memory()
        packed = packed.to(self.device, non_blocking=True)  # the step's one host-to-device copy
        ids_t, pos_t, slots_t, logits_idx_t, kv_slots_t = packed.split([len(p) for p in parts])

        meta = AttentionMetadata(
            query_lens=query_lens,
            kv_lens=kv_lens,
            query_start=query_start,
            slot_mapping=slots_t,
            kv_slots=kv_slots_t if naive else None,
            kv_start=kv_start,
            block_tables=None if naive else tables,
        )
        return ids_t, pos_t, logits_idx_t, meta

    # --- swapping --------------------------------------------------------------------------------

    def _swap_out(self, pairs: list[tuple[int, int]]) -> None:
        gpu_ids = torch.tensor([g for g, _ in pairs], dtype=torch.int64, device=self.device)
        cpu_ids = torch.tensor([c for _, c in pairs], dtype=torch.int64)
        staged = self.kv_cache.data.index_select(2, gpu_ids)  # every layer's K and V in one gather
        host = torch.empty(staged.shape, dtype=self.dtype, pin_memory=self.pin_memory)
        host.copy_(staged)  # blocking: the host copy must be complete before it is read below
        self.cpu_cache.index_copy_(2, cpu_ids, host)

    def _swap_in(self, pairs: list[tuple[int, int]]) -> None:
        cpu_ids = torch.tensor([c for c, _ in pairs], dtype=torch.int64)
        gpu_ids = torch.tensor([g for _, g in pairs], dtype=torch.int64, device=self.device)
        host = torch.empty(
            (self.num_layers, 2, len(pairs), self.block_size, self.num_kv_heads, self.head_dim),
            dtype=self.dtype,
            pin_memory=self.pin_memory,
        )
        torch.index_select(self.cpu_cache, 2, cpu_ids, out=host)
        self.kv_cache.data.index_copy_(2, gpu_ids, host.to(self.device, non_blocking=True))

    # --- sizing ----------------------------------------------------------------------------------

    @torch.inference_mode()
    def profile_num_blocks(self, sampler: Sampler) -> int:
        """Size the KV cache from measured memory, never a hardcoded count: run one dummy forward at
        the largest batch the scheduler can produce, plus the sampler at its most expensive, then give
        the cache whatever the gpu_memory_utilization budget leaves."""
        cfg = self.config
        device = self.device
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        num_tokens = cfg.max_batch_tokens
        num_seqs = min(cfg.max_num_seqs, num_tokens)
        num_dummy = min(num_seqs, 16)  # attention activations depend on tokens, not sequence count
        lens = [num_tokens // num_dummy] * num_dummy
        lens[-1] += num_tokens - sum(lens)
        input_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        positions = torch.cat([torch.arange(n, device=device) for n in lens])
        hidden = self.model(input_ids, positions, _ProfileAttention(lens))
        logits = self.model.compute_logits(hidden[:num_seqs])
        sampler(logits, sampler.prepare([SamplingParams(temperature=1.0, top_p=0.9)] * num_seqs))
        torch.cuda.synchronize(device)

        peak = torch.cuda.max_memory_allocated(device)
        free, total = torch.cuda.mem_get_info(device)
        non_torch = (total - free) - torch.cuda.memory_reserved(device)
        # Allocated after this profile, so reserve them explicitly: FlashInfer's workspace, and the
        # private memory pool the captured decode graphs keep their activations in.
        later = 0
        if cfg.attention_backend == "flashinfer":
            from engine.model.attention import FlashInferAttention

            later += FlashInferAttention.WORKSPACE_BYTES
        if cfg.cuda_graphs:
            later += 512 << 20
        budget = total * cfg.gpu_memory_utilization - peak - non_torch - later
        num_blocks = int(budget // self.bytes_per_block)
        gib = 1 << 30
        logger.info(
            "memory profile: total %.2f GiB, peak torch %.2f GiB, non-torch %.2f GiB, reserved %.2f GiB, "
            "KV budget %.2f GiB -> %d blocks of %d tokens (%.1f MiB each, %d tokens total)",
            total / gib, peak / gib, non_torch / gib, later / gib, budget / gib, num_blocks, self.block_size,
            self.bytes_per_block / (1 << 20), num_blocks * self.block_size,
        )
        del hidden, logits
        torch.cuda.empty_cache()
        if num_blocks < 1:
            raise RuntimeError("no GPU memory left for the KV cache; raise gpu_memory_utilization or lower max_batch_tokens")
        return num_blocks


class _ProfileAttention:
    """Causal attention over the dummy batch's own K/V, no cache: activation memory only."""

    def __init__(self, lens: list[int]) -> None:
        self.bounds = np.cumsum([0, *lens]).tolist()

    def __call__(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(q)
        rep = q.shape[1] // k.shape[1]
        for lo, hi in zip(self.bounds, self.bounds[1:]):
            qs = q[lo:hi].transpose(0, 1)
            ks = k[lo:hi].repeat_interleave(rep, dim=1).transpose(0, 1)
            vs = v[lo:hi].repeat_interleave(rep, dim=1).transpose(0, 1)
            out[lo:hi] = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True).transpose(0, 1)
        return out
