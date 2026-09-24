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
from engine.distributed.parallel import SINGLE, ParallelState, all_ranks_min, isend_to_next_stage, recv_from_prev_stage
from engine.core.scheduler import ScheduledBatch
from engine.core.sequence import SamplingParams
from engine.model.attention import AttentionMetadata, KVCache, NaivePagedAttention

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(self, model, config: EngineConfig, parallel: ParallelState = SINGLE) -> None:
        self.model = model
        self.config = config
        self.parallel = parallel
        self.device = torch.device(config.device)
        self.dtype = config.torch_dtype
        mcfg = model.config
        # This rank's share: its pipeline stage's layers and its tensor-parallel slice of the heads.
        self.num_layers = getattr(model, "num_local_layers", mcfg.num_hidden_layers)
        self.num_heads = getattr(model, "num_local_heads", mcfg.num_attention_heads)
        self.num_kv_heads = getattr(model, "num_local_kv_heads", mcfg.num_key_value_heads)
        self.hidden_size = getattr(mcfg, "hidden_size", None)
        self.head_dim = mcfg.head_dim
        self.vocab_size = mcfg.vocab_size
        self.block_size = config.block_size
        self.pin_memory = self.device.type == "cuda"
        self.kv_cache: KVCache | None = None
        self.cpu_cache: torch.Tensor | None = None
        self.backend = None
        self.graph_runner = None
        self._sends: list[tuple] = []  # non-blocking sends in progress: (work, tensor kept alive)
        # Recorded after each step's attention plan: see execute().
        self._staged = torch.cuda.Event() if self.device.type == "cuda" else None

    @property
    def bytes_per_block(self) -> int:
        return KVCache.bytes_per_block(self.num_layers, self.block_size, self.num_kv_heads, self.head_dim, self.dtype)

    def size_and_allocate(self, sampler: Sampler) -> tuple[int, int]:
        """Decide how many GPU and host blocks this rank holds, agree on them with every other rank,
        and allocate. Block ids are global under tensor and pipeline parallelism, so every rank must
        hold the same counts: each sizes its own cache, then all take the smallest."""
        cfg = self.config
        num_blocks = cfg.num_blocks or self.profile_num_blocks(sampler)
        num_cpu_blocks = 0
        if cfg.preemption_mode == "swap":
            num_cpu_blocks = int(cfg.swap_space_gb * (1 << 30)) // self.bytes_per_block
        num_blocks = all_ranks_min(self.parallel, num_blocks)
        num_cpu_blocks = all_ranks_min(self.parallel, num_cpu_blocks)
        self.allocate_kv_cache(num_blocks, num_cpu_blocks)
        return num_blocks, num_cpu_blocks

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
        """Run the batch's swaps and this rank's share of the forward pass. Returns logits
        [num_sampling_seqs, vocab (this rank's slice under tensor parallelism)] in batch order, or
        None when no sequence samples this step or this rank is not on the last pipeline stage."""
        if batch.swap_out:
            self._swap_out(batch.swap_out)
        if batch.swap_in:
            self._swap_in(batch.swap_in)
        if not batch.seqs:
            return None
        if self.graph_runner is not None and self.graph_runner.can_run(batch):
            return self.graph_runner.run(batch)
        input_ids, positions, logits_indices, meta = self.prepare(batch)
        if self._staged is not None:
            # begin_step may refill page-locked memory the previous step's copy has yet to read:
            # FlashInfer's plan() writes one pinned buffer per wrapper, then copies it to the GPU
            # with cudaMemcpyAsync (attention/scheduler.cuh). A rank that samples syncs every step
            # anyway; a pipeline stage that does not can get ahead of its GPU.
            self._staged.synchronize()
        self.backend.begin_step(meta)
        if self._staged is not None:
            self._staged.record(torch.cuda.current_stream(self.device))
        ps = self.parallel
        if not ps.distributed:
            hidden = self.model(input_ids, positions, self.backend)
        else:
            hidden_in = None
            if not ps.is_first_stage:
                hidden_in = recv_from_prev_stage(ps, (len(positions), self.hidden_size), self.dtype, self.device)
            hidden = self.model(input_ids, positions, self.backend, hidden_in)
            if not ps.is_last_stage:
                self.flush_sends()  # at most one hidden-state send outstanding per stage
                self.track_send(isend_to_next_stage(ps, hidden))
                return None
        if not any(batch.do_sample):
            return None
        return self.model.compute_logits(hidden.index_select(0, logits_indices))

    def track_send(self, handle: tuple) -> None:
        self._sends = [h for h in self._sends if not h[0].is_completed()]
        self._sends.append(handle)

    def flush_sends(self) -> None:
        for work, _ in self._sends:
            work.wait()
        self._sends = []

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
        ps = self.parallel
        # Later pipeline stages start from hidden states instead of token ids; stages run their own
        # layers here with no communication between them (tensor-parallel ranks all-reduce as usual).
        hidden_in = None if ps.is_first_stage else torch.zeros(num_tokens, self.hidden_size, dtype=self.dtype, device=device)
        hidden = self.model(input_ids, positions, _ProfileAttention(lens), hidden_in) if ps.distributed else \
            self.model(input_ids, positions, _ProfileAttention(lens))
        logits = None
        if ps.is_last_stage:
            logits = self.model.compute_logits(hidden[:num_seqs])
            if ps.rank == ps.sampler_rank:
                full = logits if ps.tp_size == 1 else torch.zeros(
                    num_seqs, self.vocab_size, dtype=logits.dtype, device=device
                )  # stands in for the vocab-gathered logits
                sampler(full, sampler.prepare([SamplingParams(temperature=1.0, top_p=0.9)] * num_seqs))
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
