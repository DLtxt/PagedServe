"""CUDA graphs for decode steps (FlashInfer backend only).

At 0.6B a decode step is 28 layers of small kernels, and launch overhead is a real fraction of the
step. A captured graph replays the whole forward pass with one launch. Graphs need static input
buffers, so one graph is captured per bucketed batch size and a batch is padded up to its bucket:
padding rows use token 0 at position 0, write their K/V into the scratch block, and attend over one
token of it. FlashInfer's CUDA-graph decode wrapper keeps its plan in fixed buffers that plan()
rewrites before every replay; the logits are computed outside the graph.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from engine.core.scheduler import ScheduledBatch
from engine.model.attention import KVCache, paged_kv_index

logger = logging.getLogger(__name__)

_BUCKETS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]


class _GraphDecodeAttention:
    def __init__(self, cache: KVCache, wrapper, slot_mapping: torch.Tensor) -> None:
        self.cache = cache
        self.wrapper = wrapper
        self.slot_mapping = slot_mapping

    def __call__(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        k_flat, v_flat = self.cache.flat(layer_idx)
        k_flat.index_copy_(0, self.slot_mapping, k)
        v_flat.index_copy_(0, self.slot_mapping, v)
        return self.wrapper.run(q, self.cache.layer(layer_idx))


class CudaGraphRunner:
    def __init__(self, runner) -> None:
        self.runner = runner
        self.cache: KVCache = runner.kv_cache
        cfg = runner.config
        self.buckets = [b for b in _BUCKETS if b < cfg.max_num_seqs] + [cfg.max_num_seqs]
        self.max_batch = self.buckets[-1]
        self.max_pages = self.max_batch * -(-cfg.max_model_len // cfg.block_size)
        device = runner.device
        self.input_ids = torch.zeros(self.max_batch, dtype=torch.int64, device=device)
        self.positions = torch.zeros(self.max_batch, dtype=torch.int64, device=device)
        self.slot_mapping = torch.zeros(self.max_batch, dtype=torch.int64, device=device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, torch.Tensor] = {}
        self.wrappers: dict[int, object] = {}

    def _plan(self, size: int, tables: list, kv_lens: list[int]) -> None:
        backend = self.runner.backend
        indptr, indices, last = (t.pin_memory() for t in paged_kv_index(tables, kv_lens, self.cache.block_size))
        self.wrappers[size].plan(
            indptr, indices, last, backend.num_heads, backend.num_kv_heads, backend.head_dim, self.cache.block_size,
            q_data_type=backend.dtype, kv_data_type=backend.dtype,
        )

    @torch.inference_mode()
    def capture(self) -> None:
        import flashinfer

        device = self.runner.device
        model = self.runner.model
        scratch = [np.array([self.cache.scratch_block], dtype=np.int64)]
        pool = None
        self.slot_mapping.fill_(self.cache.scratch_block * self.cache.block_size)
        for size in reversed(self.buckets):  # largest first, so smaller graphs reuse its memory pool
            self.wrappers[size] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.runner.backend.workspace,
                "NHD",
                use_cuda_graph=True,
                paged_kv_indptr_buffer=torch.zeros(size + 1, dtype=torch.int32, device=device),
                paged_kv_indices_buffer=torch.zeros(max(self.max_pages, size), dtype=torch.int32, device=device),
                paged_kv_last_page_len_buffer=torch.zeros(size, dtype=torch.int32, device=device),
            )
            self._plan(size, scratch * size, [1] * size)
            attn = _GraphDecodeAttention(self.cache, self.wrappers[size], self.slot_mapping[:size])
            model(self.input_ids[:size], self.positions[:size], attn)  # warm up outside the graph
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self.outputs[size] = model(self.input_ids[:size], self.positions[:size], attn)
            pool = graph.pool()
            self.graphs[size] = graph
        torch.cuda.synchronize(device)
        logger.info("captured CUDA graphs for decode batch sizes %s", self.buckets)

    def can_run(self, batch: ScheduledBatch) -> bool:
        return (
            len(batch.seqs) <= self.max_batch
            and all(n == 1 for n in batch.num_new_tokens)
            and all(batch.do_sample)
        )

    @torch.inference_mode()
    def run(self, batch: ScheduledBatch) -> torch.Tensor:
        n = len(batch.seqs)
        size = next(b for b in self.buckets if b >= n)
        bs = self.cache.block_size
        ids = np.zeros(size, dtype=np.int64)
        positions = np.zeros(size, dtype=np.int64)
        slots = np.full(size, self.cache.scratch_block * bs, dtype=np.int64)
        tables, kv_lens = [], []
        for i, seq in enumerate(batch.seqs):
            pos = seq.num_computed_tokens
            ids[i] = seq.token_ids[pos]
            positions[i] = pos
            slots[i] = seq.block_table[pos // bs] * bs + pos % bs
            kv_lens.append(pos + 1)
            tables.append(np.asarray(seq.block_table[: -(-(pos + 1) // bs)], dtype=np.int64))
        scratch = np.array([self.cache.scratch_block], dtype=np.int64)
        tables += [scratch] * (size - n)
        kv_lens += [1] * (size - n)
        packed = torch.from_numpy(np.concatenate([ids, positions, slots])).pin_memory().to(self.runner.device, non_blocking=True)
        self.input_ids[:size].copy_(packed[:size])
        self.positions[:size].copy_(packed[size : 2 * size])
        self.slot_mapping[:size].copy_(packed[2 * size :])
        self._plan(size, tables, kv_lens)
        self.graphs[size].replay()
        return self.runner.model.compute_logits(self.outputs[size][:n])
