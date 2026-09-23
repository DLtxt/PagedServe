"""Executors: how a scheduled batch becomes sampled tokens.

LocalExecutor is the single-process path: forward, sample, done. DistributedExecutor runs on the
driver (rank 0) when the model is split across processes:

  1. encode the batch as a compact plan and broadcast it to every rank (control plane, gloo);
  2. every rank runs its share of the forward pass through the unchanged ModelRunner: pipeline stages
     pass hidden states down the line, tensor-parallel ranks all-reduce inside each layer;
  3. the last stage gathers the vocab-split logits onto its tp rank 0, which samples;
  4. the sampled tokens come back to the driver, which runs the scheduler's update as usual.

Workers (every other rank) sit in worker_loop, turning plans back into lightweight batch views that
look to the ModelRunner exactly like a ScheduledBatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from engine.core.sampler import Sampler
from engine.core.sequence import SamplingParams
from engine.distributed.parallel import (
    ParallelState,
    broadcast_ints,
    recv_ints_on_driver,
    send_ints_to_driver,
    tp_gather_last_dim,
)

logger = logging.getLogger(__name__)

STEP, SHUTDOWN = 1, 2
_HEADER = 8  # cmd, num_seqs, num_new_tokens, num_pages, num_swap_out, num_swap_in, num_sampling, debug_topk


@dataclass
class StepResult:
    tokens: list[int]
    logits: torch.Tensor | None = None  # full logits, when this process has them (for debugging hooks)
    topk: tuple[torch.Tensor, torch.Tensor] | None = None  # (ids, values) per sampling row, when requested


class LocalExecutor:
    def __init__(self, runner, sampler: Sampler) -> None:
        self.runner = runner
        self.sampler = sampler

    def execute(self, batch) -> StepResult:
        sampling = self.sampler.prepare(batch.sampling_params)
        logits = self.runner.execute(batch)
        tokens = self.sampler(logits, sampling) if logits is not None else []
        return StepResult(tokens, logits=logits)

    def shutdown(self) -> None:
        pass


class DistributedExecutor:
    def __init__(self, parallel: ParallelState, runner, sampler: Sampler, block_size: int) -> None:
        self.parallel = parallel
        self.runner = runner
        self.sampler = sampler
        self.block_size = block_size
        self.debug_topk = 0  # > 0: the sampling rank also returns each row's top-k logits
        self._closed = False

    def execute(self, batch) -> StepResult:
        broadcast_ints(self.parallel, encode_plan(batch, self.block_size, self.debug_topk))
        return run_step(self.parallel, self.runner, self.sampler, batch, batch.sampling_params, self.debug_topk)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            broadcast_ints(self.parallel, torch.tensor([SHUTDOWN] + [0] * (_HEADER - 1), dtype=torch.int64))
        except Exception as exc:  # workers already gone (e.g. torchrun tearing everything down)
            logger.warning("could not signal workers to stop: %r", exc)


def run_step(ps: ParallelState, runner, sampler: Sampler, batch, sampling_params: list, debug_topk: int) -> StepResult | None:
    """One rank's part of one step. Returns the result on the driver, None elsewhere."""
    logits = runner.execute(batch)  # this rank's stage; forwards hidden states unless it is the last
    n = len(sampling_params)
    result = None
    if ps.is_last_stage and n:
        full = tp_gather_last_dim(ps, logits)  # [n, vocab] on tp rank 0 of the last stage
        if ps.tp_rank == 0:
            tokens = sampler(full, sampler.prepare(sampling_params))
            topk = full.float().topk(debug_topk) if debug_topk else None
            result = StepResult(tokens, logits=full if ps.is_driver else None,
                                topk=(topk.indices.cpu(), topk.values.cpu()) if topk is not None else None)
    if n and ps.sampler_rank != 0:  # the sampling rank is on another process: ship the result back
        if ps.rank == ps.sampler_rank:
            send_ints_to_driver(ps, _pack_result(result, debug_topk))
        elif ps.is_driver:
            result = _unpack_result(recv_ints_on_driver(ps, n * (1 + 2 * debug_topk), ps.sampler_rank), n, debug_topk)
    if ps.is_driver and result is None:
        result = StepResult([])
    return result if ps.is_driver else None


def worker_loop(ps: ParallelState, runner, sampler: Sampler, block_size: int) -> None:
    """Every non-driver rank: execute plans until the driver says stop."""
    while True:
        plan = broadcast_ints(ps, None)
        if int(plan[0]) == SHUTDOWN:
            return
        batch, sampling_params, debug_topk = decode_plan(plan)
        run_step(ps, runner, sampler, batch, sampling_params, debug_topk)


# --- the plan: a scheduled batch as one int64 tensor ------------------------------------------------


def encode_plan(batch, block_size: int, debug_topk: int) -> torch.Tensor:
    """Everything a rank needs to run its share of the step, and nothing else: for each sequence its
    computed-token count, new-token count, whether it samples, and the block table pages its K/V
    occupy; the new tokens themselves; swap copies; and the sampling rows' parameters."""
    seqs = batch.seqs
    starts = np.array([s.num_computed_tokens for s in seqs], dtype=np.int64)
    counts = np.array(batch.num_new_tokens, dtype=np.int64)
    samples = np.array(batch.do_sample, dtype=np.int64)
    pages = -(-(starts + counts) // block_size)
    tokens = [s.token_ids[int(a) : int(a + n)] for s, a, n in zip(seqs, starts, counts)]
    tables = [s.block_table[: int(p)] for s, p in zip(seqs, pages)]
    params = [(p.temperature, p.top_p) for p in batch.sampling_params]
    header = np.array(
        [STEP, len(seqs), int(counts.sum()), int(pages.sum()), len(batch.swap_out), len(batch.swap_in), len(params), debug_topk],
        dtype=np.int64,
    )
    parts = [
        header, starts, counts, samples, pages,
        np.fromiter((t for ts in tokens for t in ts), dtype=np.int64, count=int(counts.sum())),
        np.fromiter((b for bt in tables for b in bt), dtype=np.int64, count=int(pages.sum())),
        np.asarray(batch.swap_out, dtype=np.int64).reshape(-1),
        np.asarray(batch.swap_in, dtype=np.int64).reshape(-1),
        np.asarray(params, dtype=np.float64).reshape(-1).view(np.int64),
    ]
    return torch.from_numpy(np.concatenate(parts))


class _TokenWindow:
    """The tokens a step computes for one sequence, addressable by absolute position the way the
    ModelRunner slices Sequence.token_ids."""

    __slots__ = ("start", "tokens")

    def __init__(self, start: int, tokens: list[int]) -> None:
        self.start = start
        self.tokens = tokens

    def __getitem__(self, key: slice) -> list[int]:
        return self.tokens[key.start - self.start : key.stop - self.start]


@dataclass
class _SeqView:
    num_computed_tokens: int
    token_ids: _TokenWindow
    block_table: list[int]


@dataclass
class _BatchView:
    seqs: list[_SeqView]
    num_new_tokens: list[int]
    do_sample: list[bool]
    swap_out: list[tuple[int, int]]
    swap_in: list[tuple[int, int]]


def decode_plan(plan: torch.Tensor) -> tuple[_BatchView, list[SamplingParams], int]:
    a = plan.numpy()
    _, b, total_new, total_pages, n_out, n_in, n_sample, debug_topk = (int(x) for x in a[:_HEADER])
    pos = _HEADER

    def take(n: int) -> np.ndarray:
        nonlocal pos
        pos += n
        return a[pos - n : pos]

    starts, counts, samples, pages = take(b), take(b), take(b), take(b)
    tokens, tables = take(total_new).tolist(), take(total_pages).tolist()
    swap_out = [tuple(p) for p in take(2 * n_out).reshape(-1, 2).tolist()]
    swap_in = [tuple(p) for p in take(2 * n_in).reshape(-1, 2).tolist()]
    params = take(2 * n_sample).view(np.float64).reshape(-1, 2)
    seqs, t_off, p_off = [], 0, 0
    for start, n, p in zip(starts.tolist(), counts.tolist(), pages.tolist()):
        seqs.append(_SeqView(start, _TokenWindow(start, tokens[t_off : t_off + n]), tables[p_off : p_off + p]))
        t_off, p_off = t_off + n, p_off + p
    batch = _BatchView(seqs, counts.tolist(), [bool(x) for x in samples], swap_out, swap_in)
    sampling = [SamplingParams(temperature=float(t), top_p=float(p)) for t, p in params]
    return batch, sampling, debug_topk


def _pack_result(result: StepResult, k: int) -> torch.Tensor:
    parts = [torch.tensor(result.tokens, dtype=torch.int64)]
    if k:
        ids, values = result.topk
        parts += [ids.reshape(-1).to(torch.int64), values.reshape(-1).to(torch.float64).view(torch.int64)]
    return torch.cat(parts)


def _unpack_result(payload: torch.Tensor, n: int, k: int) -> StepResult:
    tokens = payload[:n].tolist()
    if not k:
        return StepResult(tokens)
    ids = payload[n : n + n * k].reshape(n, k)
    values = payload[n + n * k :].view(torch.float64).reshape(n, k).to(torch.float32)
    return StepResult(tokens, topk=(ids, values))
