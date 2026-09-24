"""Continuous batching: which sequences run each iteration, and how many tokens each computes.

Called once per engine step, in one of two modes (config.enable_chunked_prefill):

  prefill-priority (the plan's V1, vLLM's original design)
      If nothing is swapped out, admit waiting requests whole while blocks and the token budget
      allow; any admission makes this a prefill-only iteration. Otherwise every running sequence
      decodes one token, preempting the newest sequences when blocks run out, and swapped sequences
      come back if nothing was preempted.

  chunked (decode-first)
      Running decodes get one token each, then in-progress prefills get chunks of the remaining
      budget, then swapped sequences return, then new requests are admitted, possibly as the first
      chunk of a longer prompt. Decodes never wait behind a long prefill.

A scheduled sequence computes n of its uncomputed tokens and samples only if that chunk reaches its
last token: intermediate prefill chunks fill the cache and produce nothing.

Preemption victims are the newest running sequences. Recompute victims wait in `preempted`, ahead of
every new request, in their original order; swap victims wait in `swapped`. Nothing new is admitted
while anything is swapped out, and a step that preempts admits nothing, so one step never both
swaps out and swaps in.

Under pipeline parallelism several batches are in flight at once (config.pipeline_depth). A sequence
in a submitted batch stays in flight until that batch's result comes back: it is not scheduled again
(its next token is unknown), it is never a preemption victim, and aborting it takes effect only when
the batch returns, because pipeline stages may still be writing its blocks. Each batch takes at most
its share of the running sequences, so the batches in flight stay about equal in size.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

from engine.config import EngineConfig
from engine.core.block_manager import BlockManager
from engine.core.sampler import append_token
from engine.core.sequence import RequestOutput, Sequence, SequenceStatus


@dataclass
class ScheduledBatch:
    seqs: list[Sequence] = field(default_factory=list)
    num_new_tokens: list[int] = field(default_factory=list)
    do_sample: list[bool] = field(default_factory=list)
    swap_out: list[tuple[int, int]] = field(default_factory=list)  # (gpu_block, cpu_block)
    swap_in: list[tuple[int, int]] = field(default_factory=list)  # (cpu_block, gpu_block)
    num_preempted: int = 0
    num_tokens: int = 0
    num_decode_tokens: int = 0  # single-token steps of sequences past their prompt

    def add(self, seq: Sequence, num_new_tokens: int) -> None:
        self.seqs.append(seq)
        self.num_new_tokens.append(num_new_tokens)
        self.do_sample.append(seq.num_computed_tokens + num_new_tokens == seq.num_tokens)
        self.num_tokens += num_new_tokens
        if num_new_tokens == 1 and seq.num_computed_tokens >= seq.num_prompt_tokens:
            self.num_decode_tokens += 1

    def is_empty(self) -> bool:
        return not (self.seqs or self.swap_out or self.swap_in)

    @property
    def num_prefill_tokens(self) -> int:
        return self.num_tokens - self.num_decode_tokens

    @property
    def sampling_params(self) -> list:
        return [s.sampling_params for s, sample in zip(self.seqs, self.do_sample) if sample]


class WaitingQueue:
    """New requests, in admission-policy order (smallest key first, ties by arrival).

      fcfs      arrival time
      sjf       prompt length
      priority  priority + aging_rate * arrival, where priority defaults to the prompt length.
                A request's effective priority improves by aging_rate per second waited,
                priority - aging_rate * (now - arrival); the -aging_rate * now term is shared by
                every waiting request, so ordering by the static key is exact and never re-sorts.
                With the default priority this is SJF with aging: aging_rate = 0 is pure SJF and
                a very large aging_rate approaches FCFS.
    """

    def __init__(self, policy: str, aging_rate: float) -> None:
        self.policy = policy
        self.aging_rate = aging_rate
        self._heap: list[tuple[tuple[float, ...], int, Sequence]] = []
        self._counter = itertools.count()
        self._live: dict[int, Sequence] = {}  # entries removed by abort stay in the heap until popped

    def _key(self, seq: Sequence) -> tuple[float, ...]:
        if self.policy == "fcfs":
            return (seq.arrival_time,)
        if self.policy == "sjf":
            return (seq.num_prompt_tokens, seq.arrival_time)
        return (seq.priority_key + self.aging_rate * seq.arrival_time, seq.arrival_time)

    def push(self, seq: Sequence) -> None:
        heapq.heappush(self._heap, (self._key(seq), next(self._counter), seq))
        self._live[seq.seq_id] = seq

    def peek(self) -> Sequence | None:
        while self._heap and self._live.get(self._heap[0][2].seq_id) is not self._heap[0][2]:
            heapq.heappop(self._heap)
        return self._heap[0][2] if self._heap else None

    def pop(self) -> Sequence:
        seq = self.peek()
        heapq.heappop(self._heap)
        del self._live[seq.seq_id]
        return seq

    def remove(self, seq_id: int) -> Sequence | None:
        return self._live.pop(seq_id, None)

    def __len__(self) -> int:
        return len(self._live)

    def __iter__(self) -> Iterator[Sequence]:
        return iter(list(self._live.values()))


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager, eos_token_ids: tuple[int, ...]) -> None:
        self.config = config
        self.bm = block_manager
        self.block_size = config.block_size
        self.chunked = config.enable_chunked_prefill
        self.max_num_seqs = config.max_num_seqs
        self.max_batch_tokens = config.max_batch_tokens
        self.max_model_len = config.max_model_len
        self.preemption_mode = config.preemption_mode
        self.pipeline_depth = config.pipeline_depth or 1
        self.eos_ids = frozenset(eos_token_ids)

        self.waiting = WaitingQueue(config.admission_policy, config.aging_rate)
        self.preempted: deque[Sequence] = deque()  # recompute victims: blocks freed, prefill to redo
        self.running: list[Sequence] = []  # admission order, oldest first
        self.swapped: deque[Sequence] = deque()  # swap victims: KV in host memory
        self.seqs: dict[int, Sequence] = {}  # every live sequence
        self._outputs: list[RequestOutput] = []  # produced outside update(), e.g. failures

        self.num_preemptions = 0
        self.num_swap_outs = 0
        self.num_aborted = 0
        self.num_failed = 0

    # --- requests --------------------------------------------------------------------------------

    def add(self, seq: Sequence) -> None:
        if seq.seq_id in self.seqs:
            raise ValueError(f"duplicate seq_id {seq.seq_id}")
        seq.status = SequenceStatus.WAITING
        self.seqs[seq.seq_id] = seq
        self.waiting.push(seq)

    def abort(self, seq_id: int) -> bool:
        """Drop a request in any state and free its blocks. Idempotent: unknown, finished or
        already aborted ids are ignored. A sequence in flight is dropped when its batch returns."""
        seq = self.seqs.get(seq_id)
        if seq is None or seq.abort_requested:
            return False
        if seq.in_flight:
            seq.abort_requested = True
            return True
        del self.seqs[seq_id]
        if seq.status == SequenceStatus.RUNNING:
            self.running.remove(seq)
            self.bm.free(seq)
        elif seq.cpu_block_table:
            self.swapped.remove(seq)
            self.bm.free_cpu(seq)
        elif seq.status == SequenceStatus.PREEMPTED:
            self.preempted.remove(seq)
        else:
            self.waiting.remove(seq_id)
        seq.status = SequenceStatus.ABORTED
        self.num_aborted += 1
        return True

    def has_work(self) -> bool:
        return bool(self.running or self.swapped or self.preempted or len(self.waiting))

    def mark_in_flight(self, batch: ScheduledBatch) -> None:
        for seq, n in zip(batch.seqs, batch.num_new_tokens):
            seq.in_flight_tokens = n

    def clear_in_flight(self) -> None:
        """Forget every batch in flight, after a failed step lost them, so that aborts take effect
        at once again."""
        for seq in list(self.running):
            seq.in_flight_tokens = 0
            if seq.abort_requested:
                seq.abort_requested = False
                self.abort(seq.seq_id)

    def take_outputs(self) -> list[RequestOutput]:
        outputs, self._outputs = self._outputs, []
        return outputs

    # --- scheduling ------------------------------------------------------------------------------

    def schedule(self) -> ScheduledBatch:
        batch = ScheduledBatch()
        if self.chunked:
            self._schedule_chunked(batch)
        else:
            self._schedule_prefill_priority(batch)
        assert not (batch.swap_out and batch.swap_in), "a step must not both swap out and swap in"
        return batch

    def _schedule_prefill_priority(self, batch: ScheduledBatch) -> None:
        if not self.swapped:
            self._admit(batch, self.max_batch_tokens)
            if batch.seqs:
                return  # prefill-only iteration
        self._schedule_running(batch)
        if batch.num_preempted == 0:
            self._swap_in(batch, budget=None)

    def _schedule_chunked(self, batch: ScheduledBatch) -> None:
        self._schedule_running(batch)
        if batch.num_preempted:
            return
        self._swap_in(batch, budget=self.max_batch_tokens - batch.num_tokens)
        if not self.swapped:
            self._admit(batch, self.max_batch_tokens - batch.num_tokens)

    def _schedule_running(self, batch: ScheduledBatch) -> None:
        """Give each running sequence its tokens for this step, oldest first, preempting from the
        newest end whenever blocks run out."""
        plan = self._plan_running_tokens()
        running = self.running
        i = 0
        while i < len(running):
            seq = running[i]
            n = plan[seq]
            if n == 0:  # in flight, or no share of the budget this step
                i += 1
                continue
            while not self.bm.can_append(seq, n):
                victim = next((j for j in range(len(running) - 1, i, -1) if not running[j].in_flight), None)
                if victim is not None:
                    self._preempt(running.pop(victim), batch)
                elif i < len(running) - 1:
                    # Everything newer is in flight and cannot be preempted until its batch returns;
                    # this sequence waits a step instead.
                    i += 1
                    break
                else:
                    running.pop(i)
                    if i == 0:
                        # Alone and still out of blocks: it holds the whole pool, so neither
                        # preempting nor swapping can make room. Fail it instead of hanging.
                        self._fail(seq, self._exhausted_message(seq))
                    else:
                        self._preempt(seq, batch)
                    break
            else:
                self.bm.append_slots(seq, n)
                batch.add(seq, n)
                i += 1

    def _plan_running_tokens(self) -> dict[Sequence, int]:
        """Tokens each running sequence computes this step; 0 leaves it out."""
        ready = [s for s in self.running if not s.in_flight]
        if self.pipeline_depth > 1:
            # A batch takes at most its share of the running sequences, oldest first, so a burst
            # admitted together still spreads over the pipeline instead of travelling as one batch.
            ready = ready[: -(-len(self.running) // self.pipeline_depth)]
        plan = dict.fromkeys(self.running, 0)
        if not self.chunked:  # prefill-priority: running sequences only decode
            plan.update(dict.fromkeys(ready, 1))
            return plan
        # Decode-first: every decode gets its token before any prefill chunk gets budget.
        # len(ready) <= max_num_seqs <= max_batch_tokens, so the decodes always fit.
        left = self.max_batch_tokens - sum(1 for s in ready if s.num_uncomputed_tokens == 1)
        for seq in ready:
            if seq.num_uncomputed_tokens == 1:
                plan[seq] = 1
            else:
                plan[seq] = min(seq.num_uncomputed_tokens, left)
                left -= plan[seq]
        return plan

    def _swap_in(self, batch: ScheduledBatch, budget: int | None) -> None:
        while self.swapped and len(self.running) < self.max_num_seqs:
            seq = self.swapped[0]
            n = seq.num_uncomputed_tokens if budget is None else min(seq.num_uncomputed_tokens, budget)
            if n <= 0:
                return
            if not self.bm.can_swap_in(seq, n, use_watermark=bool(self.running)):
                if not self.running:
                    self.swapped.popleft()
                    self._fail(seq, self._exhausted_message(seq))
                    continue
                return
            self.swapped.popleft()
            batch.swap_in.extend(self.bm.swap_in(seq))
            self.bm.append_slots(seq, n)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            batch.add(seq, n)
            if budget is not None:
                budget -= n

    def _admit(self, batch: ScheduledBatch, budget: int) -> None:
        """Admit waiting requests (recompute victims first) while blocks, budget, and max_num_seqs
        allow. Stops at the first request that does not fit, so nothing overtakes it."""
        while budget > 0 and len(self.running) < self.max_num_seqs:
            seq = self.preempted[0] if self.preempted else self.waiting.peek()
            if seq is None:
                return
            if self.bm.blocks_for(seq.num_tokens) > self.bm.num_blocks:
                self._pop_waiting(seq)
                self._fail(seq, self._exhausted_message(seq))
                continue
            cached = self.bm.cached_prefix(seq)
            remaining = seq.num_tokens - len(cached) * self.block_size
            n = min(remaining, budget) if self.chunked else remaining
            if n > budget or not self.bm.can_allocate(seq, n, cached, use_watermark=bool(self.running)):
                return
            self._pop_waiting(seq)
            self.bm.allocate(seq, n, cached)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            batch.add(seq, n)
            budget -= n

    def _pop_waiting(self, seq: Sequence) -> None:
        if self.preempted and self.preempted[0] is seq:
            self.preempted.popleft()
        else:
            popped = self.waiting.pop()
            assert popped is seq

    def _preempt(self, seq: Sequence, batch: ScheduledBatch) -> None:
        assert not seq.in_flight, f"seq {seq.seq_id} preempted while in flight"
        seq.num_preemptions += 1
        self.num_preemptions += 1
        batch.num_preempted += 1
        seq.status = SequenceStatus.PREEMPTED
        if self.preemption_mode == "swap" and self.bm.can_swap_out(seq):
            batch.swap_out.extend(self.bm.swap_out(seq))
            self.swapped.appendleft(seq)  # victims go newest-first, so the front stays oldest
            self.num_swap_outs += 1
        else:
            # Recompute (also the fallback when host swap space is full): drop the KV, redo the
            # prefill over prompt + generated tokens when the sequence is readmitted.
            self.bm.free(seq)
            seq.num_computed_tokens = 0
            self.preempted.appendleft(seq)

    def _exhausted_message(self, seq: Sequence) -> str:
        capacity = self.bm.num_blocks * self.block_size
        return (
            f"KV cache exhausted: request {seq.seq_id} needs room for {seq.num_tokens + 1} tokens but the "
            f"cache holds {capacity} ({self.bm.num_blocks} blocks of {self.block_size})"
        )

    def _fail(self, seq: Sequence, message: str) -> None:
        assert not seq.in_flight, f"seq {seq.seq_id} failed while in flight"
        if seq.block_table:
            self.bm.free(seq)
        if seq.cpu_block_table:
            self.bm.free_cpu(seq)
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = "error"
        seq.error = message
        self.seqs.pop(seq.seq_id, None)
        self.num_failed += 1
        self._outputs.append(
            RequestOutput(seq.seq_id, [], "", True, "error", seq.num_prompt_tokens, seq.num_output_tokens, error=message)
        )

    # --- after the forward pass ------------------------------------------------------------------

    def update(self, batch: ScheduledBatch, sampled: list[int]) -> list[RequestOutput]:
        """Advance computed-token counts, append sampled tokens, detokenize, check stop conditions,
        and free finished sequences."""
        outputs = self.take_outputs()
        now = time.monotonic()
        tokens = iter(sampled)
        any_finished = False
        for seq, n, do_sample in zip(batch.seqs, batch.num_new_tokens, batch.do_sample):
            seq.in_flight_tokens = 0
            seq.num_computed_tokens += n
            token = next(tokens) if do_sample else None  # taken even for a dropped sequence
            if seq.abort_requested:  # aborted while its batch was in flight: drop it now
                self.bm.free(seq)
                seq.status = SequenceStatus.ABORTED
                self.seqs.pop(seq.seq_id, None)
                self.num_aborted += 1
                any_finished = True
                continue
            self.bm.cache_full_blocks(seq)
            if not do_sample:
                continue
            text, finish = append_token(seq, token, self.eos_ids, self.max_model_len)
            if seq.first_token_time is None:
                seq.first_token_time = now
            if finish is not None:
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = finish
                self.bm.free(seq)
                self.seqs.pop(seq.seq_id, None)
                any_finished = True
            outputs.append(
                RequestOutput(seq.seq_id, [token], text, finish is not None, finish, seq.num_prompt_tokens, seq.num_output_tokens)
            )
        if any_finished:
            self.running = [s for s in self.running if s.status == SequenceStatus.RUNNING]
        return outputs

    # --- debugging -------------------------------------------------------------------------------

    def check_invariants(self) -> None:
        self.bm.check_invariants(self.running, self.swapped)
        for seq in self.running:
            assert seq.status == SequenceStatus.RUNNING and not seq.cpu_block_table, f"bad running seq {seq.seq_id}"
            assert seq.num_uncomputed_tokens >= max(1, seq.in_flight_tokens), f"running seq {seq.seq_id} has nothing left to compute"
            assert seq.in_flight or not seq.abort_requested, f"seq {seq.seq_id} has an abort pending but is not in flight"
        for seq in self.swapped:
            assert seq.status == SequenceStatus.PREEMPTED and seq.cpu_block_table and not seq.block_table
        for seq in self.preempted:
            assert seq.status == SequenceStatus.PREEMPTED and not seq.block_table and not seq.cpu_block_table
            assert seq.num_computed_tokens == 0
        for seq in self.waiting:
            assert seq.status == SequenceStatus.WAITING and not seq.block_table and seq.num_computed_tokens == 0
        queued = [*self.running, *self.swapped, *self.preempted, *self.waiting]
        assert len(queued) == len({s.seq_id for s in queued}) == len(self.seqs), "a sequence is lost or queued twice"
        assert len(self.running) <= self.max_num_seqs

    def describe(self) -> str:
        return (
            f"running={[(s.seq_id, s.num_computed_tokens, s.num_tokens, s.in_flight_tokens) for s in self.running]} "
            f"swapped={[s.seq_id for s in self.swapped]} preempted={[s.seq_id for s in self.preempted]} "
            f"waiting={len(self.waiting)} free_blocks={self.bm.num_free_blocks}/{self.bm.num_blocks}"
        )
