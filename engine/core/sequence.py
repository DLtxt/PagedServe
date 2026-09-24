"""Per-request state: the sequence is the engine's process control block.

State machine, the OS process-state diagram under different names:

    WAITING ──admit──▶ RUNNING ──finish──▶ FINISHED
                        │    ▲
                preempt │    │ resume
                        ▼    │
                      PREEMPTED      recompute: blocks freed, prefill redone on resume
                                     swap:      KV parked in host memory, copied back on resume

    any live state ──abort──▶ ABORTED
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class SamplingParams:
    temperature: float = 1.0  # 0 means greedy
    top_p: float = 1.0
    max_tokens: int = 16
    stop: list[str] = field(default_factory=list)
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if any(not s for s in self.stop):
            raise ValueError("stop strings must be non-empty")


@dataclass
class RequestOutput:
    """One step's worth of output for one request."""

    seq_id: int
    token_ids: list[int]  # tokens sampled this step (empty for a pure notification)
    text: str  # newly final text since the previous output
    finished: bool
    finish_reason: str | None  # "stop" | "length" | "error" | None while running
    num_prompt_tokens: int
    num_output_tokens: int
    error: str | None = None


@dataclass(eq=False)  # identity semantics: sequences live in sets and dicts
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.monotonic)
    priority: float | None = None  # priority policy key; defaults to the prompt length
    output_token_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    # Tokens whose K/V are in the cache. Below num_tokens while a prefill is in progress, which is
    # what makes chunked prefill and prefix caching possible.
    num_computed_tokens: int = 0
    first_token_time: float | None = None
    finish_reason: str | None = None
    error: str | None = None

    # Beyond the plan's fields:
    token_ids: list[int] = field(init=False)  # prompt + output: what the model consumes
    block_hashes: list[bytes] = field(default_factory=list)  # prefix-cache hash chain over full blocks
    num_registered_blocks: int = 0  # leading blocks already published to the prefix cache
    num_cached_tokens: int = 0  # tokens served by the prefix cache at the latest admission
    cpu_block_table: list[int] = field(default_factory=list)  # KV parked in host memory by swap
    num_preemptions: int = 0
    # Pipelined execution: this sequence's tokens in a submitted batch whose result has not come back
    # (0 when it is in none), and an abort that has to wait for that batch.
    in_flight_tokens: int = 0
    abort_requested: bool = False
    output_text: str = ""  # decoded output, truncated at a stop string if one matched
    num_emitted_chars: int = 0  # prefix of output_text already returned to the client
    detokenizer: Any = None

    def __post_init__(self) -> None:
        self.token_ids = list(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        return len(self.token_ids) - self.num_computed_tokens

    @property
    def in_flight(self) -> bool:
        return self.in_flight_tokens > 0

    @property
    def is_finished(self) -> bool:
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.ABORTED)

    @property
    def priority_key(self) -> float:
        return self.priority if self.priority is not None else float(len(self.prompt_token_ids))

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.output_token_ids.append(token_id)
