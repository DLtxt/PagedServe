"""The engine loop. step() is the whole engine; everything else is the four objects it calls.

step() is synchronous and never blocks on anything but the GPU: it advances every scheduled
sequence by one iteration and returns. A stall here adds latency to every request that is decoding,
which is why the HTTP layer is async and talks to the engine only through queues.
"""

from __future__ import annotations

import csv
import itertools
import logging
import time
from collections.abc import Callable

import torch

from engine.config import EngineConfig
from engine.core.block_manager import BlockManager
from engine.core.model_runner import ModelRunner
from engine.core.sampler import IncrementalDetokenizer, Sampler
from engine.core.scheduler import ScheduledBatch, Scheduler
from engine.core.sequence import RequestOutput, SamplingParams, Sequence

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, config: EngineConfig, model=None, tokenizer=None) -> None:
        """Load the model (unless one is passed in), size and allocate the KV cache, wire the
        scheduler. Tests pass a preloaded model and tokenizer to avoid reloading weights."""
        if model is None:
            from transformers import AutoTokenizer

            from engine.model.loader import load_config, load_model, resolve_model_path

            path = resolve_model_path(config.model)
            config.resolve(load_config(path).max_position_embeddings)
            model = load_model(path, config.device, config.torch_dtype)
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(path)
        else:
            config.resolve(model.config.max_position_embeddings)
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = model.config.vocab_size
        self.sampler = Sampler(config.device, config.seed)
        self.model_runner = ModelRunner(model, config)

        num_blocks = config.num_blocks or self.model_runner.profile_num_blocks(self.sampler)
        num_cpu_blocks = 0
        if config.preemption_mode == "swap":
            num_cpu_blocks = int(config.swap_space_gb * (1 << 30)) // self.model_runner.bytes_per_block
        self.model_runner.allocate_kv_cache(num_blocks, num_cpu_blocks)
        capacity = num_blocks * config.block_size
        if config.max_model_len > capacity:
            logger.warning(
                "max_model_len %d exceeds the KV cache's %d tokens: a sequence that outgrows the cache fails with an error",
                config.max_model_len, capacity,
            )
        self.block_manager = BlockManager(
            num_blocks, config.block_size, config.enable_prefix_caching, num_cpu_blocks, config.watermark
        )
        self.scheduler = Scheduler(config, self.block_manager, model.config.eos_token_ids)

        self.logits_hook: Callable[[ScheduledBatch, torch.Tensor], None] | None = None  # debugging and tests
        self.num_steps = 0
        self.num_generated_tokens = 0
        self._ids = itertools.count()
        self._step_log = None
        if config.step_log:
            log_file = open(config.step_log, "a", newline="", buffering=1)  # line-buffered
            self._step_log = csv.writer(log_file)
            if log_file.tell() == 0:
                self._step_log.writerow(_STEP_LOG_FIELDS)
        logger.info("engine ready: %d KV blocks (%d tokens), %d host blocks", num_blocks, capacity, num_cpu_blocks)

    # --- the loop --------------------------------------------------------------------------------

    def step(self) -> list[RequestOutput]:
        batch = self.scheduler.schedule()
        if batch.is_empty():
            outputs = self.scheduler.take_outputs()
            if not outputs and self.scheduler.has_work():
                raise RuntimeError(f"scheduler made no progress with work pending: {self.scheduler.describe()}")
            return outputs
        start = time.perf_counter()
        sampling = self.sampler.prepare(batch.sampling_params)
        logits = self.model_runner.execute(batch)
        if logits is not None and self.logits_hook is not None:
            self.logits_hook(batch, logits)
        tokens = self.sampler(logits, sampling) if logits is not None else []
        outputs = self.scheduler.update(batch, tokens)
        self._after_step(batch, len(tokens), time.perf_counter() - start)
        return outputs

    def validate(self, seq: Sequence) -> None:
        """Raise ValueError for a request that can never be served."""
        n, max_tokens = seq.num_prompt_tokens, seq.sampling_params.max_tokens
        if n == 0:
            raise ValueError("the prompt is empty")
        if n + max_tokens > self.config.max_model_len:
            raise ValueError(
                f"prompt ({n} tokens) plus max_tokens ({max_tokens}) exceeds max_model_len ({self.config.max_model_len})"
            )
        if min(seq.prompt_token_ids) < 0 or max(seq.prompt_token_ids) >= self.vocab_size:
            raise ValueError(f"prompt token ids must be in [0, {self.vocab_size})")

    def add_request(self, seq: Sequence) -> None:
        self.validate(seq)
        if seq.detokenizer is None and self.tokenizer is not None:
            seq.detokenizer = IncrementalDetokenizer(self.tokenizer)
        self.scheduler.add(seq)

    def abort(self, seq_id: int) -> None:
        self.scheduler.abort(seq_id)  # frees blocks; a no-op for finished or unknown ids

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def new_seq_id(self) -> int:
        return next(self._ids)

    # --- helpers ---------------------------------------------------------------------------------

    def generate(self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams]) -> list[Sequence]:
        """Run requests to completion offline (tests, benchmarks). Returns sequences in input order."""
        if isinstance(params, SamplingParams):
            params = [params] * len(prompts)
        seqs = [Sequence(self.new_seq_id(), list(p), sp) for p, sp in zip(prompts, params)]
        for seq in seqs:
            self.add_request(seq)
        while self.has_work():
            self.step()
        return seqs

    def _after_step(self, batch: ScheduledBatch, num_sampled: int, elapsed: float) -> None:
        self.num_steps += 1
        self.num_generated_tokens += num_sampled
        if self.config.debug_invariants:
            self.scheduler.check_invariants()
        if self._step_log is not None:
            s, bm = self.scheduler, self.block_manager
            self._step_log.writerow(
                [
                    f"{time.time():.6f}", self.num_steps, f"{elapsed * 1e3:.3f}", len(s.running), len(s.waiting),
                    len(s.preempted), len(s.swapped), batch.num_tokens, batch.num_prefill_tokens,
                    batch.num_decode_tokens, bm.num_used_blocks, f"{bm.num_used_blocks / bm.num_blocks:.4f}",
                    s.num_preemptions,
                ]
            )

    def stats(self) -> dict:
        s, bm = self.scheduler, self.block_manager
        stats = {
            "num_steps": self.num_steps,
            "num_generated_tokens": self.num_generated_tokens,
            "num_running": len(s.running),
            "num_waiting": len(s.waiting),
            "num_preempted_waiting": len(s.preempted),
            "num_swapped": len(s.swapped),
            "num_blocks": bm.num_blocks,
            "num_free_blocks": bm.num_free_blocks,
            "num_cached_free_blocks": len(bm.evictable),
            "kv_usage": bm.num_used_blocks / bm.num_blocks,
            "num_cpu_blocks": bm.num_cpu_blocks,
            "num_free_cpu_blocks": len(bm.cpu_free),
            "block_size": bm.block_size,
            "total_preemptions": s.num_preemptions,
            "total_swap_outs": s.num_swap_outs,
            "total_aborted": s.num_aborted,
            "total_failed": s.num_failed,
            "prefix_cache_hit_rate": bm.prefix_hit_tokens / bm.prefix_query_tokens if bm.prefix_query_tokens else 0.0,
        }
        if torch.cuda.is_available() and self.config.is_cuda:
            stats["peak_memory_bytes"] = torch.cuda.max_memory_allocated(self.model_runner.device)
        return stats


_STEP_LOG_FIELDS = [
    "time", "step", "step_ms", "running", "waiting", "preempted_waiting", "swapped", "batch_tokens",
    "prefill_tokens", "decode_tokens", "used_blocks", "kv_usage", "total_preemptions",
]
