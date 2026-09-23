"""Engine configuration: one dataclass. The server's CLI flags are generated from its fields."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field, fields

import torch

logger = logging.getLogger(__name__)


def _opt(default, help: str):
    return field(default=default, metadata={"help": help})


@dataclass
class EngineConfig:
    # Model
    model: str = _opt("Qwen/Qwen3-0.6B-Base", "HF model id or local directory")
    device: str = _opt("auto", "auto | cpu | cuda | cuda:N | mps (auto picks cuda when available)")
    dtype: str = _opt("auto", "auto | float32 | bfloat16 | float16 (auto: bfloat16 on cuda, float32 elsewhere)")
    seed: int = _opt(0, "seed for sampling")

    # KV cache: a block is a page, block_size is the page size
    block_size: int = _opt(16, "tokens per KV-cache block")
    num_blocks: int | None = _opt(None, "KV blocks to allocate; default profiles GPU memory (required off-GPU)")
    gpu_memory_utilization: float = _opt(0.90, "fraction of GPU memory the engine may use in total")
    swap_space_gb: float = _opt(4.0, "host memory reserved for swap preemption, in GiB")
    watermark: float = _opt(0.01, "fraction of blocks admission keeps free for running sequences")

    # Scheduling
    max_num_seqs: int = _opt(256, "maximum running sequences")
    max_batch_tokens: int = _opt(2048, "token budget per iteration")
    max_model_len: int | None = _opt(None, "maximum prompt + output tokens; default is the model's context length")
    enable_chunked_prefill: bool = _opt(False, "split long prefills across iterations, decodes first")
    enable_prefix_caching: bool = _opt(False, "share full KV blocks between requests with identical prefixes")
    preemption_mode: str = _opt("recompute", "recompute | swap")
    admission_policy: str = _opt("fcfs", "fcfs | sjf | priority")
    aging_rate: float = _opt(100.0, "priority policy: priority units (tokens) gained per second of waiting")

    # Execution
    attention_backend: str = _opt("auto", "auto | naive | flashinfer (auto: flashinfer on cuda, naive elsewhere)")
    cuda_graphs: bool = _opt(False, "capture decode-only batches in CUDA graphs (flashinfer backend only)")

    # Debugging
    debug_invariants: bool = _opt(False, "check block-manager invariants after every step")
    step_log: str | None = _opt(None, "append per-step scheduler stats to this CSV file")

    def __post_init__(self) -> None:
        _check(self.block_size > 0, "block_size must be positive")
        _check(self.num_blocks is None or self.num_blocks > 0, "num_blocks must be positive")
        _check(0.0 < self.gpu_memory_utilization <= 1.0, "gpu_memory_utilization must be in (0, 1]")
        _check(self.swap_space_gb >= 0, "swap_space_gb must be non-negative")
        _check(0.0 <= self.watermark < 1.0, "watermark must be in [0, 1)")
        _check(self.max_num_seqs > 0, "max_num_seqs must be positive")
        _check(self.max_batch_tokens > 0, "max_batch_tokens must be positive")
        _check(self.max_model_len is None or self.max_model_len > 0, "max_model_len must be positive")
        _check(self.preemption_mode in ("recompute", "swap"), "preemption_mode must be recompute or swap")
        _check(self.admission_policy in ("fcfs", "sjf", "priority"), "admission_policy must be fcfs, sjf or priority")
        _check(self.aging_rate >= 0, "aging_rate must be non-negative")
        _check(self.attention_backend in ("auto", "naive", "flashinfer"), "attention_backend must be auto, naive or flashinfer")
        _check(self.dtype in ("auto", "float32", "bfloat16", "float16"), "dtype must be auto, float32, bfloat16 or float16")
        _check(
            self.device in ("auto", "cpu", "cuda", "mps") or self.device.startswith("cuda:"),
            "device must be auto, cpu, cuda, cuda:N or mps",
        )

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    def resolve(self, model_max_len: int) -> EngineConfig:
        """Fill in the automatic choices once the model's context length is known, and apply
        the cross-field rules. Idempotent."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dtype == "auto":
            self.dtype = "bfloat16" if self.is_cuda else "float32"
        if self.attention_backend == "auto":
            # FlashInfer's kernels are fp16/bf16 only; fp32 (the exact-match gates) takes the naive path.
            self.attention_backend = "flashinfer" if self.is_cuda and self.dtype != "float32" else "naive"
        _check(not (self.attention_backend == "flashinfer" and not self.is_cuda), "the flashinfer backend needs CUDA")
        _check(
            not (self.attention_backend == "flashinfer" and self.dtype == "float32"),
            "the flashinfer backend needs float16 or bfloat16",
        )
        _check(
            not self.cuda_graphs or (self.is_cuda and self.attention_backend == "flashinfer"),
            "cuda_graphs needs CUDA and the flashinfer backend",
        )
        _check(self.num_blocks is not None or self.is_cuda, "num_blocks must be set when not running on CUDA")

        if self.max_model_len is None:
            self.max_model_len = model_max_len
        _check(
            self.max_model_len <= model_max_len,
            f"max_model_len {self.max_model_len} exceeds the model's context length {model_max_len}",
        )
        if self.enable_chunked_prefill:
            # Decodes spend budget in chunked mode: one token each, and every running sequence
            # must be able to decode in the same iteration.
            _check(
                self.max_batch_tokens >= self.max_num_seqs,
                "with chunked prefill, max_batch_tokens must be >= max_num_seqs",
            )
        elif self.max_batch_tokens < self.max_model_len:
            # Without chunking a prefill is all-or-nothing, so any request that can exist must fit
            # one iteration; otherwise it (or a recompute-preempted sequence) could never be admitted.
            logger.info(
                "chunked prefill is off: raising max_batch_tokens from %d to max_model_len %d",
                self.max_batch_tokens,
                self.max_model_len,
            )
            self.max_batch_tokens = self.max_model_len
        return self

    # --- CLI -------------------------------------------------------------------------------------

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        for f in fields(cls):
            flag = "--" + f.name.replace("_", "-")
            help_text = f.metadata.get("help", "")
            kind = str(f.type)
            if kind == "bool":
                parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default, help=help_text)
            elif kind.startswith("int"):
                parser.add_argument(flag, type=int, default=f.default, help=help_text)
            elif kind.startswith("float"):
                parser.add_argument(flag, type=float, default=f.default, help=help_text)
            else:
                parser.add_argument(flag, type=str, default=f.default, help=help_text)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> EngineConfig:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
