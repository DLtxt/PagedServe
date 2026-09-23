"""Gate 9 and the other CUDA-only checks; skipped unless CUDA and FlashInfer are available.

  - FlashInfer attention matches the naive paged path at kernel level on a mixed batch (a decode, a
    prefill chunk behind cached context, and a fresh prefill), within bf16 tolerance.
  - End to end, greedy generations agree between the naive and FlashInfer engines in bf16 (the
    near-tie rule at bf16 tolerance), with chunked prefill and prefix caching on.
  - CUDA graph replay produces the same logits as eager FlashInfer decode.
  - The profiled KV cache fits inside gpu_memory_utilization.
"""

from __future__ import annotations

import gc
import importlib.util

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(importlib.util.find_spec("flashinfer") is None, reason="needs flashinfer-python"),
]

from engine.config import EngineConfig  # noqa: E402
from engine.core.engine import LLMEngine  # noqa: E402
from engine.core.sequence import SamplingParams, Sequence  # noqa: E402
from engine.model.attention import AttentionMetadata, FlashInferAttention, KVCache, NaivePagedAttention  # noqa: E402
from engine.model.loader import load_model  # noqa: E402
from greedy_check import LOGIT_TOL, LogitsRecorder, check_greedy_match  # noqa: E402
from prompts import PROMPTS  # noqa: E402

BF16_TOL = LOGIT_TOL[torch.bfloat16]


@pytest.fixture(scope="module")
def qwen_bf16(model_path):
    return load_model(model_path, "cuda", torch.bfloat16)


@pytest.mark.parametrize("block_size", [8, 16, 32])
def test_flashinfer_kernel_matches_naive(block_size):
    """Gate 9 at kernel level: same cache contents, same new K/V, compare attention outputs."""
    torch.manual_seed(0)
    heads, kv_heads, dim, dtype = 16, 8, 128, torch.bfloat16
    naive_cache = KVCache(1, 64, block_size, kv_heads, dim, dtype, "cuda")
    naive_cache.data.normal_()
    fi_cache = KVCache(1, 64, block_size, kv_heads, dim, dtype, "cuda")
    fi_cache.data.copy_(naive_cache.data)

    # (cached tokens before this step, new tokens this step): a decode, a chunk behind 45 cached
    # tokens, and a fresh 33-token prefill.
    shapes = [(36, 1), (45, 20), (0, 33)]
    rng = np.random.default_rng(0)
    free = list(rng.permutation(64))
    tables, slots, kv_slots, kv_start, query_start = [], [], [], [0], [0]
    for cached, new in shapes:
        kv_len = cached + new
        table = np.array([free.pop() for _ in range(-(-kv_len // block_size))], dtype=np.int64)
        tables.append(table)
        pos = np.arange(cached, kv_len)
        slots.append(table[pos // block_size] * block_size + pos % block_size)
        kv_slots.append((table[:, None] * block_size + np.arange(block_size)).reshape(-1)[:kv_len])
        kv_start.append(kv_start[-1] + kv_len)
        query_start.append(query_start[-1] + new)
    t = query_start[-1]
    common = dict(query_lens=[n for _, n in shapes], kv_lens=[c + n for c, n in shapes], query_start=query_start,
                  slot_mapping=torch.from_numpy(np.concatenate(slots)).cuda())
    q = torch.randn(t, heads, dim, dtype=dtype, device="cuda")
    k = torch.randn(t, kv_heads, dim, dtype=dtype, device="cuda")
    v = torch.randn(t, kv_heads, dim, dtype=dtype, device="cuda")

    naive = NaivePagedAttention(naive_cache, dim)
    naive.begin_step(AttentionMetadata(**common, kv_slots=torch.from_numpy(np.concatenate(kv_slots)).cuda(), kv_start=kv_start))
    fi = FlashInferAttention(fi_cache, heads, kv_heads, dim, dtype)
    fi.begin_step(AttentionMetadata(**common, block_tables=tables))
    want, got = naive(0, q, k, v), fi(0, q, k, v)
    torch.cuda.synchronize()
    assert torch.equal(naive_cache.data, fi_cache.data), "the two backends wrote the cache differently"
    err = (want.float() - got.float()).abs().max().item()
    assert err < 3e-2, f"FlashInfer vs naive attention: max abs error {err:.3e}"


def _engine(model, tokenizer, **overrides) -> LLMEngine:
    options = dict(device="cuda", dtype="bfloat16", num_blocks=2048, max_model_len=2048, max_num_seqs=64,
                   debug_invariants=True)
    options.update(overrides)
    return LLMEngine(EngineConfig(**options), model=model, tokenizer=tokenizer)


def _run(engine: LLMEngine, prompts: list[list[int]], max_tokens: int, refs=None):
    recorder = LogitsRecorder(refs)
    engine.logits_hook = recorder
    seqs = [Sequence(i, p, SamplingParams(temperature=0, max_tokens=max_tokens)) for i, p in enumerate(prompts)]
    for seq in seqs:
        engine.add_request(seq)
    while engine.has_work():
        engine.step()
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks
    return seqs, recorder


def _prompts(tokenizer) -> list[list[int]]:
    shared = tokenizer(PROMPTS[15]).input_ids
    return [tokenizer(p).input_ids for p in PROMPTS[:10]] + [shared + tokenizer(PROMPTS[i][:60]).input_ids for i in range(4)]


@pytest.mark.parametrize("chunked", [False, True])
def test_flashinfer_engine_matches_naive_engine(qwen_bf16, tokenizer, chunked):
    """Gate 9 end to end."""
    options = dict(enable_prefix_caching=True, enable_chunked_prefill=chunked, max_batch_tokens=256 if chunked else 2048)
    prompts = _prompts(tokenizer)
    ref, ref_rec = _run(_engine(qwen_bf16, tokenizer, attention_backend="naive", **options), prompts, 32)
    got, rec = _run(_engine(qwen_bf16, tokenizer, attention_backend="flashinfer", **options), prompts, 32,
                    refs={s.seq_id: ref_rec.steps[s.seq_id] for s in ref})
    near_ties = [check_greedy_match(a.output_token_ids, ref_rec.steps[a.seq_id], b.output_token_ids, rec.steps[b.seq_id],
                                    BF16_TOL, f"request {a.seq_id}") for a, b in zip(ref, got)]
    print(f"\ngate 9 (chunked={chunked}): near-tie divergences at {near_ties}")


def test_cuda_graphs_match_eager(qwen_bf16, tokenizer):
    prompts = _prompts(tokenizer)
    ref, ref_rec = _run(_engine(qwen_bf16, tokenizer, attention_backend="flashinfer"), prompts, 48)
    engine = _engine(qwen_bf16, tokenizer, attention_backend="flashinfer", cuda_graphs=True)
    replays = 0
    run = engine.model_runner.graph_runner.run

    def counting_run(batch):
        nonlocal replays
        replays += 1
        return run(batch)

    engine.model_runner.graph_runner.run = counting_run
    got, rec = _run(engine, prompts, 48, refs={s.seq_id: ref_rec.steps[s.seq_id] for s in ref})
    assert replays > 0, "no decode step went through a CUDA graph"
    for a, b in zip(ref, got):
        check_greedy_match(a.output_token_ids, ref_rec.steps[a.seq_id], b.output_token_ids, rec.steps[b.seq_id],
                           BF16_TOL, f"request {a.seq_id}")


def test_profiled_kv_cache_fits_the_budget(qwen_bf16, tokenizer):
    gc.collect()  # earlier tests' engines hold KV caches until collected
    torch.cuda.empty_cache()
    engine = _engine(qwen_bf16, tokenizer, attention_backend="flashinfer", num_blocks=None, gpu_memory_utilization=0.85,
                     max_batch_tokens=4096, max_model_len=4096, enable_chunked_prefill=True)
    assert engine.block_manager.num_blocks > 0
    prompts = [tokenizer(PROMPTS[i % 20] * 8).input_ids[:2000] for i in range(24)]
    _run(engine, prompts, 16)
    free, total = torch.cuda.mem_get_info()
    assert total - free <= 0.85 * total + (512 << 20), "the engine used more memory than gpu_memory_utilization allows"
