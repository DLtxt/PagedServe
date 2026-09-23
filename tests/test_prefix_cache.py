"""Gate 7: prefix caching changes nothing but speed. Shared-prefix requests produce the same output
with the cache on and off, including the case that breaks a hash without the parent chain: the same
block of tokens under two different prefixes."""

from __future__ import annotations

import pytest
import torch

from conftest import TEST_DEVICE
from engine.config import EngineConfig
from engine.core.engine import LLMEngine
from engine.core.sequence import SamplingParams, Sequence
from greedy_check import LOGIT_TOL, LogitsRecorder, check_greedy_match
from prompts import PROMPTS

pytestmark = [pytest.mark.model, pytest.mark.slow]

FP32_TOL = LOGIT_TOL[torch.float32]
BLOCK = 16


def run_waves(model, tokenizer, waves: list[list[list[int]]], prefix_caching: bool, refs=None, **overrides):
    """Run request waves back to back (a wave starts once the previous one finished, so its blocks
    are in the cache). Returns the sequences, the logits recorder, and the engine."""
    options = dict(device=TEST_DEVICE, dtype="float32", block_size=BLOCK, num_blocks=256, max_model_len=1024,
                   debug_invariants=True, enable_prefix_caching=prefix_caching)
    options.update(overrides)
    engine = LLMEngine(EngineConfig(**options), model=model, tokenizer=tokenizer)
    recorder = LogitsRecorder(refs)
    engine.logits_hook = recorder
    seqs = []
    for wave in waves:
        batch = [Sequence(len(seqs) + i, p, SamplingParams(temperature=0, max_tokens=12)) for i, p in enumerate(wave)]
        seqs.extend(batch)
        for seq in batch:
            engine.add_request(seq)
        while engine.has_work():
            engine.step()
    bm = engine.block_manager
    engine.scheduler.check_invariants()
    assert bm.num_free_blocks == bm.num_blocks
    return seqs, recorder, engine


def assert_same_outputs(ref_seqs, ref_rec, seqs, rec, label: str) -> None:
    for a, b in zip(ref_seqs, seqs):
        check_greedy_match(a.output_token_ids, ref_rec.steps[a.seq_id], b.output_token_ids, rec.steps[b.seq_id],
                           FP32_TOL, f"{label}: request {a.seq_id}")


def test_shared_system_prompt_same_output_cache_on_and_off(qwen_fp32, tokenizer):
    system = tokenizer(PROMPTS[15] + PROMPTS[16]).input_ids  # ~230 tokens, not block-aligned
    suffixes = [tokenizer(" Question %d: %s" % (i, PROMPTS[i][:80])).input_ids for i in range(6)]
    waves = [[system + suffixes[0]], [system + s for s in suffixes[1:]]]
    off, off_rec, _ = run_waves(qwen_fp32, tokenizer, waves, prefix_caching=False)
    on, on_rec, engine = run_waves(qwen_fp32, tokenizer, waves, prefix_caching=True,
                                   refs={s.seq_id: off_rec.steps[s.seq_id] for s in off})
    assert_same_outputs(off, off_rec, on, on_rec, "shared system prompt")
    full_blocks = len(system) // BLOCK
    for seq in on[1:]:  # the second wave reused every full block of the shared prompt
        assert seq.num_cached_tokens >= full_blocks * BLOCK, (seq.seq_id, seq.num_cached_tokens)
    assert engine.block_manager.prefix_hit_tokens >= 5 * full_blocks * BLOCK


def test_same_block_under_different_prefixes_is_not_shared(qwen_fp32, tokenizer):
    """The case the parent hash exists for. B = X + M + tail. B's first block X is also C's first
    block: same tokens at the same position, genuinely identical K/V, a correct hit. B's second block
    has the same tokens as A's second block M, but A's M sits after a different first block Z, so its
    K/V were computed under different context. With the parent chain B's M hashes differently and is
    recomputed; without it, B would silently reuse A's K/V and produce plausible but wrong output.
    (Lookup stops at the first miss, so the colliding block must come right after a genuine hit.)"""
    x = tokenizer(PROMPTS[0]).input_ids[:BLOCK]
    z = tokenizer(PROMPTS[10]).input_ids[:BLOCK]
    m = tokenizer(PROMPTS[2]).input_ids[:BLOCK]
    other = tokenizer(PROMPTS[5]).input_ids[:BLOCK]
    tail = tokenizer(" and so").input_ids
    waves = [[z + m + tail, x + other + tail], [x + m + tail]]
    off, off_rec, _ = run_waves(qwen_fp32, tokenizer, waves, prefix_caching=False)
    on, on_rec, engine = run_waves(qwen_fp32, tokenizer, waves, prefix_caching=True,
                                   refs={s.seq_id: off_rec.steps[s.seq_id] for s in off})
    assert on[2].num_cached_tokens == BLOCK, "B must reuse C's X block and nothing else"
    assert_same_outputs(off, off_rec, on, on_rec, "same block, different parent")


def test_prefix_hits_survive_eviction_pressure(qwen_fp32, tokenizer):
    """A cache too small to keep every prefix: LRU eviction runs, outputs stay correct."""
    heads = [tokenizer(PROMPTS[i]).input_ids[:64] for i in range(6)]
    waves = [[h + tokenizer(" first").input_ids for h in heads], [h + tokenizer(" second").input_ids for h in heads]]
    off, off_rec, _ = run_waves(qwen_fp32, tokenizer, waves, prefix_caching=False, num_blocks=40)
    on, on_rec, engine = run_waves(qwen_fp32, tokenizer, waves, prefix_caching=True, num_blocks=40,
                                   refs={s.seq_id: off_rec.steps[s.seq_id] for s in off})
    assert_same_outputs(off, off_rec, on, on_rec, "eviction pressure")
    assert engine.block_manager.prefix_hit_tokens > 0
