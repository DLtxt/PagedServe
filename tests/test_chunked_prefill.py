"""Gate 8: chunked and unchunked prefill produce the same output. Queries of one chunk attend to the
KV of every earlier chunk through the bottom-right causal mask; getting the offset wrong changes the
logits, which is what this compares."""

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


def run(model, tokenizer, prompts: list[list[int]], max_tokens: list[int], refs=None, **overrides):
    options = dict(device=TEST_DEVICE, dtype="float32", block_size=16, num_blocks=256, max_model_len=1024,
                   max_num_seqs=16, debug_invariants=True)
    options.update(overrides)
    engine = LLMEngine(EngineConfig(**options), model=model, tokenizer=tokenizer)
    recorder = LogitsRecorder(refs)
    engine.logits_hook = recorder
    seqs = [Sequence(i, p, SamplingParams(temperature=0, max_tokens=m)) for i, (p, m) in enumerate(zip(prompts, max_tokens))]
    partial_chunks = 0  # prefill chunks that did not reach the end of their prompt
    schedule = engine.scheduler.schedule

    def counting_schedule():
        nonlocal partial_chunks
        batch = schedule()
        partial_chunks += sum(1 for sample in batch.do_sample if not sample)
        return batch

    engine.scheduler.schedule = counting_schedule
    for seq in seqs:
        engine.add_request(seq)
    while engine.has_work():
        engine.step()
    bm = engine.block_manager
    engine.scheduler.check_invariants()
    assert bm.num_free_blocks == bm.num_blocks
    return seqs, recorder, partial_chunks


@pytest.mark.parametrize("budget", [48, 100])  # 100 is not a multiple of the block size: chunks end mid-block
def test_chunked_matches_unchunked(qwen_fp32, tokenizer, budget):
    long_prompt = tokenizer(PROMPTS[0] + PROMPTS[5] + PROMPTS[13] + PROMPTS[19]).input_ids  # ~490 tokens
    short = [tokenizer(PROMPTS[i]).input_ids[:24] for i in (1, 2, 3)]
    prompts = [short[0], long_prompt, short[1], short[2]]
    max_tokens = [24, 8, 20, 16]
    ref, ref_rec, _ = run(qwen_fp32, tokenizer, prompts, max_tokens)
    got, rec, partial = run(qwen_fp32, tokenizer, prompts, max_tokens, refs={s.seq_id: ref_rec.steps[s.seq_id] for s in ref},
                           enable_chunked_prefill=True, max_batch_tokens=budget)
    for a, b in zip(ref, got):
        check_greedy_match(a.output_token_ids, ref_rec.steps[a.seq_id], b.output_token_ids, rec.steps[b.seq_id],
                           FP32_TOL, f"budget {budget}: request {a.seq_id}")
        assert b.num_output_tokens == a.num_output_tokens  # intermediate chunks never sampled
    assert partial >= len(long_prompt) // budget - 1, f"the long prompt was split into only {partial + 1} chunks"
