"""Gates 3, 4 and 6 on the real model (fp32, CPU by default).

Gate 3: the block invariant holds after every step (debug_invariants=True asserts it inside step()).
Gate 4: 50 concurrent requests match their solo outputs, with zero leaked blocks, under a cache small
        enough to force repeated preemption.
Gate 6: a lone sequence that outgrows the whole cache fails cleanly instead of hanging.
"""

from __future__ import annotations

import random

import pytest

from conftest import TEST_DEVICE
from engine.config import EngineConfig
from engine.core.engine import LLMEngine
from engine.core.sequence import SamplingParams, Sequence, SequenceStatus
from engine.model.attention import greedy_generate_contiguous
from greedy_check import LOGIT_TOL, LogitsRecorder, check_greedy_match, topk
from prompts import PROMPTS

import torch

pytestmark = [pytest.mark.model, pytest.mark.slow]

FP32_TOL = LOGIT_TOL[torch.float32]


def make_engine(model, tokenizer, **overrides) -> LLMEngine:
    options = dict(
        device=TEST_DEVICE, dtype="float32", block_size=16, num_blocks=256, max_model_len=1024,
        max_num_seqs=64, debug_invariants=True, swap_space_gb=0.25,
    )
    options.update(overrides)
    return LLMEngine(EngineConfig(**options), model=model, tokenizer=tokenizer)


def assert_no_leaks(engine: LLMEngine) -> None:
    bm = engine.block_manager
    engine.scheduler.check_invariants()
    assert not engine.has_work()
    assert bm.num_free_blocks == bm.num_blocks, f"{bm.num_used_blocks} blocks leaked"
    assert len(bm.cpu_free) == bm.num_cpu_blocks, "host blocks leaked"


@pytest.fixture(scope="module")
def fifty_requests(qwen_fp32, tokenizer):
    """50 prompts of 16-90 tokens with 4-24 new tokens each, plus their solo (contiguous) outputs."""
    rng = random.Random(0)
    eos = qwen_fp32.config.eos_token_ids
    requests = []
    for i in range(50):
        prompt = tokenizer(PROMPTS[i % len(PROMPTS)]).input_ids[: rng.randrange(16, 90)]
        max_tokens = rng.randrange(4, 24)
        tokens, logits = greedy_generate_contiguous(qwen_fp32, prompt, max_tokens, eos)
        requests.append({"prompt": prompt, "max_tokens": max_tokens, "tokens": tokens, "topk": [topk(l) for l in logits]})
    return requests


@pytest.mark.parametrize("preemption", ["recompute", "swap"])
def test_50_concurrent_requests_match_solo(qwen_fp32, tokenizer, fifty_requests, preemption):
    """Gate 4 (and gate 3: invariants are checked after every step)."""
    # 96 blocks = 1,536 tokens for a working set of ~3,300: sequences are preempted over and over.
    engine = make_engine(qwen_fp32, tokenizer, num_blocks=96, preemption_mode=preemption)
    recorder = LogitsRecorder({i: r["topk"] for i, r in enumerate(fifty_requests)})
    engine.logits_hook = recorder
    seqs = [
        Sequence(i, r["prompt"], SamplingParams(temperature=0, max_tokens=r["max_tokens"]))
        for i, r in enumerate(fifty_requests)
    ]
    for seq in seqs:
        engine.add_request(seq)
    while engine.has_work():
        engine.step()

    near_ties = []
    for i, (seq, ref) in enumerate(zip(seqs, fifty_requests)):
        assert seq.status == SequenceStatus.FINISHED and seq.finish_reason in ("stop", "length")
        step = check_greedy_match(ref["tokens"], ref["topk"], seq.output_token_ids, recorder.steps[i], FP32_TOL, f"request {i}")
        if step is not None:
            near_ties.append((i, step))
        assert seq.output_text == tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
    assert engine.scheduler.num_preemptions > 0, "the cache was large enough that nothing was preempted"
    if preemption == "swap":
        assert engine.scheduler.num_swap_outs > 0
    assert_no_leaks(engine)
    print(f"\ngate 4 ({preemption}): 50/50 correct, {len(near_ties)} near-tie divergences {near_ties}, "
          f"{engine.scheduler.num_preemptions} preemptions, {engine.num_steps} steps")


def test_invariant_every_step_mixed_workload(qwen_fp32, tokenizer, fifty_requests):
    """Gate 3 across every path that frees blocks: max_tokens, stop strings, aborts, preemption,
    prefix sharing, and chunked prefill."""
    engine = make_engine(
        qwen_fp32, tokenizer, num_blocks=48, enable_prefix_caching=True, enable_chunked_prefill=True,
        max_batch_tokens=64, max_num_seqs=16,
    )
    requests = fifty_requests[:16]
    seqs = []
    for i, r in enumerate(requests):
        stop = []
        solo_text = tokenizer.decode(r["tokens"], skip_special_tokens=True)
        if i % 4 == 0 and len(solo_text) > 12:
            stop = [solo_text[6:10]]  # a string the model is known to produce: the stop path runs
        prompt = requests[0]["prompt"][:32] + r["prompt"] if i % 3 == 0 else r["prompt"]  # shared prefixes
        seqs.append(Sequence(i, prompt, SamplingParams(temperature=0, max_tokens=r["max_tokens"], stop=stop)))
    for seq in seqs:
        engine.add_request(seq)
    steps = 0
    while engine.has_work():
        engine.step()
        steps += 1
        if steps == 6:
            engine.abort(seqs[5].seq_id)
            engine.abort(seqs[7].seq_id)
    assert seqs[5].status == SequenceStatus.ABORTED or seqs[5].is_finished
    assert any(s.sampling_params.stop and s.finish_reason == "stop" for s in seqs), "no request stopped on a stop string"
    assert engine.block_manager.prefix_hit_tokens > 0
    assert_no_leaks(engine)


def test_single_sequence_oom_fails_cleanly(qwen_fp32, tokenizer):
    """Gate 6: 4 blocks of 16 tokens; a 40-token prompt can grow to 64 tokens and then nothing is
    left to preempt. It must fail with a clear error, not hang."""
    engine = make_engine(qwen_fp32, tokenizer, num_blocks=4, max_model_len=1024)
    prompt = tokenizer(PROMPTS[0]).input_ids[:40]
    seq = Sequence(0, prompt, SamplingParams(temperature=0, max_tokens=500, ignore_eos=True))
    engine.add_request(seq)
    outputs = []
    for _ in range(64):
        if not engine.has_work():
            break
        outputs.extend(engine.step())
    assert not engine.has_work(), "engine hung instead of failing the request"
    assert seq.finish_reason == "error" and "KV cache exhausted" in seq.error
    assert outputs[-1].finished and outputs[-1].error == seq.error
    # The cache holds K/V for 64 tokens. The token sampled from position 63's logits is the 65th and
    # needs no slot until it is fed back in, so the request yields 64 - 40 + 1 tokens, then fails.
    assert seq.num_output_tokens == 64 - 40 + 1, "the request should run until the cache is actually full"
    assert_no_leaks(engine)
