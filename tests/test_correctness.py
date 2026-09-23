"""Correctness gates 1, 2 and 9.

Gate 1: our model (contiguous KV cache, one sequence) matches transformers greedily on 20 prompts.
Gate 2: the paged path (block tables, slot mapping, naive paged attention, batched) matches the
        contiguous path.
Gate 9: the FlashInfer path matches the naive paged path within bf16 tolerance (GPU only).

All comparisons use the near-tie rule in greedy_check.py.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import pytest
import torch
import transformers

from conftest import TEST_DEVICE
from engine.config import EngineConfig
from engine.core.block_manager import BlockManager
from engine.core.model_runner import ModelRunner
from engine.core.scheduler import ScheduledBatch
from engine.core.sequence import SamplingParams, Sequence
from engine.model.attention import greedy_generate_contiguous
from greedy_check import LOGIT_TOL, TopK, at, check_greedy_match, topk
from prompts import PROMPTS

pytestmark = [pytest.mark.model, pytest.mark.slow]

MAX_NEW_TOKENS = 100
CACHE_DIR = Path(__file__).parent / ".cache"
FP32_TOL = LOGIT_TOL[torch.float32]


def _cached(name: str, key_parts: list, compute):
    """Reference outputs from transformers are expensive to recompute and never change for a given
    model snapshot, transformers version and input, so they are cached under tests/.cache/."""
    key = hashlib.sha256(json.dumps(key_parts).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{name}_{key}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    value = compute()
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(value, path)
    return value


def _hf_greedy(model_path, prompts: list[list[int]], max_new_tokens: int) -> list[dict]:
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, attn_implementation="eager").to(TEST_DEVICE)
    refs = []
    with torch.inference_mode():
        for ids in prompts:
            out = hf.generate(
                torch.tensor([ids], device=TEST_DEVICE),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=hf.generation_config.eos_token_id,
            )
            refs.append(
                {
                    "tokens": out.sequences[0, len(ids) :].tolist(),
                    "topk": [tuple(topk(step[0])) for step in out.logits],
                }
            )
    del hf
    gc.collect()
    return refs


@pytest.fixture(scope="module")
def prompt_ids(tokenizer):
    ids = [tokenizer(p).input_ids for p in PROMPTS]
    assert len(ids) >= 20, "gate 1 needs at least 20 prompts"
    assert min(map(len, ids)) >= 100, "gate 1 needs prompts of at least 100 tokens"
    return ids


@pytest.fixture(scope="module")
def hf_reference(model_path, prompt_ids):
    key = [model_path.name, transformers.__version__, prompt_ids, MAX_NEW_TOKENS]
    refs = _cached("hf_greedy", key, lambda: _hf_greedy(model_path, prompt_ids, MAX_NEW_TOKENS))
    return [{"tokens": r["tokens"], "topk": [TopK(*t) for t in r["topk"]]} for r in refs]


@pytest.fixture(scope="module")
def contiguous_results(qwen_fp32, prompt_ids, hf_reference):
    """The week-10 path: one sequence at a time, plain contiguous KV cache."""
    results = []
    for ids, ref in zip(prompt_ids, hf_reference):
        tokens, logits = greedy_generate_contiguous(qwen_fp32, ids, MAX_NEW_TOKENS, qwen_fp32.config.eos_token_ids)
        results.append(
            {
                "tokens": tokens,
                "at_hf": [at(step, r) for step, r in zip(logits, ref["topk"])],
                "topk": [topk(step) for step in logits],
            }
        )
    return results


def test_greedy_matches_transformers(prompt_ids, hf_reference, contiguous_results):
    """Gate 1."""
    near_ties = []
    for i, (ref, ours) in enumerate(zip(hf_reference, contiguous_results)):
        step = check_greedy_match(ref["tokens"], ref["topk"], ours["tokens"], ours["at_hf"], FP32_TOL, f"prompt {i}")
        if step is not None:
            near_ties.append((i, step))
    print(f"\ngate 1: {len(prompt_ids) - len(near_ties)}/{len(prompt_ids)} prompts identical; near-tie divergences (prompt, step): {near_ties}")


def test_long_context_matches_transformers(qwen_fp32, model_path, prompt_ids):
    """Position bugs hide at short context. One ~2,500-token prompt, 16 greedy tokens."""
    long_ids = [t for ids in prompt_ids for t in ids]
    assert len(long_ids) > 2000
    key = [model_path.name, transformers.__version__, long_ids, 16]
    ref = _cached("hf_greedy_long", key, lambda: _hf_greedy(model_path, [long_ids], 16))[0]
    ref_topk = [TopK(*t) for t in ref["topk"]]
    tokens, logits = greedy_generate_contiguous(qwen_fp32, long_ids, 16, qwen_fp32.config.eos_token_ids)
    at_ref = [at(step, r) for step, r in zip(logits, ref_topk)]
    check_greedy_match(ref["tokens"], ref_topk, tokens, at_ref, FP32_TOL, "long prompt")


def _paged_greedy(model, prompts: list[list[int]], block_size: int, max_new_tokens: int, refs: list[dict]) -> list[dict]:
    """A static batch through the paging layer alone: block manager, slot mapping, flat batching,
    naive paged attention. No scheduler, so a failure here points at paging."""
    eos = set(model.config.eos_token_ids)
    num_blocks = sum(-(-(len(p) + max_new_tokens) // block_size) for p in prompts)
    config = EngineConfig(device=TEST_DEVICE, dtype="float32", block_size=block_size, num_blocks=num_blocks,
                          attention_backend="naive", max_model_len=4096)
    config.resolve(model.config.max_position_embeddings)
    runner = ModelRunner(model, config)
    runner.allocate_kv_cache(num_blocks, 0)
    bm = BlockManager(num_blocks, block_size)
    seqs = [Sequence(i, list(p), SamplingParams(temperature=0, max_tokens=max_new_tokens)) for i, p in enumerate(prompts)]
    results = [{"tokens": [], "at_ref": []} for _ in prompts]
    batch = ScheduledBatch()
    for seq in seqs:  # one prefill iteration for every prompt at once
        bm.allocate(seq, seq.num_tokens, [])
        batch.add(seq, seq.num_tokens)
    while batch.seqs:
        logits = runner.execute(batch)
        live = []
        for row, (seq, n) in enumerate(zip(batch.seqs, batch.num_new_tokens)):
            seq.num_computed_tokens += n
            step = seq.num_output_tokens
            ref_topk = refs[seq.seq_id]["topk"]
            if step < len(ref_topk):
                results[seq.seq_id]["at_ref"].append(at(logits[row], ref_topk[step]))
            token = int(logits[row].argmax())
            seq.append_token(token)
            results[seq.seq_id]["tokens"].append(token)
            if token in eos or seq.num_output_tokens >= max_new_tokens:
                bm.free(seq)
            else:
                live.append(seq)
        bm.check_invariants(live)
        batch = ScheduledBatch()
        for seq in live:
            bm.append_slots(seq, 1)
            batch.add(seq, 1)
    bm.check_invariants([])
    assert bm.num_free_blocks == num_blocks
    return results


@pytest.mark.parametrize("block_size", [8, 16, 32])
def test_paged_matches_contiguous(qwen_fp32, prompt_ids, contiguous_results, block_size):
    """Gate 2."""
    paged = _paged_greedy(qwen_fp32, prompt_ids, block_size, MAX_NEW_TOKENS, contiguous_results)
    near_ties = []
    for i, (ref, ours) in enumerate(zip(contiguous_results, paged)):
        step = check_greedy_match(ref["tokens"], ref["topk"], ours["tokens"], ours["at_ref"], FP32_TOL, f"prompt {i}, block_size {block_size}")
        if step is not None:
            near_ties.append((i, step))
    print(f"\ngate 2 (block_size {block_size}): {len(paged) - len(near_ties)}/{len(paged)} identical; near-ties: {near_ties}")
