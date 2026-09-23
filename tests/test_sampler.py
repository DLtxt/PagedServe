"""Sampler and incremental-detokenizer tests."""

from __future__ import annotations

import random

import pytest
import torch

from engine.core.sampler import IncrementalDetokenizer, Sampler, top_p_filter
from engine.core.sequence import SamplingParams


def test_greedy_rows_take_the_argmax():
    torch.manual_seed(0)
    logits = torch.randn(5, 100)
    sampler = Sampler("cpu", seed=0)
    params = [SamplingParams(temperature=0), SamplingParams(temperature=1.0), SamplingParams(temperature=0),
              SamplingParams(temperature=0.7, top_p=0.9), SamplingParams(temperature=0)]
    tokens = sampler(logits, sampler.prepare(params))
    argmax = logits.argmax(-1).tolist()
    assert [tokens[i] for i in (0, 2, 4)] == [argmax[i] for i in (0, 2, 4)]
    assert all(0 <= t < 100 for t in tokens)


def test_all_greedy_batch_skips_sampling():
    sampler = Sampler("cpu")
    tensors = sampler.prepare([SamplingParams(temperature=0)] * 3)
    assert tensors.all_greedy and tensors.temperatures is None
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    assert sampler(logits, tensors) == [1, 0, 2]


def test_temperature_sampling_matches_softmax():
    """20,000 identical rows sampled in one call: frequencies must match softmax(logits / T)."""
    sampler = Sampler("cpu", seed=1)
    row = torch.tensor([1.0, 0.0, -1.0, 0.5])
    temperature = 0.7
    n = 20_000
    tokens = sampler(row.repeat(n, 1), sampler.prepare([SamplingParams(temperature=temperature)] * n))
    freq = torch.bincount(torch.tensor(tokens), minlength=4).float() / n
    want = torch.softmax(row / temperature, -1)
    assert torch.allclose(freq, want, atol=0.015), (freq, want)


def test_top_p_keeps_the_smallest_set_reaching_p():
    probs = torch.tensor([[0.5, 0.25, 0.15, 0.1], [0.1, 0.6, 0.2, 0.1]])
    out = top_p_filter(probs, torch.tensor([0.7, 0.3]))
    # Row 0: 0.5 then 0.25 reaches 0.7; the rest go. Row 1: 0.6 alone reaches 0.3.
    assert torch.allclose(out[0], torch.tensor([0.5, 0.25, 0.0, 0.0]) / 0.75)
    assert torch.allclose(out[1], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(top_p_filter(probs, torch.tensor([1.0, 1.0])), probs)


def test_top_p_sampling_never_draws_filtered_tokens():
    sampler = Sampler("cpu", seed=2)
    row = torch.log(torch.tensor([0.5, 0.25, 0.15, 0.1]))
    tokens = sampler(row.repeat(5000, 1), sampler.prepare([SamplingParams(temperature=1.0, top_p=0.7)] * 5000))
    assert set(tokens) == {0, 1}


@pytest.mark.model
def test_detokenizer_stream_equals_full_decode(tokenizer):
    """The concatenated stream must equal a full decode exactly, including multi-byte characters split
    across tokens, and must never contain a replacement character a full decode does not."""
    rng = random.Random(0)
    text = "Weekend recap 🌄🥾 丝绸之路 日本の鉄道 😵‍💫 naïve café — “quotes” ⛈️ end"
    natural = tokenizer(text).input_ids
    specials = [151643, 151644, 151645]
    for trial in range(300):
        if trial % 3 == 0:
            ids = natural[: rng.randrange(1, len(natural) + 1)]
        elif trial % 3 == 1:
            ids = [rng.randrange(151643) for _ in range(rng.randrange(1, 40))]  # arbitrary byte-level tokens
        else:
            ids = [rng.choice(natural + specials) for _ in range(rng.randrange(1, 40))]
        detok = IncrementalDetokenizer(tokenizer)
        pieces = [detok.add(t) for t in ids]
        pieces.append(detok.flush())
        full = tokenizer.decode(ids, skip_special_tokens=True)
        assert "".join(pieces) == full, (ids, pieces, full)
        if "�" not in full:
            assert all("�" not in p for p in pieces)
