"""Batched sampling, incremental detokenization, and stop conditions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from engine.core.sequence import SamplingParams, Sequence


@dataclass
class SamplingTensors:
    temperatures: torch.Tensor | None  # [N]; 1.0 in greedy rows
    top_ps: torch.Tensor | None  # [N], only when some row uses top-p
    greedy: torch.Tensor | None  # [N] bool
    all_greedy: bool


class Sampler:
    """Samples every row of a batch in one pass. Greedy rows (temperature 0) take the argmax; the
    others are temperature-scaled, top-p filtered, and drawn with a single torch.multinomial. The
    closing .tolist() is the only GPU sync in a step."""

    def __init__(self, device: torch.device | str, seed: int = 0) -> None:
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def prepare(self, params: list[SamplingParams]) -> SamplingTensors:
        """Build per-row parameter tensors on the CPU and start their copy to the device, ahead of
        the forward pass, so sampling never waits on a host-to-device transfer."""
        if all(p.temperature == 0 for p in params):
            return SamplingTensors(None, None, None, all_greedy=True)
        pin = self.device.type == "cuda"
        temps = torch.tensor([p.temperature if p.temperature > 0 else 1.0 for p in params], dtype=torch.float32)
        greedy = torch.tensor([p.temperature == 0 for p in params], dtype=torch.bool)
        top_ps = None
        if any(p.top_p < 1.0 for p in params):
            top_ps = torch.tensor([p.top_p for p in params], dtype=torch.float32)
        move = lambda t: None if t is None else (t.pin_memory() if pin else t).to(self.device, non_blocking=True)  # noqa: E731
        return SamplingTensors(move(temps), move(top_ps), move(greedy), all_greedy=False)

    def __call__(self, logits: torch.Tensor, tensors: SamplingTensors) -> list[int]:
        logits = logits.float()
        if tensors.all_greedy:
            return logits.argmax(dim=-1).tolist()
        probs = torch.softmax(logits / tensors.temperatures.unsqueeze(1), dim=-1)
        if tensors.top_ps is not None:
            probs = top_p_filter(probs, tensors.top_ps)
        sampled = torch.multinomial(probs, 1, generator=self.generator).squeeze(1)
        return torch.where(tensors.greedy, logits.argmax(dim=-1), sampled).tolist()


def top_p_filter(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Keep the smallest set of highest-probability tokens whose mass reaches top_p (per row), then
    renormalize. The most likely token always survives."""
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    sorted_probs = sorted_probs.masked_fill(mass_before > top_p.unsqueeze(1), 0.0)
    filtered = torch.zeros_like(probs).scatter_(-1, order, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True)


class IncrementalDetokenizer:
    """Streams text for a growing token list without re-decoding all of it every step.

    Byte-level BPE tokens do not align with characters (a multi-byte UTF-8 character often spans
    two tokens), so decoding tokens one at a time and concatenating emits U+FFFD mid-stream. Instead
    keep a small window: decode tokens[prefix:read] and tokens[prefix:], and emit the difference once
    it no longer ends in an incomplete character. Work per token is O(window), not O(length).
    tests/test_sampler.py checks the concatenated stream against a full decode.

    Output tokens are decoded without prompt context. That is exact for byte-level BPE, where a
    token's bytes do not depend on its neighbours; SentencePiece-style tokenizers would need the
    last few prompt tokens as a seed.
    """

    def __init__(self, tokenizer, skip_special_tokens: bool = True) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=self.skip_special_tokens)

    def add(self, token_id: int) -> str:
        self.ids.append(token_id)
        prefix_text = self._decode(self.ids[self.prefix_offset : self.read_offset])
        new_text = self._decode(self.ids[self.prefix_offset :])
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
            return new_text[len(prefix_text) :]
        return ""

    def flush(self) -> str:
        """Whatever is still held back when the sequence ends. A trailing incomplete character comes
        out as U+FFFD, exactly as a full decode renders it."""
        prefix_text = self._decode(self.ids[self.prefix_offset : self.read_offset])
        new_text = self._decode(self.ids[self.prefix_offset :])
        self.prefix_offset = self.read_offset = len(self.ids)
        return new_text[len(prefix_text) :]


def _find_stop(text: str, prev_len: int, stops: list[str]) -> int | None:
    """Earliest index of a stop string that ends in the text added after prev_len."""
    found = None
    for stop in stops:
        idx = text.find(stop, max(0, prev_len - len(stop) + 1))
        if idx != -1 and (found is None or idx < found):
            found = idx
    return found


def append_token(seq: Sequence, token_id: int, eos_ids: frozenset[int], max_model_len: int) -> tuple[str, str | None]:
    """Append a sampled token, extend the decoded text, and evaluate stop conditions.

    Returns (text that is now safe to send, finish_reason or None). Stop strings are matched
    against decoded text rather than token ids, because the same string tokenizes differently
    depending on what precedes it. While stop strings are set, the last max(len(stop)) - 1
    characters are held back, since they could turn out to be the start of a stop string.
    """
    params = seq.sampling_params
    seq.append_token(token_id)
    prev_len = len(seq.output_text)
    if seq.detokenizer is not None:
        seq.output_text += seq.detokenizer.add(token_id)

    finish = None
    stop_at = _find_stop(seq.output_text, prev_len, params.stop) if params.stop else None
    if stop_at is not None:
        seq.output_text = seq.output_text[:stop_at]  # the stop string itself is not returned
        finish = "stop"
    elif token_id in eos_ids and not params.ignore_eos:
        finish = "stop"
    elif seq.num_output_tokens >= params.max_tokens or seq.num_tokens >= max_model_len:
        finish = "length"
    if finish is not None and stop_at is None and seq.detokenizer is not None:
        seq.output_text += seq.detokenizer.flush()

    holdback = 0 if finish is not None or not params.stop else max(map(len, params.stop)) - 1
    end = max(seq.num_emitted_chars, len(seq.output_text) - holdback)
    text = seq.output_text[seq.num_emitted_chars : end]
    seq.num_emitted_chars = end
    return text, finish
