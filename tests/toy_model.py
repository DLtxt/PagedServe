"""A weight-free stand-in for Qwen3 that makes paging bugs visible, for fast scheduler tests.

Every token's K vector stores its token id and its V vector stores its position. Instead of
attention, the model gathers each sequence's K/V back out of the paged cache through the naive
backend (block tables, slot mapping, and whatever the prefix cache shared), asserts the positions
come back as 0..n-1, and emits a token that hashes the entire gathered history. A wrong slot, a
stale or wrongly shared block, a skipped prefill chunk, or a bad swap therefore changes some
sequence's output compared with running it alone, which is what the tests check.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

EOS = 0


@dataclass(frozen=True)
class ToyConfig:
    num_hidden_layers: int = 1
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    head_dim: int = 2
    vocab_size: int = 64
    max_position_embeddings: int = 8192
    eos_token_ids: tuple[int, ...] = (EOS,)


def next_token(history: list[int], vocab_size: int) -> int:
    return hash(tuple(history)) % vocab_size


class ToyModel(nn.Module):
    def __init__(self, config: ToyConfig = ToyConfig()) -> None:
        super().__init__()
        self.config = config

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, attn) -> torch.Tensor:
        t = input_ids.shape[0]
        k = input_ids.to(torch.float32).view(t, 1, 1).expand(t, 1, 2).contiguous()
        v = positions.to(torch.float32).view(t, 1, 1).expand(t, 1, 2).contiguous()
        attn.write(0, k, v)
        meta = attn.meta
        out = torch.empty(t, dtype=torch.float32)
        for i in range(meta.num_seqs):
            keys, values = attn.gather(0, i)
            history = keys[:, 0, 0].long().tolist()
            got = values[:, 0, 0].long().tolist()
            assert got == list(range(len(got))), f"gathered positions out of order: {got[:40]}"
            lo, hi, kv_len = meta.query_start[i], meta.query_start[i + 1], meta.kv_lens[i]
            first = kv_len - (hi - lo)  # tokens cached before this step's chunk
            for j in range(hi - lo):
                out[lo + j] = next_token(history[: first + j + 1], self.config.vocab_size)
        return out.view(t, 1)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        tokens = hidden[:, 0].long()
        logits = torch.full((len(tokens), self.config.vocab_size), -1e9)
        logits[torch.arange(len(tokens)), tokens] = 0.0
        return logits


class ToyTokenizer:
    """Token i decodes to one letter (EOS is special and skipped), so stop strings can be tested."""

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(ord("a") + i % 26) for i in ids if not (skip_special_tokens and i == EOS))


def reference_tokens(prompt: list[int], max_tokens: int, ignore_eos: bool, vocab_size: int = ToyConfig.vocab_size) -> list[int]:
    """What any correct engine must generate for this prompt under the toy model (greedy)."""
    tokens = list(prompt)
    out = []
    while len(out) < max_tokens:
        tok = next_token(tokens, vocab_size)
        tokens.append(tok)
        out.append(tok)
        if tok == EOS and not ignore_eos:
            break
    return out
