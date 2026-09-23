"""The greedy-equivalence rule shared by the correctness gates.

Two correct implementations never produce bit-identical logits: they sum in different orders. So
exact token equality is required everywhere except at a genuine near-tie. If every logit agrees with
the reference within `logit_tol`, the argmax can only flip where the reference's top-2 gap is below
2 * logit_tol; a first mismatch there is numerical noise, and the sequences legitimately diverge
after it. Any mismatch at a larger gap, or any logit error above the tolerance, is a bug.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

TOP_K = 8
# Max |logit error| accepted against the reference. fp32 paths agree to ~3e-5 in practice; bf16
# reductions are ~1e3x noisier.
LOGIT_TOL = {torch.float32: 1e-3, torch.bfloat16: 0.5, torch.float16: 0.25}


class TopK(NamedTuple):
    ids: torch.Tensor  # [TOP_K] token ids, highest logit first
    values: torch.Tensor  # [TOP_K] fp32 logits


def topk(logits: torch.Tensor, k: int = TOP_K) -> TopK:
    values, ids = logits.float().topk(k)
    return TopK(ids.cpu(), values.cpu())


def at(logits: torch.Tensor, ref: TopK) -> torch.Tensor:
    """Our fp32 logits at the reference's top-k ids."""
    return logits.float()[ref.ids.to(logits.device)].cpu()


def check_greedy_match(
    ref_tokens: list[int],
    ref_topk: list[TopK],
    tokens: list[int],
    logits_at_ref: list[torch.Tensor],
    logit_tol: float,
    label: str,
) -> int | None:
    """Assert the rule above for one sequence. Returns the step of a tolerated near-tie divergence,
    or None when the sequences match exactly."""
    for t in range(min(len(ref_tokens), len(tokens))):
        err = float((logits_at_ref[t] - ref_topk[t].values).abs().max())
        assert err <= logit_tol, f"{label}: step {t}: logit error {err:.3e} exceeds {logit_tol:.0e}"
        if tokens[t] != ref_tokens[t]:
            gap = float(ref_topk[t].values[0] - ref_topk[t].values[1])
            assert gap < 2 * logit_tol, (
                f"{label}: token mismatch at step {t} ({tokens[t]} vs reference {ref_tokens[t]}) "
                f"where the reference's top-2 gap is {gap:.3e}: not a near-tie"
            )
            return t
    assert len(tokens) == len(ref_tokens), (
        f"{label}: identical prefixes but lengths differ ({len(tokens)} vs reference {len(ref_tokens)})"
    )
    return None


class LogitsRecorder:
    """An engine.logits_hook that records, per request and output step, either the top-k of the
    logits (a reference run: refs=None) or the logits at a reference run's top-k ids."""

    def __init__(self, refs: dict[int, list[TopK]] | None = None) -> None:
        self.refs = refs
        self.steps: dict[int, list] = {}

    def __call__(self, batch, logits: torch.Tensor) -> None:
        sampling = [seq for seq, sample in zip(batch.seqs, batch.do_sample) if sample]
        for row, seq in enumerate(sampling):
            steps = self.steps.setdefault(seq.seq_id, [])
            assert len(steps) == seq.num_output_tokens, "logits recorded out of order"
            if self.refs is None:
                steps.append(topk(logits[row]))
            elif seq.num_output_tokens < len(self.refs[seq.seq_id]):
                steps.append(at(logits[row], self.refs[seq.seq_id][seq.num_output_tokens]))
            else:
                steps.append(None)  # past the end of the reference: nothing to compare


def at_topk(ids: torch.Tensor, values: torch.Tensor, ref: TopK) -> torch.Tensor:
    """Our logits at the reference's top-k ids, looked up in our own (wider) top-k. A reference id
    missing from ours comes back NaN, which fails any tolerance check: it means the logits differ by
    more than the gap between our top-k and the rest."""
    lookup = dict(zip(ids.tolist(), values.tolist()))
    return torch.tensor([lookup.get(i, float("nan")) for i in ref.ids.tolist()])


class TopKRecorder:
    """An engine.topk_hook for distributed runs, where full logits live on another rank: records, per
    request and output step, our logits at the reference run's top-k ids."""

    def __init__(self, refs: dict[int, list[TopK]]) -> None:
        self.refs = refs
        self.steps: dict[int, list] = {}

    def __call__(self, batch, ids: torch.Tensor, values: torch.Tensor) -> None:
        sampling = [seq for seq, sample in zip(batch.seqs, batch.do_sample) if sample]
        for row, seq in enumerate(sampling):
            steps = self.steps.setdefault(seq.seq_id, [])
            assert len(steps) == seq.num_output_tokens, "top-k recorded out of order"
            ref = self.refs[seq.seq_id]
            step = seq.num_output_tokens
            steps.append(at_topk(ids[row], values[row], ref[step]) if step < len(ref) else None)
