"""Baseline: HuggingFace generate() with static batching, on the same workload and arrival process.

A single worker takes up to --batch-size requests that have arrived, left-pads them into one batch,
and runs generate() until the longest one is done; requests arriving meanwhile wait for the next
batch. That is what continuous batching replaces. Every generated token is timestamped (a stopping
criterion runs once per step), so TTFT and ITL are measured as if the batch streamed its tokens,
which is generous to the baseline: a real static-batching server would return nothing until the
whole batch finished. Results use the same CSV schema as run_bench.py.

    python -m bench.hf_baseline --workload bench/data/sharegpt.jsonl --rates 1,2,4 --duration 60 --out bench/results/latency
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

from bench.run_bench import MODEL, _requests_for, summarize, write_results
from bench.workload import load, poisson_arrivals


class _StepClock(StoppingCriteria):
    """Records the wall time at which each generation step's tokens exist. Never stops generation."""

    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.times: list[float] = []

    def __call__(self, input_ids, scores, **kwargs) -> torch.Tensor:
        if input_ids.is_cuda:
            torch.cuda.synchronize()
        self.times.append(time.perf_counter() - self.t0)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


@torch.inference_mode()
def run_rate(model, requests, rate: float, seed: int, batch_size: int, pad_id: int, device: str) -> dict:
    arrivals = [r.arrival for r in requests] if all(r.arrival is not None for r in requests) else poisson_arrivals(len(requests), rate, seed)
    order = sorted(range(len(requests)), key=lambda i: arrivals[i])
    records: list[dict | None] = [None] * len(requests)
    t0 = time.perf_counter()
    nxt = 0
    pending: list[int] = []
    while nxt < len(order) or pending:
        now = time.perf_counter() - t0
        while nxt < len(order) and arrivals[order[nxt]] <= now:
            pending.append(order[nxt])
            nxt += 1
        if not pending:
            time.sleep(max(0.0, arrivals[order[nxt]] - now))
            continue
        batch, pending = pending[:batch_size], pending[batch_size:]
        prompts = [requests[i].prompt_ids for i in batch]
        width = max(map(len, prompts))
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, p in enumerate(prompts):  # left padding
            ids[row, width - len(p):] = torch.tensor(p)
            mask[row, width - len(p):] = 1
        new_tokens = max(requests[i].output_len for i in batch)
        clock = _StepClock(t0)
        model.generate(
            input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=new_tokens, min_new_tokens=new_tokens,
            do_sample=False, pad_token_id=pad_id, stopping_criteria=StoppingCriteriaList([clock]),
        )
        end = time.perf_counter() - t0
        for i in batch:
            n = requests[i].output_len
            records[i] = {
                "idx": i, "group": requests[i].group, "prompt_len": len(requests[i].prompt_ids), "output_len": n,
                "sent": arrivals[i], "token_times": clock.times[:n], "output_tokens": n, "error": "", "end": end,
                "success": len(clock.times) >= n,
            }
    return {"records": records, "arrivals": arrivals, "duration": time.perf_counter() - t0, "before": None, "after": None, "series": []}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--rates", default="1,2,4")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--label", default="hf-static")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="bench/results")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(args.device).eval()
    pad_id = model.generation_config.eos_token_id
    pad_id = pad_id[0] if isinstance(pad_id, list) else pad_id
    workload = load(args.workload)
    for rate in (float(x) for x in args.rates.split(",")):
        requests = _requests_for(workload, rate, args.duration, args.num_requests)
        result = run_rate(model, requests, rate, args.seed, args.batch_size, pad_id, args.device)
        summary = summarize(args.label, rate, result)
        write_results(Path(args.out), args.label, rate, result, summary)
        print(f"[{args.label}] rate {rate:g}: {summary['throughput_tok_s']:.0f} tok/s, "
              f"TTFT p50/p99 {summary['ttft_p50']:.2f}/{summary['ttft_p99']:.2f}s, ITL p99 {summary['itl_p99'] * 1e3:.1f}ms", flush=True)


if __name__ == "__main__":
    main()
