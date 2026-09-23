"""Benchmark workloads: realistic, non-uniform request lengths, with Poisson arrivals.

Uniform lengths hide exactly the head-of-line blocking the scheduler experiments exist to expose, so
lengths come from a ShareGPT trace (with vLLM's benchmark filtering) or from a lognormal. Prompts are
token ids, sent as such, so every server under test sees exactly the same lengths.

A workload file (JSON lines) fixes the requests; arrival times are drawn per run from a seeded
Poisson process at the requested rate, so a rate sweep replays identical requests at each rate.
Structured workloads (the long-prompt mix) store their own arrival times.

    python -m bench.workload sharegpt --sharegpt bench/data/ShareGPT_V3_unfiltered_cleaned_split.json --n 1000 --out bench/data/sharegpt.jsonl
    python -m bench.workload lognormal --n 1000 --out bench/data/lognormal.jsonl
    python -m bench.workload fit --sharegpt bench/data/ShareGPT_V3_unfiltered_cleaned_split.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

MODEL = "Qwen/Qwen3-0.6B-Base"
FIRST_SPECIAL_TOKEN = 151643  # Qwen3: ids at or above this are special tokens

# Lognormal defaults for when no ShareGPT file is available. These are assumptions, not a fit:
# median prompt 128 tokens (mean ~210), median output 96 tokens (mean ~145), with long right tails.
# Replace them with `python -m bench.workload fit` on a real trace before quoting results.
DEFAULT_LOGNORMAL = {"in_mu": math.log(128), "in_sigma": 1.0, "out_mu": math.log(96), "out_sigma": 0.9}


@dataclass
class Request:
    prompt_ids: list[int]
    output_len: int
    arrival: float | None = None  # seconds after the start of the run; None: drawn per run
    group: str = ""  # a request class for per-class plots, e.g. "long" / "short" / a prefix id


def random_prompt(rng: np.random.Generator, n: int) -> list[int]:
    return rng.integers(0, FIRST_SPECIAL_TOKEN, size=n).tolist()


def poisson_arrivals(n: int, rate: float, seed: int) -> list[float]:
    """Arrival times of a Poisson process at `rate` requests/s. rate=inf sends everything at t=0."""
    if math.isinf(rate):
        return [0.0] * n
    gaps = np.random.default_rng(seed).exponential(1.0 / rate, size=n)
    gaps[0] = 0.0
    return np.cumsum(gaps).tolist()


# --- length sources --------------------------------------------------------------------------------


def load_sharegpt(path: str, tokenizer, n: int, seed: int, max_prompt: int = 1024, max_total: int = 2048) -> list[Request]:
    """First human turn as the prompt, first model turn's length as the output length, filtered the
    way vLLM's benchmark filters ShareGPT: both at least 4 tokens, prompt at most 1024, total 2048."""
    with open(path) as f:
        conversations = json.load(f)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(conversations))
    requests = []
    for idx in order:
        turns = conversations[idx].get("conversations", [])
        if len(turns) < 2:
            continue
        prompt = tokenizer(turns[0]["value"]).input_ids
        output_len = len(tokenizer(turns[1]["value"]).input_ids)
        if len(prompt) < 4 or output_len < 4 or len(prompt) > max_prompt or len(prompt) + output_len > max_total:
            continue
        requests.append(Request(prompt, output_len))
        if len(requests) == n:
            break
    return requests


def lognormal(n: int, seed: int, in_mu: float, in_sigma: float, out_mu: float, out_sigma: float,
              max_prompt: int = 1024, max_output: int = 1024) -> list[Request]:
    rng = np.random.default_rng(seed)
    ins = np.clip(np.rint(rng.lognormal(in_mu, in_sigma, n)), 4, max_prompt).astype(int)
    outs = np.clip(np.rint(rng.lognormal(out_mu, out_sigma, n)), 4, max_output).astype(int)
    return [Request(random_prompt(rng, int(i)), int(o)) for i, o in zip(ins, outs)]


def fit_lognormal(requests: list[Request]) -> dict:
    ins = np.log([len(r.prompt_ids) for r in requests])
    outs = np.log([r.output_len for r in requests])
    return {"in_mu": float(ins.mean()), "in_sigma": float(ins.std()), "out_mu": float(outs.mean()), "out_sigma": float(outs.std())}


# --- structured workloads ----------------------------------------------------------------------------


def shared_prefix(n: int, seed: int, prefix_len: int, suffix_range: tuple[int, int] = (16, 64), output_len: int = 64,
                  num_prefixes: int = 1) -> list[Request]:
    """A long shared system prompt plus short unique suffixes: what prefix caching is for."""
    rng = np.random.default_rng(seed)
    prefixes = [random_prompt(rng, prefix_len) for _ in range(num_prefixes)]
    requests = []
    for i in range(n):
        p = i % num_prefixes
        suffix = random_prompt(rng, int(rng.integers(suffix_range[0], suffix_range[1] + 1)))
        requests.append(Request(prefixes[p] + suffix, output_len, group=f"prefix{p}"))
    return requests


def long_prompt_mix(duration: float, seed: int, short_rate: float = 4.0, short_in: int = 64, short_out: int = 256,
                    long_every: float = 5.0, long_in: int = 4000, long_out: int = 16) -> list[Request]:
    """A steady Poisson stream of short chat requests with a long prompt landing every `long_every`
    seconds: without chunked prefill each long prompt monopolizes an iteration and every decoding
    request stalls behind it."""
    rng = np.random.default_rng(seed)
    requests = []
    t = 0.0
    while t < duration:
        requests.append(Request(random_prompt(rng, short_in), short_out, arrival=t, group="short"))
        t += float(rng.exponential(1.0 / short_rate))
    t = long_every
    while t < duration:
        requests.append(Request(random_prompt(rng, long_in), long_out, arrival=t, group="long"))
        t += long_every
    return sorted(requests, key=lambda r: r.arrival)


# --- files -------------------------------------------------------------------------------------------


def save(requests: list[Request], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in requests:
            f.write(json.dumps(asdict(r)) + "\n")


def load(path: str) -> list[Request]:
    with open(path) as f:
        return [Request(**json.loads(line)) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("sharegpt")
    p.add_argument("--sharegpt", required=True)
    p = sub.add_parser("lognormal")
    for key, value in DEFAULT_LOGNORMAL.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=float, default=value)
    p = sub.add_parser("shared-prefix")
    p.add_argument("--prefix-len", type=int, required=True)
    p.add_argument("--num-prefixes", type=int, default=1)
    p.add_argument("--output-len", type=int, default=64)
    p = sub.add_parser("long-mix")
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--short-rate", type=float, default=4.0)
    p.add_argument("--long-every", type=float, default=5.0)
    p.add_argument("--long-in", type=int, default=4000)
    p.add_argument("--short-in", type=int, default=64)
    p.add_argument("--short-out", type=int, default=256)
    p.add_argument("--long-out", type=int, default=16)
    p = sub.add_parser("fit")
    p.add_argument("--sharegpt", required=True)
    for p in sub.choices.values():
        p.add_argument("--n", type=int, default=1000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--model", default=MODEL)
        p.add_argument("--out", default=None)
    args = parser.parse_args()

    tokenizer = None
    if args.kind in ("sharegpt", "fit"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.kind == "fit":
        print(json.dumps(fit_lognormal(load_sharegpt(args.sharegpt, tokenizer, args.n, args.seed)), indent=2))
        return
    if args.kind == "sharegpt":
        requests = load_sharegpt(args.sharegpt, tokenizer, args.n, args.seed)
    elif args.kind == "lognormal":
        requests = lognormal(args.n, args.seed, args.in_mu, args.in_sigma, args.out_mu, args.out_sigma)
    elif args.kind == "shared-prefix":
        requests = shared_prefix(args.n, args.seed, args.prefix_len, output_len=args.output_len, num_prefixes=args.num_prefixes)
    else:
        requests = long_prompt_mix(args.duration, args.seed, short_rate=args.short_rate, short_in=args.short_in,
                                   short_out=args.short_out, long_every=args.long_every, long_in=args.long_in,
                                   long_out=args.long_out)
    save(requests, args.out)
    lens = [len(r.prompt_ids) for r in requests]
    outs = [r.output_len for r in requests]
    print(f"{len(requests)} requests -> {args.out}: prompt mean {np.mean(lens):.0f} p50 {np.median(lens):.0f} max {max(lens)}; "
          f"output mean {np.mean(outs):.0f} p50 {np.median(outs):.0f} max {max(outs)}")


if __name__ == "__main__":
    main()
