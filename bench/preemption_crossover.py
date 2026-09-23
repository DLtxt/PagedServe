"""Preemption cost versus sequence length: recompute against swap.

Recompute throws a victim's KV away and redoes its prefill on resume: GPU work that grows with
sequence length (and quadratically in attention). Swap copies the victim's blocks to pinned host
memory and back: PCIe traffic linear in length. This measures both, per length, on the real engine
(the same runner code paths the scheduler uses) and finds where they cross.

    python -m bench.preemption_crossover --out bench/results/preemption/crossover.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from engine.config import EngineConfig
from engine.core.engine import LLMEngine
from engine.core.scheduler import ScheduledBatch
from engine.core.sequence import SamplingParams, Sequence

LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _timed(fn, repeats: int, cuda: bool) -> float:
    fn()  # warm-up
    samples = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-blocks", type=int, default=None, help="required off-GPU")
    parser.add_argument("--out", default="bench/results/preemption/crossover.csv")
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    max_len = max(lengths)
    config = EngineConfig(model=args.model, device=args.device, block_size=args.block_size,
                          max_model_len=max(max_len + 16, 1024), max_batch_tokens=max_len + 16, preemption_mode="swap",
                          swap_space_gb=8.0 if args.num_blocks is None else 0.5, gpu_memory_utilization=0.85,
                          num_blocks=args.num_blocks)
    engine = LLMEngine(config)
    cuda = config.is_cuda
    runner, bm = engine.model_runner, engine.block_manager
    rng = np.random.default_rng(0)
    rows = []
    for length in lengths:
        seq = Sequence(0, rng.integers(0, 151643, size=length).tolist(), SamplingParams(temperature=0, max_tokens=1))

        def recompute() -> None:  # a prefill over the whole sequence, as a recompute victim's resume does
            bm.allocate(seq, length, [])
            batch = ScheduledBatch()
            batch.add(seq, length)
            runner.execute(batch)
            bm.free(seq)
            seq.num_computed_tokens = 0

        bm.allocate(seq, length, [])
        seq.num_computed_tokens = length
        blocks = list(seq.block_table)
        cpu_blocks = list(range(len(blocks)))
        out_pairs = list(zip(blocks, cpu_blocks))
        in_pairs = list(zip(cpu_blocks, blocks))
        swap_out = _timed(lambda: runner._swap_out(out_pairs), args.repeats, cuda)
        swap_in = _timed(lambda: runner._swap_in(in_pairs), args.repeats, cuda)
        bm.free(seq)
        seq.num_computed_tokens = 0
        recompute_ms = _timed(recompute, args.repeats, cuda)
        mib = len(blocks) * runner.bytes_per_block / (1 << 20)
        rows.append({"length": length, "blocks": len(blocks), "kv_mib": round(mib, 3), "recompute_ms": round(recompute_ms, 4),
                     "swap_out_ms": round(swap_out, 4), "swap_in_ms": round(swap_in, 4), "swap_ms": round(swap_out + swap_in, 4)})
        print(rows[-1], flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    crossings = [r["length"] for r in rows if r["swap_ms"] < r["recompute_ms"]]
    print(f"swap is cheaper than recompute from length {crossings[0]}" if crossings else "recompute was cheaper at every length")


if __name__ == "__main__":
    main()
