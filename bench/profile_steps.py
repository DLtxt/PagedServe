"""Profile engine steps with torch.profiler: where a step's time goes, and whether anything syncs.

Loads the engine, keeps it busy with a batch of requests, skips warm-up steps, then records a window of
steps. Prints the top operators by device and host time and writes a Chrome trace (open it at
https://ui.perfetto.dev). A hidden GPU sync shows up in the trace as the host thread waiting inside a
copy (aten::item, aten::_local_scalar_dense, cudaStreamSynchronize) while the GPU idles, anywhere other
than the sampler's one .tolist() per step.

    python -m bench.profile_steps --enable-chunked-prefill --num-requests 128 --steps 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from engine.config import EngineConfig
from engine.core.engine import LLMEngine
from engine.core.sequence import SamplingParams, Sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    EngineConfig.add_cli_args(parser)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--trace", default="profiles/engine_steps.trace.json")
    args = parser.parse_args()

    engine = LLMEngine(EngineConfig.from_cli_args(args))
    rng = np.random.default_rng(0)
    for _ in range(args.num_requests):
        prompt = rng.integers(0, 151643, size=args.prompt_len).tolist()
        engine.add_request(Sequence(engine.new_seq_id(), prompt,
                                    SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)))
    for _ in range(args.warmup):
        engine.step()

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if engine.config.is_cuda else [])
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(args.steps):
            engine.step()
        if engine.config.is_cuda:
            torch.cuda.synchronize()
    sort = "cuda_time_total" if engine.config.is_cuda else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort, row_limit=25))
    Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(args.trace)
    print(f"trace: {args.trace}")


if __name__ == "__main__":
    main()
