# GPU runbook

Everything that can run on a CPU was built and verified on one: gates 1 through 8 pass in fp32 against
the real Qwen3-0.6B-Base weights. This runbook covers the rest, which needs an NVIDIA GPU:

- gate 9: the FlashInfer paged-attention path matches the naive oracle (`tests/test_gpu.py`)
- CUDA graphs against eager decode, and the profiled KV-cache size
- gates 1 through 8 again, on CUDA
- every benchmark: the latency-throughput sweep against vLLM and HF static batching, and the block-size,
  admission-policy, preemption, prefix-caching, chunked-prefill and kernel sweeps

One script does all of it: `scripts/gpu_suite.sh`.

## 1. Rent a machine

- **GPU**: one consumer card with 16 to 24 GB, such as an RTX 4090, 3090 or 4080. Marketplace prices run
  about $0.25 to $0.40 an hour. A datacenter card is unnecessary for a 0.6B model.
- **Image**: a PyTorch image with CUDA 12 and its toolkit (`nvcc`), for example a "PyTorch ... devel"
  template. FlashInfer downloads prebuilt kernels when it can and compiles them with `nvcc` when it can't.
- **Disk**: 40 GB or more. The two Python environments take most of it (vLLM pins its own torch, so it
  gets a separate one), plus the model (1.2 GB) and the ShareGPT trace (about 650 MB).
- **Python**: 3.10 or newer.

## 2. Run the suite

```bash
git clone https://github.com/DLtxt/PagedServe.git && cd PagedServe
tmux new -s suite                      # keeps the run alive if SSH drops
scripts/gpu_suite.sh 2>&1 | tee gpu_suite.log
```

Expect about three hours on an RTX 4090, roughly $1 to $2 of compute. The stages, which can also run one
at a time (`scripts/gpu_suite.sh gates`):

| stage | what it does | time |
|---|---|---|
| `setup` | both environments, FlashInfer's prebuilt kernels, model, ShareGPT, all workload files | ~15 min |
| `gates` | `tests/test_gpu.py` first (fail fast), then gates 1-8 on CUDA in fp32 | ~15 min |
| `bench` | every sweep in `bench/sweeps/`, the preemption crossover, vLLM, HF static batching | ~2.5 h |
| `plots` | figures and tables into `bench/figures/` | ~1 min |

The suite stops at the first failure. Benchmarks never run on a path that failed its gate.

## 3. If a gate fails

Don't benchmark around it. Send me `gpu_suite.log`, or the failing test's output. These narrow it down:

```bash
export PAGEDSERVE_TEST_DEVICE=cuda
.venv/bin/python -m pytest tests/test_gpu.py -q -s -k kernel    # FlashInfer vs naive, one attention call
.venv/bin/python -m pytest tests/test_gpu.py -q -s -k engine    # FlashInfer vs naive, end to end
.venv/bin/python -m pytest tests/test_gpu.py -q -s -k graphs    # CUDA graphs vs eager
.venv/bin/python -m pytest tests -q -s -m "not gpu" -x          # gates 1-8
```

If only the CUDA-graph test fails, the benchmarks can still run without graphs: remove `--cuda-graphs`
from `bench/sweeps/latency.json` and `bench/sweeps/kernels.json`.

## 4. Bring the results back

```bash
git add bench/results bench/figures
git commit -m "GPU results on <GPU model>"
git push
```

`bench/results/` holds every CSV (per request, per step, and summaries); `bench/figures/` holds the
figures and `results.md`, the same numbers as tables. Server logs stay out of git (`*.log` is ignored),
so copy any you want to keep. Once the results are pushed, the write-up's results section can be filled
from them (`docs/WRITEUP.md`).

## Troubleshooting

- **FlashInfer fails to import or compile.** `.venv/bin/flashinfer show-config` shows what it found.
  The CUDA toolkit's major version must match the one torch was built with (`python -c "import torch;
  print(torch.version.cuda)"`).
- **Out of memory at startup.** Lower `--gpu-memory-utilization` in the spec's `server_args` (the
  default is 0.90).
- **The highest request rates run long.** Past saturation the queue grows without bound: that knee is
  the point of the latency-throughput plot. Trim `rates` in `bench/sweeps/latency.json` once you know
  where it is.
- **vLLM will not start.** Its log is `bench/results/latency/vllm_server.log`. The baseline only needs
  `vllm serve` with an OpenAI-compatible completions endpoint; any recent release works.
