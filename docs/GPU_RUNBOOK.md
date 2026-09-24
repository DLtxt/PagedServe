# GPU runbook

Everything that can run on a CPU was built and verified on one: gates 1 through 8 pass in fp32 against
the real Qwen3-0.6B-Base weights. This runbook covers the rest, which needs an NVIDIA GPU:

- gate 9: the FlashInfer paged-attention path matches the naive oracle (`tests/test_gpu.py`)
- CUDA graphs against eager decode, and the profiled KV-cache size
- gates 1 through 8 again, on CUDA
- every benchmark: the latency-throughput sweep against vLLM and HF static batching, and the block-size,
  admission-policy, preemption, prefix-caching, chunked-prefill and kernel sweeps
- on a machine with two GPUs, tensor and pipeline parallelism: the split-model tests over NCCL, then
  Qwen3-8B on one GPU, split both ways, and against vLLM, and Qwen3-14B, which only fits split

One script does all of it: `scripts/gpu_suite.sh`. A second, `scripts/multinode.sh`, runs pipeline
parallelism across two machines (section 5).

## 1. Rent a machine

- **GPU**: one consumer card with 16 to 24 GB, such as an RTX 4090, 3090 or 4080. Marketplace prices run
  about $0.25 to $0.40 an hour. A datacenter card is unnecessary for a 0.6B model.
- **Image**: a PyTorch image with CUDA 12 and its toolkit (`nvcc`), for example a "PyTorch ... devel"
  template. FlashInfer downloads prebuilt kernels when it can and compiles them with `nvcc` when it can't.
- **Disk**: 40 GB or more. The two Python environments take most of it (vLLM pins its own torch, so it
  gets a separate one), plus the model (1.2 GB) and the ShareGPT trace (about 650 MB).
- **Python**: 3.10 or newer.

For the distributed stage, rent **two GPUs in one machine** (24 GB each is enough for all of it: 14B
needs both) and **100 GB of disk**: Qwen3-8B-Base is about 16 GB and Qwen3-14B-Base about 28 GB. The
interconnect decides how tensor parallelism looks. Consumer cards such as the RTX 4090 have no NVLink,
and their driver does not enable peer-to-peer between them, so NCCL's all-reduces go through host
memory; NVLink-connected datacenter GPUs (A100 or H100 SXM) show tensor parallelism at its best.
`nvidia-smi topo -m` shows what a machine has. Either is a fair measurement, as long as the write-up
names it. One option is a two-GPU machine for the whole suite; the other is the single-GPU
suite on a one-GPU machine, where `dist` skips itself, and then `setup`, `dist` and `plots` on a two-GPU
machine.

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
| `dist` | two GPUs only: `tests/test_distributed.py` over NCCL, the 8B and 14B models, their sweeps, vLLM at TP=2 | ~2 h |
| `plots` | figures and tables into `bench/figures/` | ~1 min |

The suite stops at the first failure. Benchmarks never run on a path that failed its gate.

## 3. If a gate fails

Don't benchmark around it. Send me `gpu_suite.log`, or the failing test's output. These narrow it down:

```bash
export PAGEDSERVE_TEST_DEVICE=cuda
.venv/bin/python -m pytest tests/test_gpu.py -q -s -k kernel    # FlashInfer vs naive, one attention call
.venv/bin/python -m pytest tests/test_gpu.py -q -s -k engine    # FlashInfer vs naive, end to end
.venv/bin/python -m pytest tests/test_gpu.py -q -s -k graphs    # CUDA graphs vs eager
.venv/bin/python -m pytest tests -q -s -m "not gpu and not distributed" -x   # gates 1-8
```

If only the CUDA-graph test fails, the benchmarks can still run without graphs: remove `--cuda-graphs`
from `bench/sweeps/latency.json` and `bench/sweeps/kernels.json`.

For the distributed stage:

```bash
PAGEDSERVE_TEST_DEVICE=cuda .venv/bin/python -m pytest tests/test_distributed.py -q -s -m "not model"  # tiny model, NCCL
PAGEDSERVE_TEST_DEVICE=cuda .venv/bin/python -m pytest tests/test_distributed.py -q -s -k bf16         # the benchmark configuration
NCCL_DEBUG=INFO scripts/gpu_suite.sh dist                                                              # what NCCL picked, and why
```

## 4. Bring the results back

Copy them to your own machine and commit there, so the rented box never holds GitHub credentials and
the commit carries your usual identity:

```bash
# on your machine, in the repository, on the branch the suite ran
rsync -av <user>@<gpu-box>:PagedServe/bench/results/ bench/results/
rsync -av <user>@<gpu-box>:PagedServe/bench/figures/ bench/figures/
scp <user>@<gpu-box>:PagedServe/gpu_suite.log .   # for reading, not for committing
git add bench/results bench/figures
git commit -m "GPU results on <GPU model>"
git push
```

`bench/results/` holds every CSV (per request, per step, and summaries); `bench/figures/` holds the
figures and `results.md`, the same numbers as tables. Server logs stay out of git (`*.log` is ignored),
so copy any you want to keep. Once the results are pushed, the write-up's results section can be filled
from them (`docs/WRITEUP.md`).

## 5. Two machines

`scripts/multinode.sh` splits Qwen3-8B into two pipeline stages on two machines, one GPU each, and
benchmarks it from the first; the results land next to the single-machine layouts, as `pp2-2nodes`.

The machines need a **private network** between them: the same cloud VPC, or Tailscale. torchrun's
rendezvous and the gloo and NCCL connections have no authentication, so the script refuses a public
`MASTER_ADDR`, and pins gloo and NCCL to the interface that reaches it. Two more things the script
cannot check for you:

- torchrun's rendezvous listens on `MASTER_PORT` (29500) on every interface of machine 0. Keep
  inbound traffic from the internet blocked (the usual cloud default, apart from SSH) and allow TCP
  between the two machines only.
- Tailscale needs a real network device: a VM, or a container with `/dev/net/tun`. In userspace
  networking mode NCCL and gloo cannot reach the tailnet.

```bash
# on both machines
git clone https://github.com/DLtxt/PagedServe.git && cd PagedServe && scripts/gpu_suite.sh setup
# then one on each, in either order (each waits for the other); MASTER_ADDR is machine 0's private IP
NODE_RANK=1 MASTER_ADDR=10.0.0.5 scripts/multinode.sh
NODE_RANK=0 MASTER_ADDR=10.0.0.5 scripts/multinode.sh
```

Machine 0 serves HTTP on its own loopback only, runs the rates, and shuts both machines' ranks down.

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
- **A distributed server hangs at startup.** Every rank must reach the rendezvous and every other rank:
  `NCCL_DEBUG=INFO` shows the interface NCCL chose. On one machine with several network interfaces, set
  `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to the right one, as `scripts/multinode.sh` does.
- **"needs 2 GPUs" skips.** `nvidia-smi --list-gpus` must show both, and `CUDA_VISIBLE_DEVICES`, if set,
  must list both.
