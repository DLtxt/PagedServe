#!/usr/bin/env bash
# The GPU half of PagedServe in one script: environment, correctness gates, benchmarks, baselines,
# figures. Run from anywhere on a Linux machine with NVIDIA GPUs (16-24 GB each); see docs/GPU_RUNBOOK.md.
#
#   scripts/gpu_suite.sh            # everything, in order (about 3 hours on an RTX 4090)
#   scripts/gpu_suite.sh setup      # or one stage: setup | gates | bench | dist | plots
#
# `dist` (tensor and pipeline parallelism, Qwen3-8B and 14B) needs two GPUs in the machine and about
# 45 GB more disk for the two models; on a single-GPU machine it skips itself.
#
# Stages stop at the first failure. Gates run before any benchmark: a number measured on a path that
# fails its gate means nothing.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-all}"
PY=.venv/bin/python
VLLM_PY=.venv-vllm/bin/python
MODEL="${MODEL:-Qwen/Qwen3-0.6B-Base}"
RESULTS="${RESULTS:-bench/results}"
SHAREGPT_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT=bench/data/ShareGPT_V3_unfiltered_cleaned_split.json

log() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

setup() {
  log "environment (.venv: this engine; .venv-vllm: the vLLM baseline, which pins its own torch)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  python3 -m venv .venv
  $PY -m pip install -q -U pip
  $PY -m pip install -q -e ".[gpu,bench,test]"
  # Prebuilt FlashInfer kernels skip a long JIT compile on first use; the JIT path still works without them.
  .venv/bin/flashinfer install-cubin-wheel || echo "flashinfer cubin wheel unavailable; kernels will JIT-compile on first use"
  .venv/bin/flashinfer install-jit-cache-wheel || echo "flashinfer jit-cache wheel unavailable; kernels will JIT-compile on first use"
  python3 -m venv .venv-vllm
  $VLLM_PY -m pip install -q -U pip
  $VLLM_PY -m pip install -q vllm

  log "model weights (about 1.2 GB) and the ShareGPT trace (about 650 MB)"
  $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"
  mkdir -p bench/data
  [ -f "$SHAREGPT" ] || curl -L --fail -o "$SHAREGPT" "$SHAREGPT_URL"

  log "workloads"
  $PY -m bench.workload sharegpt --sharegpt "$SHAREGPT" --n 4000 --out bench/data/sharegpt.jsonl
  $PY -m bench.workload fit --sharegpt "$SHAREGPT" --n 4000 | tee bench/data/sharegpt_lognormal_fit.json
  $PY -m bench.workload lognormal --n 2000 --out-mu 5.5 --out-sigma 0.6 --out bench/data/long_outputs.jsonl
  for L in 1000 2000 4000; do
    $PY -m bench.workload long-mix --duration 120 --short-rate 4 --long-every 5 --long-in "$L" --out "bench/data/long_mix_L$L.jsonl"
  done
  for L in 0 128 256 512 1024 2048; do
    $PY -m bench.workload shared-prefix --n 2000 --prefix-len "$L" --out "bench/data/prefix_L$L.jsonl"
  done
}

gates() {
  log "fail fast: gate 9 (FlashInfer vs naive), CUDA graphs vs eager, memory profiling"
  PAGEDSERVE_TEST_DEVICE=cuda $PY -m pytest tests/test_gpu.py -x -q -s
  log "gates 1-8 on CUDA, fp32 (exact greedy, near-tie rule); the split-model tests run in the dist stage"
  PAGEDSERVE_TEST_DEVICE=cuda $PY -m pytest tests -x -q -s -m "not gpu and not distributed"
}

wait_healthy() {  # url, pid
  for _ in $(seq 1 900); do
    curl -sf "$1/health" >/dev/null 2>&1 && return 0
    kill -0 "$2" 2>/dev/null || { echo "server exited during startup"; return 1; }
    sleep 1
  done
  echo "server did not become healthy"; return 1
}

bench() {
  log "sweeps of this engine (each config launches its own server)"
  for spec in latency block_size policies preemption prefix chunked kernels; do
    $PY -m bench.run_bench sweep --spec "bench/sweeps/$spec.json" --out "$RESULTS/$spec"
  done
  log "preemption crossover: recompute vs swap cost against sequence length"
  $PY -m bench.preemption_crossover --out "$RESULTS/preemption/crossover.csv"
  log "profile of 20 busy engine steps (operator table saved; Chrome trace under profiles/)"
  $PY -m bench.profile_steps --enable-chunked-prefill --enable-prefix-caching --num-requests 128 --steps 20 \
    --trace profiles/engine_steps.trace.json | tee "$RESULTS/profile_steps.txt"

  log "baseline: vLLM, same GPU, same traffic, same client"
  .venv-vllm/bin/vllm serve "$MODEL" --host 127.0.0.1 --port 8001 \
    --max-model-len 4096 --gpu-memory-utilization 0.9 > "$RESULTS/latency/vllm_server.log" 2>&1 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
  wait_healthy http://127.0.0.1:8001 "$VLLM_PID"
  $VLLM_PY -c "import vllm; print('vllm', vllm.__version__)" | tee "$RESULTS/latency/vllm_version.txt"
  $PY -m bench.run_bench run --url http://127.0.0.1:8001 --workload bench/data/sharegpt.jsonl \
    --rates 1,2,4,8,16,24,32,48,64 --duration 60 --label vllm --out "$RESULTS/latency"
  kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null || true
  trap - EXIT

  log "baseline: HuggingFace generate() with static batching"
  $PY -m bench.hf_baseline --workload bench/data/sharegpt.jsonl --rates 1,2,4,8 --duration 60 \
    --batch-size 32 --label hf-static --out "$RESULTS/latency"
}

dist() {
  local gpus
  gpus=$(nvidia-smi --list-gpus | wc -l)
  if [ "$gpus" -lt 2 ]; then
    echo "distributed stage skipped: it needs 2 GPUs in this machine, found $gpus"
    return 0
  fi
  log "distributed correctness on $gpus GPUs: tensor and pipeline parallelism over NCCL match one GPU"
  PAGEDSERVE_TEST_DEVICE=cuda $PY -m pytest tests/test_distributed.py -x -q -s

  log "Qwen3-8B-Base (about 16 GB) and Qwen3-14B-Base (about 28 GB)"
  for m in Qwen/Qwen3-8B-Base Qwen/Qwen3-14B-Base; do
    $PY -c "from engine.model.loader import resolve_model_path; print(resolve_model_path('$m'))"
  done

  log "Qwen3-8B: one GPU, tensor parallel 2, pipeline parallel 2 (pipelined, then one batch at a time)"
  $PY -m bench.run_bench sweep --spec bench/sweeps/distributed_8b.json --out "$RESULTS/distributed_8b"

  log "baseline: vLLM, Qwen3-8B, tensor parallel 2, same traffic, same client"
  mkdir -p "$RESULTS/distributed_8b"
  .venv-vllm/bin/vllm serve Qwen/Qwen3-8B-Base --tensor-parallel-size 2 --host 127.0.0.1 --port 8001 \
    --max-model-len 4096 --gpu-memory-utilization 0.9 > "$RESULTS/distributed_8b/vllm_server.log" 2>&1 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
  wait_healthy http://127.0.0.1:8001 "$VLLM_PID"
  $PY -m bench.run_bench run --url http://127.0.0.1:8001 --model Qwen/Qwen3-8B-Base --workload bench/data/sharegpt.jsonl \
    --rates 1,2,4,8,12,16 --duration 45 --label vllm-tp2 --out "$RESULTS/distributed_8b"
  kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null || true
  trap - EXIT

  log "Qwen3-14B: too big for one 24 GB GPU; tensor parallel 2 against pipeline parallel 2"
  $PY -m bench.run_bench sweep --spec bench/sweeps/distributed_14b.json --out "$RESULTS/distributed_14b"
}

plots() {
  log "figures and tables"
  $PY -m bench.plot --results "$RESULTS" --out bench/figures
  echo "figures: bench/figures/   tables: bench/figures/results.md   raw CSVs: $RESULTS/"
}

case "$STAGE" in
  setup) setup ;;
  gates) gates ;;
  bench) bench ;;
  dist) dist ;;
  plots) plots ;;
  all) setup; gates; bench; dist; plots ;;
  *) echo "usage: $0 [setup|gates|bench|dist|plots|all]"; exit 2 ;;
esac
