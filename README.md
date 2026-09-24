# PagedServe

An LLM inference engine: paged KV-cache memory management, iteration-level continuous
batching, prefix caching, chunked prefill, recompute and swap preemption, and a streaming
OpenAI-compatible HTTP API. It serves `Qwen/Qwen3-0.6B-Base`.

## Status

| Part | State |
|---|---|
| Model, paged KV cache, scheduler, sampler, HTTP server | Verified on CPU: correctness gates 1 to 8 pass in fp32 against the real weights |
| FlashInfer attention path, CUDA graphs, GPU memory profiling | Written against FlashInfer's current API; not yet run on a GPU (gate 9 is `tests/test_gpu.py`) |
| Benchmarks | Harness smoke-tested end to end on CPU; no GPU measurements yet |

`scripts/gpu_suite.sh` runs the GPU half in one go; see [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md).

## Architecture

```
   HTTP clients
        │  POST /v1/completions (SSE stream or JSON)
        ▼
┌──────────────────────────────────────────────────┐
│  FastAPI handlers (async)                        │
│    tokenize, build a Sequence, enqueue it        │
│    await the request's output queue              │
│    client disconnect -> abort in the engine      │
└───────────────┬──────────────────────────────────┘
                │  per-request asyncio.Queue
┌───────────────▼──────────────────────────────────┐
│  Engine loop (synchronous, never awaits)         │
│     batch   = scheduler.schedule()               │
│     logits  = model_runner.execute(batch)        │
│     tokens  = sampler(logits)                    │
│     outputs = scheduler.update(batch, tokens)    │
│                                                  │
│  Scheduler ── BlockManager                       │
│  (admission, preemption, token budget)           │
│  (free list, block tables, refcounts, prefix     │
│   hash chain, LRU of cached blocks, swap pool)   │
│                                                  │
│  ModelRunner -> Qwen3 -> paged attention         │
│  KV cache: [layers, 2, blocks + 1, block_size,   │
│             kv_heads, head_dim]                  │
└──────────────────────────────────────────────────┘
```

The engine loop advances every scheduled sequence one iteration per `step()` and never blocks on
anything but the GPU; the HTTP layer is async and reaches it only through queues. [docs/DESIGN.md](docs/DESIGN.md)
explains every decision, the alternatives, and where this departs from the original plan.

## Quickstart (CPU)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[test,bench]"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B-Base')"  # ~1.2 GB

.venv/bin/python -m engine.api.server --device cpu --num-blocks 512 --max-model-len 2048
```

```bash
curl -N http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-0.6B-Base", "prompt": "The printing press changed Europe because",
  "max_tokens": 32, "temperature": 0, "stream": true}'
```

On a GPU the defaults do the right thing: `python -m engine.api.server` profiles memory to size the KV
cache and uses the FlashInfer backend. The server binds to 127.0.0.1; it has no authentication.

## Configuration

Every flag maps to a field of `EngineConfig` (`engine/config.py`); `--help` lists them all.

| Flag | Default | Meaning |
|---|---|---|
| `--block-size` | 16 | tokens per KV block (the page size) |
| `--num-blocks` | profiled | KV blocks; required off-GPU |
| `--gpu-memory-utilization` | 0.90 | share of GPU memory the engine may use |
| `--max-num-seqs` | 256 | running sequences per iteration |
| `--max-batch-tokens` | 2048 | token budget per iteration (raised to `max_model_len` without chunking) |
| `--enable-chunked-prefill` | off | split long prompts across iterations, decodes first |
| `--enable-prefix-caching` | off | share full KV blocks between requests with a common prefix |
| `--preemption-mode` | recompute | `recompute` or `swap` |
| `--admission-policy` | fcfs | `fcfs`, `sjf`, or `priority` (SJF with aging, `--aging-rate`) |
| `--attention-backend` | auto | `naive` (oracle) or `flashinfer` (CUDA) |
| `--cuda-graphs` | off | capture decode steps as CUDA graphs |
| `--debug-invariants` | off | check block-manager invariants after every step |
| `--step-log` | none | append per-step scheduler stats to a CSV |

## API

`POST /v1/completions` follows the OpenAI schema, streaming or not, and accepts the extensions vLLM's
benchmark client sends: `ignore_eos`, `stream_options.include_usage`, token-id prompts, and `priority`.
Unsupported parameters (`n > 1`, logprobs, echo, penalties, `top_k`, `seed`) return a 400 rather than
being ignored. Also `GET /v1/models`, `/health`, and `/stats` (queue depths, KV usage, preemptions,
prefix-cache hit rate, peak memory).

## Correctness gates

| # | Gate | Test |
|---|---|---|
| 1 | Greedy output matches transformers, 20 prompts | `tests/test_correctness.py` |
| 2 | Paged output matches the contiguous-cache path (block sizes 8/16/32) | `tests/test_correctness.py` |
| 3 | Block invariant holds after every step | every engine test; `tests/test_block_leak.py` |
| 4 | 50 concurrent requests match their solo runs, zero leaked blocks | `tests/test_block_leak.py` |
| 5 | Client disconnect frees blocks | `tests/test_server.py` |
| 6 | A single sequence that outgrows the cache fails cleanly | `tests/test_block_leak.py` |
| 7 | Prefix caching on and off give identical output | `tests/test_prefix_cache.py` |
| 8 | Chunked and unchunked prefill give identical output | `tests/test_chunked_prefill.py` |
| 9 | FlashInfer matches the naive path within bf16 tolerance | `tests/test_gpu.py` |

"Identical" means token for token in fp32, except where the reference's top two logits are within
numerical noise of each other; see `tests/greedy_check.py` and DESIGN.md.

```bash
.venv/bin/python -m pytest -m "not model"   # scheduler, sampler, block manager: seconds, no weights
.venv/bin/python -m pytest -m "not gpu"     # gates 1-8 on the real model: 95 tests, about 7 minutes on an M2 laptop
PAGEDSERVE_TEST_DEVICE=cuda .venv/bin/python -m pytest   # everything, on a GPU
```

Model-backed tests skip, and never download, when the weights are not in the local Hugging Face cache.
The first run also builds the transformers reference outputs for gate 1 (cached under `tests/.cache/`),
which adds several minutes once.

## Benchmarks

`bench/` replays ShareGPT-length or lognormal workloads with Poisson arrivals against any
OpenAI-compatible server, so this engine and vLLM are measured by the same client on the same traffic.
It records TTFT, inter-token latency, end-to-end latency, throughput (whole run and steady state),
KV utilization, preemptions and peak memory. The sweeps in `bench/sweeps/` cover latency against
throughput, block size, admission policy, preemption strategy, prefix length and chunked prefill;
`bench/plot.py` turns the CSVs into figures and tables. No numbers are quoted here until they are
measured on a GPU.

## Layout

```
engine/
  config.py              EngineConfig: every setting, and the CLI flags
  model/qwen3.py         Qwen3 forward pass
  model/loader.py        strict safetensors loading
  model/attention.py     KV-cache layout; contiguous, naive paged and FlashInfer backends
  core/sequence.py       request state
  core/block_manager.py  paged allocator, prefix cache, swap pool, invariants
  core/scheduler.py      continuous batching, admission, preemption
  core/model_runner.py   batch -> flat tensors -> forward -> logits; swaps; memory profiling
  core/cuda_graphs.py    captured decode steps
  core/sampler.py        batched sampling, incremental detokenization, stop strings
  core/engine.py         step(), add_request(), abort()
  api/server.py          FastAPI app, SSE streaming, the async bridge
  api/protocol.py        request and response schemas
bench/                   workloads, client, sweeps, baselines, crossover microbenchmark, profiler, plots
tests/                   the gates, plus a weight-free toy model for scheduler stress tests
scripts/gpu_suite.sh     the GPU half, end to end
docs/                    DESIGN, BUGLOG, GPU_RUNBOOK, WRITEUP
```
