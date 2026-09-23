# PagedServe — Implementation Plan

**An LLM inference engine with paged KV-cache memory management and continuous batching.**

Repo: `paged-serve`

A single-GPU LLM serving engine implementing the core mechanics behind vLLM: paged KV-cache
memory management, iteration-level continuous batching, prefix caching, chunked prefill, and a
streaming OpenAI-compatible HTTP API.

**Scope boundary:** everything inside one replica. Multi-replica routing, autoscaling, and
load balancing are explicitly out of scope (that is the cluster layer, handled by a separate
project). This engine is the thing a cluster-layer router would put N copies of behind it.

**Domain:** operating systems — virtual memory and scheduling — applied to LLM serving.
Nearly zero machine learning. You need to know mechanically what attention does so you know
what shape the KV tensors are. That is the extent of it.

---

## Table of contents

1. [Environment](#1-environment)
2. [Architecture](#2-architecture)
3. [Repo layout](#3-repo-layout)
4. [Component 1: model forward pass](#4-component-1-model-forward-pass)
5. [Component 2: block manager](#5-component-2-block-manager-paged-kv-cache)
6. [Component 3: sequence state](#6-component-3-sequence-state)
7. [Component 4: scheduler](#7-component-4-scheduler-continuous-batching)
8. [Component 5: model runner](#8-component-5-model-runner-batched-forward)
9. [Component 6: sampler](#9-component-6-sampler)
10. [Component 7: engine loop](#10-component-7-engine-loop)
11. [Component 8: async API server](#11-component-8-async-api-server)
12. [Component 9: prefix caching](#12-component-9-prefix-caching)
13. [Component 10: chunked prefill](#13-component-10-chunked-prefill)
14. [Component 11: paged attention kernel](#14-component-11-paged-attention-kernel)
15. [Benchmarking](#15-benchmarking)
16. [Correctness gates](#16-correctness-gates)
17. [Debugging playbook](#17-debugging-playbook)
18. [Schedule](#18-schedule)
19. [Write-up outline](#19-write-up-outline)

---

## 1. Environment

**Hardware.** One consumer GPU with 16–24 GB. Rent rather than fight a local install: consumer
cards run roughly $0.25–0.35/hour on Vast.ai, RunPod, or SaladCloud. Total project compute
should land near $15. Do not rent datacenter cards; you do not need one for a 0.6B model and
they cost 8× more.

**Model.** `Qwen/Qwen3-0.6B-Base`. Small enough that iteration is fast, large enough that the
KV cache is a real constraint at high concurrency.

**Dependencies.** Deliberately short:

```
torch
transformers      # tokenizer + correctness reference only, not for inference
safetensors
flash-attn        # week 12, not before
fastapi
uvicorn
numpy
matplotlib
```

No Docker, no Kubernetes, no Prometheus, no Ray, no Grafana. Every one of those is either
cluster-layer concern or ceremony that produces no graph you cannot produce with a CSV and
matplotlib. Config is a dataclass, not a YAML framework.

**Reference reading.** Read the vLLM paper for the idea, then *Operating Systems: Three Easy
Pieces* chapters 18–22 (paging) and 7–9 (scheduling) for the actual concepts. Read nano-vLLM's
source only **after** you have your own version working — otherwise you will absorb its
decisions instead of making your own, and you will have nothing to say when asked why you built
it a particular way.

---

## 2. Architecture

```
   HTTP clients
        │
        ▼
┌──────────────────────────────────────────────────┐
│  FastAPI handlers (async)                        │
│    POST /v1/completions   → SSE stream           │
│    - tokenize, build Sequence                    │
│    - push to request queue                       │
│    - await per-request output queue              │
│    - on disconnect → engine.abort(seq_id)        │
└───────────────┬──────────────────────────────────┘
                │  asyncio.Queue in / per-seq Queue out
┌───────────────▼──────────────────────────────────┐
│  Engine loop (synchronous, never awaits)         │
│                                                  │
│   while True:                                    │
│     batch   = scheduler.schedule()   ← policy    │
│     logits  = model_runner.execute(batch)        │
│     tokens  = sampler.sample(logits, batch)      │
│     outputs = scheduler.update(tokens)           │
│                                                  │
│   ┌──────────────┐  ┌───────────────┐            │
│   │  Scheduler   │◄─┤ BlockManager  │            │
│   │ waiting/     │  │ free list,    │            │
│   │ running sets │  │ block tables, │            │
│   │ preemption   │  │ refcounts,    │            │
│   │ token budget │  │ prefix hashes │            │
│   └──────────────┘  └───────────────┘            │
│                                                  │
│   ┌──────────────────────────────────┐           │
│   │ ModelRunner → paged attention    │           │
│   │ KV cache: [num_blocks, block_sz, │           │
│   │            kv_heads, head_dim]    │          │
│   └──────────────────────────────────┘           │
└──────────────────────────────────────────────────┘
```

### The central design decision

**The engine loop is synchronous and must never block.** `step()` advances every running
sequence by exactly one token and returns. If a single iteration stalls, latency is added to
every request currently decoding, not just the one that caused the stall. This is why the
frontend is async and the engine is not, and why they communicate through queues rather than
direct calls.

This split is the single most important thing to understand in the project, and the thing most
people who have only read the paper do not know exists.

---

## 3. Repo layout

```
engine/
  config.py            # dataclass: model path, block_size, gpu_util, max_batch_tokens
  model/
    qwen3.py           # forward pass, ~250 lines
    loader.py          # safetensors → state dict → your modules
    attention.py       # naive paged (reference) + flash paged (fast)
  core/
    block_manager.py   # free list, block tables, refcounts, prefix hash map
    sequence.py        # Sequence, SequenceStatus, SamplingParams
    scheduler.py       # admission, preemption, batch construction
    sampler.py         # temperature, top-p, stop conditions
    model_runner.py    # batch → flat tensors → forward → logits
    engine.py          # step(), add_request(), abort()
  api/
    server.py          # FastAPI, SSE, async engine wrapper
    protocol.py        # request/response pydantic models
bench/
  workload.py          # Poisson arrival generator
  run_bench.py         # sweep arrival rates, emit CSV
  plot.py              # CSV → figures
tests/
  test_correctness.py  # vs transformers, greedy
  test_block_leak.py   # invariant checks
  test_prefix_cache.py # shared-prefix correctness
```

Roughly 1,500 lines total. nano-vLLM does comparable scope in ~1,200, so this is calibrated,
not aspirational.

---

## 4. Component 1: model forward pass

Write your own. You need to own the attention module anyway because HuggingFace's is entangled
with its own cache abstraction, and once you own attention you may as well own the 200 lines
around it.

### Qwen3-0.6B configuration

```
hidden_size            1024
intermediate_size      3072
num_hidden_layers      28
num_attention_heads    16
num_key_value_heads    8       ← GQA, 2:1 ratio
head_dim               128
vocab_size             151936
rope_theta             1000000
rms_norm_eps           1e-6
hidden_act             silu
tie_word_embeddings    true    ← verify in the actual config.json you download
```

### Three traps specific to this model

**1. `head_dim` is decoupled from `hidden_size / num_heads`.** Note that
`16 × 128 = 2048 ≠ 1024`. So `q_proj` maps 1024 → 2048, and `o_proj` maps 2048 → 1024. If you
assume `head_dim = hidden_size // num_heads` you will get 64 and the shapes will not line up.

**2. Qwen3 applies QK-norm.** There is an RMSNorm over `head_dim` applied per-head to Q and to
K, **before** RoPE. The weights are in the state dict as `q_norm` and `k_norm`. Miss this and
the model produces fluent-looking garbage rather than an obvious crash, which is the worst
possible failure mode.

**3. GQA head expansion.** 8 KV heads serve 16 query heads. Either `repeat_interleave` the KV
heads by 2 before attention, or use a kernel that handles GQA natively. FlashAttention does the
latter. Get the naive version working first with an explicit repeat so you can see the shapes.

### Build order

Write the model with a plain contiguous KV cache, one sequence, no batching. Load weights from
safetensors. Generate greedily.

**Do not proceed until greedy output at temperature 0 matches `transformers` token-for-token
for at least 20 prompts of 100+ tokens.** A scheduler bug sitting on top of a silently wrong
model is essentially unfindable — you will chase a memory-management ghost for days when the
real problem is a missing norm. This gate is non-negotiable.

---

## 5. Component 2: block manager (paged KV cache)

This is a page table. Everything here has a direct OS analogue, and naming them that way in
your write-up is worth doing.

### Cache allocation

One tensor pair per layer, or one fused tensor:

```python
# per layer
k_cache = torch.empty(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16)
v_cache = torch.empty(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16)
```

**Sizing math for Qwen3-0.6B**, worth computing yourself once so the numbers are not magic:

```
bytes per token per layer = 2 (K,V) × 8 kv_heads × 128 head_dim × 2 bytes (bf16) = 4,096 B
bytes per token, all 28 layers                                                  = 112 KiB
bytes per block (block_size = 16)                                               = 1.75 MiB
```

On a 24 GB card with ~1.2 GB of weights and ~2 GB of activation headroom, ~20 GB is available
for cache → roughly 11,400 blocks → roughly 182,000 tokens resident. That is a lot of
concurrency, which is exactly what you want for the benchmarks to be interesting.

Determine `num_blocks` at startup by profiling: run one dummy forward at max batch size, read
`torch.cuda.max_memory_allocated()`, subtract from `total × gpu_memory_utilization`, divide by
bytes-per-block. Do not hardcode it.

### Core structure

```python
class BlockManager:
    free_blocks: deque[int]              # free list
    ref_counts: dict[int, int]           # for CoW and prefix sharing
    hash_to_block: dict[int, int]        # prefix cache, week 12
    block_size: int = 16                 # vLLM's default; make it a swept parameter
```

`block_size` is the page-size tradeoff: smaller means less internal fragmentation but longer
block tables and more gather overhead. Sweep it (8/16/32) and put the curve in your write-up.
This is a nice, cheap result that most reimplementations skip.

### Writing KV into paged storage

The slot-mapping trick, which is what vLLM's `reshape_and_cache` kernel does:

```python
# for a token at position `pos` in a sequence with `block_table`
block_idx = pos // block_size
offset    = pos %  block_size
slot      = block_table[block_idx] * block_size + offset

# flatten the cache to [num_blocks * block_size, kv_heads, head_dim] once at startup,
# then a whole batch of tokens is one scatter:
k_cache_flat.index_copy_(0, slot_tensor, k_new)
v_cache_flat.index_copy_(0, slot_tensor, v_new)
```

Compute `slot_tensor` for the entire batch on CPU during batch construction, transfer once.
Doing this per-sequence in a Python loop on the GPU will dominate your step time.

### Invariant

Assert this after every step in debug mode. It catches leaks immediately rather than 40 minutes
into a benchmark run when the pool mysteriously starves:

```python
assert len(free_blocks) + sum(len(s.block_table) for s in all_live_seqs) == num_blocks
```

---

## 6. Component 3: sequence state

```python
class SequenceStatus(Enum):
    WAITING, RUNNING, PREEMPTED, FINISHED, ABORTED

@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    block_table: list[int]
    status: SequenceStatus
    num_computed_tokens: int      # < len(prompt) while prefill is in progress
    sampling_params: SamplingParams
    arrival_time: float
    first_token_time: float | None
```

`num_computed_tokens` is what makes chunked prefill possible later. Put it in from day one even
though nothing uses it in week 10 — retrofitting it means touching every scheduler path.

**Abort matters more than it sounds.** When a client disconnects mid-stream you must free that
sequence's blocks and drop it from the running set. Get this wrong and blocks leak until the
pool starves, and the symptom appears long after the cause. This is also your OS process state
diagram, which is worth naming as such in the write-up.

---

## 7. Component 4: scheduler (continuous batching)

The heart of the project. Called once per iteration.

### Version 1 — prefill-priority (build this first)

Simpler and matches vLLM's original design:

```python
def schedule(self) -> Batch:
    # 1. try to admit waiting requests
    scheduled_prefills = []
    while self.waiting:
        seq = self.waiting[0]
        blocks_needed = ceil(len(seq.prompt_token_ids) / block_size)
        if not self.block_manager.can_allocate(blocks_needed):
            break
        if tokens_in_batch + len(seq.prompt_token_ids) > max_batch_tokens:
            break
        self.block_manager.allocate(seq)
        scheduled_prefills.append(self.waiting.popleft())

    if scheduled_prefills:
        return Batch(prefills=scheduled_prefills)   # prefill-only iteration

    # 2. otherwise decode every running sequence by one token
    for seq in list(self.running):
        while not self.block_manager.can_append_slot(seq):
            if len(self.running) > 1:
                self.preempt(self.running.pop())    # evict the newest
            else:
                self.preempt(seq)                   # nothing else to evict
                break
        else:
            self.block_manager.append_slot(seq)
    return Batch(decodes=self.running)
```

### The deadlock case

If exactly one sequence is running and there is no free block for its next token, there is
nothing left to preempt. You must handle this explicitly: either swap its cache to host memory,
or fail the request with a clear error. Silently hanging here is a real bug that real engines
have shipped. Write a test that forces it (tiny `num_blocks`, one long request).

### Preemption strategies — one of your headline experiments

| Strategy | Mechanism | Cost | Best when |
|---|---|---|---|
| **Recompute** | Free blocks, return seq to front of waiting queue, redo prefill later | O(prompt length) GPU work | Short sequences |
| **Swap** | Copy blocks to a pinned host buffer, restore on resume | O(bytes) PCIe transfer both ways | Long sequences |

There is a crossover point in sequence length. Find it empirically and plot it. This is a
genuinely interesting result and almost nobody's reimplementation measures it.

### Admission policies — your other headline experiment

Implement all three behind one interface, sweep on identical traffic:

- **FCFS** — fair, but a long request head-of-line-blocks short ones. Watch p99 TTFT degrade.
- **SJF** (shortest prompt first) — better mean latency, starves long requests. Show the
  starvation in a tail-latency plot, do not just assert it.
- **Priority queue** — with an aging term to bound starvation.

This section is where an OS course pays off directly, and where you can say something concrete
that is not just "I implemented the paper."

---

## 8. Component 5: model runner (batched forward)

Converts a scheduler `Batch` into flat tensors and runs the forward pass.

**Flatten, do not pad.** All tokens across all sequences go into one 1-D tensor, with
`cu_seqlens` (cumulative sequence lengths) marking boundaries. This is the varlen convention
FlashAttention expects, and it avoids wasting compute on padding.

```
3 sequences with 4, 2, 7 tokens →
  input_ids  : [t0..t3, t0..t1, t0..t6]     shape [13]
  cu_seqlens : [0, 4, 6, 13]                shape [4]
  positions  : [0,1,2,3, 0,1, 0,1,2,3,4,5,6]
  slot_map   : [13 slot indices into the flattened cache]
```

**Position IDs are a classic paging bug.** Positions are the token's index *within its own
sequence*, not its index in the batch and not its slot in the cache. RoPE consumes these. Get
it wrong and output degrades subtly with longer contexts rather than failing loudly.

**Build all index tensors on CPU, transfer once per step.** Any `.item()`, `.tolist()`, or
`.cpu()` call inside the hot loop forces a GPU sync and will silently halve your throughput.
This is the single most common performance bug in hand-rolled engines, and finding one with
`torch.profiler` is a good paragraph in your write-up.

---

## 9. Component 6: sampler

Keep it lean: greedy, temperature, top-p, `max_tokens`, EOS, stop strings.

```python
logits = logits / temperature            # temperature == 0 → argmax path instead
probs  = softmax(logits, dim=-1)
# top-p: sort desc, cumsum, mask everything past the threshold, renormalize
next_token = torch.multinomial(probs, 1)
```

**Sample the whole batch in one kernel.** Per-sequence Python loops over `torch.multinomial`
will cost more than the model forward at high batch sizes.

### The detokenization trap

You cannot decode tokens independently and concatenate — BPE tokens do not map cleanly to
character boundaries, and multi-byte UTF-8 characters routinely span two tokens. Naively you
emit replacement characters mid-stream.

Simplest correct approach: keep the full token list, decode all of it, emit the delta versus
what you last emitted. Slightly wasteful, obviously correct. Optimize with a sliding window only
if profiling says it matters (it probably will not at 0.6B).

Stop strings must be checked against decoded *text*, not token IDs, since a stop string can
tokenize differently depending on what precedes it.

---

## 10. Component 7: engine loop

```python
class Engine:
    def step(self) -> list[RequestOutput]:
        batch = self.scheduler.schedule()
        if batch.is_empty():
            return []
        logits  = self.model_runner.execute(batch)
        tokens  = self.sampler.sample(logits, batch)
        return self.scheduler.update(tokens)   # append, check stop, free finished

    def add_request(self, seq: Sequence) -> None:
        self.scheduler.waiting.append(seq)

    def abort(self, seq_id: int) -> None:
        self.scheduler.abort(seq_id)           # free blocks, drop from running
```

That is the entire engine. Everything else is the four objects it calls. If `step()` grows past
~20 lines, logic has leaked into it that belongs in the scheduler.

---

## 11. Component 8: async API server

### What "adding an HTTP layer" means here

It means the engine is reachable over `POST http://localhost:8000/v1/completions` and streams
tokens back as server-sent events. It runs on localhost. You are not deploying it publicly, and
you should not — an unauthenticated LLM endpoint on the open internet is a bad idea and adds
nothing to the project.

The reason to do it is architectural, not deployment. Three concrete payoffs:

1. **It forces the async/sync split**, which is the design decision that separates a serving
   engine from a batch inference script. Without it you never build the request lifecycle.
2. **It forces you to handle abort**, streaming backpressure, and per-request state — the
   things that only exist when requests arrive independently rather than as a fixed list.
3. **It makes your benchmarks directly comparable.** If you match the OpenAI schema, you can
   point vLLM's own `benchmark_serving.py` at your server and at real vLLM with identical
   traffic and identical measurement code. That removes any argument that your numbers were
   generated by a harness tuned to flatter you. This alone justifies the schema compatibility.

### The bridge

```python
class AsyncEngine:
    def __init__(self, engine):
        self.engine = engine
        self.output_queues: dict[int, asyncio.Queue] = {}
        self.has_new_work = asyncio.Event()

    async def run_loop(self):
        while True:
            if not self.engine.has_work():
                await self.has_new_work.wait()      # idle without spinning
                self.has_new_work.clear()
            for out in self.engine.step():
                self.output_queues[out.seq_id].put_nowait(out)
            await asyncio.sleep(0)                  # yield so handlers get scheduled

    async def generate(self, seq):
        q = asyncio.Queue()
        self.output_queues[seq.seq_id] = q
        self.engine.add_request(seq)
        self.has_new_work.set()
        try:
            while True:
                out = await q.get()
                yield out
                if out.finished:
                    return
        finally:
            self.engine.abort(seq.seq_id)           # runs on disconnect too
            self.output_queues.pop(seq.seq_id, None)
```

The `finally` block is the whole abort story: FastAPI cancels the generator when the client
disconnects, which raises `CancelledError` at the `await`, which runs `finally`, which frees the
blocks. Test it by killing a `curl` mid-stream and asserting the block invariant still holds.

`await asyncio.sleep(0)` between steps is what keeps the HTTP handlers responsive. Start here.
If you later want the engine on a dedicated thread, `loop.call_soon_threadsafe` is the bridge —
but only do that if profiling shows the yield is actually costing you.

---

## 12. Component 9: prefix caching

Page sharing and deduplication, under a different name.

```python
# hash chains so identical suffixes under different prefixes do not collide
block_hash = hash((parent_block_hash, tuple(token_ids_in_this_block)))
```

Rules:

- **Only full blocks are hashable.** A partial block is still mutable, so it cannot be shared.
- On allocation, look up each full block's hash. On hit, bump the refcount, reuse the block ID,
  and advance `num_computed_tokens` past it — that prefix is now free.
- **Including the parent hash is mandatory.** Without it, the same 16 tokens appearing under two
  different prefixes collide and you serve corrupted context. This bug produces plausible
  output, so it will not announce itself.
- Eviction: LRU over blocks with refcount zero.

**Benchmark it honestly.** Construct a workload with a long shared system prompt (say 512
tokens) plus short unique suffixes, and sweep the shared-prefix length. Report TTFT improvement
as a curve against prefix length, not as a single number from your best-case workload.

---

## 13. Component 10: chunked prefill

Without it, a 4,000-token prompt monopolizes an iteration and every sequence already decoding
waits for it. That shows up directly as a p99 inter-token latency spike.

With it, you cap tokens per iteration (`max_batch_tokens`, e.g. 2048) and split long prefills
across steps, filling leftover budget with decode tokens:

```
step N:   [prefill chunk: 2000 tokens of seq A] + [decode: 48 sequences × 1 token]
step N+1: [prefill chunk: next 2000 of seq A]   + [decode: 48 sequences × 1 token]
step N+2: [prefill chunk: final 500 of seq A → samples a token] + [decode: 49 × 1]
```

Implementation notes:

- Advance `num_computed_tokens` by the chunk size each step.
- **Only the final chunk samples a token.** Intermediate chunks populate the KV cache and
  produce no output. Sampling from a mid-prefill chunk is a real bug people hit.
- Attention must handle a query of length 2000 against a KV cache holding 2000 already-computed
  tokens from earlier chunks. The causal mask has to account for the offset.

**The money plot:** p99 ITL over time, with and without chunked prefill, on a workload mixing
long prompts into a steady stream of short decodes. The spikes disappear. It is a visually
obvious result and it demonstrates you understand *why* the optimization exists rather than that
you copied it.

---

## 14. Component 11: paged attention kernel

Build two implementations and keep both.

**Naive (week 10, keep forever).** Gather each sequence's blocks into a contiguous tensor, call
`scaled_dot_product_attention`. Slow, obviously correct, and it is your reference oracle — every
optimization after this is validated against it.

**FlashAttention paged (week 12).** `flash_attn_with_kvcache` accepts a `block_table` directly
and handles GQA natively. Signatures have shifted across releases, so check the one you install
rather than trusting any snippet, including this document's:

```python
out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=k_new, v=v_new,          # appends into the cache for you
    cache_seqlens=seq_lens,
    block_table=block_table,   # [batch, max_blocks_per_seq], int32
    causal=True,
)
```

**Gate:** outputs must match the naive path within bf16 tolerance before you trust any number
the fast path produces. Then measure and report the speedup — that is a real, earned benchmark
result.

**CUDA graphs (stretch).** At 0.6B, kernel launch overhead is a meaningful fraction of each
decode step, so capturing decode at bucketed batch sizes gives a genuine measurable win. Needs
static input buffers, which is real work. Only after everything above is done.

---

## 15. Benchmarking

### Workload generator

Poisson arrivals at rate λ. Sample request lengths from a realistic distribution — a ShareGPT
trace is standard, or a lognormal fitted to one. **Do not use uniform lengths**; uniform hides
exactly the head-of-line blocking your scheduler experiments are supposed to expose.

### Metrics

Every one of these is a measurement, not arithmetic on a constant you chose:

| Metric | How | Why it matters |
|---|---|---|
| Throughput | output tokens/sec at steady state | headline number |
| TTFT | p50/p95/p99 | interactive responsiveness |
| ITL | p50/p95/p99 | streaming smoothness |
| KV utilization | live blocks / total, sampled per step | memory efficiency |
| Preemption rate | preemptions/sec vs λ | scheduler pressure |
| Peak memory | `torch.cuda.max_memory_allocated()` | real, not modeled |

### The plot that defines the project

**Latency versus throughput.** Sweep λ, plot p99 TTFT on the y-axis against achieved throughput
on the x-axis. Every serving system produces a hockey stick: flat while the system absorbs load,
then a knee where the queue grows without bound. Where your knee sits relative to vLLM's is the
single most informative result you will produce.

### Baselines

- **vLLM**, same model, same GPU, same traffic. Expect to be slower. Being within a factor of
  1.5–2 on a 0.6B model is a good result and an honest one.
- **HuggingFace `generate()` with static batching.** Expect to beat this decisively. This is the
  comparison that shows what continuous batching buys.

Report both. Only showing the baseline you beat is the thing that makes an interviewer stop
trusting the rest of your numbers.

### Sweeps worth running

`block_size` ∈ {8, 16, 32} · admission policy ∈ {FCFS, SJF, priority} · preemption ∈ {recompute,
swap} × sequence length · prefix caching on/off × shared-prefix length · chunked prefill on/off ×
prompt-length mix.

---

## 16. Correctness gates

Do not advance past a failing gate. Each one exists because the bug it catches is much harder to
find later.

| # | Gate | Catches |
|---|---|---|
| 1 | Greedy output matches `transformers` exactly, 20 prompts | model bugs (QK-norm, RoPE, GQA) |
| 2 | Paged output matches contiguous-cache output | block table / slot mapping |
| 3 | Block invariant holds after every step | leaks |
| 4 | 50 concurrent requests all correct, zero leaked blocks | scheduler state |
| 5 | Client disconnect frees blocks | abort path |
| 6 | Single-sequence OOM fails cleanly, no hang | preemption deadlock |
| 7 | Shared-prefix requests produce identical output with cache on/off | hash collisions |
| 8 | Chunked and unchunked prefill produce identical output | mask offset |
| 9 | Flash path matches naive path within bf16 tolerance | kernel misuse |

---

## 17. Debugging playbook

**Output is fluent but wrong.** Almost always the model, not the cache. Check QK-norm is applied
per-head before RoPE. Check `head_dim=128` not `hidden_size//num_heads=64`. Check GQA repeat
factor. Check position IDs are per-sequence.

**Output degrades only at long context.** Position IDs or the causal mask offset in chunked
prefill. Compare against the naive path at increasing lengths to find where they diverge.

**Blocks leak.** Enable the invariant assert every step. Then check the paths that free blocks:
normal finish, EOS, `max_tokens`, stop string, abort, preemption. One of them is missing a
`free()`. Abort is the usual culprit.

**Throughput far below expectation.** Profile with `torch.profiler` before theorizing. Look for
GPU syncs from `.item()` in the loop, per-sequence Python loops that should be batched tensor
ops, and index tensors being rebuilt on GPU instead of CPU.

**Hangs under load.** The single-sequence preemption deadlock. Log scheduler state every step
and find the iteration where `running` stops changing.

**Nondeterminism at temperature 0.** Usually a batch-composition dependency — the same sequence
producing different logits depending on what else is in the batch. Small bf16 reduction-order
differences are expected; anything that changes the argmax is a bug.

---

## 18. Schedule

**Week 10 — the model is correct.**
Forward pass, weight loading, contiguous cache, single sequence. Then block manager, slot
mapping, naive paged attention. *Exit: gates 1 and 2.*

**Week 11 — it is a server.**
Sequence state machine, scheduler with preemption, batched model runner, batched sampler, engine
loop, FastAPI, SSE streaming, abort. *Exit: gates 3, 4, 5, 6 — `curl` streams tokens, 50
concurrent requests, no leaks.*

**Week 12 — it is fast.**
Prefix caching, chunked prefill, FlashAttention paged kernel. CUDA graphs if time allows.
Benchmark harness. *Exit: gates 7, 8, 9, plus a measured win for each optimization on a workload
chosen to expose it.*

**Week 13 — it is a portfolio piece.**
Admission policy sweep, preemption strategy sweep, block-size sweep, vLLM and HF baselines,
latency-throughput curves, write-up. Read nano-vLLM and vLLM's scheduler *now* and add a section
on what they do differently and why.

If you slip, cut in this order: CUDA graphs → block-size sweep → chunked prefill → prefix
caching. **Never cut week 13.** The measurement work is the differentiator; without it you have
a reimplementation, which is a much weaker artifact.

---

## 19. Write-up outline

The blog post is not optional — it is what makes the repo legible to someone who will not read
1,500 lines of your code.

1. **What this is and is not.** State plainly that it is a reimplementation for learning, that
   nano-vLLM and vLLM exist, and that you read them after building. Claiming novelty is the
   fastest way to lose a reader who knows the space.
2. **The OS framing.** Block table as page table, block size as page size, eviction as page
   replacement, preemption as context switching, prefix sharing as page dedup. This is the
   section that shows you understand the ideas rather than the API.
3. **Architecture**, with the async/sync split and why the engine loop must never block.
4. **Results.** Latency-throughput curves against both baselines. Every sweep. The preemption
   crossover point.
5. **Three bugs and how you found them.** Pick real ones — a GPU sync found in the profiler, the
   prefix hash collision, the preemption deadlock. Debugging narrative is more convincing
   evidence of competence than a clean architecture diagram, because anyone can draw the diagram.
6. **What you would do next and what you deliberately left out**, with reasons.

---

## Resume bullet

Fill the placeholders with real measured numbers before using it. If a number is not measured,
delete that clause rather than estimating.

> **PagedServe** — Built a single-GPU LLM inference engine in PyTorch pairing a paged KV-cache
> allocator with a continuous-batching **scheduler** that admits and preempts requests at token
> granularity; added prefix caching and chunked prefill behind a streaming OpenAI-compatible API,
> reaching **X%** of vLLM's throughput on Qwen3-0.6B and cutting p99 inter-token latency **Z%**
> versus static batching.

Shorter variant, if the above wraps past two lines:

> Built a single-GPU LLM inference engine in PyTorch — paged KV-cache allocator plus a
> continuous-batching **scheduler** with token-level admission and preemption — hitting **X%** of
> vLLM's throughput on Qwen3-0.6B and cutting p99 inter-token latency **Z%** vs. static batching.

Notes:

- The HuggingFace static-batching comparison is deliberately cut. Beating it by a large multiple
  is expected rather than impressive, so it is the weakest of the three numbers and the first
  thing to drop for space. Keep it in the write-up.
- "Scheduler" alone is a generic term; "admits and preempts at token granularity" is what makes
  it specific to this domain and proves you know what continuous batching actually means. Do not
  cut that clause to save the word count.
- Two lines maximum on the resume. If it runs to three, cut the prefix-caching clause next.
- If your resume format doesn't use named project headers, drop "PagedServe" and start at
  "Built a single-GPU LLM inference engine…" — the mechanics carry the bullet, not the codename.
