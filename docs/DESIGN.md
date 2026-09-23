# PagedServe design

The decisions behind the engine, what each one trades away, and where it differs from
`pagedserve-plan.md`. Read this before the code: every choice below has an alternative that someone
will ask about.

## The one idea: a synchronous engine behind an async front end

`LLMEngine.step()` (`engine/core/engine.py`) advances every scheduled sequence by one iteration and
returns. It never awaits and never blocks on anything but the GPU:

```python
batch   = scheduler.schedule()      # which sequences run, how many tokens each computes
logits  = model_runner.execute(batch)
tokens  = sampler(logits, sampling)
outputs = scheduler.update(batch, tokens)   # append, detokenize, check stops, free finished
```

Any stall inside a step is added to the latency of every request currently decoding, not just the one
that caused it. That is why HTTP handling is async and the engine is not, and why they talk only
through queues: `AsyncEngine.run_loop` (`engine/api/server.py`) runs `step()` as a task on the event
loop, yields once between steps so handlers can run, and pushes each output into its request's
`asyncio.Queue`. Starting the engine on a dedicated thread (`loop.call_soon_threadsafe` as the bridge)
is the next step only if profiling shows the per-step yield costs anything.

## Operating-systems vocabulary

| Serving concept | OS analogue |
|---|---|
| KV block | physical page frame |
| `block_size` | page size |
| a sequence's `block_table` | its page table |
| `slot = block_table[pos // bs] * bs + pos % bs` | virtual-to-physical address translation |
| free block list | free-frame list |
| refcount > 1 (shared prefix blocks) | shared pages / copy-on-write |
| prefix cache (hash -> block) | page deduplication |
| LRU of cached, unreferenced blocks | page cache and page replacement |
| preemption by recompute | killing a process and restarting it from its inputs |
| preemption by swap | swapping a process's pages to disk |
| `WAITING / RUNNING / PREEMPTED / FINISHED / ABORTED` | the process state diagram |
| admission policy (FCFS, SJF, priority with aging) | CPU scheduling, and starvation |
| scheduler watchdog on "work pending, nothing scheduled" | deadlock detection |

## Model (`engine/model/qwen3.py`, `loader.py`)

Our own Qwen3 forward pass, because the attention module has to own the KV cache and transformers'
is entangled with its own cache abstraction. Operation order follows transformers' `modeling_qwen3.py`
exactly (RMSNorm in fp32 then cast, residual additions in the same order, RoPE table built the same
way) so fp32 logits agree to about 3e-5.

The three traps the plan names are handled in the code where they occur:

1. `head_dim` is 128 while `hidden_size / num_heads` is 64: `q_proj` maps 1024 -> 2048 and `o_proj`
   2048 -> 1024.
2. QK-norm: RMSNorm over `head_dim`, per head, on Q and K, before RoPE.
3. GQA: 8 KV heads serve 16 query heads; backends expand (`repeat_interleave`) or handle it natively.

Two departures from transformers, both standard in serving engines: q/k/v and gate/up projections are
fused into single matmuls (fewer kernel launches per layer), and the model takes a flat token batch
(`[T]` ids and positions) rather than padded `[batch, seq]` tensors. The loader is strict: every
checkpoint tensor must be consumed and every parameter filled, because a silently skipped weight
produces fluent garbage rather than a crash.

## KV cache layout (`engine/model/attention.py`)

One tensor: `[num_layers, 2, num_blocks + 1, block_size, num_kv_heads, head_dim]`.

- Per-layer K and V views are contiguous `[num_blocks + 1, block_size, kv_heads, head_dim]`, which is
  FlashInfer's NHD paged layout, so FlashInfer reads our cache directly.
- Flattening a layer's first two dimensions gives `[slots, kv_heads, head_dim]`, so a batch's new K/V
  are written with one `index_copy_` per layer (the slot-mapping trick; vLLM's `reshape_and_cache`).
- Swapping gathers along the block dimension, which moves every layer of a block in one operation.
- Block `num_blocks` is a scratch page the block manager never hands out: padded rows of a CUDA-graph
  batch write there.

Alternative: one tensor per layer (vLLM's choice). Rejected because swapping would then need one copy
per layer per direction.

Sizing, for Qwen3-0.6B in bf16:

```
per token, per layer   2 (K, V) x 8 kv_heads x 128 head_dim x 2 bytes = 4,096 B
per token, 28 layers                                                   = 112 KiB
per block of 16 tokens                                                 = 1.75 MiB
```

A 24 GB card leaves roughly 20 GB for the cache after weights and activations: on the order of 11,000
blocks, or 180,000 tokens resident at once. The engine never hardcodes the count; it measures it at
startup (see the model runner).

## Block manager (`engine/core/block_manager.py`)

Every block is in exactly one pool: free (refcount 0, no cached content), evictable (refcount 0 but
still hashed, so a later request can reuse it; LRU order; counts as free), or in use (refcount >= 1).
The invariant, asserted after every step with `debug_invariants=True`:

```
|free| + |evictable| + |in use| == num_blocks
refcount(b) == number of live block tables containing b, for every block b
```

The plan's `len(free) + sum(len(table)) == num_blocks` is the special case with no sharing; with prefix
caching it is false by design (shared blocks are counted once per table, cached blocks are in no
table). Each running sequence also satisfies `len(block_table) == ceil(num_computed_tokens / block_size)`
at step boundaries. The host swap pool has the same accounting.

### Prefix caching

- Only full blocks whose K/V are computed are published (`cache_full_blocks`, after the forward pass);
  a partial block is still being written.
- A block's hash is `sha256(parent_hash + token_ids)`. The parent chain means the same tokens under two
  different prefixes never share a block. SHA-256 rather than Python's 64-bit `hash()`: a collision
  would silently serve another request's context, the failure that "produces plausible output".
- Lookups never reuse the block holding the prompt's last token: at least one token must be recomputed
  to produce logits, and its K/V must not be written into a block others may be reading.
- A cache hit bumps the refcount; a hit on an evictable block takes it off the LRU list. Admission
  accounting counts those pinned hits against free space.
- Eviction is LRU over evictable blocks, and a finished sequence releases its blocks tail first, so its
  tail is evicted before its prefix (the part other requests are likely to share).
- Identical content computed concurrently by two sequences is published once; the second copy stays
  private and is freed normally.

## Sequence state (`engine/core/sequence.py`)

The plan's fields plus what the scheduler needs. The key one is `num_computed_tokens`: tokens whose K/V
are in the cache. Every scheduling decision is phrased as "compute `n` of this sequence's
`num_tokens - num_computed_tokens` uncomputed tokens; sample only if that reaches the end". A fresh
prefill, a later prefill chunk, a prefill behind a cached prefix, a decode step, and the recompute of a
preempted sequence (prompt plus everything it had generated) are all the same operation.

## Scheduler (`engine/core/scheduler.py`)

### Two modes

**Prefill-priority** (`enable_chunked_prefill=False`, the plan's V1): if nothing is swapped out, admit
waiting requests whole while blocks and the token budget allow; any admission makes the iteration
prefill-only. Otherwise decode every running sequence one token, then bring swapped sequences back if
nothing was preempted.

Without chunking a prefill is all-or-nothing, so any request that can exist must fit one iteration.
The config therefore raises `max_batch_tokens` to at least `max_model_len` in this mode (vLLM's rule);
otherwise a long prompt, or a recompute-preempted sequence whose prompt plus output outgrew the budget,
could never be admitted and would block the head of the queue forever.

**Chunked, decode-first**: running decodes get one token each, then in-progress prefills get chunks of
the remaining budget, then swapped sequences come back, then new requests are admitted (the first chunk
of a long prompt, if that is all the budget allows). Decodes never wait behind a long prompt, which is
the point of chunked prefill. In practice at most one sequence is ever mid-prefill and it is always the
newest, so decode-first and admission order give the same plan; the ordering is still explicit and
tested so the guarantee does not depend on that coincidence.

### Preemption

Victims are the newest running sequences. The plan's pseudocode walks a copy of `running` while evicting
from its end, which can evict the very sequence being extended and then give it a slot; the loop here
walks oldest to newest, evicts from the newest end, and when the victim would be the current sequence,
preempts it and stops.

- **Recompute**: free the blocks, reset `num_computed_tokens`, and park the sequence in `preempted`,
  which is served before any new request, in original order. Its output so far is kept; the resume is a
  prefill over prompt plus generated tokens, with prefix-cache hits if its blocks are still cached.
- **Swap**: copy its blocks to the host pool and park it in `swapped`. Falls back to recompute when host
  space is full. Nothing new is admitted while anything is swapped out, and a step that preempts admits
  nothing, so one step never both swaps out and swaps in (a host block freed by a swap-in could
  otherwise be overwritten before it is read).

The crossover between the two is an experiment (`bench/preemption_crossover.py`): recompute costs
prefill compute, swap costs PCIe bytes.

### Things that must fail instead of hang

- A request whose tokens need more blocks than the whole pool fails at admission.
- A lone running sequence with no free block holds the entire pool; neither preemption nor swapping
  can make room, so it fails with "KV cache exhausted" (gate 6). The plan's "swap it to host memory"
  option cannot work here.
- `LLMEngine.step()` raises, with a scheduler dump, if a step schedules nothing while work is pending.
  Every such state above is handled, so this turns any remaining bug from a hang into an error.

### Admission policies

One heap, three static keys:

| policy | key |
|---|---|
| FCFS | arrival time |
| SJF | prompt length, then arrival |
| priority | `priority + aging_rate * arrival`, `priority` defaulting to the prompt length |

A request's effective priority improves linearly while it waits: `priority - aging_rate * (now -
arrival)`. The `-aging_rate * now` term is common to every waiting request, so ordering by the static key
is exact and the heap never needs re-sorting. With the default priority this is SJF with aging:
`aging_rate = 0` is pure SJF, and a large rate approaches FCFS. The admission loop stops at the first
request that does not fit, so nothing overtakes it; preempted sequences always go first.

A small watermark (1% of blocks) is kept free during admission while anything is running, so freshly
admitted prefills do not immediately force decodes to preempt.

## Model runner (`engine/core/model_runner.py`)

Flatten, don't pad: every scheduled token goes into one 1-D batch with offsets marking sequences. Input
ids, positions, slot mapping, logits indices, and (for the naive backend) each sequence's cache slots
are built with numpy on the CPU and moved in **one** host-to-device copy per step. Positions are indices
within each token's own sequence. Logits are computed only for rows that sample (one per sequence
finishing its chunk), not for every prefill token: at a 151,936-token vocabulary that is the difference
between a few megabytes and gigabytes.

`num_blocks` is measured, never hardcoded: a dummy forward at `max_batch_tokens` plus a sampling pass
over `max_num_seqs` rows gives the activation peak, and the cache gets
`total * gpu_memory_utilization - peak - non-torch overhead`.

## Attention backends

| backend | use | how |
|---|---|---|
| `ContiguousAttention` | gate 1/2 reference | one sequence, per-layer tensors grown by `torch.cat` |
| `NaivePagedAttention` | oracle, CPU default | gather each sequence's slots, repeat KV heads, SDPA with an explicit mask |
| `FlashInferAttention` | CUDA default | `plan()` once per step, `run()` per layer |

The causal mask is aligned bottom-right: query `i` of a chunk of `q` tokens over a cache of `k` tokens
sees keys `j <= i + (k - q)`. With `q == k` that is the usual triangle; with `q < k` (a decode, a later
prefill chunk, a prefill behind cached context) the offset is what keeps earlier context visible.
PyTorch's `is_causal=True` is top-left aligned, which is why the naive path builds the mask explicitly.
FlashInfer's kernel uses `q_idx + kv_len - qo_len` (`include/flashinfer/attention/prefill.cuh`), the same
alignment.

**Why FlashInfer rather than FlashAttention** (the plan's choice): upstream FlashAttention's paged
kernels reject any block size that is not a multiple of 256 (`csrc/flash_attn/flash_api.cpp`, checked in
both `mha_varlen_fwd` and `mha_fwd_kvcache`), which rules out the default of 16 and the 8/16/32 sweep.
FlashInfer accepts any page size, reads the same layout, handles GQA natively, and has a CUDA-graph
decode wrapper.

**CUDA graphs** (`engine/core/cuda_graphs.py`): decode-only batches, one graph per bucketed batch size,
padding rows pointed at the scratch block. FlashInfer's plan lives in fixed buffers that `plan()`
rewrites before each replay; logits are computed outside the graph.

## Sampler and detokenization (`engine/core/sampler.py`)

The whole batch is sampled at once: argmax for temperature-0 rows, temperature scaling, top-p (sort,
cumulative sum, mask, renormalize) and one `torch.multinomial`. The closing `.tolist()` is the step's
only GPU sync; per-row parameter tensors are copied to the device before the forward pass starts.

Detokenization is incremental with a small window (the approach vLLM and TGI use) instead of the plan's
"decode everything each step": decoding the full output every step is O(n^2) per sequence on the same
thread that serves HTTP, and at 0.6B a GPU step is short enough for that to show. Output tokens are
decoded without prompt context, which is exact for byte-level BPE. A trailing incomplete UTF-8 character
is held back until it completes. Stop strings are matched on decoded text, never on token ids, and while
stop strings are set the last `max(len(stop)) - 1` characters are held back, since they could turn out
to be the start of a stop string; the stop string itself is never returned. The full decode is the
test oracle (`tests/test_sampler.py`).

## HTTP layer (`engine/api/`)

OpenAI `/v1/completions`, streaming or not, plus the vLLM extensions its benchmark client sends
(`ignore_eos`, `stream_options.include_usage`, token-id prompts, `priority`). One SSE event per generated
token, so a client's inter-token latency is the engine's. Parameters the engine does not implement
(`n > 1`, logprobs, echo, penalties, `top_k`, per-request seeds) are rejected with a 400 rather than
silently ignored, so a benchmark can never believe a setting applied when it did not.

**Client disconnects.** The plan's story is that Starlette cancels the generator and its `finally` frees
the blocks. Gate 5 showed that is not reliable: Starlette cancels through an anyio cancel scope, anyio
only delivers cancellation to a task blocked on a pending future, and the engine resolves the stream's
queue future every iteration, so the cancellation starved for the whole generation (see
`docs/BUGLOG.md`). The response class therefore aborts the request in the engine directly from the
disconnect signal and hands the generator a terminal output. Non-streaming requests, which Starlette does
not watch at all, get a disconnect watcher that does the same.

## Testing strategy

The gates of the plan, each a test file, run in fp32 against the real weights:

| gate | test |
|---|---|
| 1 | `test_correctness.py::test_greedy_matches_transformers` (plus a ~2,500-token long-context check) |
| 2 | `test_correctness.py::test_paged_matches_contiguous`, block sizes 8/16/32 |
| 3 | `debug_invariants=True` in every engine test; `test_block_leak.py::test_invariant_every_step_mixed_workload` |
| 4 | `test_block_leak.py::test_50_concurrent_requests_match_solo` (recompute and swap) |
| 5 | `test_server.py::test_client_disconnect_mid_stream_frees_blocks` (and the non-streaming variant) |
| 6 | `test_block_leak.py::test_single_sequence_oom_fails_cleanly` |
| 7 | `test_prefix_cache.py` |
| 8 | `test_chunked_prefill.py` |
| 9 | `test_gpu.py` (FlashInfer vs naive, kernel level and end to end; CUDA graphs vs eager) |

**The near-tie rule** (`tests/greedy_check.py`). Two correct implementations never produce bit-identical
logits; they sum in different orders. So the gates require exact token equality except where a genuine
near-tie makes a flip legitimate: logits are compared at every step against the reference within a
tolerance (1e-3 in fp32, where observed errors are ~3e-5), and a first token mismatch is accepted only
where the reference's top-2 gap is under twice that tolerance. Exact-match gates run in fp32; in bf16
the same rule runs with a bf16 tolerance. The plan's "anything that changes the argmax is a bug" is not
true in bf16: batch composition legitimately flips near-ties there.

**The toy-model oracle** (`tests/toy_model.py`). A weight-free model whose K stores each token's id and
whose V stores its position. Instead of attention it reads each sequence's history back out of the
paged cache through the naive backend and emits a token that hashes the whole history, so any paging,
prefix-sharing, chunking or swap bug changes some sequence's output compared with a direct computation.
Randomized stress runs cover every mode, policy, preemption strategy and prefix setting with requests
arriving and aborting mid-run, checking invariants after every step. Mutation testing (planting known
bugs one at a time) confirmed the suite catches paging bugs; attention-math bugs are left to the
real-model gates.

## Debugging

The plan's playbook, mapped to what the engine provides:

| Symptom | Tool |
|---|---|
| Output is fluent but wrong | gate 1 against transformers, then `greedy_generate_contiguous` step by step; the three model traps are commented where they occur |
| Output degrades only at long context | the long-context gate (~2,500 tokens); gate 8 for the chunk offset |
| Blocks leak | `--debug-invariants` asserts the accounting after every step and names the block |
| Throughput far below expectation | `python -m bench.profile_steps`: operator table and a Chrome trace of busy steps |
| Hangs under load | the step watchdog raises with a scheduler dump; `--step-log` records queue depths and KV usage every step |
| Nondeterminism at temperature 0 | the near-tie rule separates reduction-order noise from bugs |

## Benchmark methodology (`bench/`)

- Lengths come from a ShareGPT trace with vLLM's filtering, or a lognormal fitted to it
  (`bench.workload fit`); never uniform, which would hide head-of-line blocking. The lognormal defaults
  in `bench/workload.py` are assumptions, labeled as such, for when no trace is available.
- Arrivals are Poisson at each rate, from a fixed seed, so every system replays identical requests.
- Prompts are sent as token ids so every server sees exactly the same lengths; `ignore_eos` fixes output
  lengths.
- The same client measures this engine and vLLM. HF static batching uses the same CSV schema and is
  measured generously (per-step token timestamps, as if the batch streamed).
- Throughput is reported over the whole run and at steady state (tokens produced between the 10th and
  90th percentile arrival times). TTFT, ITL and end-to-end latency are percentiles over requests and
  tokens; KV utilization, preemptions and peak memory come from `/stats`.

## Deliberately left out

- Multi-GPU (tensor or pipeline parallelism) and multi-replica routing: cluster-layer concerns, out of
  scope by the plan.
- Speculative decoding, quantized KV caches, LoRA: each a project of its own.
- Per-request seeds, logprobs, `n > 1`, penalties: rejected explicitly rather than half-supported.
- Swapping of shared prefix blocks as shared: a swapped sequence copies its shared blocks too and gets
  private copies back. Simple and correct; slightly wasteful when swapping sequences that share a prefix.
- CUDA graphs for mixed prefill/decode batches (vLLM's piecewise graphs): decode-only graphs capture most
  of the launch-overhead win at this model size.
