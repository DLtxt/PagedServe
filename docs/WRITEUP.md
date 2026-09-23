# PagedServe: paged KV caches and continuous batching, rebuilt from the OS up

*Draft. Sections marked TODO need measured numbers from the GPU runs, or statements only the author can
make. Nothing below the results heading should be filled in by estimate: if a number was not measured,
the sentence goes.*

## 1. What this is, and what it is not

PagedServe is a single-GPU inference engine for Qwen3-0.6B-Base: a paged KV cache, iteration-level
continuous batching, prefix caching, chunked prefill, preemption by recompute or swap, and a streaming
OpenAI-compatible API. It is a reimplementation for learning. vLLM introduced these ideas and is the
production system; nano-vLLM does comparable scope in a small codebase. Nothing here is new.

> TODO (author): how you used vLLM's and nano-vLLM's source, stated plainly.

What it is not: a cluster-layer system (no routing, autoscaling, or multiple replicas), a multi-GPU
engine, or a faster vLLM.

## 2. The operating-systems framing

Serving an LLM is a memory-management and scheduling problem, and almost every mechanism in the engine
has a textbook name:

| Serving | Operating systems |
|---|---|
| KV block | physical page frame |
| block size | page size |
| a sequence's block table | its page table |
| `slot = table[pos // bs] * bs + pos % bs` | address translation |
| shared prefix blocks with refcounts | shared pages |
| prefix cache keyed by a hash chain | page deduplication |
| LRU of cached, unreferenced blocks | the page cache and page replacement |
| preemption by recompute / by swap | killing and restarting a process / swapping it to disk |
| waiting, running, preempted, finished, aborted | the process state diagram |
| FCFS, SJF, priority with aging | CPU scheduling, and starvation |
| a lone sequence that outgrows memory | a process larger than physical memory: it cannot be scheduled, only failed |

Page size is the clearest trade-off. Small blocks waste less of each sequence's last block (internal
fragmentation) but make block tables longer and attention gather more scattered pages.

> TODO (results): the block-size sweep (`bench/figures/block_size.png`).

## 3. Architecture: the loop that must never block

`step()` is the whole engine: schedule, run the model on one flat batch, sample, update. It is
synchronous and advances every scheduled sequence by one iteration. A stall inside it is added to the
latency of every request currently decoding, so the HTTP layer is async and talks to the engine only
through per-request queues. The engine loop runs as a task on the same event loop, never awaits inside
a step, and yields once between steps so handlers can run.

The KV cache is one tensor, `[layers, 2, blocks + 1, block_size, kv_heads, head_dim]`. Each layer's K and
V views are the paged layout FlashInfer reads directly; a batch's new K/V are written with one scatter
per layer through a slot mapping computed on the CPU; and one gather moves a block across every layer
when swapping.

The scheduler works in tokens, not phases. Every sequence has `num_computed_tokens`, the tokens whose
K/V are already cached, and each iteration computes some of the rest. A fresh prefill, a later chunk of
a long prompt, a prefill behind a cached prefix, a decode step and the resume of a preempted sequence
are the same operation, and a sequence samples only when its chunk reaches its last token.

## 4. Results

> TODO (results): every number from `bench/results/`, measured on the GPU named here, with the vLLM
> version and flags (`bench/results/latency/vllm_version.txt`).

- **Latency against throughput**, this engine against vLLM and HF static batching on identical traffic
  (`latency_throughput.png`): where each knee sits.
- **Admission policies** (`admission_policies.png`): FCFS head-of-line blocking, SJF starving long
  prompts in the tail, aging bounding it.
- **Preemption** (`preemption.png`): the recompute/swap crossover in sequence length, and the end-to-end
  effect under memory pressure.
- **Prefix caching** (`prefix_caching.png`): TTFT against shared-prefix length, cache on and off.
- **Chunked prefill** (`chunked_prefill.png`): p99 inter-token latency over time with long prompts
  arriving, with and without chunking.
- **Kernels**: naive paged attention against FlashInfer, with and without CUDA graphs.

## 5. Three bugs and how they were found

> TODO (author): pick three real ones. `docs/BUGLOG.md` records every bug found while building this, how
> each was found, and the fix; add your own from the GPU runs. Strong candidates, because each was found by
> a method rather than by luck:
>
> - the client-disconnect abort that silently never happened: gate 5, then instrumentation, down to how
>   anyio delivers cancellation
> - the hash-collision test that could not fail: found by planting the bug it was meant to catch
> - the plan's decode-preemption loop that evicts the sequence it is extending: found by tracing two
>   sequences through the pseudocode

## 6. What's next, and what was left out

Left out on purpose: multiple GPUs and replicas (cluster-layer work), speculative decoding, quantized KV
caches and LoRA (each a project of its own), and per-request seeds, logprobs, `n > 1` and penalties,
which the API rejects rather than half-supports.

Next, in order of what the measurements would justify: piecewise CUDA graphs so mixed prefill/decode
batches also avoid launch overhead; moving the engine loop to its own thread if profiling shows the
per-step yield costs anything; and sharing swapped prefix blocks instead of swapping private copies.

> TODO (author): revise once the results say what matters most.
