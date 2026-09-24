# Bug log

Real bugs found while building PagedServe, how each was found, and what fixed it. Nothing here is
hypothetical: each entry happened. The write-up's "three bugs and how you found them" section should
draw from real entries only, from this log or from your own GPU runs.

## Found before running anything

**The plan's decode-preemption loop evicts the sequence it is extending.**
Found by reading the pseudocode against a two-sequence trace. It iterates over a copy of `running`
while popping victims off the end. When the sequence being extended is itself the newest, `pop()`
evicts it, the `while` re-checks `can_append_slot` for an evicted sequence, and `append_slot` then
hands a block to a sequence that is no longer running. The loop also goes on to visit sequences it
already evicted. Fix: walk running from oldest to newest, take victims from the newest end, and when
the victim would be the current sequence, preempt it (or fail it, if it is alone) and stop
(`Scheduler._schedule_running`).

**FlashAttention's paged kernels cannot run the planned block sizes.**
Found by reading `csrc/flash_attn/flash_api.cpp` before writing any kernel code: both
`mha_varlen_fwd` and `mha_fwd_kvcache` reject a paged KV cache whose block size is not a multiple of
256. The plan's default block size of 16 and its 8/16/32 sweep were impossible on that kernel. Fix:
the fast path uses FlashInfer, which accepts any page size and the same `[pages, page_size, kv_heads,
head_dim]` layout.

**Swapping cannot rescue a lone sequence that is out of blocks.**
The plan offers "swap its cache to host memory, or fail the request". A single running sequence with
no free block already holds the whole pool; moving it to host memory frees nothing it can use. The
only correct outcome is a clean failure, which `test_single_sequence_oom_fails_cleanly` forces.

## Found by the gates

**A disconnected client kept its request decoding to the end (gate 5).** The first run of gate 5 closed
a streaming client after three tokens; the blocks came back, but only because the request had run all
400 tokens to completion, and `total_aborted` stayed at 0. Instrumentation showed uvicorn noticed the
reset and Starlette's disconnect listener received `http.disconnect` and cancelled its anyio cancel
scope, yet the streaming task kept sending. anyio delivers a scope cancellation only to a task blocked
on a pending future, retrying every loop iteration. Our engine loop resolves the stream's queue future
once per iteration, so each delivery attempt ran right after the engine had woken the task and before
the task ran: it was never blocked on a pending future when anyio looked, and the cancellation starved
for the whole generation. That is the plan's abort story ("`finally` frees the blocks") failing in
exactly the case it exists for, silently, since nothing leaks: the work is simply wasted. Fix: the
response subclass aborts the request in the engine directly from the disconnect signal (the listener
under ASGI 2.3, a failed send under 2.4) and hands the generator a terminal output, so the abort no
longer depends on cancellation timing. Reproduced first with the toy model behind the real server,
decoding at real-model speed.

## Found in review of my own code

**The fp32 gates would have crashed on the GPU.** Reviewing the CUDA paths before handing them over:
every engine test builds its engine in fp32 with `attention_backend="auto"`, and on CUDA "auto" meant
FlashInfer, whose kernels exist only for fp16 and bf16. Gates 3 through 8 pass on a Mac, where "auto"
means the naive backend, and would have failed at the first FlashInfer call on the GPU box. Fix:
"auto" picks FlashInfer only for fp16/bf16 on CUDA, and asking for FlashInfer in fp32 is a config
error rather than a kernel crash.

**An `assert` with a side effect.** `_pop_waiting` was written as `assert self.waiting.pop() is seq`.
Under `python -O` asserts are stripped, the pop never happens, and the same request would be admitted
over and over. Fix: pop first, assert on the result.

**A test harness that would have allocated gigabytes of swap space.** The toy-model stress test sized
host swap space only when a nonzero block count was requested, so "zero host blocks" silently fell
back to the 4 GiB default, which is tens of millions of toy blocks. Fix: always size swap space from
the block count, including zero.

**Irreproducible stress seeds.** The randomized stress test derived its seed from `hash()` of a tuple
containing strings. String hashing is randomized per process (PYTHONHASHSEED), so a failing seed could
never be replayed. Fix: `zlib.crc32` of the parameters.

## Found by testing the tests

**Half of the swap configurations never swapped.** Instrumenting the stress runs (preemptions, swap-outs
and prefix-cache hits per configuration) showed that the random host-space choice often gave zero or two
blocks, so preemption always fell back to recompute and the swap path went untested in exactly the
configurations named after it. Fix: host space is drawn from {8, 128} tokens, which covers both real
swapping and the fallback when host space runs out; every swap configuration now swaps.

**Mutation testing: two planted bugs survived the scheduler suite.** Ten deliberate bugs were planted
one at a time (hash chain without the parent, reuse of the last-token block, never publishing blocks,
leaking on finish, sampling on every chunk, skipping the swap-in copy, an off-by-one slot, batch-row
positions, a top-left causal mask, prefill-before-decode ordering). The toy-model suite caught eight.
The causal-mask bug survives because the toy model never runs attention math; the real-model gates 7
and 8 are what catch it. The ordering change turned out to be an equivalent mutant: at most one
sequence is ever mid-prefill, and it is always the newest in admission order, so running order and
decode-first order produce the same plan. A dedicated test now asserts that decodes never wait behind
a prefill chunk.

**A test assertion that could never pass.** The chunked-prefill gate first checked that "some step
produced no output", but short requests decode in the same iterations as the long prompt's chunks,
so almost no step is silent. The assertion now counts prefill chunks that did not reach the end of
their prompt.

**An off-by-one in my own gate-6 test, not in the engine.** The first real-model run of gate 6 failed:
the test expected a 40-token prompt in a 64-token cache (4 blocks of 16) to yield 24 tokens before
failing, and the engine yielded 25. The engine was right. The cache holds K/V for 64 tokens; the token
sampled from position 63's logits is the 65th and needs no slot until it is fed back in as input. So a
request can emit one more token than the cache has slots for its context. The expectation is now
`64 - 40 + 1`, with the reasoning in a comment.

**Gate 7's hash-collision test could not detect a missing parent hash.** Mutation testing against the
real-model gates planted two bugs aimed at them. A top-left causal mask was caught by gate 8 (logit error
16 at the first chunked step). A block hash without its parent was not: the test's request B differed from
the cached request A in its first block, and prefix lookup stops at the first miss, so B never reached
the colliding middle blocks. A missing parent only matters when a request's first blocks are genuine
hits and the next block has the same tokens as a block cached under a different prefix. The test now
builds exactly that (B = X + M, where X is shared with request C and M's cached copy sits after A's Z).
With the planted bug, B reuses 32 cached tokens instead of 16 and its first-step logits are off by 1.0:
plausible-looking, wrong output, the failure the plan warns about.

## Found building tensor and pipeline parallelism

**Two planted sharding bugs survived because the tiny model made them no-ops.** Mutation testing the
split model planted nine bugs one at a time: no all-reduce after o_proj, none after down_proj, a
vocabulary-parallel embedding without its mask, k/v sliced like q, o_proj sliced by rows, zeros sent
between stages, every stage running layers 0 to n, swap-in copies dropped from the plan, and logit
shards gathered in reverse. Seven were caught. The other two were equivalent mutants for the test
model: with twice as many query heads as KV heads, half of each rank's q slice is exactly its kv
slice, and with a hidden size of 64, rows 0 to 64 of o_proj are the whole matrix. Planted again as
real bugs (each rank taking the other rank's k heads, the other rank's o_proj columns, or down_proj's
columns reversed), all three were caught.

**Aborting a sequence in flight would have handed its token to its neighbour.** Found in review of the
pipelining patch, before it ran. With several batches in flight, aborting a sequence has to wait for
its batch to come back, and the draft dropped the sequence in `update()` before taking its sampled
token from the step's token list. Every later sequence in that batch would have received the token
meant for the one before it: fluent, wrong text, and no crash. Fix: take the token, then drop the
sequence. `test_abort_in_flight_waits_for_its_batch_and_keeps_tokens_aligned` fails on the draft's
order, and so does the randomized stress at depth 2.

**Without balancing, the pipeline would never have filled.** Also found in review before running. A
sequence in flight is not scheduled again, so sequences that return together are scheduled together,
and a burst admitted in one batch would travel as one batch for good: stage 0 idle while stage 1
works, the bubble pipelining exists to remove, with every output still correct, so no correctness test
would have noticed. Fix: each batch takes at most its share of the running sequences. A test asserts
that eight sequences admitted together run as two batches of four, and every pipelined distributed
test asserts the pipeline really held `pipeline_depth` batches.

**A pipeline stage could overwrite FlashInfer's plan before its GPU had read it.** The pipelining draft
had this race; it was found by reading FlashInfer's source against the pipelined timeline and has not
been observed, since it needs a GPU. FlashInfer 0.7.0's `plan()` writes into one pinned buffer per
wrapper and copies it to the GPU with `cudaMemcpyAsync` (`attention/scheduler.cuh`). On one GPU that is
safe because sampling syncs every step, but a pipeline stage that does not sample can plan step k+1
before its GPU has executed step k's copy. Fix: the runner records an event after each plan and waits
on it before the next.

**Stopping a distributed server hung for 30 seconds, then killed the driver.** Found by the first
smoke run of a benchmark sweep under torchrun: every configuration's server took exactly 30 seconds to
stop, and its log ended in a worker's KeyboardInterrupt and torchrun's "forcefully exiting via 9".
Dumping the driver's stack during the hang put it in the shutdown path, waiting to send the shutdown
plan. torchrun forwards SIGINT to every rank (a Ctrl-C in a terminal reaches them all anyway), so the
worker died of the signal inside its wait for the next plan, while the driver, whose HTTP server shuts
down gracefully, then sent the shutdown plan to a dead peer; gloo never noticed, and the driver waited
until torchrun SIGKILLed it. The bug predates pipelining: the broadcast version would have hung the
same way. Fix: workers ignore SIGINT and SIGTERM, since their lifetime is the driver's (it stops them
once the batches in flight have drained, and torchrun kills them if the driver dies), and the driver
waits at most 30 seconds on the shutdown send. The torchrun server test now asserts every rank stops
within 20 seconds of a SIGINT with no rank dying of it; the whole shutdown takes about a second.

**A planted pipelining bug that only a direct check can see.** Mutation testing the pipelined scheduler
planted ten bugs; nine were caught at once. The tenth, letting the engine keep one more batch in flight
than `pipeline_depth`, passed everything, because it is harmless to correctness. The stress test now
asserts the bound at every submission and catches it. One catch was for the wrong reason: the first
version of the token-order mutant deleted the token read instead of moving it, so the tests failed on
a NameError, not on the bug. Planted faithfully, it is caught by the dedicated abort test and by the
stress test on its own.
