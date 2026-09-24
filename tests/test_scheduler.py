"""Scheduler, block manager, and engine-loop tests on the weight-free toy model: fast, and exact.

The toy model's output is a hash of the history it reads back out of the paged cache, so comparing
against `expected()` catches any paging, prefix-sharing, chunking, or swap bug, and
debug_invariants=True checks the block accounting after every step (gate 3).
"""

from __future__ import annotations

import random
import zlib

import pytest

from engine.config import EngineConfig
from engine.core.block_manager import BlockManager
from engine.core.engine import LLMEngine
from engine.core.scheduler import WaitingQueue
from engine.core.sequence import SamplingParams, Sequence, SequenceStatus
from toy_model import EOS, ToyConfig, ToyModel, ToyTokenizer, next_token

VOCAB = ToyConfig.vocab_size
TOY_BLOCK_BYTES = 1 * 2 * 2 * 4  # layers * (K, V) * head_dim * fp32, per token


def make_engine(**overrides) -> LLMEngine:
    options = dict(
        device="cpu", dtype="float32", block_size=4, num_blocks=64, max_model_len=256, max_num_seqs=16,
        max_batch_tokens=64, debug_invariants=True,
    )
    options.update(overrides)
    # Size host swap space in blocks: the 4 GiB default would be millions of toy blocks.
    options["swap_space_gb"] = options.pop("host_blocks", 0) * options["block_size"] * TOY_BLOCK_BYTES / (1 << 30)
    return LLMEngine(EngineConfig(**options), model=ToyModel(), tokenizer=ToyTokenizer())


def expected(prompt: list[int], params: SamplingParams) -> tuple[list[int], str]:
    """Exactly what a correct engine returns for this request under the toy model."""
    tok = ToyTokenizer()
    history, out = list(prompt), []
    while True:
        token = next_token(history, VOCAB)
        history.append(token)
        out.append(token)
        text = tok.decode(out)
        hits = [text.find(s) for s in params.stop if s in text]
        if hits:
            return out, text[: min(hits)]
        if (token == EOS and not params.ignore_eos) or len(out) >= params.max_tokens:
            return out, text


def run(engine: LLMEngine, seqs: list[Sequence], max_steps: int = 100_000) -> dict[int, str]:
    """Drive the engine until idle; return the concatenated streamed text per request."""
    texts: dict[int, str] = {s.seq_id: "" for s in seqs}
    for seq in seqs:
        engine.add_request(seq)
    for _ in range(max_steps):
        if not engine.has_work():
            break
        for out in engine.step():
            texts[out.seq_id] += out.text
    else:
        pytest.fail(f"engine did not finish in {max_steps} steps: {engine.scheduler.describe()}")
    return texts


def assert_idle_and_clean(engine: LLMEngine) -> None:
    bm = engine.block_manager
    engine.scheduler.check_invariants()
    assert bm.num_free_blocks == bm.num_blocks, "blocks leaked"
    assert len(bm.cpu_free) == bm.num_cpu_blocks, "host blocks leaked"
    assert not engine.scheduler.seqs


def random_prompt(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(1, VOCAB) for _ in range(n)]


# --- block manager ----------------------------------------------------------------------------------


def seq_with(tokens: list[int], seq_id: int = 0) -> Sequence:
    return Sequence(seq_id, tokens, SamplingParams(temperature=0))


def test_allocate_append_free_roundtrip():
    bm = BlockManager(num_blocks=8, block_size=4)
    s = seq_with(list(range(1, 11)))  # 10 tokens -> 3 blocks
    assert bm.can_allocate(s, 10, [], use_watermark=False)
    bm.allocate(s, 10, [])
    assert len(s.block_table) == 3 and bm.num_free_blocks == 5
    s.num_computed_tokens = 10
    s.append_token(5)
    bm.append_slots(s, 1)  # token 11 still fits in block 3
    assert len(s.block_table) == 3
    s.num_computed_tokens = 11
    s.append_token(6)
    s.append_token(7)
    bm.append_slots(s, 2)  # tokens 12-13: the second one needs block 4
    assert len(s.block_table) == 4
    s.num_computed_tokens = 13
    bm.check_invariants([s])
    bm.free(s)
    assert bm.num_free_blocks == 8 and not s.block_table
    bm.check_invariants([])


def test_invariant_catches_a_leak():
    bm = BlockManager(num_blocks=4, block_size=4)
    s = seq_with([1, 2, 3, 4, 5])
    bm.allocate(s, 5, [])
    s.num_computed_tokens = 5
    s.block_table.pop()  # "forget" a block: it is neither free nor held
    with pytest.raises(AssertionError):
        bm.check_invariants([s])


def test_prefix_cache_hits_skip_the_last_token_block():
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    a = seq_with(list(range(1, 13)), 0)  # 12 tokens = exactly 3 full blocks
    bm.allocate(a, 12, bm.cached_prefix(a))
    a.num_computed_tokens = 12
    bm.cache_full_blocks(a)
    b = seq_with(list(range(1, 13)), 1)  # identical prompt
    hits = bm.cached_prefix(b)
    # Block 3 holds b's last token, which must be recomputed, so only 2 blocks are shared.
    assert hits == a.block_table[:2]
    bm.allocate(b, 12 - 8, hits)
    assert b.num_computed_tokens == 8 and b.block_table[:2] == a.block_table[:2]
    assert b.block_table[2] != a.block_table[2]
    assert bm.ref_counts[a.block_table[0]] == 2
    b.num_computed_tokens = 12
    bm.cache_full_blocks(b)  # b's copy of block 3 duplicates a's: it stays private
    bm.check_invariants([a, b])
    bm.free(a)
    bm.free(b)
    bm.check_invariants([])
    assert len(bm.evictable) == 3 and bm.num_free_blocks == 16


def test_same_block_under_different_prefix_is_not_shared():
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    shared_tail = [9, 9, 9, 9]
    a = seq_with([1, 2, 3, 4] + shared_tail + [5], 0)
    bm.allocate(a, a.num_tokens, bm.cached_prefix(a))
    a.num_computed_tokens = a.num_tokens
    bm.cache_full_blocks(a)
    b = seq_with([7, 7, 7, 7] + shared_tail + [5], 1)  # same second block, different parent
    assert bm.cached_prefix(b) == []
    assert a.block_hashes[1] != b.block_hashes[1]


def test_lru_eviction_prefers_oldest_and_tail_blocks():
    bm = BlockManager(num_blocks=4, block_size=2, enable_prefix_caching=True)
    a = seq_with([1, 2, 3, 4, 5], 0)  # blocks: [1,2] [3,4] [5]
    bm.allocate(a, 5, [])
    a.num_computed_tokens = 5
    bm.cache_full_blocks(a)
    first, second = a.block_table[0], a.block_table[1]
    bm.free(a)  # [5] -> free list; [3,4] then [1,2] -> evictable, tail first
    assert list(bm.evictable) == [second, first]
    b = seq_with([8, 8, 8, 8, 8, 8, 8], 1)  # needs 4 blocks: 2 free + evict both cached ones
    bm.allocate(b, 7, bm.cached_prefix(b))
    assert not bm.evictable and not bm.hash_to_block
    b.num_computed_tokens = 7
    bm.check_invariants([b])


def test_swap_out_and_in_roundtrip():
    bm = BlockManager(num_blocks=4, block_size=2, num_cpu_blocks=4)
    s = seq_with([1, 2, 3], 0)
    bm.allocate(s, 3, [])
    s.num_computed_tokens = 3
    gpu = list(s.block_table)
    out = bm.swap_out(s)
    assert [g for g, _ in out] == gpu and not s.block_table and len(s.cpu_block_table) == 2
    bm.check_invariants([], [s])
    assert bm.can_swap_in(s, 1, use_watermark=False)
    back = bm.swap_in(s)
    assert [c for c, _ in back] == [c for _, c in out] and len(s.block_table) == 2
    bm.check_invariants([s], [])


# --- admission policies ------------------------------------------------------------------------------


def _queued(policy: str, aging_rate: float, specs: list[tuple[int, float, float | None]]) -> list[int]:
    q = WaitingQueue(policy, aging_rate)
    for i, (prompt_len, arrival, priority) in enumerate(specs):
        q.push(Sequence(i, [1] * prompt_len, SamplingParams(), arrival_time=arrival, priority=priority))
    return [q.pop().seq_id for _ in range(len(specs))]


def test_policy_orders():
    specs = [(50, 0.0, None), (10, 1.0, None), (30, 2.0, None)]
    assert _queued("fcfs", 0, specs) == [0, 1, 2]
    assert _queued("sjf", 0, specs) == [1, 2, 0]
    assert _queued("priority", 0.0, specs) == [1, 2, 0]  # no aging: SJF
    # 30 priority units per second of waiting: the 50-token request that arrived 1-2 s earlier
    # overtakes both (keys 50, 40, 90).
    assert _queued("priority", 30.0, specs) == [1, 0, 2]
    assert _queued("priority", 1e6, specs) == [0, 1, 2]  # aging dominates: FCFS
    # explicit priorities override the length default
    assert _queued("priority", 0.0, [(10, 0.0, 5.0), (10, 0.0, 1.0)]) == [1, 0]


def test_waiting_queue_remove_is_lazy_but_exact():
    q = WaitingQueue("fcfs", 0)
    for i in range(3):
        q.push(Sequence(i, [1], SamplingParams(), arrival_time=float(i)))
    assert q.remove(0).seq_id == 0 and q.remove(0) is None
    assert len(q) == 2 and q.pop().seq_id == 1 and q.pop().seq_id == 2 and q.peek() is None


# --- engine scenarios --------------------------------------------------------------------------------


@pytest.mark.parametrize("chunked", [False, True])
def test_single_request_matches_reference(chunked):
    engine = make_engine(enable_chunked_prefill=chunked, max_batch_tokens=16)
    prompt = random_prompt(random.Random(1), 37)
    params = SamplingParams(temperature=0, max_tokens=30, ignore_eos=True)
    seq = Sequence(0, prompt, params)
    texts = run(engine, [seq])
    assert (seq.output_token_ids, texts[0]) == expected(prompt, params)
    assert seq.finish_reason == "length"
    assert_idle_and_clean(engine)


def test_chunked_prefill_samples_only_on_the_final_chunk():
    engine = make_engine(enable_chunked_prefill=True, max_batch_tokens=16, max_num_seqs=4)
    seq = Sequence(0, random_prompt(random.Random(2), 50), SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
    engine.add_request(seq)
    outputs_per_step = []
    while engine.has_work():
        outputs_per_step.append(len(engine.step()))
    # 50 prompt tokens in chunks of 16: 16, 16, 16, 2 -> three silent steps, then a token per step.
    assert outputs_per_step == [0, 0, 0, 1, 1, 1]
    assert_idle_and_clean(engine)


def test_chunked_decodes_are_never_stalled_by_prefill():
    """The point of chunked prefill: a long prompt never costs running decodes a step."""
    engine = make_engine(enable_chunked_prefill=True, max_batch_tokens=16, max_num_seqs=8, num_blocks=128)
    rng = random.Random(8)
    params = SamplingParams(temperature=0, max_tokens=40, ignore_eos=True)
    short = [Sequence(i, random_prompt(rng, 3), params) for i in range(4)]
    for s in short:
        engine.add_request(s)
    engine.step()  # all four prefill together
    engine.add_request(Sequence(9, random_prompt(rng, 150), SamplingParams(temperature=0, max_tokens=2)))
    for _ in range(12):  # the long prompt needs ~12 chunks of 12 tokens
        batch = engine.scheduler.schedule()
        decoding = [s for s in engine.scheduler.running if s.num_uncomputed_tokens == 1 and s.seq_id < 9]
        assert all(s in batch.seqs for s in decoding), "a decode waited behind a prefill chunk"
        assert batch.num_tokens <= 16
        sampling = engine.sampler.prepare(batch.sampling_params)
        logits = engine.model_runner.execute(batch)
        engine.scheduler.update(batch, engine.sampler(logits, sampling) if logits is not None else [])
    while engine.has_work():
        engine.step()
    assert_idle_and_clean(engine)


def test_lone_sequence_out_of_blocks_fails_cleanly():
    # 4 blocks of 4 tokens: a 10-token prompt can grow to 16 tokens, then nothing is left to evict.
    engine = make_engine(num_blocks=4, max_model_len=128)
    seq = Sequence(0, random_prompt(random.Random(3), 10), SamplingParams(temperature=0, max_tokens=100, ignore_eos=True))
    engine.add_request(seq)
    outputs = []
    for _ in range(50):
        if not engine.has_work():
            break
        outputs.extend(engine.step())
    assert not engine.has_work(), "engine hung instead of failing the request"
    assert seq.finish_reason == "error" and "KV cache exhausted" in seq.error
    assert outputs[-1].finished and outputs[-1].finish_reason == "error"
    assert_idle_and_clean(engine)


@pytest.mark.parametrize("preemption", ["recompute", "swap"])
def test_oversized_request_fails_while_others_complete(preemption):
    engine = make_engine(num_blocks=8, max_model_len=128, preemption_mode=preemption, host_blocks=16)
    rng = random.Random(4)
    big = Sequence(0, random_prompt(rng, 40), SamplingParams(temperature=0, max_tokens=5))  # 40 tokens > 32
    small = Sequence(1, random_prompt(rng, 6), SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
    texts = run(engine, [big, small])
    assert big.finish_reason == "error"
    assert (small.output_token_ids, texts[1]) == expected(small.prompt_token_ids, small.sampling_params)
    assert_idle_and_clean(engine)


@pytest.mark.parametrize("preemption", ["recompute", "swap"])
def test_preemption_evicts_newest_and_output_is_unchanged(preemption):
    # 6 blocks of 4 tokens; three 5-token prompts start with 2 blocks each and must grow.
    engine = make_engine(num_blocks=6, max_model_len=64, preemption_mode=preemption, host_blocks=12, watermark=0.0)
    rng = random.Random(5)
    params = SamplingParams(temperature=0, max_tokens=12, ignore_eos=True)
    seqs = [Sequence(i, random_prompt(rng, 5), params) for i in range(3)]
    texts = run(engine, seqs)
    for s in seqs:
        assert (s.output_token_ids, texts[s.seq_id]) == expected(s.prompt_token_ids, params)
    assert seqs[2].num_preemptions >= 1 and seqs[0].num_preemptions == 0
    assert engine.scheduler.num_preemptions >= 1
    if preemption == "swap":
        assert engine.scheduler.num_swap_outs >= 1
    assert_idle_and_clean(engine)


def test_abort_in_every_state_frees_everything():
    engine = make_engine(num_blocks=6, max_model_len=64, preemption_mode="swap", host_blocks=4, max_num_seqs=3)
    rng = random.Random(6)
    params = SamplingParams(temperature=0, max_tokens=15, ignore_eos=True)  # 20 tokens: fits alone
    seqs = [Sequence(i, random_prompt(rng, 5), params) for i in range(6)]
    for s in seqs:
        engine.add_request(s)
    seen = set()
    for _ in range(200):
        if not engine.has_work():
            break
        engine.step()
        for s in list(engine.scheduler.seqs.values()):
            state = "swapped" if s.cpu_block_table else s.status.value
            if state not in seen:
                seen.add(state)
                engine.abort(s.seq_id)
                engine.abort(s.seq_id)  # idempotent
                assert s.status == SequenceStatus.ABORTED
                engine.scheduler.check_invariants()
    assert {"waiting", "running"} <= seen
    while engine.has_work():
        engine.step()
    assert_idle_and_clean(engine)


def test_stop_strings_hold_back_and_truncate():
    engine = make_engine()
    rng = random.Random(7)
    prompt = random_prompt(rng, 8)
    free_run, _ = expected(prompt, SamplingParams(temperature=0, max_tokens=60, ignore_eos=True))
    text = ToyTokenizer().decode(free_run)
    stop = text[20:23]  # a 3-letter string the model is known to produce
    params = SamplingParams(temperature=0, max_tokens=60, stop=[stop], ignore_eos=True)
    seq = Sequence(0, prompt, params)
    streamed = run(engine, [seq])[0]
    want_tokens, want_text = expected(prompt, params)
    assert seq.output_token_ids == want_tokens and streamed == want_text and stop not in streamed
    assert seq.finish_reason == "stop"
    assert_idle_and_clean(engine)


def test_prompt_too_long_for_max_model_len_is_rejected():
    engine = make_engine(max_model_len=32)
    with pytest.raises(ValueError, match="max_model_len"):
        engine.add_request(Sequence(0, [1] * 30, SamplingParams(max_tokens=5)))


def test_pipeline_depth_defaults_to_the_pipeline_size():
    base = dict(device="cpu", num_blocks=8, max_model_len=64)
    assert EngineConfig(**base, pipeline_parallel_size=2).resolve(4096).pipeline_depth == 2
    assert EngineConfig(**base, pipeline_parallel_size=2, pipeline_depth=1).resolve(4096).pipeline_depth == 1
    assert EngineConfig(**base).resolve(4096).pipeline_depth == 1
    with pytest.raises(ValueError, match="pipeline_depth"):
        EngineConfig(**base, pipeline_depth=0)


def test_no_chunking_raises_budget_to_max_model_len():
    cfg = EngineConfig(device="cpu", num_blocks=8, max_model_len=4096, max_batch_tokens=2048).resolve(32768)
    assert cfg.max_batch_tokens == 4096
    cfg = EngineConfig(device="cpu", num_blocks=8, max_model_len=4096, max_batch_tokens=2048, enable_chunked_prefill=True).resolve(32768)
    assert cfg.max_batch_tokens == 2048


# --- pipelined scheduling: several batches in flight -------------------------------------------------
#
# The local executor runs a batch the moment it is submitted, which a pipeline does too as far as any
# one stage's cache is concerned: every stage executes batches in submission order. What these tests
# exercise is the scheduler's side: batches submitted before earlier ones have returned.


def test_pipeline_splits_running_sequences_into_disjoint_balanced_batches():
    engine = make_engine(pipeline_depth=2, max_num_seqs=8)
    rng = random.Random(10)
    params = SamplingParams(temperature=0, max_tokens=20, ignore_eos=True)
    seqs = [Sequence(i, random_prompt(rng, 6), params) for i in range(8)]
    submitted = []  # per submission: its sequence ids, the ids already in flight, the running count
    submit = engine.executor.submit

    def recording_submit(batch):
        in_flight = {s.seq_id for b, _ in engine.in_flight for s in b.seqs}
        submitted.append(({s.seq_id for s in batch.seqs}, in_flight, len(engine.scheduler.running)))
        return submit(batch)

    engine.executor.submit = recording_submit
    texts = run(engine, seqs)
    for s in seqs:
        assert (s.output_token_ids, texts[s.seq_id]) == expected(s.prompt_token_ids, params)
    assert all(not ids & in_flight for ids, in_flight, _ in submitted), "a sequence was in two batches at once"
    assert any(in_flight for _, in_flight, _ in submitted), "the pipeline never held two batches"
    # All eight are admitted in one prefill batch; after that each batch takes half of them.
    assert {len(ids) for ids, _, running in submitted[1:] if running == 8} == {4}
    assert_idle_and_clean(engine)


def test_abort_in_flight_waits_for_its_batch_and_keeps_tokens_aligned():
    """Aborting a sequence whose batch is in flight only marks it, since pipeline stages may still be
    writing its blocks. When the batch returns the sequence is dropped and the token sampled for it
    is discarded, so the next sequence in that batch still gets its own token."""
    engine = make_engine(pipeline_depth=2, max_num_seqs=4)
    rng = random.Random(11)
    params = SamplingParams(temperature=0, max_tokens=12, ignore_eos=True)
    seqs = [Sequence(i, random_prompt(rng, 5), params) for i in range(4)]
    victim, neighbour = seqs[2], seqs[3]
    # Otherwise handing the neighbour the victim's token would go unnoticed.
    assert expected(victim.prompt_token_ids, params)[0][1] != expected(neighbour.prompt_token_ids, params)[0][1]
    texts = {s.seq_id: "" for s in seqs}

    def step():
        for out in engine.step():
            texts[out.seq_id] += out.text

    for s in seqs:
        engine.add_request(s)
    step()  # one prefill batch for all four
    step()  # decodes in two batches: {0, 1} has returned, {2, 3} is still in flight
    ((batch, _),) = engine.in_flight
    assert [s.seq_id for s in batch.seqs] == [2, 3]
    free_before = engine.block_manager.num_free_blocks
    engine.abort(victim.seq_id)
    engine.abort(victim.seq_id)  # idempotent while pending
    assert victim.status == SequenceStatus.RUNNING and victim.abort_requested
    assert engine.block_manager.num_free_blocks == free_before, "freed blocks a batch in flight is using"
    step()  # {2, 3} returns
    assert victim.status == SequenceStatus.ABORTED and not victim.block_table
    assert victim.seq_id not in engine.scheduler.seqs and engine.scheduler.num_aborted == 1
    while engine.has_work():
        step()
    for s in (seqs[0], seqs[1], neighbour):
        assert (s.output_token_ids, texts[s.seq_id]) == expected(s.prompt_token_ids, params)
    assert len(victim.output_token_ids) == 1, "the victim kept a token sampled after its abort"
    assert_idle_and_clean(engine)


def test_out_of_blocks_waits_for_a_batch_in_flight_instead_of_preempting_it():
    """a runs out of blocks while b, the only newer sequence, is in flight. b cannot be preempted
    until its batch returns, and a is not alone, so a must neither preempt b nor fail: it waits a
    step, then preempts b once b is back."""
    engine = make_engine(pipeline_depth=2, num_blocks=4, max_num_seqs=2, max_model_len=64, watermark=0.0)
    rng = random.Random(12)
    params = SamplingParams(temperature=0, max_tokens=10, ignore_eos=True)  # 14 tokens: 4 blocks each
    a, b = (Sequence(i, random_prompt(rng, 4), params) for i in range(2))
    texts = run(engine, [a, b])
    for s in (a, b):
        assert (s.output_token_ids, texts[s.seq_id]) == expected(s.prompt_token_ids, params)
    assert a.num_preemptions == 0 and b.num_preemptions >= 1
    assert_idle_and_clean(engine)


def test_failed_step_forgets_the_batches_in_flight():
    engine = make_engine(pipeline_depth=2, max_num_seqs=4)
    rng = random.Random(13)
    params = SamplingParams(temperature=0, max_tokens=30, ignore_eos=True)
    seqs = [Sequence(i, random_prompt(rng, 5), params) for i in range(4)]
    for s in seqs:
        engine.add_request(s)
    engine.step()
    engine.step()
    ((batch, _),) = engine.in_flight
    pending = batch.seqs[0]
    engine.abort(pending.seq_id)  # deferred: its batch is in flight

    def dead_rank(handle):
        raise RuntimeError("rank 1 is gone")

    engine.executor.wait = dead_rank
    with pytest.raises(RuntimeError, match="rank 1 is gone"):
        engine.step()
    assert not engine.in_flight and not any(s.in_flight for s in engine.scheduler.running)
    assert pending.status == SequenceStatus.ABORTED, "the pending abort was lost with its batch"
    for s in seqs:  # what the server does after a failed step
        engine.abort(s.seq_id)
    assert not engine.has_work()
    assert_idle_and_clean(engine)


# --- randomized stress: every mode x policy x preemption x prefix caching ----------------------------


@pytest.mark.parametrize("depth", [1, 2, 3])  # batches in flight; > 1 is pipelined scheduling
@pytest.mark.parametrize("trial", [0, 1])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("preemption", ["recompute", "swap"])
@pytest.mark.parametrize("policy", ["fcfs", "sjf", "priority"])
def test_randomized_stress(chunked, prefix, preemption, policy, trial, depth):
    # Stable across runs, and the same workload at every depth.
    seed = zlib.crc32(repr((chunked, prefix, preemption, policy, trial)).encode())
    rng = random.Random(seed)
    block_size = rng.choice([1, 2, 4, 8])
    engine = make_engine(
        block_size=block_size,
        num_blocks=max(8, 96 // block_size),  # ~96 tokens of cache: heavy preemption
        max_model_len=80,
        max_num_seqs=rng.choice([4, 8]),
        max_batch_tokens=rng.choice([8, 24]) if chunked else 80,
        enable_chunked_prefill=chunked,
        enable_prefix_caching=prefix,
        preemption_mode=preemption,
        admission_policy=policy,
        aging_rate=rng.choice([0.0, 50.0]),
        # 8 host tokens: swap space runs out and preemption falls back to recompute; 128: it doesn't.
        host_blocks=max(1, rng.choice([8, 128]) // block_size) if preemption == "swap" else 0,
        pipeline_depth=depth,
    )
    submit = engine.executor.submit

    def bounded_submit(batch):
        assert len(engine.in_flight) < depth, "more batches in flight than pipeline_depth"
        return submit(batch)

    engine.executor.submit = bounded_submit
    shared = [random_prompt(rng, rng.randrange(1, 30)) for _ in range(3)]  # common prefixes
    pending = []
    for i in range(40):
        prompt = rng.choice(shared) + random_prompt(rng, rng.randrange(1, 20)) if rng.random() < 0.6 else random_prompt(rng, rng.randrange(1, 40))
        params = SamplingParams(
            temperature=0,
            max_tokens=rng.randrange(1, 80 - len(prompt) + 1),
            stop=[ToyTokenizer().decode(random_prompt(rng, 2))] if rng.random() < 0.2 else [],
            ignore_eos=rng.random() < 0.5,
        )
        pending.append(Sequence(i, prompt, params, priority=rng.choice([None, float(rng.randrange(50))])))

    texts: dict[int, str] = {s.seq_id: "" for s in pending}
    aborted, arrivals = set(), list(pending)
    for step in range(20_000):
        for _ in range(rng.choice([0, 0, 1, 2])):  # requests arrive while the engine runs
            if arrivals:
                engine.add_request(arrivals.pop(0))
        if rng.random() < 0.03 and engine.scheduler.seqs:
            victim = rng.choice(sorted(engine.scheduler.seqs))
            engine.abort(victim)
            aborted.add(victim)
        if not engine.has_work() and not arrivals:
            break
        for out in engine.step():
            texts[out.seq_id] += out.text
    else:
        pytest.fail(f"did not finish: {engine.scheduler.describe()}")

    for seq in pending:
        if seq.seq_id in aborted and seq.status == SequenceStatus.ABORTED:
            continue
        assert seq.finish_reason in ("stop", "length"), f"seq {seq.seq_id} ended with {seq.finish_reason}: {seq.error}"
        want_tokens, want_text = expected(seq.prompt_token_ids, seq.sampling_params)
        assert seq.output_token_ids == want_tokens, f"seq {seq.seq_id} tokens diverged (seed {seed})"
        assert texts[seq.seq_id] == want_text, f"seq {seq.seq_id} text diverged (seed {seed})"
    assert_idle_and_clean(engine)
    if prefix:
        assert engine.block_manager.prefix_hit_tokens > 0, "prefix cache never hit on a shared-prefix workload"
