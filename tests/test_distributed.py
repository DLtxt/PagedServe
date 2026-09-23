"""Tensor and pipeline parallelism: a split model must give the answers of the whole one.

Every rank is a real process talking over torch.distributed (gloo on CPU here, NCCL on GPUs). The
reference is the single-process engine on the same weights; comparisons use the near-tie rule, since
a tensor-parallel all-reduce sums partial results in a different order than one big matmul. The tiny
random-weight Qwen3 (tests/tiny_qwen3.py) makes most of these fast; the last tests split the real
Qwen3-0.6B and serve it through torchrun.
"""

from __future__ import annotations

import gc
import json
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
import torch

from engine.config import EngineConfig
from engine.core.engine import LLMEngine
from engine.core.scheduler import ScheduledBatch
from engine.core.sequence import SamplingParams, Sequence
from engine.distributed.executor import decode_plan, encode_plan
from engine.distributed.parallel import check_parallel, layer_range
from engine.distributed.worker import local_cluster
from engine.model.loader import load_config
from greedy_check import LOGIT_TOL, LogitsRecorder, TopKRecorder, check_greedy_match
from prompts import PROMPTS
from tiny_qwen3 import make_tiny_qwen3

pytestmark = pytest.mark.distributed

FP32_TOL = LOGIT_TOL[torch.float32]
LAYOUTS = [(2, 1), (1, 2), (2, 2)]  # (tensor_parallel_size, pipeline_parallel_size)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_qwen3(tmp_path_factory.mktemp("tiny_untied"))


@pytest.fixture(scope="module")
def tiny_tied(tmp_path_factory):
    return make_tiny_qwen3(tmp_path_factory.mktemp("tiny_tied"), tie_word_embeddings=True, seed=1)


def _config(model: str, **overrides) -> EngineConfig:
    options = dict(model=model, device="cpu", dtype="float32", block_size=8, num_blocks=96, max_model_len=256,
                   debug_invariants=True)
    options.update(overrides)
    return EngineConfig(**options)


def _drive(engine: LLMEngine, prompts: list[list[int]], params: SamplingParams) -> list[Sequence]:
    seqs = [Sequence(i, list(p), params) for i, p in enumerate(prompts)]
    for seq in seqs:
        engine.add_request(seq)
    while engine.has_work():
        engine.step()
    bm = engine.block_manager
    assert bm.num_free_blocks == bm.num_blocks and len(bm.cpu_free) == bm.num_cpu_blocks, "blocks leaked"
    return seqs


def _single(config: EngineConfig, prompts, params):
    engine = LLMEngine(config)
    recorder = LogitsRecorder()
    engine.logits_hook = recorder
    seqs = _drive(engine, prompts, params)
    return seqs, recorder


def _split(config: EngineConfig, prompts, params, refs):
    with local_cluster(config) as parallel:
        engine = LLMEngine(config, parallel=parallel)
        try:
            engine.executor.debug_topk = 16
            recorder = TopKRecorder(refs)
            engine.topk_hook = recorder
            seqs = _drive(engine, prompts, params)
            shard = {
                "layers": engine.model.num_local_layers,
                "heads": engine.model.num_local_heads,
                "kv_heads": engine.model.num_local_kv_heads,
                "cache": tuple(engine.model_runner.kv_cache.data.shape),
            }
        finally:
            engine.shutdown()
    return seqs, recorder, shard


def _assert_same(ref_seqs, ref_rec, seqs, rec, label: str) -> list:
    near_ties = []
    for a, b in zip(ref_seqs, seqs):
        step = check_greedy_match(a.output_token_ids, ref_rec.steps[a.seq_id], b.output_token_ids,
                                  rec.steps[b.seq_id], FP32_TOL, f"{label}: request {a.seq_id}")
        if step is not None:
            near_ties.append((a.seq_id, step))
    return near_ties


def _prompts(seed: int, n: int, vocab: int, shared: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    prefix = [rng.randrange(1, vocab) for _ in range(shared)]
    return [prefix + [rng.randrange(1, vocab) for _ in range(rng.randrange(3, 40))] for _ in range(n)]


# --- pure functions --------------------------------------------------------------------------------


def test_layer_range_covers_every_layer_once():
    for layers, stages in [(28, 2), (28, 3), (36, 4), (4, 4), (5, 2)]:
        ranges = [layer_range(layers, stages, s) for s in range(stages)]
        assert ranges[0][0] == 0 and ranges[-1][1] == layers
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
        sizes = [e - s for s, e in ranges]
        assert max(sizes) - min(sizes) <= 1


def test_check_parallel_names_the_problem(tiny):
    cfg = load_config(tiny)
    check_parallel(cfg, 2, 4)
    with pytest.raises(ValueError, match="num_key_value_heads=2"):
        check_parallel(cfg, 4, 1)
    with pytest.raises(ValueError, match="exceeds the model's 4 layers"):
        check_parallel(cfg, 1, 5)


def test_plan_roundtrip_reproduces_the_batch():
    seqs = [Sequence(0, list(range(1, 21)), SamplingParams(temperature=0)),
            Sequence(1, list(range(5, 12)), SamplingParams(temperature=0.7, top_p=0.9))]
    seqs[0].num_computed_tokens, seqs[0].block_table = 16, [3, 9, 4]
    seqs[1].block_table = [7]
    batch = ScheduledBatch()
    batch.add(seqs[0], 4)  # the last 4 of 20 tokens: samples
    batch.add(seqs[1], 5)  # a chunk: 5 of 7 tokens, does not sample
    batch.swap_out = [(2, 0), (5, 1)]
    view, params, topk = decode_plan(encode_plan(batch, block_size=8, debug_topk=3))
    assert [s.num_computed_tokens for s in view.seqs] == [16, 0]
    assert view.num_new_tokens == [4, 5] and view.do_sample == [True, False]
    assert view.seqs[0].token_ids[16:20] == [17, 18, 19, 20] and view.seqs[1].token_ids[0:5] == [5, 6, 7, 8, 9]
    assert view.seqs[0].block_table == [3, 9, 4] and view.seqs[1].block_table == [7]
    assert view.swap_out == [(2, 0), (5, 1)] and view.swap_in == [] and topk == 3
    assert [(p.temperature, p.top_p) for p in params] == [(0.0, 1.0)]


# --- the tiny model, every layout --------------------------------------------------------------------


@pytest.mark.parametrize("tp,pp", LAYOUTS)
def test_split_tiny_model_matches_single_process(tiny, tp, pp):
    prompts = _prompts(0, 6, 512)
    params = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)
    ref, ref_rec = _single(_config(str(tiny)), prompts, params)
    got, rec, shard = _split(_config(str(tiny), tensor_parallel_size=tp, pipeline_parallel_size=pp),
                             prompts, params, {s.seq_id: ref_rec.steps[s.seq_id] for s in ref})
    # The work really is split: the driver holds its stage's layers and its slice of the heads.
    assert shard["layers"] == 4 // pp and shard["heads"] == 4 // tp and shard["kv_heads"] == 2 // tp
    assert shard["cache"][0] == 4 // pp and shard["cache"][4] == 2 // tp
    near_ties = _assert_same(ref, ref_rec, got, rec, f"tp={tp} pp={pp}")
    assert len(near_ties) <= 1, near_ties


def test_split_with_prefix_caching_chunking_and_swap(tiny):
    """The scheduler runs only on the driver and is unchanged; this checks the plan carries everything
    it decides, prefix-cache hits (non-zero computed tokens), prefill chunks, and swap copies, to
    every rank."""
    prompts = _prompts(1, 10, 512, shared=20)
    params = SamplingParams(temperature=0, max_tokens=30, ignore_eos=True)
    features = dict(enable_prefix_caching=True, enable_chunked_prefill=True, max_batch_tokens=32, max_num_seqs=8,
                    num_blocks=40, preemption_mode="swap", swap_space_gb=0.002)
    ref, ref_rec = _single(_config(str(tiny), **features), prompts, params)
    config = _config(str(tiny), tensor_parallel_size=2, pipeline_parallel_size=2, **features)
    with local_cluster(config) as parallel:
        engine = LLMEngine(config, parallel=parallel)
        try:
            engine.executor.debug_topk = 16
            rec = TopKRecorder({s.seq_id: ref_rec.steps[s.seq_id] for s in ref})
            engine.topk_hook = rec
            got = _drive(engine, prompts, params)
            assert engine.scheduler.num_preemptions > 0 and engine.scheduler.num_swap_outs > 0
            assert engine.block_manager.prefix_hit_tokens > 0
        finally:
            engine.shutdown()
    assert len(_assert_same(ref, ref_rec, got, rec, "tp=2 pp=2 with features")) <= 1


def test_pipeline_with_tied_embeddings_loads_the_table_on_both_ends(tiny_tied):
    prompts = _prompts(2, 5, 512)
    params = SamplingParams(temperature=0, max_tokens=20, ignore_eos=True)
    ref, ref_rec = _single(_config(str(tiny_tied)), prompts, params)
    got, rec, shard = _split(_config(str(tiny_tied), pipeline_parallel_size=2), prompts, params,
                             {s.seq_id: ref_rec.steps[s.seq_id] for s in ref})
    assert shard["layers"] == 2
    assert len(_assert_same(ref, ref_rec, got, rec, "tied pp=2")) <= 1


# --- the real model ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen_reference(model_path, tokenizer):
    """Single-process greedy outputs of Qwen3-0.6B (fp32), freed before the split runs."""
    prompts = [tokenizer(p).input_ids[:40] for p in PROMPTS[:6]]
    params = SamplingParams(temperature=0, max_tokens=24)
    config = EngineConfig(model=str(model_path), device="cpu", dtype="float32", num_blocks=64, max_model_len=512)
    seqs, rec = _single(config, prompts, params)
    gc.collect()
    return prompts, params, seqs, rec


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 2)])
def test_split_qwen_matches_single_process(model_path, qwen_reference, tp, pp):
    prompts, params, ref, ref_rec = qwen_reference
    config = EngineConfig(model=str(model_path), device="cpu", dtype="float32", num_blocks=64, max_model_len=512,
                          tensor_parallel_size=tp, pipeline_parallel_size=pp, debug_invariants=True)
    got, rec, shard = _split(config, prompts, params, {s.seq_id: ref_rec.steps[s.seq_id] for s in ref})
    assert shard["layers"] == 28 // pp and shard["heads"] == 16 // tp and shard["kv_heads"] == 8 // tp
    near_ties = _assert_same(ref, ref_rec, got, rec, f"Qwen3-0.6B tp={tp} pp={pp}")
    print(f"\nQwen3-0.6B tp={tp} pp={pp}: {len(ref) - len(near_ties)}/{len(ref)} identical; near-ties {near_ties}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.model
@pytest.mark.slow
def test_torchrun_server_with_tensor_parallelism(model_path, tokenizer, qwen_reference, tmp_path):
    """The launch path a GPU box uses, end to end: torchrun starts two ranks, rank 0 serves HTTP."""
    prompts, _, ref, _ = qwen_reference
    port, master = _free_port(), _free_port()
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2", "--master-addr", "127.0.0.1",
           "--master-port", str(master), "-m", "engine.api.server", "--model", str(model_path),
           "--tensor-parallel-size", "2", "--device", "cpu", "--num-blocks", "64", "--max-model-len", "512",
           "--port", str(port), "--log-level", "warning"]
    log = open(tmp_path / "torchrun.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**__import__("os").environ, "OMP_NUM_THREADS": "2"})
    try:
        deadline = time.time() + 300
        while True:
            assert proc.poll() is None, (tmp_path / "torchrun.log").read_text()[-3000:]
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                    break
            except Exception:
                assert time.time() < deadline, "server did not start"
                time.sleep(1)
        body = json.dumps({"model": str(model_path), "prompt": prompts[0], "max_tokens": 24, "temperature": 0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            text = json.loads(resp.read())["choices"][0]["text"]
        assert text == tokenizer.decode(ref[0].output_token_ids, skip_special_tokens=True)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
