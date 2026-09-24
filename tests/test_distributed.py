"""Tensor and pipeline parallelism: a split model must give the answers of the whole one.

Every rank is a real process talking over torch.distributed: gloo on CPU, NCCL with one GPU per rank
under PAGEDSERVE_TEST_DEVICE=cuda (layouts needing more GPUs than the machine has are skipped). The
reference is the single-process engine on the same weights; comparisons use the near-tie rule, since
a tensor-parallel all-reduce sums partial results in a different order than one big matmul. Pipeline
layouts run pipelined (several batches in flight) unless a test sets pipeline_depth=1, and check that
the pipeline really filled. The tiny random-weight Qwen3 (tests/tiny_qwen3.py) makes most of these
fast; the last tests split the real Qwen3-0.6B and serve it through torchrun.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest
import torch

from conftest import TEST_DEVICE
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
BF16_TOL = LOGIT_TOL[torch.bfloat16]
# (tensor_parallel_size, pipeline_parallel_size, pipeline_depth). pp=3 gives the tiny model a middle
# stage, the only kind that both receives and sends hidden states.
LAYOUTS = [(2, 1, 1), (1, 2, 1), (1, 2, 2), (2, 2, 2), (1, 3, 3)]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_qwen3(tmp_path_factory.mktemp("tiny_untied"))


@pytest.fixture(scope="module")
def tiny_tied(tmp_path_factory):
    return make_tiny_qwen3(tmp_path_factory.mktemp("tiny_tied"), tie_word_embeddings=True, seed=1)


def _needs_gpus(world: int) -> None:
    """NCCL needs a GPU per rank; on CPU (gloo) every layout runs."""
    if TEST_DEVICE.startswith("cuda") and torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs, found {torch.cuda.device_count()}")


def _config(model: str, **overrides) -> EngineConfig:
    options = dict(model=model, device=TEST_DEVICE, dtype="float32", block_size=8, num_blocks=96, max_model_len=256,
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


def _count_batches_in_flight(engine: LLMEngine) -> list[int]:
    """The most batches the engine has had in flight at once, updated as it runs."""
    peak = [0]
    submit = engine.executor.submit

    def counting_submit(batch):
        peak[0] = max(peak[0], len(engine.in_flight) + 1)
        return submit(batch)

    engine.executor.submit = counting_submit
    return peak


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
            peak = _count_batches_in_flight(engine)
            seqs = _drive(engine, prompts, params)
            shard = {
                "layers": engine.model.num_local_layers,
                "heads": engine.model.num_local_heads,
                "kv_heads": engine.model.num_local_kv_heads,
                "cache": tuple(engine.model_runner.kv_cache.data.shape),
                "peak_in_flight": peak[0],
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


@pytest.mark.parametrize("tp,pp,depth", LAYOUTS)
def test_split_tiny_model_matches_single_process(tiny, tp, pp, depth):
    _needs_gpus(tp * pp)
    prompts = _prompts(0, 6, 512)
    params = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)
    ref, ref_rec = _single(_config(str(tiny)), prompts, params)
    config = _config(str(tiny), tensor_parallel_size=tp, pipeline_parallel_size=pp, pipeline_depth=depth)
    got, rec, shard = _split(config, prompts, params, {s.seq_id: ref_rec.steps[s.seq_id] for s in ref})
    # The work really is split: the driver holds its stage's layers and its slice of the heads.
    first, end = layer_range(4, pp, 0)
    assert shard["layers"] == end - first and shard["heads"] == 4 // tp and shard["kv_heads"] == 2 // tp
    assert shard["cache"][0] == end - first and shard["cache"][4] == 2 // tp
    assert shard["peak_in_flight"] == depth, "the pipeline never filled"
    near_ties = _assert_same(ref, ref_rec, got, rec, f"tp={tp} pp={pp} depth={depth}")
    assert len(near_ties) <= 1, near_ties


def test_split_with_prefix_caching_chunking_and_swap(tiny):
    """The scheduler runs only on the driver; this checks the plan carries everything it decides,
    prefix-cache hits (non-zero computed tokens), prefill chunks, and swap copies, to every rank, with
    two batches in flight."""
    _needs_gpus(4)
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
            peak = _count_batches_in_flight(engine)
            got = _drive(engine, prompts, params)
            assert engine.scheduler.num_preemptions > 0 and engine.scheduler.num_swap_outs > 0
            assert engine.block_manager.prefix_hit_tokens > 0
            assert peak[0] == 2
        finally:
            engine.shutdown()
    assert len(_assert_same(ref, ref_rec, got, rec, "tp=2 pp=2 with features")) <= 1


def test_pipeline_with_tied_embeddings_loads_the_table_on_both_ends(tiny_tied):
    _needs_gpus(2)
    prompts = _prompts(2, 5, 512)
    params = SamplingParams(temperature=0, max_tokens=20, ignore_eos=True)
    ref, ref_rec = _single(_config(str(tiny_tied)), prompts, params)
    got, rec, shard = _split(_config(str(tiny_tied), pipeline_parallel_size=2), prompts, params,
                             {s.seq_id: ref_rec.steps[s.seq_id] for s in ref})
    assert shard["layers"] == 2 and shard["peak_in_flight"] == 2
    assert len(_assert_same(ref, ref_rec, got, rec, "tied pp=2")) <= 1


# --- the real model ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen_reference(model_path, tokenizer):
    """Single-process greedy outputs of Qwen3-0.6B (fp32), freed before the split runs."""
    prompts = [tokenizer(p).input_ids[:40] for p in PROMPTS[:6]]
    params = SamplingParams(temperature=0, max_tokens=24)
    config = EngineConfig(model=str(model_path), device=TEST_DEVICE, dtype="float32", num_blocks=64, max_model_len=512)
    seqs, rec = _single(config, prompts, params)
    gc.collect()
    return prompts, params, seqs, rec


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 2)])
def test_split_qwen_matches_single_process(model_path, qwen_reference, tp, pp):
    _needs_gpus(tp * pp)
    prompts, params, ref, ref_rec = qwen_reference
    config = EngineConfig(model=str(model_path), device=TEST_DEVICE, dtype="float32", num_blocks=64, max_model_len=512,
                          tensor_parallel_size=tp, pipeline_parallel_size=pp, debug_invariants=True)
    got, rec, shard = _split(config, prompts, params, {s.seq_id: ref_rec.steps[s.seq_id] for s in ref})
    assert shard["layers"] == 28 // pp and shard["heads"] == 16 // tp and shard["kv_heads"] == 8 // tp
    assert shard["peak_in_flight"] == pp  # pipelined by default
    near_ties = _assert_same(ref, ref_rec, got, rec, f"Qwen3-0.6B tp={tp} pp={pp}")
    print(f"\nQwen3-0.6B tp={tp} pp={pp}: {len(ref) - len(near_ties)}/{len(ref)} identical; near-ties {near_ties}")


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.skipif(importlib.util.find_spec("flashinfer") is None, reason="needs flashinfer-python")
@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 2)])
def test_split_qwen_bf16_flashinfer_matches_one_gpu(model_path, tokenizer, tp, pp):
    """What the distributed benchmarks run: bf16, FlashInfer, NCCL, chunked prefill and prefix caching,
    and under pipeline parallelism two batches in flight. Compared with one GPU by the bf16 near-tie
    rule."""
    if not TEST_DEVICE.startswith("cuda"):
        pytest.skip("needs PAGEDSERVE_TEST_DEVICE=cuda")
    _needs_gpus(tp * pp)
    prompts = [tokenizer(p).input_ids for p in PROMPTS[:12]]
    params = SamplingParams(temperature=0, max_tokens=32)
    base = dict(model=str(model_path), device="cuda", dtype="bfloat16", attention_backend="flashinfer", num_blocks=1024,
                max_model_len=2048, enable_chunked_prefill=True, max_batch_tokens=512, enable_prefix_caching=True,
                debug_invariants=True)
    ref, ref_rec = _single(EngineConfig(**base), prompts, params)
    gc.collect()
    torch.cuda.empty_cache()
    config = EngineConfig(**base, tensor_parallel_size=tp, pipeline_parallel_size=pp)
    got, rec, shard = _split(config, prompts, params, {s.seq_id: ref_rec.steps[s.seq_id] for s in ref})
    assert shard["peak_in_flight"] == pp
    near_ties = [check_greedy_match(a.output_token_ids, ref_rec.steps[a.seq_id], b.output_token_ids, rec.steps[b.seq_id],
                                    BF16_TOL, f"bf16 tp={tp} pp={pp}: request {a.seq_id}") for a, b in zip(ref, got)]
    print(f"\nQwen3-0.6B bf16 FlashInfer tp={tp} pp={pp}: near-tie divergences {near_ties}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 2)])
def test_torchrun_server(model_path, tokenizer, qwen_reference, tmp_path, tp, pp):
    """The launch path a GPU box uses, end to end: torchrun starts two ranks and rank 0 serves HTTP.
    One client disconnects mid-stream while another request runs. The survivor must get the reference
    text, and the aborted request's blocks must come back; under pipelining the abort can land while
    its batch is in flight."""
    _needs_gpus(tp * pp)
    prompts, _, ref, _ = qwen_reference
    port, master = _free_port(), _free_port()
    base = f"http://127.0.0.1:{port}"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2", "--master-addr", "127.0.0.1",
           "--master-port", str(master), "-m", "engine.api.server", "--model", str(model_path),
           "--tensor-parallel-size", str(tp), "--pipeline-parallel-size", str(pp), "--device", TEST_DEVICE,
           "--dtype", "float32", "--num-blocks", "64", "--max-model-len", "512", "--port", str(port),
           "--log-level", "warning"]
    log = open(tmp_path / "torchrun.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**__import__("os").environ, "OMP_NUM_THREADS": "2"})
    try:
        deadline = time.time() + 300
        while True:
            assert proc.poll() is None, (tmp_path / "torchrun.log").read_text()[-3000:]
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=2):
                    break
            except Exception:
                assert time.time() < deadline, "server did not start"
                time.sleep(1)

        survivor: dict = {}

        def complete() -> None:
            body = json.dumps({"model": str(model_path), "prompt": prompts[0], "max_tokens": 24, "temperature": 0}).encode()
            req = urllib.request.Request(f"{base}/v1/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                survivor["text"] = json.loads(resp.read())["choices"][0]["text"]

        thread = threading.Thread(target=complete)
        thread.start()
        body = json.dumps({"model": str(model_path), "prompt": prompts[1], "max_tokens": 400, "temperature": 0,
                           "ignore_eos": True, "stream": True}).encode()
        sock = socket.create_connection(("127.0.0.1", port))
        sock.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                     b"Content-Length: %d\r\n\r\n" % len(body) + body)
        received, deadline = b"", time.time() + 300
        while received.count(b"data: ") < 3:
            assert time.time() < deadline, "no tokens streamed"
            received += sock.recv(65536)
        sock.close()  # disconnect mid-stream
        thread.join(timeout=300)
        assert survivor.get("text") == tokenizer.decode(ref[0].output_token_ids, skip_special_tokens=True)
        deadline = time.time() + 120
        while True:
            stats = _get_json(f"{base}/stats")
            if (stats["num_running"] == 0 and stats["num_batches_in_flight"] == 0
                    and stats["num_free_blocks"] == stats["num_blocks"]):
                break
            assert time.time() < deadline, f"blocks were not freed: {stats}"
            time.sleep(0.2)
        assert stats["total_aborted"] == 1
    finally:
        start = time.time()
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        stopped_in = time.time() - start
    # A Ctrl-C, forwarded by torchrun to every rank, must stop the workers through the driver's shutdown
    # plan: no rank may die of the signal and leave the driver waiting for it.
    text = (tmp_path / "torchrun.log").read_text()
    assert "KeyboardInterrupt" not in text and "forcefully exiting" not in text, text[-3000:]
    assert stopped_in < 20, f"shutdown took {stopped_in:.0f}s"
