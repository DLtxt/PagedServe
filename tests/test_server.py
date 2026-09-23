"""The HTTP layer on the real model: OpenAI-compatible streaming and errors, and gate 5 (a client
disconnect frees its blocks). The server runs in a background thread; tests talk to it over
sockets only, the way a real client would."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
import uvicorn

from conftest import TEST_DEVICE
from engine.api.server import build_app
from engine.config import EngineConfig
from engine.core.engine import LLMEngine
from engine.model.attention import greedy_generate_contiguous
from prompts import PROMPTS

pytestmark = [pytest.mark.model, pytest.mark.slow]

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"


@pytest.fixture(scope="module")
def server(qwen_fp32, tokenizer):
    config = EngineConfig(device=TEST_DEVICE, dtype="float32", num_blocks=128, max_model_len=1024, debug_invariants=True)
    engine = LLMEngine(config, model=qwen_fp32, tokenizer=tokenizer)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(build_app(engine, MODEL_NAME), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not srv.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=10)


def post(base: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(f"{base}/v1/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def stream(base: str, payload: dict) -> list:
    req = urllib.request.Request(f"{base}/v1/completions", json.dumps({**payload, "stream": True}).encode(), {"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data: "):
                data = line[len("data: "):]
                events.append(data if data == "[DONE]" else json.loads(data))
    return events


def stats(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/stats", timeout=30) as resp:
        return json.loads(resp.read())


def wait_until_idle(base: str, timeout: float = 60) -> dict:
    deadline = time.time() + timeout
    while True:
        s = stats(base)
        if s["num_running"] == 0 and s["num_waiting"] == 0 and s["num_free_blocks"] == s["num_blocks"]:
            return s
        assert time.time() < deadline, f"blocks were not freed: {s}"
        time.sleep(0.1)


@pytest.fixture(scope="module")
def expected(qwen_fp32, tokenizer):
    """Greedy continuation of PROMPTS[1] from the contiguous reference path, computed up front so it
    never runs concurrently with the server."""
    ids = tokenizer(PROMPTS[1]).input_ids
    tokens, _ = greedy_generate_contiguous(qwen_fp32, ids, 12, qwen_fp32.config.eos_token_ids)
    return {"prompt_ids": ids, "tokens": tokens, "text": tokenizer.decode(tokens, skip_special_tokens=True)}


def test_streaming_matches_offline_greedy(server, expected):
    events = stream(server, {"model": MODEL_NAME, "prompt": PROMPTS[1], "max_tokens": 12, "temperature": 0,
                             "stream_options": {"include_usage": True}})
    assert events[-1] == "[DONE]"
    token_events = [e for e in events[:-1] if e["choices"]]
    assert len(token_events) == 12, "one event per generated token"
    assert "".join(e["choices"][0]["text"] for e in token_events) == expected["text"]
    assert [e["choices"][0]["finish_reason"] for e in token_events] == [None] * 11 + ["length"]
    usage = events[-2]["usage"]
    assert events[-2]["choices"] == []
    assert usage == {"prompt_tokens": len(expected["prompt_ids"]), "completion_tokens": 12, "total_tokens": len(expected["prompt_ids"]) + 12}
    wait_until_idle(server)


def test_non_streaming_and_token_id_prompts(server, expected):
    status, body = post(server, {"model": MODEL_NAME, "prompt": expected["prompt_ids"], "max_tokens": 12, "temperature": 0})
    assert status == 200
    assert body["object"] == "text_completion" and body["choices"][0]["text"] == expected["text"]
    assert body["choices"][0]["finish_reason"] == "length" and body["usage"]["completion_tokens"] == 12


def test_openai_style_errors(server):
    base = {"model": MODEL_NAME, "prompt": "hello"}
    for payload, status in [
        ({**base, "n": 2}, 400),
        ({**base, "logprobs": 1}, 400),
        ({**base, "model": "no-such-model"}, 404),
        ({**base, "max_tokens": 5000}, 400),  # beyond max_model_len
        ({**base, "temperature": -1}, 400),
        ({**base, "top_p": 0}, 400),
        ({"model": MODEL_NAME}, 400),  # no prompt
        ({**base, "prompt": []}, 400),
        ({**base, "prompt": [10**9]}, 400),  # token id out of range
    ]:
        got, body = post(server, payload)
        assert got == status, (payload, body)
        assert body["error"]["message"]
    wait_until_idle(server)


def _open_raw_request(base: str, payload: dict) -> socket.socket:
    host, port = base.removeprefix("http://").split(":")
    body = json.dumps(payload).encode()
    sock = socket.create_connection((host, int(port)))
    sock.sendall(
        b"POST /v1/completions HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
        % (host.encode(), len(body)) + body
    )
    return sock


def test_client_disconnect_mid_stream_frees_blocks(server):
    """Gate 5: kill the client mid-stream; the request must be aborted and its blocks freed."""
    before = stats(server)
    sock = _open_raw_request(server, {"model": MODEL_NAME, "prompt": PROMPTS[3], "max_tokens": 400,
                                      "temperature": 0, "ignore_eos": True, "stream": True})
    received = b""
    while received.count(b"data: ") < 3:
        received += sock.recv(65536)
    sock.close()
    after = wait_until_idle(server)
    assert after["total_aborted"] == before["total_aborted"] + 1
    assert after["num_generated_tokens"] - before["num_generated_tokens"] < 400, "generation kept going after the client left"


def test_client_disconnect_non_streaming_frees_blocks(server):
    before = stats(server)
    sock = _open_raw_request(server, {"model": MODEL_NAME, "prompt": PROMPTS[4], "max_tokens": 400,
                                      "temperature": 0, "ignore_eos": True})
    deadline = time.time() + 60
    while stats(server)["num_generated_tokens"] < before["num_generated_tokens"] + 3:
        assert time.time() < deadline
        time.sleep(0.05)
    sock.close()
    after = wait_until_idle(server)
    assert after["total_aborted"] == before["total_aborted"] + 1
    assert after["num_generated_tokens"] - before["num_generated_tokens"] < 400


def test_concurrent_streams_complete_without_leaks(server):
    results: dict[int, list] = {}

    def client(i: int) -> None:
        results[i] = stream(server, {"model": MODEL_NAME, "prompt": PROMPTS[i], "max_tokens": 6 + i, "temperature": 0,
                                     "ignore_eos": True, "stream_options": {"include_usage": True}})

    threads = [threading.Thread(target=client, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    for i in range(8):
        events = results[i]
        assert events[-1] == "[DONE]" and events[-2]["usage"]["completion_tokens"] == 6 + i
        assert sum(1 for e in events[:-1] if e["choices"]) == 6 + i
    wait_until_idle(server)
