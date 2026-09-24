"""Benchmark client: replays a workload against an OpenAI-compatible server and measures it.

Per request: time to first token (TTFT), inter-token latencies (ITL), end-to-end latency, output
tokens. Per run: output-token throughput over the whole run and at steady state (tokens produced
between the 10th and 90th percentile arrival times, excluding warm-up and drain), TTFT/ITL/E2E
percentiles, and, from this engine's /stats, KV-cache utilization, preemptions and peak memory. The
same client drives this engine and vLLM, so both are measured on identical traffic with identical code.

    # one server that is already running (this engine, or vLLM for the baseline)
    python -m bench.run_bench run --url http://127.0.0.1:8000 --workload bench/data/sharegpt.jsonl \\
        --rates 2,4,8,inf --duration 60 --label ours --out bench/results/latency

    # launch this engine once per configuration and run every rate against each
    python -m bench.run_bench sweep --spec bench/sweeps/policies.json --out bench/results/policies

A configuration whose arguments ask for tensor or pipeline parallelism is launched under torchrun,
one process per GPU on this machine.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

from bench.workload import Request, load, poisson_arrivals

MODEL = "Qwen/Qwen3-0.6B-Base"
SUMMARY_FIELDS = [
    "label", "rate", "num_requests", "num_failed", "duration_s", "throughput_tok_s", "steady_tok_s", "request_rate",
    "ttft_mean", "ttft_p50", "ttft_p95", "ttft_p99", "itl_mean", "itl_p50", "itl_p95", "itl_p99",
    "e2e_p50", "e2e_p99", "preemptions", "preemptions_per_s", "kv_usage_mean", "kv_usage_max", "peak_memory_gib",
]


async def _send(session: aiohttp.ClientSession, url: str, model: str, req: Request, idx: int, t0: float) -> dict:
    payload = {
        "model": model, "prompt": req.prompt_ids, "max_tokens": req.output_len, "temperature": 0.0,
        "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
    }
    rec = {"idx": idx, "group": req.group, "prompt_len": len(req.prompt_ids), "output_len": req.output_len,
           "sent": time.perf_counter() - t0, "token_times": [], "output_tokens": 0, "error": ""}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
            else:
                buffer = b""
                async for chunk in resp.content.iter_any():
                    buffer += chunk
                    while b"\n\n" in buffer:
                        event, buffer = buffer.split(b"\n\n", 1)
                        now = time.perf_counter() - t0
                        event = event.strip()
                        if not event.startswith(b"data:") or event[5:].strip() == b"[DONE]":
                            continue
                        msg = json.loads(event[5:])
                        if "error" in msg:
                            rec["error"] = str(msg["error"].get("message", msg["error"]))
                        elif msg.get("choices"):
                            rec["token_times"].append(now)
                        elif msg.get("usage"):
                            rec["output_tokens"] = msg["usage"]["completion_tokens"]
    except Exception as exc:  # a failed request is data, not a crash
        rec["error"] = repr(exc)
    rec["end"] = time.perf_counter() - t0
    rec["success"] = bool(rec["token_times"]) and not rec["error"]
    if rec["success"] and not rec["output_tokens"]:
        rec["output_tokens"] = len(rec["token_times"])
    return rec


async def _fetch_stats(session: aiohttp.ClientSession, base: str) -> dict | None:
    try:
        async with session.get(f"{base}/stats") as resp:
            return await resp.json() if resp.status == 200 else None
    except Exception:
        return None


async def _poll_stats(session: aiohttp.ClientSession, base: str, t0: float, series: list, period: float = 0.25) -> None:
    while True:
        stats = await _fetch_stats(session, base)
        if stats is not None:
            series.append({"t": time.perf_counter() - t0, **stats})
        await asyncio.sleep(period)


async def run_rate(base: str, model: str, requests: list[Request], rate: float, seed: int) -> dict:
    arrivals = (
        [r.arrival for r in requests] if all(r.arrival is not None for r in requests)
        else poisson_arrivals(len(requests), rate, seed)
    )
    url = f"{base}/v1/completions"
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=None)) as session:
        before = await _fetch_stats(session, base)
        series: list[dict] = []
        t0 = time.perf_counter()
        poller = asyncio.create_task(_poll_stats(session, base, t0, series)) if before is not None else None
        tasks = []
        for idx, (req, at) in enumerate(zip(requests, arrivals)):
            delay = t0 + at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(_send(session, url, model, req, idx, t0)))
        records = await asyncio.gather(*tasks)
        duration = time.perf_counter() - t0
        if poller is not None:
            poller.cancel()
        after = await _fetch_stats(session, base)
    return {"records": records, "arrivals": arrivals, "duration": duration, "before": before, "after": after, "series": series}


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def summarize(label: str, rate: float, result: dict) -> dict:
    records = result["records"]
    ok = [r for r in records if r["success"]]
    ttft = [r["token_times"][0] - r["sent"] for r in ok]
    itl = [b - a for r in ok for a, b in zip(r["token_times"], r["token_times"][1:])]
    e2e = [r["token_times"][-1] - r["sent"] for r in ok]
    tokens = sum(r["output_tokens"] for r in ok)
    span = max((r["end"] for r in records), default=0.0) - min(result["arrivals"], default=0.0)
    lo, hi = np.percentile(result["arrivals"], [10, 90]) if len(records) > 1 else (0.0, 0.0)
    steady = (
        sum(1 for r in ok for t in r["token_times"] if lo <= t <= hi) / (hi - lo) if hi > lo else float("nan")
    )
    before, after, series = result["before"], result["after"], result["series"]
    preemptions = after["total_preemptions"] - before["total_preemptions"] if before and after else float("nan")
    kv = [s["kv_usage"] for s in series]
    return {
        "label": label, "rate": rate, "num_requests": len(records), "num_failed": len(records) - len(ok),
        "duration_s": result["duration"], "throughput_tok_s": tokens / span if span > 0 else float("nan"),
        "steady_tok_s": steady, "request_rate": len(ok) / span if span > 0 else float("nan"),
        "ttft_mean": float(np.mean(ttft)) if ttft else float("nan"), "ttft_p50": _pct(ttft, 50),
        "ttft_p95": _pct(ttft, 95), "ttft_p99": _pct(ttft, 99), "itl_mean": float(np.mean(itl)) if itl else float("nan"),
        "itl_p50": _pct(itl, 50), "itl_p95": _pct(itl, 95), "itl_p99": _pct(itl, 99),
        "e2e_p50": _pct(e2e, 50), "e2e_p99": _pct(e2e, 99), "preemptions": preemptions,
        "preemptions_per_s": preemptions / result["duration"] if result["duration"] else float("nan"),
        "kv_usage_mean": float(np.mean(kv)) if kv else float("nan"), "kv_usage_max": max(kv) if kv else float("nan"),
        "peak_memory_gib": after.get("peak_memory_bytes", float("nan")) / (1 << 30) if after else float("nan"),
    }


def write_results(out: Path, label: str, rate: float, result: dict, summary: dict, save_tokens: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_rate{rate:g}"
    with open(out / f"{stem}_requests.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "group", "prompt_len", "output_len", "output_tokens", "sent", "ttft", "e2e", "tpot", "success", "error"])
        for r in result["records"]:
            tt = r["token_times"]
            ttft = tt[0] - r["sent"] if tt else ""
            e2e = tt[-1] - r["sent"] if tt else ""
            tpot = (tt[-1] - tt[0]) / (len(tt) - 1) if len(tt) > 1 else ""
            w.writerow([r["idx"], r["group"], r["prompt_len"], r["output_len"], r["output_tokens"], f"{r['sent']:.6f}",
                        ttft, e2e, tpot, r["success"], r["error"]])
    if save_tokens:  # every token's arrival time, for ITL over time; large at high rates, so opt-in
        with open(out / f"{stem}_tokens.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "group", "token", "t"])
            for r in result["records"]:
                for i, t in enumerate(r["token_times"]):
                    w.writerow([r["idx"], r["group"], i, f"{t:.6f}"])
    if result["series"]:
        keys = ["t", "num_running", "num_waiting", "num_swapped", "kv_usage", "total_preemptions", "num_free_blocks"]
        with open(out / f"{stem}_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(result["series"])
    summary_path = out / "summary.csv"
    new = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(summary)


def _requests_for(workload: list[Request], rate: float, duration: float | None, num_requests: int | None) -> list[Request]:
    if all(r.arrival is not None for r in workload):  # structured workload: its own clock
        return workload if duration is None else [r for r in workload if r.arrival < duration]
    n = num_requests or len(workload)
    if duration is not None and not math.isinf(rate):
        n = min(len(workload), max(1, math.ceil(rate * duration)))
    return workload[:n]


def run_all(base: str, model: str, workload_path: str, rates: list[float], label: str, out: Path, seed: int,
            duration: float | None, num_requests: int | None, save_tokens: bool = False) -> None:
    workload = load(workload_path)
    for rate in rates:
        requests = _requests_for(workload, rate, duration, num_requests)
        result = asyncio.run(run_rate(base, model, requests, rate, seed))
        summary = summarize(label, rate, result)
        write_results(out, label, rate, result, summary, save_tokens)
        print(f"[{label}] rate {rate:g}: {summary['num_requests']} requests ({summary['num_failed']} failed), "
              f"{summary['throughput_tok_s']:.0f} tok/s (steady {summary['steady_tok_s']:.0f}), "
              f"TTFT p50/p99 {summary['ttft_p50']:.3f}/{summary['ttft_p99']:.3f}s, "
              f"ITL p50/p99 {summary['itl_p50'] * 1e3:.1f}/{summary['itl_p99'] * 1e3:.1f}ms", flush=True)


def _wait_healthy(base: str, proc: subprocess.Popen, timeout: float) -> None:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} during startup")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise TimeoutError("server did not become healthy")


def _world_size(args: list[str]) -> int:
    size = 1
    for flag in ("--tensor-parallel-size", "--pipeline-parallel-size"):
        if flag in args:
            size *= int(args[args.index(flag) + 1])
    return size


def _server_command(args: list[str]) -> list[str]:
    """This engine's server, under torchrun when the arguments split the model across GPUs."""
    world = _world_size(args)
    if world == 1:
        return [sys.executable, "-m", "engine.api.server", *args]
    with socket.socket() as s:  # the ranks' rendezvous, on loopback
        s.bind(("127.0.0.1", 0))
        master_port = s.getsockname()[1]
    return [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(world), "--master-addr", "127.0.0.1",
            "--master-port", str(master_port), "-m", "engine.api.server", *args]


def sweep(spec_path: str, out: Path, port: int) -> None:
    """Launch this engine once per configuration in the spec and run every rate against it."""
    spec = json.loads(Path(spec_path).read_text())
    out.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{port}"
    for config in spec["configs"]:
        label = config["label"]
        cmd = _server_command(["--port", str(port), "--log-level", "warning", "--step-log", str(out / f"{label}_steps.csv"),
                               *spec.get("server_args", []), *config.get("args", [])])
        print(f"[{label}] launching: {' '.join(cmd)}", flush=True)
        with open(out / f"{label}_server.log", "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                _wait_healthy(base, proc, timeout=spec.get("startup_timeout", 900))
                rates = [float(r) for r in config.get("rates", spec["rates"])]  # JSON has no Infinity: "inf"
                run_all(base, spec.get("model", MODEL), config.get("workload", spec["workload"]), rates, label, out,
                        spec.get("seed", 0), spec.get("duration"), spec.get("num_requests"), spec.get("save_tokens", False))
            finally:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="benchmark a server that is already running")
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument("--model", default=MODEL)
    r.add_argument("--workload", required=True)
    r.add_argument("--rates", default="inf", help="comma-separated request rates (req/s); inf sends everything at once")
    r.add_argument("--duration", type=float, default=None, help="seconds of arrivals per rate (sets the request count)")
    r.add_argument("--num-requests", type=int, default=None)
    r.add_argument("--label", required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--save-tokens", action="store_true", help="also write every token's arrival time")
    r.add_argument("--out", default="bench/results")
    s = sub.add_parser("sweep", help="launch this engine per configuration from a JSON spec")
    s.add_argument("--spec", required=True)
    s.add_argument("--port", type=int, default=int(os.environ.get("PAGEDSERVE_BENCH_PORT", 8100)))
    s.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "run":
        rates = [float(x) for x in args.rates.split(",")]
        run_all(args.url.rstrip("/"), args.model, args.workload, rates, args.label, Path(args.out), args.seed,
                args.duration, args.num_requests, args.save_tokens)
    else:
        sweep(args.spec, Path(args.out), args.port)


if __name__ == "__main__":
    main()
