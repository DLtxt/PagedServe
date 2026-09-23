"""Benchmark CSVs -> figures and tables.

Reads the directories run_bench.py, hf_baseline.py and preemption_crossover.py write under
--results, and renders whatever is present into --out:

    latency_throughput.png   p99 TTFT against achieved throughput per system: the hockey stick
    block_size.png           throughput, p99 ITL and KV usage against request rate per block size
    admission_policies.png   TTFT distribution, and p99 TTFT by prompt length (starvation)
    preemption.png           recompute vs swap cost against sequence length, crossover marked
    prefix_caching.png       TTFT against shared-prefix length with the cache on and off
    chunked_prefill.png      p99 inter-token latency over time with and without chunked prefill
    kv_usage.png             KV-cache utilization over time per request rate
    results.md               every summary as a table (the table view of every chart)

    python -m bench.plot --results bench/results --out bench/figures
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Validated reference palette (categorical, fixed order) and chart chrome, light surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # one hue, light to dark
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["system-ui", "-apple-system", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "lines.linewidth": 2.0, "lines.markersize": 6, "legend.frameon": False, "legend.labelcolor": INK2,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 150,
    })


def color_of(label: str, fixed: list[str], present: list[str] = ()) -> tuple[str, str]:
    """Colors follow the entity, never its rank: a label's slot is its position in the figure's fixed
    order, so a missing series never repaints the others. Unknown labels take the slots after it."""
    extras = [l for l in present if l not in fixed]
    slot = fixed.index(label) if label in fixed else len(fixed) + extras.index(label)
    return SERIES[slot % len(SERIES)], MARKERS[slot % len(MARKERS)]


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def num(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def end_label(ax, x: float, y: float, text: str) -> None:
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)


def plain_ticks(axis) -> None:
    """Plain numbers on a log axis ("0.5", "2", "100"), never "5 x 10^-1"."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    axis.set_major_formatter(fmt)
    axis.set_minor_formatter(fmt)


def finish(fig, path: Path, source: str) -> None:
    fig.text(0.01, 0.005, f"Source: {source}", fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def by_label(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    for rows_ in groups.values():
        rows_.sort(key=lambda r: num(r["rate"]))
    return groups


# --- figures -----------------------------------------------------------------------------------------


def latency_throughput(results: Path, out: Path) -> None:
    path = results / "latency" / "summary.csv"
    if not path.exists():
        return
    groups = by_label(read_csv(path))
    fixed = ["ours", "vllm", "hf-static"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for label, rows in groups.items():
        color, marker = color_of(label, fixed, list(groups))
        pts = [(num(r["steady_tok_s"]) if not math.isnan(num(r["steady_tok_s"])) else num(r["throughput_tok_s"]),
                num(r["ttft_p99"])) for r in rows]
        pts = [(x, y) for x, y in pts if not (math.isnan(x) or math.isnan(y))]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, marker=marker, label=label)
        end_label(ax, xs[-1], ys[-1], label)
    ax.set_yscale("log")
    plain_ticks(ax.yaxis)
    ax.set_xlabel("Achieved throughput (output tokens/s, steady state)")
    ax.set_ylabel("p99 time to first token (s, log scale)")
    ax.set_title("Latency vs throughput: each point is one request rate")
    if len(groups) > 1:  # a lone series is named by the title and its end label
        ax.legend(loc="best")
    finish(fig, out / "latency_throughput.png", str(path))


def block_size(results: Path, out: Path) -> None:
    path = results / "block_size" / "summary.csv"
    if not path.exists():
        return
    groups = by_label(read_csv(path))
    order = sorted(groups, key=lambda l: int(re.sub(r"\D", "", l) or 0))
    fixed = ["bs8", "bs16", "bs32"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [("throughput_tok_s", "Output tokens/s", 1.0), ("itl_p99", "p99 inter-token latency (ms)", 1e3),
              ("kv_usage_mean", "Mean KV-cache usage (%)", 100.0)]
    for ax, (key, ylabel, scale) in zip(axes, panels):
        for label in order:
            rows = [r for r in groups[label] if not math.isinf(num(r["rate"]))]
            color, marker = color_of(label, fixed, order)
            ax.plot([num(r["rate"]) for r in rows], [num(r[key]) * scale for r in rows], color=color, marker=marker, label=label)
        ax.set_xlabel("Request rate (req/s)")
        ax.set_ylabel(ylabel)
    axes[0].set_title("Block size: the page-size trade-off")
    axes[0].legend(loc="upper left")
    finish(fig, out / "block_size.png", str(path))


def admission_policies(results: Path, out: Path, rate: str | None) -> None:
    folder = results / "policies"
    files = sorted(folder.glob("*_requests.csv")) if folder.exists() else []
    if not files:
        return
    rates = sorted({re.search(r"_rate([^_]+)_requests", f.name).group(1) for f in files}, key=num)
    rate = rate or next((r for r in reversed(rates) if not math.isinf(num(r))), rates[-1])
    per_label = {}
    for f in files:
        m = re.match(r"(.+)_rate([^_]+)_requests\.csv", f.name)
        if m and m.group(2) == rate:
            per_label[m.group(1)] = [r for r in read_csv(f) if r["success"] == "True"]
    if not per_label:
        return
    fixed = ["fcfs", "sjf", "priority"]
    order = [l for l in fixed if l in per_label] + sorted(l for l in per_label if l not in fixed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    all_lens = np.array([num(r["prompt_len"]) for rows in per_label.values() for r in rows])
    edges = np.unique(np.percentile(all_lens, np.linspace(0, 100, 6)))
    for label in order:
        rows = per_label[label]
        color, marker = color_of(label, fixed, order)
        ttft = np.sort([num(r["ttft"]) for r in rows])
        ax1.plot(ttft, np.arange(1, len(ttft) + 1) / len(ttft), color=color, label=label)
        lens = np.array([num(r["prompt_len"]) for r in rows])
        vals = np.array([num(r["ttft"]) for r in rows])
        xs, ys = [], []
        for lo, hi in zip(edges, edges[1:]):
            sel = (lens >= lo) & (lens <= hi)
            if sel.any():
                xs.append((lo + hi) / 2)
                ys.append(np.percentile(vals[sel], 99))
        ax2.plot(xs, ys, color=color, marker=marker, label=label)
        end_label(ax2, xs[-1], ys[-1], label)
    ax1.set_xscale("log")
    plain_ticks(ax1.xaxis)
    ax1.set_xlabel("Time to first token (s, log scale)")
    ax1.set_ylabel("Fraction of requests")
    ax1.set_title(f"TTFT distribution at {rate} req/s")
    ax1.legend(loc="lower right")
    ax2.set_yscale("log")
    plain_ticks(ax2.yaxis)
    ax2.set_xlabel("Prompt length (tokens, quintile midpoints)")
    ax2.set_ylabel("p99 TTFT (s, log scale)")
    ax2.set_title("Who waits: p99 TTFT by prompt length")
    finish(fig, out / "admission_policies.png", f"{folder}/*_rate{rate}_requests.csv")


def preemption(results: Path, out: Path) -> None:
    path = results / "preemption" / "crossover.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    lengths = [num(r["length"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for label, key in (("recompute", "recompute_ms"), ("swap", "swap_ms")):
        color, marker = color_of(label, ["recompute", "swap"])
        ys = [num(r[key]) for r in rows]
        ax.plot(lengths, ys, color=color, marker=marker, label=label)
        end_label(ax, lengths[-1], ys[-1], label)
    cheaper = [l for l, r in zip(lengths, rows) if num(r["swap_ms"]) < num(r["recompute_ms"])]
    if cheaper and cheaper[0] > lengths[0]:
        ax.axvline(cheaper[0], color=AXIS, linewidth=1)
        ax.annotate(f"swap cheaper from {int(cheaper[0])} tokens", (cheaper[0], ax.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points", fontsize=9, color=INK2)
    elif len(cheaper) == len(lengths):
        ax.text(0.99, 0.02, "swap was cheaper at every measured length", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=INK2)
    elif not cheaper:
        ax.text(0.99, 0.02, "recompute was cheaper at every measured length", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths, [f"{int(l)}" for l in lengths])
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_yscale("log")
    plain_ticks(ax.yaxis)
    ax.set_xlabel("Sequence length (tokens, log scale)")
    ax.set_ylabel("Cost of one preemption and resume (ms, log scale)")
    ax.set_title("Preemption: recompute vs swap")
    ax.legend(loc="upper left")
    finish(fig, out / "preemption.png", str(path))


def prefix_caching(results: Path, out: Path) -> None:
    path = results / "prefix" / "summary.csv"
    if not path.exists():
        return
    points: dict[str, dict[int, float]] = defaultdict(dict)
    for r in read_csv(path):
        m = re.match(r"(cache-on|cache-off)-L(\d+)", r["label"])
        if m:
            points[m.group(1)][int(m.group(2))] = num(r["ttft_mean"])
    if not points:
        return
    fixed = ["cache-on", "cache-off"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for label in fixed:
        if label in points:
            xs = sorted(points[label])
            color, marker = color_of(label, fixed)
            ax1.plot(xs, [points[label][x] * 1e3 for x in xs], color=color, marker=marker, label=label)
            end_label(ax1, xs[-1], points[label][xs[-1]] * 1e3, label)
    shared = sorted(set(points.get("cache-on", {})) & set(points.get("cache-off", {})))
    if shared:
        speedup = [points["cache-off"][x] / points["cache-on"][x] for x in shared]
        ax2.plot(shared, speedup, color=SERIES[0], marker=MARKERS[0])
        end_label(ax2, shared[-1], speedup[-1], f"{speedup[-1]:.1f}x")
        ax2.axhline(1.0, color=AXIS, linewidth=1)
    ax1.set_xlabel("Shared prefix length (tokens)")
    ax1.set_ylabel("Mean time to first token (ms)")
    ax1.set_title("Prefix caching: TTFT vs shared prefix length")
    ax1.legend(loc="upper left")
    ax2.set_xlabel("Shared prefix length (tokens)")
    ax2.set_ylabel("TTFT speedup (cache off / cache on)")
    ax2.set_title("Speedup grows with the shared prefix")
    finish(fig, out / "prefix_caching.png", str(path))


def _itl_over_time(tokens_csv: Path, window: float) -> tuple[np.ndarray, list[float]]:
    """p99 inter-token latency per time window, over the short (decoding) requests only."""
    by_req: dict[str, list[float]] = defaultdict(list)
    groups = {}
    for r in read_csv(tokens_csv):
        by_req[r["idx"]].append(num(r["t"]))
        groups[r["idx"]] = r["group"]
    buckets: dict[int, list[float]] = defaultdict(list)
    for idx, times in by_req.items():
        if groups[idx] == "long":
            continue  # the curve is what the *decoding* requests experience
        times.sort()
        for a, b in zip(times, times[1:]):
            buckets[int(b // window)].append(b - a)
    xs = sorted(buckets)
    return np.array(xs) * window, [float(np.percentile(buckets[x], 99)) * 1e3 for x in xs]


def chunked_prefill(results: Path, out: Path, window: float = 1.0) -> None:
    """Left: the money plot, p99 ITL over time at the longest prompt mix. Right: p99 ITL of the
    decoding requests over the whole run, against the long prompts' length."""
    folder = results / "chunked"
    if not folder.exists():
        return
    runs: dict[str, dict[int, Path]] = defaultdict(dict)  # mode -> long-prompt length -> tokens csv
    for f in sorted(folder.glob("*_tokens.csv")):
        m = re.match(r"(chunked|unchunked)-L(\d+)_rate[^_]+_tokens\.csv", f.name)
        if m:
            runs[m.group(1)][int(m.group(2))] = f
    if not runs:
        return
    fixed = ["chunked", "unchunked"]
    longest = max(l for mode in runs.values() for l in mode)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax1.grid(axis="x", visible=False)  # vertical rules below mean one thing: a long prompt arrived
    source = runs.get("unchunked", {}).get(longest) or runs.get("chunked", {}).get(longest)
    req_file = source.with_name(source.name.replace("_tokens.csv", "_requests.csv"))
    long_arrivals = [num(r["sent"]) for r in read_csv(req_file) if r["group"] == "long"] if req_file.exists() else []
    for t in long_arrivals:
        ax1.axvline(t, color=AXIS, linewidth=1.2, zorder=0)
    handles = []
    for mode in fixed:
        if longest in runs.get(mode, {}):
            xs, ys = _itl_over_time(runs[mode][longest], window)
            handles += ax1.plot(xs, ys, color=color_of(mode, fixed)[0], label=mode, linewidth=1.6)
    if long_arrivals:
        handles.append(Line2D([], [], color=AXIS, linewidth=1.2, label="long prompt arrives"))
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(f"p99 inter-token latency per {window:g}s window (ms)")
    ax1.set_title(f"Chunked prefill: decode stalls behind {longest:,}-token prompts")
    ax1.legend(handles=handles, loc="upper right")

    summary = folder / "summary.csv"
    if summary.exists():
        p99: dict[str, dict[int, float]] = defaultdict(dict)
        for r in read_csv(summary):
            m = re.match(r"(chunked|unchunked)-L(\d+)$", r["label"])
            if m:
                p99[m.group(1)][int(m.group(2))] = num(r["itl_p99"]) * 1e3
        for mode in fixed:
            if p99.get(mode):
                xs = sorted(p99[mode])
                color, marker = color_of(mode, fixed)
                ax2.plot(xs, [p99[mode][x] for x in xs], color=color, marker=marker, label=mode)
                end_label(ax2, xs[-1], p99[mode][xs[-1]], mode)
        ax2.set_xlabel("Long prompt length (tokens)")
        ax2.set_ylabel("p99 inter-token latency, whole run (ms)")
        ax2.set_title("Stall grows with prompt length")
    finish(fig, out / "chunked_prefill.png", f"{folder}/*_tokens.csv, summary.csv")


def kv_usage(results: Path, out: Path, label: str = "ours") -> None:
    files = sorted((results / "latency").glob(f"{label}_rate*_stats.csv")) if (results / "latency").exists() else []
    files = [f for f in files if not math.isinf(num(re.search(r"_rate([^_]+)_stats", f.name).group(1)))]
    if not files:
        return
    files.sort(key=lambda f: num(re.search(r"_rate([^_]+)_stats", f.name).group(1)))
    files = files[-len(ORDINAL):]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for f, color in zip(files, ORDINAL[-len(files):]):
        rows = read_csv(f)
        rate = re.search(r"_rate([^_]+)_stats", f.name).group(1)
        ts = [num(r["t"]) for r in rows]
        ys = [num(r["kv_usage"]) * 100 for r in rows]
        ax.plot(ts, ys, color=color, linewidth=1.6, label=f"{rate} req/s")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("KV-cache blocks in use (%)")
    ax.set_title(f"KV-cache utilization over time ({label})")
    if len(files) > 1:
        ax.legend(loc="lower right", ncols=min(5, len(files)))
    finish(fig, out / "kv_usage.png", f"{results / 'latency'}/{label}_rate*_stats.csv")


def tables(results: Path, out: Path) -> None:
    cols = ["label", "rate", "num_requests", "num_failed", "throughput_tok_s", "steady_tok_s", "ttft_p50", "ttft_p99",
            "itl_p50", "itl_p99", "e2e_p99", "preemptions_per_s", "kv_usage_mean", "peak_memory_gib"]
    lines = ["# Benchmark results", "", "Every number below is measured; see bench/results for the raw CSVs.", ""]
    for path in sorted(results.glob("*/summary.csv")):
        lines += [f"## {path.parent.name}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in read_csv(path):
            cells = []
            for c in cols:
                v = r.get(c, "")
                cells.append(v if c == "label" or not v else (f"{num(v):.3g}" if abs(num(v)) < 1000 else f"{num(v):.0f}"))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    crossover = results / "preemption" / "crossover.csv"
    if crossover.exists():
        rows = read_csv(crossover)
        lines += ["## preemption crossover", "", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        lines += ["| " + " | ".join(r.values()) + " |" for r in rows]
        lines.append("")
    (out / "results.md").write_text("\n".join(lines))
    print(f"wrote {out / 'results.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="bench/results")
    parser.add_argument("--out", default="bench/figures")
    parser.add_argument("--policy-rate", default=None, help="rate whose runs the admission-policy figure uses")
    args = parser.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    latency_throughput(results, out)
    block_size(results, out)
    admission_policies(results, out, args.policy_rate)
    preemption(results, out)
    prefix_caching(results, out)
    chunked_prefill(results, out)
    kv_usage(results, out)
    tables(results, out)


if __name__ == "__main__":
    main()
