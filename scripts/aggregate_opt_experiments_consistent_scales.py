#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

from log_utils import Logger, LogLevel


PROFILE_ORDER = ["A", "B", "C"]
RESOURCE_PROFILES = ["B", "C"]
PHASE_METRICS = ["forward_ms", "backward_ms", "optimizer_step_ms"]
RESOURCE_METRICS = [
    "gpu_util",
    "gpu_mem_used",
    "proc_cpu_percent",
    "sys_cpu_percent",
    "step_energy_kwh",
    "cumulative_gpu_energy_kwh",
]


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def maybe_float(v: object) -> Optional[float]:
    try:
        if v in ("", None):
            return None
        return float(v)
    except Exception:
        return None


def rolling_mean(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    out: List[float] = []
    acc = 0.0
    q: List[float] = []
    for v in values:
        val = float("nan") if v is None else float(v)
        q.append(val)
        if not math.isnan(val):
            acc += val
        if len(q) > window:
            old = q.pop(0)
            if not math.isnan(old):
                acc -= old
        valid = [x for x in q if not math.isnan(x)]
        out.append(acc / len(valid) if valid else float("nan"))
    return out


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def build_timeline(rows: List[Dict[str, str]], metrics: List[str]) -> Dict[int, Dict[str, float]]:
    by_sec: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        elapsed = maybe_float(row.get("elapsed_sec"))
        if elapsed is None:
            continue
        sec = int(math.floor(elapsed))
        for metric in metrics:
            val = maybe_float(row.get(metric))
            if val is None:
                continue
            by_sec[sec][metric].append(val)

    out: Dict[int, Dict[str, float]] = {}
    for sec, metric_values in by_sec.items():
        out[sec] = {}
        for metric, vals in metric_values.items():
            out[sec][metric] = statistics.mean(vals)
    return out


def aggregate_timelines(per_repeat: List[Dict[int, Dict[str, float]]], metrics: List[str]) -> List[Dict[str, float]]:
    secs = sorted({sec for series in per_repeat for sec in series.keys()})
    aggregated: List[Dict[str, float]] = []
    for sec in secs:
        row: Dict[str, float] = {"elapsed_sec": float(sec)}
        for metric in metrics:
            vals = [series[sec][metric] for series in per_repeat if sec in series and metric in series[sec]]
            m, s = mean_std(vals)
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
        aggregated.append(row)
    return aggregated


def detect_gpu_idle_windows(timeline_rows: List[Dict[str, float]], threshold: float) -> List[Dict[str, float]]:
    windows: List[Dict[str, float]] = []
    active_start: Optional[int] = None
    active_values: List[float] = []

    for row in timeline_rows:
        sec = int(row.get("elapsed_sec", 0.0))
        util = row.get("gpu_util_mean")
        if util is None or math.isnan(util):
            continue
        if util < threshold:
            if active_start is None:
                active_start = sec
                active_values = [util]
            else:
                active_values.append(util)
        elif active_start is not None:
            windows.append(
                {
                    "start_sec": active_start,
                    "end_sec": sec - 1,
                    "duration_sec": max(1, sec - active_start),
                    "avg_gpu_util": statistics.mean(active_values),
                }
            )
            active_start = None
            active_values = []

    if active_start is not None:
        end_sec = int(timeline_rows[-1]["elapsed_sec"]) if timeline_rows else active_start
        windows.append(
            {
                "start_sec": active_start,
                "end_sec": end_sec,
                "duration_sec": max(1, end_sec - active_start + 1),
                "avg_gpu_util": statistics.mean(active_values) if active_values else float("nan"),
            }
        )
    return windows


def find_codecarbon_full_csv(run_dir: str) -> str:
    candidates = sorted(glob(os.path.join(run_dir, "*cc_full_rank_*.csv")))
    return candidates[-1] if candidates else ""


def read_codecarbon_energy_kwh(run_dir: str) -> float:
    path = find_codecarbon_full_csv(run_dir)
    if not path:
        return float("nan")
    rows = read_csv_rows(path)
    if not rows:
        return float("nan")
    row = rows[-1]
    for key, value in row.items():
        if "energy" in key.lower() and "consum" in key.lower():
            val = maybe_float(value)
            if val is not None:
                return val
    return float("nan")


def resolve_run_dir(exp_dir: str, run: Dict) -> str:
    run_dir = str(run.get("run_dir", ""))
    if run_dir and os.path.isdir(run_dir):
        return run_dir
    return os.path.join(
        exp_dir,
        f"batch_{int(run['batch_size'])}",
        f"profile_{run['profile']}",
        f"repeat_{int(run['repeat'])}",
    )


def resolve_manifest_path(input_dir: str) -> str:
    direct = os.path.join(input_dir, "manifest.json")
    if os.path.isfile(direct):
        return direct

    candidates = []
    try:
        for name in os.listdir(input_dir):
            child = os.path.join(input_dir, name)
            if not os.path.isdir(child):
                continue
            candidate = os.path.join(child, "manifest.json")
            if os.path.isfile(candidate):
                candidates.append(candidate)
    except FileNotFoundError:
        return ""

    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    return ""


def save_bar_with_error(plt, out_path: str, labels: List[str], means: List[float], stds: List[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, means, yerr=stds, capsize=4, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_overhead_bar(plt, out_path: str, labels: List[str], means: List[float], stds: List[float], threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, means, yerr=stds, capsize=4, color=["#ff7f0e", "#2ca02c"][: len(labels)])
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"{threshold}% threshold")
    ax.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    ax.set_title("Measurement Overhead vs Profile A")
    ax.set_ylabel("Overhead (%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_phase_stacked_bar(plt, out_path: str, phase_row: Dict[str, float]) -> None:
    labels = ["Step"]
    forward = [phase_row.get("forward_pct", float("nan"))]
    backward = [phase_row.get("backward_pct", float("nan"))]
    optimizer = [phase_row.get("optimizer_step_pct", float("nan"))]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(labels, forward, label="Forward", color="#1f77b4")
    ax.bar(labels, backward, bottom=forward, label="Backward", color="#ff7f0e")
    bottom2 = [a + b for a, b in zip(forward, backward)]
    ax.bar(labels, optimizer, bottom=bottom2, label="Optimizer", color="#2ca02c")
    ax.set_title("Step Phase Composition")
    ax.set_ylabel("Share of Step Time (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def finite_values(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def padded_limits(values: Sequence[float], pad_ratio: float = 0.05) -> Optional[Tuple[float, float]]:
    clean = finite_values(values)
    if not clean:
        return None
    lo = min(clean)
    hi = max(clean)
    if hi == lo:
        delta = max(abs(hi) * pad_ratio, 1e-9)
        return lo - delta, hi + delta
    span = hi - lo
    return lo - span * pad_ratio, hi + span * pad_ratio


def compute_global_timeline_limits(
    series_by_batch: Dict[int, List[Dict[str, float]]],
    metrics: List[str],
) -> Tuple[Optional[Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    elapsed_vals: List[float] = []
    metric_ranges: Dict[str, Tuple[float, float]] = {}
    for rows in series_by_batch.values():
        elapsed_vals.extend([float(r.get("elapsed_sec", float("nan"))) for r in rows])

    x_lim = None
    if elapsed_vals:
        max_elapsed = max(finite_values(elapsed_vals), default=float("nan"))
        if not math.isnan(max_elapsed):
            x_lim = (0.0, max_elapsed)

    for metric in metrics:
        vals: List[float] = []
        for rows in series_by_batch.values():
            for row in rows:
                mean = float(row.get(f"{metric}_mean", float("nan")))
                std = float(row.get(f"{metric}_std", 0.0))
                if not math.isnan(mean):
                    vals.append(mean + (0.0 if math.isnan(std) else std))
                    vals.append(mean - (0.0 if math.isnan(std) else std))
        lim = padded_limits(vals)
        if lim is not None:
            metric_ranges[metric] = lim
    return x_lim, metric_ranges


def save_timeline_with_band(
    plt,
    out_path: str,
    rows: List[Dict[str, float]],
    metrics: List[Tuple[str, str, str]],
    title: str,
    xlabel: str = "Elapsed Time (sec)",
    x_lim: Optional[Tuple[float, float]] = None,
    y_lims: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, max(3, 2.4 * len(metrics))), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    x = [float(r["elapsed_sec"]) for r in rows]
    for ax, (metric, label, color) in zip(axes, metrics):
        y = [float(r.get(f"{metric}_mean", float("nan"))) for r in rows]
        ys = [float(r.get(f"{metric}_std", 0.0)) for r in rows]
        upper = [a + b for a, b in zip(y, ys)]
        lower = [a - b for a, b in zip(y, ys)]
        ax.plot(x, y, color=color, linewidth=1.8, label="mean")
        ax.fill_between(x, lower, upper, color=color, alpha=0.2, label="std")
        ax.set_ylabel(label)
        if y_lims and metric in y_lims:
            lo, hi = y_lims[metric]
            ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel(xlabel)
    if x_lim is not None:
        axes[-1].set_xlim(*x_lim)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_overlay_timeline_with_band(
    plt,
    out_path: str,
    rows: List[Dict[str, float]],
    metrics: List[Tuple[str, str, str]],
    title: str,
    ylabel: str,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = [float(r["elapsed_sec"]) for r in rows]
    for metric, label, color in metrics:
        y = [float(r.get(f"{metric}_mean", float("nan"))) for r in rows]
        ys = [float(r.get(f"{metric}_std", 0.0)) for r in rows]
        upper = [a + b for a, b in zip(y, ys)]
        lower = [a - b for a, b in zip(y, ys)]
        ax.plot(x, y, color=color, linewidth=1.8, label=f"{label} mean")
        ax.fill_between(x, lower, upper, color=color, alpha=0.15, label=f"{label} std")
    ax.set_xlabel("Elapsed Time (sec)")
    ax.set_ylabel(ylabel)
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_dual_axis_timeline(
    plt,
    out_path: str,
    rows: List[Dict[str, float]],
    left_metric: Tuple[str, str, str],
    right_metric: Tuple[str, str, str],
    title: str,
    x_lim: Optional[Tuple[float, float]] = None,
    left_y_lim: Optional[Tuple[float, float]] = None,
    right_y_lim: Optional[Tuple[float, float]] = None,
) -> None:
    if not rows:
        return
    lm, llabel, lcolor = left_metric
    rm, rlabel, rcolor = right_metric
    x = [float(r["elapsed_sec"]) for r in rows]
    y1 = [float(r.get(f"{lm}_mean", float("nan"))) for r in rows]
    y2 = [float(r.get(f"{rm}_mean", float("nan"))) for r in rows]

    fig, ax1 = plt.subplots(figsize=(11.5, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(x, y1, color=lcolor, linewidth=1.8, label=llabel)
    ax2.plot(x, y2, color=rcolor, linewidth=1.8, label=rlabel)
    ax1.set_xlabel("Elapsed Time (sec)")
    ax1.set_ylabel(llabel, color=lcolor)
    ax2.set_ylabel(rlabel, color=rcolor)
    if x_lim is not None:
        ax1.set_xlim(*x_lim)
    if left_y_lim is not None:
        ax1.set_ylim(*left_y_lim)
    if right_y_lim is not None:
        ax2.set_ylim(*right_y_lim)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_gpu_idle_highlight(
    plt,
    out_path: str,
    rows: List[Dict[str, float]],
    windows: List[Dict[str, float]],
    threshold: float,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
) -> None:
    if not rows:
        return
    x = [float(r["elapsed_sec"]) for r in rows]
    y = [float(r.get("gpu_util_mean", float("nan"))) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(x, y, color="#1f77b4", linewidth=1.8, label="GPU util mean")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={threshold}%")
    for w in windows:
        ax.axvspan(float(w["start_sec"]), float(w["end_sec"]), color="#ff7f0e", alpha=0.2)
    ax.set_xlabel("Elapsed Time (sec)")
    ax.set_ylabel("GPU Util (%)")
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_title("GPU Underutilization Windows")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_histogram(plt, out_path: str, values: List[float], title: str, xlabel: str) -> None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(clean, bins=20, color="#17becf", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_underutil_summary_plot(plt, out_path: str, windows: List[Dict[str, float]]) -> None:
    count = len(windows)
    avg_duration, _ = mean_std([float(w.get("duration_sec", float("nan"))) for w in windows])
    avg_util, _ = mean_std([float(w.get("avg_gpu_util", float("nan"))) for w in windows])
    labels = ["window_count", "avg_duration_sec", "avg_gpu_util_pct"]
    vals = [float(count), avg_duration if not math.isnan(avg_duration) else 0.0, avg_util if not math.isnan(avg_util) else 0.0]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Underutilization Window Summary")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_cross_batch_lines(
    plt,
    out_path: str,
    rows: List[Dict[str, float]],
    y_key: str,
    y_label: str,
    title: str,
    profiles: List[str],
) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    grouped: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for row in rows:
        prof = str(row.get("profile", ""))
        batch = int(row.get("batch_size", 0))
        y = float(row.get(y_key, float("nan")))
        y_std = float(row.get(f"{y_key}_std", float("nan")))
        grouped[prof].append((batch, y, y_std))

    for p in profiles:
        vals = sorted(grouped.get(p, []), key=lambda x: x[0])
        if not vals:
            continue
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]
        es = [0.0 if math.isnan(v[2]) else v[2] for v in vals]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, linewidth=1.8, color=colors.get(p), label=f"Profile {p}")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_cross_batch_phase_lines(plt, out_path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    c_rows = [r for r in rows if str(r.get("profile")) == "C"]
    if not c_rows:
        return
    c_rows.sort(key=lambda r: int(r["batch_size"]))
    x = [int(r["batch_size"]) for r in c_rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, label, color in [
        ("forward_ms_mean", "Forward", "#1f77b4"),
        ("backward_ms_mean", "Backward", "#ff7f0e"),
        ("optimizer_step_ms_mean", "Optimizer", "#2ca02c"),
    ]:
        y = [float(r.get(metric, float("nan"))) for r in c_rows]
        y_std = [
            float(r.get(metric.replace("_mean", "_std"), 0.0))
            if not math.isnan(float(r.get(metric, float("nan"))))
            else 0.0
            for r in c_rows
        ]
        ax.errorbar(x, y, yerr=y_std, marker="o", capsize=4, linewidth=1.8, label=label, color=color)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Phase Time (ms)")
    ax.set_title("Profile C Phase Times vs Batch Size")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_resource_metric(rows: List[Dict[str, float]], metric: str) -> Tuple[float, float]:
    values = [float(r.get(f"{metric}_mean", float("nan"))) for r in rows]
    clean = finite_values(values)
    if not clean:
        return float("nan"), float("nan")
    return statistics.mean(clean), max(clean)


def save_cross_batch_resource_summary(
    plt,
    out_path: str,
    rows: List[Dict[str, float]],
) -> None:
    if not rows:
        return
    ordered = sorted(rows, key=lambda r: int(r["batch_size"]))
    x = [int(r["batch_size"]) for r in ordered]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(x, [float(r.get("gpu_util_avg_pct", float("nan"))) for r in ordered], marker="o", linewidth=1.8, color="#1f77b4", label="GPU util avg")
    axes[0].plot(x, [float(r.get("gpu_util_peak_pct", float("nan"))) for r in ordered], marker="s", linewidth=1.8, color="#ff7f0e", label="GPU util peak")
    axes[0].set_ylabel("GPU Util (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(x, [float(r.get("proc_cpu_avg_pct", float("nan"))) for r in ordered], marker="o", linewidth=1.8, color="#2ca02c", label="Process CPU avg")
    axes[1].plot(x, [float(r.get("sys_cpu_avg_pct", float("nan"))) for r in ordered], marker="s", linewidth=1.8, color="#8c564b", label="System CPU avg")
    axes[1].set_ylabel("CPU (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(x, [float(r.get("gpu_mem_avg_mib", float("nan"))) for r in ordered], marker="o", linewidth=1.8, color="#d62728", label="GPU mem avg")
    axes[2].plot(x, [float(r.get("gpu_mem_peak_mib", float("nan"))) for r in ordered], marker="s", linewidth=1.8, color="#9467bd", label="GPU mem peak")
    axes[2].set_ylabel("GPU Mem (MiB)")
    axes[2].set_xlabel("Batch Size")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.suptitle("Profile C Resource Usage vs Batch Size")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_cross_batch_resource_lines(
    plt,
    out_path: str,
    rows: List[Dict[str, float]],
    y_specs: List[Tuple[str, str, str]],
    ylabel: str,
    title: str,
) -> None:
    if not rows:
        return
    ordered = sorted(rows, key=lambda r: int(r["batch_size"]))
    x = [int(r["batch_size"]) for r in ordered]
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color in y_specs:
        y = [float(r.get(key, float("nan"))) for r in ordered]
        ax.plot(x, y, marker="o", linewidth=1.8, color=color, label=label)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_global_overlay_limits(
    series_by_batch: Dict[int, List[Dict[str, float]]],
    metrics: List[str],
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    x_lim, _ = compute_global_timeline_limits(series_by_batch, metrics)
    vals: List[float] = []
    for rows in series_by_batch.values():
        for row in rows:
            for metric in metrics:
                mean = float(row.get(f"{metric}_mean", float("nan")))
                std = float(row.get(f"{metric}_std", 0.0))
                if math.isnan(mean):
                    continue
                vals.append(mean + (0.0 if math.isnan(std) else std))
                vals.append(mean - (0.0 if math.isnan(std) else std))
    return x_lim, padded_limits(vals)


def save_cross_batch_timeline_metric_comparison(
    plt,
    out_path: str,
    series_by_batch: Dict[int, List[Dict[str, float]]],
    metrics: List[Tuple[str, str, str]],
    title: str,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lims: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    if not series_by_batch:
        return
    ordered_batches = sorted(series_by_batch.keys())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, max(4, 2.6 * len(metrics))), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    batch_colors = {batch: cmap(idx % 10) for idx, batch in enumerate(ordered_batches)}

    for ax, (metric, label, _) in zip(axes, metrics):
        for batch in ordered_batches:
            rows = series_by_batch.get(batch, [])
            if not rows:
                continue
            x = [float(r["elapsed_sec"]) for r in rows]
            y = [float(r.get(f"{metric}_mean", float("nan"))) for r in rows]
            y_std = [float(r.get(f"{metric}_std", 0.0)) for r in rows]
            color = batch_colors[batch]
            upper = [a + b for a, b in zip(y, y_std)]
            lower = [a - b for a, b in zip(y, y_std)]
            ax.plot(x, y, color=color, linewidth=1.8, label=f"Batch {batch}")
            ax.fill_between(x, lower, upper, color=color, alpha=0.12)
        ax.set_ylabel(label)
        if y_lims and metric in y_lims:
            ax.set_ylim(*y_lims[metric])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    if x_lim is not None:
        axes[-1].set_xlim(*x_lim)
    axes[-1].set_xlabel("Elapsed Time (sec)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate OPT experiment outputs across repeats.")
    parser.add_argument("experiment_dir", type=str, help="Path to one run directory generated by run_opt_experiments.py")
    parser.add_argument("--gpu-util-threshold", type=float, default=99.0, help="GPU util threshold for idle window detection.")
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--strict-integrity", type=int, default=1)
    parser.add_argument("--overhead-threshold-pct", type=float, default=5.0)
    parser.add_argument("--rolling-window-sec", type=int, default=5)
    parser.add_argument("--skip-plots", type=int, default=0)
    args = parser.parse_args()

    logger = Logger("AGGREGATE", LogLevel.INFO)
    logger.info("Starting aggregation of OPT experiment outputs...")

    exp_dir = os.path.abspath(args.experiment_dir)
    manifest_path = resolve_manifest_path(exp_dir)
    if not manifest_path:
        raise FileNotFoundError(f"manifest.json not found under {exp_dir} or immediate children")
    exp_dir = os.path.dirname(manifest_path)
    logger.info(f"Found manifest at {manifest_path}")

    manifest = read_json(manifest_path)
    reports_dir = os.path.join(exp_dir, "reports-new")
    figures_root = os.path.join(reports_dir, "figures")
    figures_cross = os.path.join(figures_root, "cross_batch")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_root, exist_ok=True)
    os.makedirs(figures_cross, exist_ok=True)

    runs_all = manifest.get("runs", [])
    require_valid_duration = bool(int(manifest.get("params", {}).get("require_valid_duration", 1)))
    expected_profiles = [x.strip() for x in str(manifest.get("params", {}).get("profiles", "A,B,C")).split(",") if x.strip()]
    expected_batches = [int(x) for x in manifest.get("batch_sweep", [])]
    if not expected_batches:
        expected_batches = sorted({int(r["batch_size"]) for r in runs_all}, reverse=True)

    grouped: Dict[Tuple[int, str], List[Dict]] = defaultdict(list)
    integrity = {
        "expected_repeats": args.expected_repeats,
        "require_valid_duration": require_valid_duration,
        "cells": {},
        "matrix_complete": True,
    }

    for b in expected_batches:
        for p in expected_profiles:
            matching = [r for r in runs_all if int(r["batch_size"]) == b and str(r["profile"]) == p]
            success = [r for r in matching if int(r.get("returncode", 1)) == 0]
            valid = [r for r in success if bool(r.get("valid_duration", False)) or not require_valid_duration]
            cell_complete = len(valid) == args.expected_repeats
            key = f"batch_{b}_profile_{p}"
            integrity["matrix_complete"] = integrity["matrix_complete"] and cell_complete
            integrity["cells"][key] = {
                "batch_size": b,
                "profile": p,
                "run_count": len(matching),
                "success_count": len(success),
                "valid_count": len(valid),
                "missing_runs": max(0, args.expected_repeats - len(matching)),
                "failed_runs": max(0, len(matching) - len(success)),
                "invalid_duration_runs": max(0, len(success) - len(valid)),
                "complete": cell_complete,
            }
            grouped[(b, p)] = valid[: args.expected_repeats]

    with open(os.path.join(reports_dir, "integrity_report.json"), "w", encoding="utf-8") as fp:
        json.dump(integrity, fp, indent=2)

    if bool(args.strict_integrity) and not bool(integrity["matrix_complete"]):
        raise ValueError(
            f"Integrity failed. Expected {args.expected_repeats} valid runs per batch/profile. "
            f"See {os.path.join(reports_dir, 'integrity_report.json')}"
        )

    plt = None if bool(args.skip_plots) else ensure_matplotlib()
    if plt is None and not bool(args.skip_plots):
        logger.warning("matplotlib unavailable; continuing with CSV/JSON outputs only.")

    batch_sizes = sorted({k[0] for k in grouped.keys()}, reverse=True)
    logger.info(f"Processing batches: {batch_sizes}")

    overall_report = {
        "batches": {},
        "overhead": {},
        "overhead_compliance": {},
        "integrity": integrity,
    }

    cross_rows: List[Dict[str, object]] = []
    phase_timeline_by_batch: Dict[int, List[Dict[str, float]]] = {}
    resource_timeline_by_profile: Dict[str, Dict[int, List[Dict[str, float]]]] = {
        profile: {} for profile in RESOURCE_PROFILES
    }
    gpu_idle_windows_by_profile: Dict[str, Dict[int, List[Dict[str, float]]]] = {
        profile: {} for profile in RESOURCE_PROFILES
    }
    cross_resource_rows_by_profile: Dict[str, List[Dict[str, object]]] = {
        profile: [] for profile in RESOURCE_PROFILES
    }

    for batch_size in batch_sizes:
        logger.info(f"Processing batch_size={batch_size}")
        batch_dir = os.path.join(reports_dir, f"batch_{batch_size}")
        fig_batch_dir = os.path.join(figures_root, f"batch_{batch_size}")
        os.makedirs(batch_dir, exist_ok=True)
        os.makedirs(fig_batch_dir, exist_ok=True)

        profile_reports: Dict[str, Dict] = {}
        duration_rows: List[Dict[str, object]] = []
        duration_means: Dict[str, float] = {}
        duration_stds: Dict[str, float] = {}
        baseline_mean = float("nan")
        baseline_std = float("nan")

        # First pass durations by profile.
        for profile in expected_profiles:
            runs = grouped.get((batch_size, profile), [])
            dvals = [float(r.get("elapsed_sec", float("nan"))) for r in runs]
            d_mean, d_std = mean_std(dvals)
            duration_means[profile] = d_mean
            duration_stds[profile] = d_std
            if profile == "A":
                baseline_mean, baseline_std = d_mean, d_std
            duration_rows.append(
                {
                    "batch_size": batch_size,
                    "profile": profile,
                    "repeat_count": len(dvals),
                    "duration_mean_sec": d_mean,
                    "duration_std_sec": d_std,
                }
            )

        # Second pass full profile artifacts.
        for profile in expected_profiles:
            runs = grouped.get((batch_size, profile), [])
            if not runs:
                continue
            logger.info(f"  Profile {profile}: {len(runs)} runs")
            p_report: Dict[str, object] = {
                "duration_sec": {
                    "mean": duration_means.get(profile, float("nan")),
                    "std": duration_stds.get(profile, float("nan")),
                    "n": len(runs),
                }
            }

            energy_vals: List[float] = []
            phase_vals: Dict[str, List[float]] = {k: [] for k in PHASE_METRICS + ["step_ms"]}
            phase_timeline_series: List[Dict[int, Dict[str, float]]] = []
            resource_timeline_series: List[Dict[int, Dict[str, float]]] = []
            steps_per_sec_vals: List[float] = []
            samples_per_sec_vals: List[float] = []
            energy_per_step_vals: List[float] = []

            for run in runs:
                run_dir = resolve_run_dir(exp_dir, run)

                if profile == "B":
                    e = read_codecarbon_energy_kwh(run_dir)
                    if not math.isnan(e):
                        energy_vals.append(e)

                if profile in RESOURCE_PROFILES:
                    resource_steps_path = os.path.join(run_dir, "resource_steps.csv")
                    resource_summary_path = os.path.join(run_dir, "resource_summary.json")

                    if os.path.isfile(resource_steps_path):
                        rrows = read_csv_rows(resource_steps_path)
                        resource_timeline_series.append(build_timeline(rrows, RESOURCE_METRICS))

                    if os.path.isfile(resource_summary_path):
                        rs = read_json(resource_summary_path)
                        iterations = maybe_float(rs.get("iterations"))
                        total_time = maybe_float(rs.get("total_training_time_sec"))
                        total_energy = maybe_float(rs.get("cumulative_gpu_energy_kwh"))
                        if total_time and total_time > 0 and iterations is not None:
                            steps_per_sec_vals.append(iterations / total_time)
                            samples_per_sec_vals.append((iterations * batch_size) / total_time)
                        if total_energy is not None:
                            energy_vals.append(total_energy)
                            if iterations and iterations > 0:
                                energy_per_step_vals.append(total_energy / iterations)

                if profile == "C":
                    simple_summary_path = os.path.join(run_dir, "simple_summary.json")
                    simple_steps_path = os.path.join(run_dir, "simple_steps.csv")

                    if os.path.isfile(simple_summary_path):
                        ss = read_json(simple_summary_path)
                        av = ss.get("averages_ms", {})
                        for m in PHASE_METRICS + ["step_ms"]:
                            v = maybe_float(av.get(m))
                            if v is not None:
                                phase_vals[m].append(v)

                    if os.path.isfile(simple_steps_path):
                        srows = read_csv_rows(simple_steps_path)
                        phase_timeline_series.append(build_timeline(srows, PHASE_METRICS))

            if energy_vals:
                e_mean, e_std = mean_std(energy_vals)
                p_report["end_to_end_energy_kwh"] = {"mean": e_mean, "std": e_std, "n": len(energy_vals)}

            phase_summary_row: Dict[str, float] = {}
            if profile == "C":
                for m in PHASE_METRICS + ["step_ms"]:
                    m_mean, m_std = mean_std(phase_vals[m])
                    phase_summary_row[f"{m}_mean"] = m_mean
                    phase_summary_row[f"{m}_std"] = m_std
                step_mean = phase_summary_row.get("step_ms_mean", float("nan"))
                if step_mean and not math.isnan(step_mean) and step_mean > 0:
                    phase_summary_row["forward_pct"] = 100.0 * phase_summary_row.get("forward_ms_mean", float("nan")) / step_mean
                    phase_summary_row["backward_pct"] = 100.0 * phase_summary_row.get("backward_ms_mean", float("nan")) / step_mean
                    phase_summary_row["optimizer_step_pct"] = 100.0 * phase_summary_row.get("optimizer_step_ms_mean", float("nan")) / step_mean
                else:
                    phase_summary_row["forward_pct"] = float("nan")
                    phase_summary_row["backward_pct"] = float("nan")
                    phase_summary_row["optimizer_step_pct"] = float("nan")

                p_report["phase_time_ms_per_step"] = {
                    "forward_ms": {
                        "mean": phase_summary_row.get("forward_ms_mean", float("nan")),
                        "std": phase_summary_row.get("forward_ms_std", float("nan")),
                    },
                    "backward_ms": {
                        "mean": phase_summary_row.get("backward_ms_mean", float("nan")),
                        "std": phase_summary_row.get("backward_ms_std", float("nan")),
                    },
                    "optimizer_step_ms": {
                        "mean": phase_summary_row.get("optimizer_step_ms_mean", float("nan")),
                        "std": phase_summary_row.get("optimizer_step_ms_std", float("nan")),
                    },
                }

                phase_csv_path = os.path.join(batch_dir, "phase_time_summary.csv")
                phase_fields = [
                    "batch_size",
                    "profile",
                    "step_ms_mean",
                    "step_ms_std",
                    "forward_ms_mean",
                    "forward_ms_std",
                    "backward_ms_mean",
                    "backward_ms_std",
                    "optimizer_step_ms_mean",
                    "optimizer_step_ms_std",
                    "forward_pct",
                    "backward_pct",
                    "optimizer_step_pct",
                ]
                write_csv(phase_csv_path, [{"batch_size": batch_size, "profile": "C", **phase_summary_row}], phase_fields)

                phase_bars_summary = {
                    "forward_ms": {
                        "mean": phase_summary_row.get("forward_ms_mean", float("nan")),
                        "std": phase_summary_row.get("forward_ms_std", float("nan")),
                    },
                    "backward_ms": {
                        "mean": phase_summary_row.get("backward_ms_mean", float("nan")),
                        "std": phase_summary_row.get("backward_ms_std", float("nan")),
                    },
                    "optimizer_step_ms": {
                        "mean": phase_summary_row.get("optimizer_step_ms_mean", float("nan")),
                        "std": phase_summary_row.get("optimizer_step_ms_std", float("nan")),
                    },
                }
                profile_c_dir = os.path.join(batch_dir, "profile_C")
                os.makedirs(profile_c_dir, exist_ok=True)
                with open(os.path.join(profile_c_dir, "phase_bars_summary.json"), "w", encoding="utf-8") as fp:
                    json.dump(phase_bars_summary, fp, indent=2)

                if phase_timeline_series:
                    phase_agg = aggregate_timelines(phase_timeline_series, PHASE_METRICS)
                    for m in PHASE_METRICS:
                        vals = [float(r.get(f"{m}_mean", float("nan"))) for r in phase_agg]
                        smooth = rolling_mean(vals, max(1, int(args.rolling_window_sec)))
                        for i, v in enumerate(smooth):
                            phase_agg[i][f"{m}_rolling_mean"] = v
                    phase_timeline_csv = os.path.join(batch_dir, "phase_timeline_aggregate.csv")
                    phase_fields = ["elapsed_sec"]
                    for m in PHASE_METRICS:
                        phase_fields.extend([f"{m}_mean", f"{m}_std", f"{m}_rolling_mean"])
                    write_csv(phase_timeline_csv, phase_agg, phase_fields)
                    phase_timeline_by_batch[batch_size] = phase_agg

                if resource_timeline_series:
                    resource_agg = aggregate_timelines(resource_timeline_series, RESOURCE_METRICS)
                    profile_dir = os.path.join(batch_dir, f"profile_{profile}")
                    os.makedirs(profile_dir, exist_ok=True)
                    resource_timeline_csv = os.path.join(profile_dir, "resource_timeline_aggregate.csv")
                    resource_fields = ["elapsed_sec"]
                    for m in RESOURCE_METRICS:
                        resource_fields.extend([f"{m}_mean", f"{m}_std"])
                    write_csv(resource_timeline_csv, resource_agg, resource_fields)

                    # Backward-compatible names for profile C.
                    write_csv(os.path.join(profile_dir, "timeline_aggregate.csv"), resource_agg, resource_fields)
                    if profile == "C":
                        write_csv(os.path.join(batch_dir, "resource_timeline_aggregate.csv"), resource_agg, resource_fields)
                    resource_timeline_by_profile[profile][batch_size] = resource_agg

                    windows = detect_gpu_idle_windows(resource_agg, args.gpu_util_threshold)
                    underutil_csv = os.path.join(profile_dir, "underutilization_windows.csv")
                    write_csv(
                        underutil_csv,
                        windows,
                        ["start_sec", "end_sec", "duration_sec", "avg_gpu_util"],
                    )
                    with open(os.path.join(profile_dir, "gpu_idle_windows.json"), "w", encoding="utf-8") as fp:
                        json.dump(windows, fp, indent=2)
                    p_report["gpu_idle_windows"] = windows
                    gpu_idle_windows_by_profile[profile][batch_size] = windows

                    narrative_path = os.path.join(profile_dir, "gpu_idle_narrative.txt")
                    with open(narrative_path, "w", encoding="utf-8") as fp:
                        if not windows:
                            fp.write("No low GPU-utilization windows were detected for this batch.\n")
                        else:
                            fp.write("Detected low GPU-utilization windows (potential efficiency opportunities):\n")
                            for w in windows:
                                fp.write(
                                    f"- {w['start_sec']}s to {w['end_sec']}s "
                                    f"(duration={w['duration_sec']}s, avg_gpu_util={w['avg_gpu_util']:.2f}%)\n"
                                )

                    gpu_util_avg, gpu_util_peak = summarize_resource_metric(resource_agg, "gpu_util")
                    gpu_mem_avg, gpu_mem_peak = summarize_resource_metric(resource_agg, "gpu_mem_used")
                    proc_cpu_avg, _ = summarize_resource_metric(resource_agg, "proc_cpu_percent")
                    sys_cpu_avg, _ = summarize_resource_metric(resource_agg, "sys_cpu_percent")
                    cross_resource_rows_by_profile[profile].append(
                        {
                            "batch_size": batch_size,
                            "profile": profile,
                            "gpu_util_avg_pct": gpu_util_avg,
                            "gpu_util_peak_pct": gpu_util_peak,
                            "proc_cpu_avg_pct": proc_cpu_avg,
                            "sys_cpu_avg_pct": sys_cpu_avg,
                            "gpu_mem_avg_mib": gpu_mem_avg,
                            "gpu_mem_peak_mib": gpu_mem_peak,
                        }
                    )

            # profile-level report
            if steps_per_sec_vals:
                sps_mean, sps_std = mean_std(steps_per_sec_vals)
                p_report["steps_per_sec"] = {"mean": sps_mean, "std": sps_std, "n": len(steps_per_sec_vals)}
            if samples_per_sec_vals:
                ss_mean, ss_std = mean_std(samples_per_sec_vals)
                p_report["samples_per_sec"] = {"mean": ss_mean, "std": ss_std, "n": len(samples_per_sec_vals)}
            if energy_per_step_vals:
                eps_mean, eps_std = mean_std(energy_per_step_vals)
                p_report["energy_per_step_kwh"] = {"mean": eps_mean, "std": eps_std, "n": len(energy_per_step_vals)}

            # overhead vs profile A
            overhead_pct = float("nan")
            overhead_std_pct = float("nan")
            if profile in ("B", "C") and baseline_mean and not math.isnan(baseline_mean):
                d_mean = duration_means.get(profile, float("nan"))
                d_std = duration_stds.get(profile, float("nan"))
                if not math.isnan(d_mean) and baseline_mean != 0:
                    overhead_pct = (d_mean - baseline_mean) / baseline_mean * 100.0
                    overhead_std_pct = (d_std / baseline_mean) * 100.0 if not math.isnan(d_std) else float("nan")
                    overall_report["overhead"][f"batch_{batch_size}_profile_{profile}"] = overhead_pct
                    overall_report["overhead_compliance"][f"batch_{batch_size}_profile_{profile}"] = {
                        "overhead_pct": overhead_pct,
                        "threshold_pct": args.overhead_threshold_pct,
                        "pass": overhead_pct <= args.overhead_threshold_pct,
                    }

            profile_reports[profile] = p_report

            cross_rows.append(
                {
                    "batch_size": batch_size,
                    "profile": profile,
                    "duration_mean_sec": duration_means.get(profile, float("nan")),
                    "duration_std_sec": duration_stds.get(profile, float("nan")),
                    "overhead_pct": overhead_pct,
                    "overhead_pct_std": overhead_std_pct,
                    "end_to_end_energy_kwh_mean": p_report.get("end_to_end_energy_kwh", {}).get("mean", float("nan")),
                    "end_to_end_energy_kwh_std": p_report.get("end_to_end_energy_kwh", {}).get("std", float("nan")),
                    "steps_per_sec_mean": p_report.get("steps_per_sec", {}).get("mean", float("nan")),
                    "steps_per_sec_std": p_report.get("steps_per_sec", {}).get("std", float("nan")),
                    "samples_per_sec_mean": p_report.get("samples_per_sec", {}).get("mean", float("nan")),
                    "samples_per_sec_std": p_report.get("samples_per_sec", {}).get("std", float("nan")),
                    "energy_per_step_kwh_mean": p_report.get("energy_per_step_kwh", {}).get("mean", float("nan")),
                    "energy_per_step_kwh_std": p_report.get("energy_per_step_kwh", {}).get("std", float("nan")),
                    "forward_ms_mean": p_report.get("phase_time_ms_per_step", {}).get("forward_ms", {}).get("mean", float("nan")),
                    "forward_ms_std": p_report.get("phase_time_ms_per_step", {}).get("forward_ms", {}).get("std", float("nan")),
                    "backward_ms_mean": p_report.get("phase_time_ms_per_step", {}).get("backward_ms", {}).get("mean", float("nan")),
                    "backward_ms_std": p_report.get("phase_time_ms_per_step", {}).get("backward_ms", {}).get("std", float("nan")),
                    "optimizer_step_ms_mean": p_report.get("phase_time_ms_per_step", {}).get("optimizer_step_ms", {}).get("mean", float("nan")),
                    "optimizer_step_ms_std": p_report.get("phase_time_ms_per_step", {}).get("optimizer_step_ms", {}).get("std", float("nan")),
                }
            )

        # Per-batch CSV.
        duration_csv = os.path.join(batch_dir, "duration_by_profile.csv")
        write_csv(
            duration_csv,
            duration_rows,
            ["batch_size", "profile", "repeat_count", "duration_mean_sec", "duration_std_sec"],
        )

        # Batch-level plots.
        if plt is not None:
            labels = [p for p in PROFILE_ORDER if p in expected_profiles]
            save_bar_with_error(
                plt,
                os.path.join(fig_batch_dir, "duration_by_profile.png"),
                labels,
                [duration_means.get(p, float("nan")) for p in labels],
                [duration_stds.get(p, float("nan")) for p in labels],
                title=f"End-to-End Duration by Profile (Batch {batch_size})",
                ylabel="Duration (sec)",
            )

            overhead_labels = [p for p in ["B", "C"] if p in expected_profiles]
            overhead_means: List[float] = []
            overhead_stds: List[float] = []
            for p in overhead_labels:
                m = duration_means.get(p, float("nan"))
                s = duration_stds.get(p, float("nan"))
                if baseline_mean and not math.isnan(baseline_mean) and not math.isnan(m):
                    overhead_means.append((m - baseline_mean) / baseline_mean * 100.0)
                    overhead_stds.append((s / baseline_mean) * 100.0 if not math.isnan(s) else 0.0)
                else:
                    overhead_means.append(float("nan"))
                    overhead_stds.append(float("nan"))
            if overhead_labels:
                save_overhead_bar(
                    plt,
                    os.path.join(fig_batch_dir, "overhead_vs_profile_A.png"),
                    overhead_labels,
                    overhead_means,
                    overhead_stds,
                    args.overhead_threshold_pct,
                )

            energy_profiles = [p for p in PROFILE_ORDER if p in profile_reports and "end_to_end_energy_kwh" in profile_reports[p]]
            if energy_profiles:
                save_bar_with_error(
                    plt,
                    os.path.join(fig_batch_dir, "end_to_end_energy_by_profile.png"),
                    energy_profiles,
                    [float(profile_reports[p]["end_to_end_energy_kwh"]["mean"]) for p in energy_profiles],
                    [float(profile_reports[p]["end_to_end_energy_kwh"]["std"]) for p in energy_profiles],
                    title=f"End-to-End Energy by Profile (Batch {batch_size})",
                    ylabel="Energy (kWh)",
                )

            phase_row_path = os.path.join(batch_dir, "phase_time_summary.csv")
            if os.path.isfile(phase_row_path):
                prow = read_csv_rows(phase_row_path)[0]
                save_bar_with_error(
                    plt,
                    os.path.join(fig_batch_dir, "profileC_phase_time_per_step.png"),
                    ["Forward", "Backward", "Optimizer"],
                    [
                        maybe_float(prow.get("forward_ms_mean")) or float("nan"),
                        maybe_float(prow.get("backward_ms_mean")) or float("nan"),
                        maybe_float(prow.get("optimizer_step_ms_mean")) or float("nan"),
                    ],
                    [
                        maybe_float(prow.get("forward_ms_std")) or 0.0,
                        maybe_float(prow.get("backward_ms_std")) or 0.0,
                        maybe_float(prow.get("optimizer_step_ms_std")) or 0.0,
                    ],
                    title=f"Profile C Phase Time per Step (Batch {batch_size})",
                    ylabel="Phase Time (ms)",
                )
                save_phase_stacked_bar(
                    plt,
                    os.path.join(fig_batch_dir, "profileC_phase_composition_pct.png"),
                    {
                        "forward_pct": maybe_float(prow.get("forward_pct")) or float("nan"),
                        "backward_pct": maybe_float(prow.get("backward_pct")) or float("nan"),
                        "optimizer_step_pct": maybe_float(prow.get("optimizer_step_pct")) or float("nan"),
                    },
                )

        batch_report = {"profiles": profile_reports}
        overall_report["batches"][f"batch_{batch_size}"] = batch_report
        with open(os.path.join(batch_dir, "summary.json"), "w", encoding="utf-8") as fp:
            json.dump(batch_report, fp, indent=2)

    # Profile C phase timelines with consistent scales across batch sizes.
    if plt is not None and phase_timeline_by_batch:
        phase_x_lim, phase_y_lim = compute_global_overlay_limits(phase_timeline_by_batch, PHASE_METRICS)
        for batch_size, phase_rows in phase_timeline_by_batch.items():
            fig_batch_dir = os.path.join(figures_root, f"batch_{batch_size}")
            save_overlay_timeline_with_band(
                plt,
                os.path.join(fig_batch_dir, "profileC_phase_timeline.png"),
                phase_rows,
                [
                    ("forward_ms", "Forward (ms)", "#1f77b4"),
                    ("backward_ms", "Backward (ms)", "#ff7f0e"),
                    ("optimizer_step_ms", "Optimizer (ms)", "#2ca02c"),
                ],
                title=f"Profile C Phase Timelines (Batch {batch_size})",
                ylabel="Phase Time (ms)",
                x_lim=phase_x_lim,
                y_lim=phase_y_lim,
            )
        _, phase_metric_lims = compute_global_timeline_limits(phase_timeline_by_batch, PHASE_METRICS)
        save_cross_batch_timeline_metric_comparison(
            plt,
            os.path.join(figures_cross, "profileC_phase_timeline_compare_batches.png"),
            phase_timeline_by_batch,
            [
                ("forward_ms", "Forward (ms)", "#1f77b4"),
                ("backward_ms", "Backward (ms)", "#ff7f0e"),
                ("optimizer_step_ms", "Optimizer (ms)", "#2ca02c"),
            ],
            title="Profile C Phase Timelines by Batch Size",
            x_lim=phase_x_lim,
            y_lims=phase_metric_lims,
        )

    # Resource plots with consistent scales across batch sizes.
    resource_label = {"B": "Profile B", "C": "Profile C"}
    resource_slug = {"B": "profileB", "C": "profileC"}
    for profile in RESOURCE_PROFILES:
        resource_timeline_by_batch = resource_timeline_by_profile.get(profile, {})
        if plt is not None and resource_timeline_by_batch:
            x_lim, metric_lims = compute_global_timeline_limits(resource_timeline_by_batch, RESOURCE_METRICS)
            for batch_size, resource_agg in resource_timeline_by_batch.items():
                fig_batch_dir = os.path.join(figures_root, f"batch_{batch_size}")
                windows = gpu_idle_windows_by_profile.get(profile, {}).get(batch_size, [])
                save_timeline_with_band(
                    plt,
                    os.path.join(fig_batch_dir, f"{resource_slug[profile]}_resource_timeline.png"),
                    resource_agg,
                    [
                        ("gpu_util", "GPU Util (%)", "#1f77b4"),
                        ("gpu_mem_used", "GPU Memory Used (MiB)", "#ff7f0e"),
                        ("proc_cpu_percent", "Process CPU (%)", "#2ca02c"),
                        ("sys_cpu_percent", "System CPU (%)", "#8c564b"),
                        ("step_energy_kwh", "Step Energy (kWh)", "#9467bd"),
                        ("cumulative_gpu_energy_kwh", "Cumulative GPU Energy (kWh)", "#d62728"),
                    ],
                    title=f"{resource_label[profile]} Resource Timelines (Batch {batch_size})",
                    x_lim=x_lim,
                    y_lims=metric_lims,
                )
                save_dual_axis_timeline(
                    plt,
                    os.path.join(fig_batch_dir, f"{resource_slug[profile]}_gpu_util_vs_mem.png"),
                    resource_agg,
                    ("gpu_util", "GPU Util (%)", "#1f77b4"),
                    ("gpu_mem_used", "GPU Memory (MiB)", "#ff7f0e"),
                    title=f"{resource_label[profile]} GPU Util and Memory (Batch {batch_size})",
                    x_lim=x_lim,
                    left_y_lim=metric_lims.get("gpu_util"),
                    right_y_lim=metric_lims.get("gpu_mem_used"),
                )
                save_dual_axis_timeline(
                    plt,
                    os.path.join(fig_batch_dir, f"{resource_slug[profile]}_gpu_util_vs_step_energy.png"),
                    resource_agg,
                    ("gpu_util", "GPU Util (%)", "#1f77b4"),
                    ("step_energy_kwh", "Step Energy (kWh)", "#9467bd"),
                    title=f"{resource_label[profile]} GPU Util and Step Energy (Batch {batch_size})",
                    x_lim=x_lim,
                    left_y_lim=metric_lims.get("gpu_util"),
                    right_y_lim=metric_lims.get("step_energy_kwh"),
                )
                save_gpu_idle_highlight(
                    plt,
                    os.path.join(fig_batch_dir, f"{resource_slug[profile]}_gpu_idle_windows_timeline.png"),
                    resource_agg,
                    windows,
                    args.gpu_util_threshold,
                    x_lim=x_lim,
                    y_lim=metric_lims.get("gpu_util"),
                )
                save_histogram(
                    plt,
                    os.path.join(fig_batch_dir, f"{resource_slug[profile]}_gpu_util_histogram.png"),
                    [float(r.get("gpu_util_mean", float("nan"))) for r in resource_agg],
                    title=f"{resource_label[profile]} GPU Util Distribution (Batch {batch_size})",
                    xlabel="GPU Util (%)",
                )
                save_underutil_summary_plot(
                    plt,
                    os.path.join(fig_batch_dir, f"{resource_slug[profile]}_underutilization_summary.png"),
                    windows,
                )
            save_cross_batch_timeline_metric_comparison(
                plt,
                os.path.join(figures_cross, f"{resource_slug[profile]}_resource_timeline_compare_batches.png"),
                resource_timeline_by_batch,
                [
                    ("gpu_util", "GPU Util (%)", "#1f77b4"),
                    ("gpu_mem_used", "GPU Memory Used (MiB)", "#ff7f0e"),
                    ("proc_cpu_percent", "Process CPU (%)", "#2ca02c"),
                    ("sys_cpu_percent", "System CPU (%)", "#8c564b"),
                    ("step_energy_kwh", "Step Energy (kWh)", "#9467bd"),
                    ("cumulative_gpu_energy_kwh", "Cumulative GPU Energy (kWh)", "#d62728"),
                ],
                title=f"{resource_label[profile]} Resource Timelines by Batch Size",
                x_lim=x_lim,
                y_lims=metric_lims,
            )

        cross_resource_rows = cross_resource_rows_by_profile.get(profile, [])
        if cross_resource_rows:
            cross_resource_fields = [
                "batch_size",
                "profile",
                "gpu_util_avg_pct",
                "gpu_util_peak_pct",
                "proc_cpu_avg_pct",
                "sys_cpu_avg_pct",
                "gpu_mem_avg_mib",
                "gpu_mem_peak_mib",
            ]
            cross_resource_sorted = sorted(cross_resource_rows, key=lambda r: int(r["batch_size"]))
            write_csv(
                os.path.join(reports_dir, f"cross_batch_{resource_slug[profile]}_resources.csv"),
                cross_resource_sorted,
                cross_resource_fields,
            )
            if plt is not None:
                save_cross_batch_resource_summary(
                    plt,
                    os.path.join(figures_cross, f"{resource_slug[profile]}_resources_vs_batch.png"),
                    cross_resource_sorted,
                )
                save_cross_batch_resource_lines(
                    plt,
                    os.path.join(figures_cross, f"{resource_slug[profile]}_gpu_vs_batch.png"),
                    cross_resource_sorted,
                    [
                        ("gpu_util_avg_pct", "GPU util avg (%)", "#1f77b4"),
                        ("gpu_util_peak_pct", "GPU util peak (%)", "#ff7f0e"),
                    ],
                    ylabel="GPU Util (%)",
                    title=f"{resource_label[profile]} GPU Utilization vs Batch Size",
                )
                save_cross_batch_resource_lines(
                    plt,
                    os.path.join(figures_cross, f"{resource_slug[profile]}_cpu_vs_batch.png"),
                    cross_resource_sorted,
                    [
                        ("proc_cpu_avg_pct", "Process CPU avg (%)", "#2ca02c"),
                        ("sys_cpu_avg_pct", "System CPU avg (%)", "#8c564b"),
                    ],
                    ylabel="CPU (%)",
                    title=f"{resource_label[profile]} CPU Usage vs Batch Size",
                )
                save_cross_batch_resource_lines(
                    plt,
                    os.path.join(figures_cross, f"{resource_slug[profile]}_memory_vs_batch.png"),
                    cross_resource_sorted,
                    [
                        ("gpu_mem_avg_mib", "GPU mem avg (MiB)", "#d62728"),
                        ("gpu_mem_peak_mib", "GPU mem peak (MiB)", "#9467bd"),
                    ],
                    ylabel="GPU Memory (MiB)",
                    title=f"{resource_label[profile]} GPU Memory vs Batch Size",
                )

    # cross-batch csv
    cross_fields = [
        "batch_size",
        "profile",
        "duration_mean_sec",
        "duration_std_sec",
        "overhead_pct",
        "overhead_pct_std",
        "end_to_end_energy_kwh_mean",
        "end_to_end_energy_kwh_std",
        "steps_per_sec_mean",
        "steps_per_sec_std",
        "samples_per_sec_mean",
        "samples_per_sec_std",
        "energy_per_step_kwh_mean",
        "energy_per_step_kwh_std",
        "forward_ms_mean",
        "forward_ms_std",
        "backward_ms_mean",
        "backward_ms_std",
        "optimizer_step_ms_mean",
        "optimizer_step_ms_std",
    ]
    cross_rows_sorted = sorted(cross_rows, key=lambda r: (int(r["batch_size"]), str(r["profile"])))
    write_csv(os.path.join(reports_dir, "cross_batch_summary.csv"), cross_rows_sorted, cross_fields)

    # cross-batch plots
    if plt is not None and cross_rows_sorted:
        save_cross_batch_lines(
            plt,
            os.path.join(figures_cross, "duration_vs_batch_by_profile.png"),
            cross_rows_sorted,
            "duration_mean_sec",
            "Duration (sec)",
            "End-to-End Duration vs Batch Size",
            profiles=[p for p in PROFILE_ORDER if p in expected_profiles],
        )
        save_cross_batch_lines(
            plt,
            os.path.join(figures_cross, "throughput_steps_per_sec_vs_batch.png"),
            cross_rows_sorted,
            "steps_per_sec_mean",
            "Steps / sec",
            "Throughput Proxy (Steps/sec) vs Batch Size",
            profiles=[p for p in PROFILE_ORDER if p in expected_profiles],
        )
        save_cross_batch_lines(
            plt,
            os.path.join(figures_cross, "energy_per_step_vs_batch.png"),
            cross_rows_sorted,
            "energy_per_step_kwh_mean",
            "Energy per Step (kWh)",
            "Energy per Step vs Batch Size",
            profiles=[p for p in ["B", "C"] if p in expected_profiles],
        )
        save_cross_batch_phase_lines(
            plt,
            os.path.join(figures_cross, "profileC_phase_times_vs_batch.png"),
            cross_rows_sorted,
        )

    with open(os.path.join(reports_dir, "overall_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(overall_report, fp, indent=2)

    logger.info("Aggregation complete.")
    logger.info(f"Reports directory: {reports_dir}")
    logger.info(f"Figures directory: {figures_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
