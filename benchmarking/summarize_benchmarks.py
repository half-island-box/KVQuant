#!/usr/bin/env python3
"""Summarize benchmark CSV output and create a matplotlib dashboard."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CSV_FIELDS = [
    "experiment",
    "metric_name",
    "metric_unit",
    "context",
    "count",
    "mean",
    "min",
    "max",
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def row_numeric(row: Dict[str, str]) -> Optional[float]:
    return numeric(row.get("metric_value"))


def summarize(csv_path: Path, output_dir: Optional[Path] = None) -> int:
    output_dir = (output_dir or csv_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        value = row_numeric(row)
        if value is None:
            continue
        key = (
            row["experiment"],
            row["metric_name"],
            row.get("metric_unit", ""),
            row.get("context", ""),
        )
        grouped[key].append(value)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for (experiment, metric, unit, context), values in sorted(grouped.items()):
            writer.writerow(
                {
                    "experiment": experiment,
                    "metric_name": metric,
                    "metric_unit": unit,
                    "context": context,
                    "count": len(values),
                    "mean": format(sum(values) / len(values), ".12g"),
                    "min": format(min(values), ".12g"),
                    "max": format(max(values), ".12g"),
                }
            )

    # A profiler table contains parent and child rows. Use the largest average
    # CUDA time per profiler entry as the dominant per-call CUDA activity;
    # totals are not directly comparable when experiments issue different
    # numbers of kernel invocations.
    dominant_by_run: Dict[str, float] = defaultdict(lambda: math.nan)
    run_experiment: Dict[str, str] = {}
    for row in rows:
        run_id = row.get("run_id") or f"{row['experiment']}:{row.get('timestamp', '')}"
        run_experiment[run_id] = row["experiment"]
        if row.get("metric_name") != "cuda_time_avg":
            continue
        value = row_numeric(row)
        if value is None:
            continue
        dominant_by_run[run_id] = value if math.isnan(dominant_by_run[run_id]) else max(dominant_by_run[run_id], value)

    by_experiment: Dict[str, List[float]] = defaultdict(list)
    for run_id, value in dominant_by_run.items():
        by_experiment[run_experiment[run_id]].append(value)
    # Fallback for imported CSVs whose run_id naming is different.
    if not by_experiment:
        for row in rows:
            if row.get("metric_name") == "cuda_time_avg":
                value = row_numeric(row)
                if value is not None:
                    by_experiment[row["experiment"]].append(value)

    run_wall: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row.get("metric_name") == "run.wall_time":
            value = row_numeric(row)
            if value is not None:
                run_wall[row["experiment"]].append(value)

    status_by_experiment: Dict[str, str] = {}
    for row in rows:
        experiment = row["experiment"]
        status = row.get("status", "")
        if status == "FAIL":
            status_by_experiment[experiment] = "FAIL"
        elif experiment not in status_by_experiment and status in {"PASS", "SKIPPED"}:
            status_by_experiment[experiment] = status

    experiments = sorted(set(run_wall) | set(by_experiment) | set(status_by_experiment))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Summary CSV written to {summary_path}, but matplotlib is unavailable: {exc}")
        return 2

    figure, axes = plt.subplots(1, 2, figsize=(16, max(5, 0.55 * len(experiments))))
    statuses = [status_by_experiment.get(experiment, "UNKNOWN") for experiment in experiments]
    colors = {"PASS": "#2878B5", "FAIL": "#E8494C", "SKIPPED": "#B8B8B8", "UNKNOWN": "#777777"}
    wall_means = [sum(run_wall[name]) / len(run_wall[name]) if run_wall.get(name) else math.nan for name in experiments]
    bars = axes[0].barh(experiments, wall_means, color=[colors[status] for status in statuses])
    axes[0].set_xlabel("Mean runner wall time (s)")
    axes[0].set_title("Experiment wall time")
    axes[0].set_xscale("symlog")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, wall_means):
        if not math.isnan(value):
            axes[0].text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:.3g}s", va="center", fontsize=8)

    dominant_means = [sum(by_experiment[name]) / len(by_experiment[name]) if by_experiment.get(name) else math.nan for name in experiments]
    bars = axes[1].barh(experiments, dominant_means, color=[colors[status] for status in statuses])
    axes[1].set_xlabel("Mean dominant per-call CUDA time (us)")
    axes[1].set_title("Dominant per-call CUDA activity")
    axes[1].set_xscale("symlog")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, dominant_means):
        if not math.isnan(value):
            axes[1].text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:.3g}us", va="center", fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in colors]
    figure.legend(handles, list(colors), loc="lower center", ncol=4, frameon=False)
    figure.suptitle("KVQuant benchmark summary")
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    plot_path = output_dir / "benchmark_summary.png"
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Summary CSV: {summary_path}")
    print(f"Summary plot: {plot_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="results.csv generated by run_all.py")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for summary outputs")
    args = parser.parse_args()
    return summarize(args.csv, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
