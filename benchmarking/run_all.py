#!/usr/bin/env python3
"""Run the benchmarking experiments sequentially and collect metrics into CSV.

The scripts benchmark CUDA kernels and expect the working directory to be
``benchmarking/``.  This runner always launches them from that directory, even
if it is invoked from the repository root.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_FIELDS = [
    "timestamp",
    "run_id",
    "repeat",
    "experiment",
    "metric_name",
    "metric_value",
    "metric_unit",
    "context",
    "status",
    "returncode",
    "wall_time_seconds",
    "command",
    "raw_output",
]


@dataclass(frozen=True)
class Experiment:
    name: str
    script: Path
    prerequisites: Tuple[str, ...] = ()


MODEL_MODELSCOPE_IDS = {
    "llama-7b": "LLM-Research/Llama-2-7b",
    "llama2-7b": "LLM-Research/Llama-2-7b",
    "llama-2-7b": "LLM-Research/Llama-2-7b",
    "llama-2-7b-hf": "LLM-Research/Llama-2-7b",
    "llama2-7b-chat": "LLM-Research/Llama-2-7b-chat",
    "llama-2-7b-chat": "LLM-Research/Llama-2-7b-chat",
    "llama-2-7b-chat-hf": "LLM-Research/Llama-2-7b-chat",
}


def resolve_model_path(model_name: Path, cache_dir: Path) -> Path:
    """Return a local model directory, downloading from ModelScope if needed."""
    model_name = Path(model_name).expanduser()
    if model_name.is_dir():
        return model_name.resolve()

    model_id = MODEL_MODELSCOPE_IDS.get(str(model_name), str(model_name))
    try:
        try:
            from modelscope import snapshot_download
        except ImportError:
            from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"Model {model_name!r} is not a local directory; automatic download requires "
            "the modelscope package. Install it with `pip install modelscope`."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(model_id, cache_dir=str(cache_dir))
    return Path(downloaded).resolve()


EXPERIMENTS: Tuple[Experiment, ...] = (
    Experiment("fp16_matmul_baseline", BENCHMARK_DIR / "scripts" / "test_kernel_baselines.py"),
    Experiment("fp16_k_plus_rope", BENCHMARK_DIR / "scripts" / "test_kernel_benchmark_K_plus_rope.py"),
    Experiment("offload_topk", BENCHMARK_DIR / "scripts" / "test_offload_topk.py"),
    Experiment(
        "kvquant_key_matmul",
        BENCHMARK_DIR / "scripts" / "test_kernels_key.py",
        ("quantizers.pickle", "activations-seqlen2048.pickle"),
    ),
    Experiment(
        "kvquant_key_packing",
        BENCHMARK_DIR / "scripts" / "test_kernels_key_packing.py",
        ("quantizers.pickle", "activations-seqlen2048.pickle"),
    ),
    Experiment(
        "kvquant_value_matmul",
        BENCHMARK_DIR / "scripts" / "test_kernels_value.py",
        ("quantizers.pickle", "activations-seqlen2048.pickle"),
    ),
    Experiment(
        "kvquant_value_packing",
        BENCHMARK_DIR / "scripts" / "test_kernels_value_packing.py",
        ("quantizers.pickle", "activations-seqlen2048.pickle"),
    ),
)


def parse_duration(value: str) -> Optional[float]:
    """Parse a PyTorch profiler duration into microseconds."""
    text = value.strip().replace(",", "")
    multiplier = 1.0
    if text.endswith(("µs", "us")):
        text = text[:-2]
    elif text.endswith("ms"):
        text = text[:-2]
        multiplier = 1_000.0
    elif text.endswith("ns"):
        text = text[:-2]
        multiplier = 0.001
    elif text.endswith("s"):
        text = text[:-1]
        multiplier = 1_000_000.0
    else:
        return None
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def parse_number(value: str) -> Optional[float]:
    text = value.strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def normalize_header(header: str) -> str:
    text = header.strip().replace("µs", "us")
    text = re.sub(r"[^A-Za-z0-9%]+", "_", text).strip("_").lower()
    return text


def profiler_metric_name(header: str) -> Tuple[Optional[str], Optional[str]]:
    normalized = normalize_header(header)
    if normalized == "name" or normalized == "source file" or normalized == "module":
        return None, None
    if normalized in {"self_cuda", "self_cuda_time", "self_cuda_time_total", "self_cuda_time_us"}:
        return "self_cuda_time", "us"
    if normalized in {"cuda_time_avg", "cuda_time_average", "cuda_avg"}:
        return "cuda_time_avg", "us"
    if normalized in {"cuda_total", "cuda_time_total", "cuda_time_total_us"}:
        return "cuda_time_total", "us"
    if normalized in {"self_cpu", "self_cpu_time", "self_cpu_time_total", "self_cpu_time_us"}:
        return "self_cpu_time", "us"
    if normalized in {"cpu_time_avg", "cpu_time_average", "cpu_avg"}:
        return "cpu_time_avg", "us"
    if normalized in {"cpu_total", "cpu_time_total", "cpu_time_total_us"}:
        return "cpu_time_total", "us"
    if normalized == "calls":
        return "calls", "count"
    if normalized.endswith("%"):
        return f"percent.{normalized[:-1].rstrip('_')}", "percent"
    return f"profiler.{normalized}", "unknown"


def _metrics_from_table_cells(
    cells: List[str], headers: List[str], canonical: List[Tuple[Optional[str], Optional[str]]]
) -> List[Dict[str, str]]:
    metrics: List[Dict[str, str]] = []
    if len(cells) != len(headers) or not cells:
        return metrics
    context = cells[0]
    for value, (metric_name, unit) in zip(cells[1:], canonical[1:]):
        if metric_name is None:
            continue
        numeric_value = parse_number(value)
        if numeric_value is None and unit == "us":
            numeric_value = parse_duration(value)
        if numeric_value is None:
            continue
        metrics.append(
            {
                "metric_name": metric_name,
                "metric_value": format(numeric_value, ".12g"),
                "metric_unit": unit or "unknown",
                "context": context,
            }
        )
    return metrics


def parse_profiler_table(output: str) -> List[Dict[str, str]]:
    """Parse pipe-delimited and PyTorch's whitespace-aligned profiler tables."""
    metrics: List[Dict[str, str]] = []
    headers: List[str] = []
    canonical: List[Tuple[Optional[str], Optional[str]]] = []
    table_active = False
    separator_count = 0

    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        is_separator = bool(line) and bool(re.fullmatch(r"[-: ]+", line))

        if table_active and is_separator:
            separator_count += 1
            if separator_count >= 2:
                table_active = False
                headers = []
                canonical = []
            continue
        if is_separator:
            continue

        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(re.fullmatch(r"[-: ]*", cell) for cell in cells):
                continue
            if not table_active:
                headers = cells
                canonical = [profiler_metric_name(header) for header in headers]
                table_active = True
                separator_count = 0
                continue
            metrics.extend(_metrics_from_table_cells(cells, headers, canonical))
            continue

        # PyTorch often emits a fixed-width table without pipe delimiters.
        if not table_active and re.match(r"^Name(?:\s|$)", line):
            header_names = re.findall(
                r"Self CPU %|Self CPU Time|Self CPU|CPU total %|CPU total Time|CPU total|"
                r"CPU time avg|Self CUDA %|Self CUDA Time|Self CUDA|CUDA total %|CUDA total Time|"
                r"CUDA total|CUDA time avg|Calls|Events",
                line,
            )
            headers = ["Name"] + header_names
            canonical = [profiler_metric_name(header) for header in headers]
            table_active = True
            separator_count = 0
            continue

        if table_active and headers:
            # The remaining fields are numeric, while the first field can contain
            # spaces and template arguments.  Take the final N numeric-looking
            # tokens so numbers inside a kernel name do not break parsing.
            values = re.findall(
                r"(?<![A-Za-z0-9_])(?:nan|inf|-?\d[\d,.]*(?:[eE][-+]?\d+)?)"
                r"(?:%|ns|us|µs|ms|s)?(?![A-Za-z0-9_])",
                line,
            )
            expected = len(headers) - 1
            if len(values) < expected:
                continue
            value_matches = list(re.finditer(
                r"(?<![A-Za-z0-9_])(?:nan|inf|-?\d[\d,.]*(?:[eE][-+]?\d+)?)"
                r"(?:%|ns|us|µs|ms|s)?(?![A-Za-z0-9_])",
                line,
            ))[-expected:]
            first_value = value_matches[0].start()
            context = line[:first_value].strip()
            cells = [context] + [match.group(0) for match in value_matches]
            metrics.extend(_metrics_from_table_cells(cells, headers, canonical))

    return metrics


PLAIN_METRICS = {
    "baseline": ("custom.baseline_iteration", "s"),
    "offload": ("custom.offload_iteration", "s"),
    "median": ("e2e.median_token_latency", "s"),
    "ppl": ("e2e.perplexity", "ppl"),
    "max memory(mib)": ("e2e.max_memory", "MiB"),
}


def parse_plain_metrics(output: str) -> List[Dict[str, str]]:
    metrics: List[Dict[str, str]] = []
    for line in output.splitlines():
        match = re.match(r"^([A-Za-z][A-Za-z0-9 _()\-]*?)\s*:\s*([-+eE.,0-9]+)\s*$", line.strip())
        if not match:
            continue
        key = match.group(1).strip().lower()
        if key not in PLAIN_METRICS:
            continue
        metric_name, unit = PLAIN_METRICS[key]
        value = parse_number(match.group(2))
        if value is not None:
            metrics.append(
                {
                    "metric_name": metric_name,
                    "metric_value": format(value, ".12g"),
                    "metric_unit": unit,
                    "context": "stdout",
                }
            )
    return metrics


def missing_prerequisites(experiment: Experiment, check: bool) -> List[str]:
    if not check:
        return []
    return [str(path) for path in experiment.prerequisites if not (BENCHMARK_DIR / path).is_file()]


def run_process(
    command: Sequence[str],
    timeout: Optional[float],
    cwd: Path = BENCHMARK_DIR,
) -> Tuple[int, str, float]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    output_lines: List[str] = []

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        output_lines.append(f"\n[runner] timed out after {timeout} seconds\n")
        returncode = 124
    else:
        returncode = process.returncode
    thread.join(timeout=2)
    return returncode, "".join(output_lines), time.monotonic() - started


def preparation_commands(
    args: argparse.Namespace,
    fisher_path: Path,
    quantizer_path: Path,
    activation_path: Path,
) -> List[Tuple[str, Sequence[str], Path, Path]]:
    if not args.model_name:
        raise SystemExit("--prepare-artifacts requires --model-name")
    root = BENCHMARK_DIR.parent
    commands: List[Tuple[str, Sequence[str], Path, Path]] = []

    fisher_outputs = list(fisher_path.glob("pytorch_model*")) + list(fisher_path.glob("*.safetensors"))
    if args.force_artifacts or not fisher_outputs:
        commands.append(
            (
                "prepare_fisher",
                [
                    str(args.gradient_python),
                    "run-fisher.py",
                    "--model_name_or_path",
                    str(args.model_path),
                    "--output_dir",
                    str(fisher_path),
                    "--dataset",
                    "wikitext2",
                    "--dataset_source",
                    args.dataset_source,
                    "--seqlen",
                    str(args.seqlen),
                    "--maxseqlen",
                    str(args.seqlen),
                    "--num_examples",
                    str(args.fisher_num_examples),
                ],
                root / "gradients",
                fisher_path,
            )
        )

    if args.force_artifacts or not quantizer_path.is_file():
        commands.append(
            (
                "prepare_quantizers",
                [
                    str(args.quant_python),
                    "llama_simquant.py",
                    str(args.model_path),
                    "--abits",
                    "4",
                    "--nsamples",
                    str(args.fisher_num_examples),
                    "--seqlen",
                    str(args.seqlen),
                    "--maxseqlen",
                    str(args.seqlen),
                    "--nuq",
                    "--fisher",
                    str(fisher_path),
                    "--quantize",
                    "--include_sparse",
                    "--sparsity-threshold",
                    "0.99",
                    "--dataset_source",
                    args.dataset_source,
                    "--quantizer-path",
                    str(quantizer_path),
                ],
                root / "quant",
                quantizer_path,
            )
        )

    if args.force_artifacts or not activation_path.is_file():
        commands.append(
            (
                "prepare_activations",
                [
                    args.python,
                    "cache-llama-activations.py",
                    str(args.model_path),
                    "--wbits",
                    "4",
                    "--nsamples",
                    "1",
                    "--seqlen",
                    str(args.seqlen),
                    "--maxseqlen",
                    str(args.seqlen),
                    "--dataset_source",
                    args.dataset_source,
                    "--quantizer-path",
                    str(quantizer_path),
                    "--output-path",
                    str(activation_path),
                ],
                BENCHMARK_DIR,
                activation_path,
            )
        )
    return commands


def select_experiments(only: Iterable[str], exclude: Iterable[str]) -> List[Experiment]:
    selected = list(EXPERIMENTS)
    only_names = set(only)
    exclude_names = set(exclude)
    if only_names:
        selected = [experiment for experiment in selected if experiment.name in only_names]
    if exclude_names:
        selected = [experiment for experiment in selected if experiment.name not in exclude_names]
    unknown = (only_names | exclude_names) - {experiment.name for experiment in EXPERIMENTS}
    if unknown:
        valid = ", ".join(experiment.name for experiment in EXPERIMENTS)
        raise SystemExit(f"Unknown experiment(s): {', '.join(sorted(unknown))}. Valid names: {valid}")
    return selected


def write_csv_row(writer: csv.DictWriter, base: Dict[str, str], metrics: Sequence[Dict[str, str]]) -> None:
    if not metrics:
        writer.writerow({**base, "metric_name": "run.no_metrics", "metric_value": "", "metric_unit": "", "context": ""})
        return
    for metric in metrics:
        writer.writerow({**base, **metric})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for CSV, raw logs, and plots")
    parser.add_argument("--csv", type=Path, default=None, help="CSV path (default: OUTPUT_DIR/results.csv)")
    parser.add_argument("--only", action="append", default=[], help="Run only this experiment; repeatable")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude this experiment; repeatable")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repetitions per experiment")
    parser.add_argument("--timeout", type=float, default=None, help="Per-run timeout in seconds")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--no-prerequisite-check", action="store_true", help="Do not skip runs when artifacts are missing")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed run")
    parser.add_argument("--append", action="store_true", help="Append to an existing CSV instead of creating it")
    parser.add_argument("--plot", action="store_true", help="Create summary.csv and benchmark_summary.png after runs")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for benchmark scripts")
    parser.add_argument("--prepare-artifacts", action="store_true", help="Generate Fisher info, quantizers.pickle, and activations before benchmarks")
    parser.add_argument(
        "--model-name", "--model-path", dest="model_name", type=Path, default=Path("llama-7b"),
        help="Project model name or ModelScope model ID; a local directory is also accepted",
    )
    parser.add_argument("--model-cache", type=Path, default=None, help="ModelScope cache directory (default: benchmarking/modelscope_models)")
    parser.add_argument("--fisher-path", type=Path, default=None, help="Fisher output directory (default: OUTPUT_DIR/fisher)")
    parser.add_argument("--fisher-num-examples", type=int, default=16, help="Calibration examples for Fisher/quantizer generation")
    parser.add_argument("--seqlen", type=int, choices=(2048,), default=2048, help="Calibration and activation sequence length")
    parser.add_argument("--dataset-source", choices=("hf", "modelscope"), default="modelscope", help="Dataset backend for artifact preparation")
    parser.add_argument("--gradient-python", type=Path, default=None, help="Python executable for gradients environment")
    parser.add_argument("--quant-python", type=Path, default=None, help="Python executable for quant environment")
    parser.add_argument("--force-artifacts", action="store_true", help="Regenerate artifacts even when outputs already exist")
    args = parser.parse_args()

    if args.list:
        for experiment in EXPERIMENTS:
            prerequisites = ", ".join(experiment.prerequisites) or "-"
            print(f"{experiment.name:28s} {experiment.script.relative_to(BENCHMARK_DIR)} [{prerequisites}]")
        return 0
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (BENCHMARK_DIR / "benchmark_results" / f"run_{timestamp}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (args.csv or output_dir / "results.csv").resolve()
    model_cache = (args.model_cache or BENCHMARK_DIR / "modelscope_models").resolve()
    if args.prepare_artifacts and not args.dry_run:
        args.model_path = resolve_model_path(args.model_name, model_cache)
    else:
        args.model_path = args.model_name
    fisher_path = (args.fisher_path or output_dir / "fisher").resolve()
    quantizer_path = BENCHMARK_DIR / "quantizers.pickle"
    activation_path = BENCHMARK_DIR / f"activations-seqlen{args.seqlen}.pickle"
    if args.gradient_python is None:
        args.gradient_python = Path(args.python)
    if args.quant_python is None:
        args.quant_python = Path(args.python)

    args.gradient_python = args.gradient_python.resolve()
    args.quant_python = args.quant_python.resolve()

    experiments = select_experiments(args.only, args.exclude)
    commands = [
        (
            experiment,
            [args.python, str(experiment.script.relative_to(BENCHMARK_DIR))],
        )
        for experiment in experiments
    ]
    if args.dry_run:
        print(f"Output CSV: {csv_path}")
        if args.prepare_artifacts:
            for name, command, cwd, output_path in preparation_commands(
                args, fisher_path, quantizer_path, activation_path
            ):
                print(f"PREP {name:20s} cwd={cwd} output={output_path}\n      {' '.join(command)}")
        for experiment, command in commands:
            missing = missing_prerequisites(experiment, not args.no_prerequisite_check and not args.prepare_artifacts)
            state = "SKIP" if missing else "RUN"
            print(f"{state:4s} {' '.join(command)}")
        return 0

    csv_fields = list(DEFAULT_CSV_FIELDS)
    if args.append and csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            existing_fields = csv.DictReader(handle).fieldnames or []
        csv_fields = existing_fields or csv_fields

    should_write_header = not (args.append and csv_path.exists())
    failures = 0
    with csv_path.open("a" if args.append else "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields, extrasaction="ignore")
        if should_write_header:
            writer.writeheader()

        if args.prepare_artifacts:
            for name, command, cwd, output_path in preparation_commands(
                args, fisher_path, quantizer_path, activation_path
            ):
                run_id = f"{timestamp}-{name}"
                timestamp_value = dt.datetime.now().isoformat(timespec="seconds")
                display_command = " ".join(command)
                print(f"\n[runner] {name}\n[runner] cwd={cwd}\n[runner] {display_command}\n", flush=True)
                raw_path = output_dir / f"{name}.log"
                returncode, output, wall_time = run_process(command, args.timeout, cwd=cwd)
                raw_path.write_text(output, encoding="utf-8")
                try:
                    raw_relative = str(raw_path.relative_to(BENCHMARK_DIR))
                except ValueError:
                    raw_relative = str(raw_path)
                status = "PASS" if returncode == 0 else "FAIL"
                if returncode != 0:
                    failures += 1
                writer.writerow(
                    {
                        "timestamp": timestamp_value,
                        "run_id": run_id,
                        "repeat": 1,
                        "experiment": name,
                        "metric_name": "run.wall_time",
                        "metric_value": f"{wall_time:.9f}",
                        "metric_unit": "s",
                        "context": str(output_path),
                        "status": status,
                        "returncode": returncode,
                        "wall_time_seconds": f"{wall_time:.9f}",
                        "command": display_command,
                        "raw_output": raw_relative,
                    }
                )
                csv_file.flush()
                print(f"\n[runner] {status}; returncode={returncode}; log={raw_relative}\n", flush=True)
                if returncode != 0:
                    print(f"[runner] stopping; artifact generation failed for {name}", flush=True)
                    return 1

        for repeat in range(1, args.repeat + 1):
            for experiment, command in commands:
                run_id = f"{timestamp}-{experiment.name}-r{repeat}"
                timestamp_value = dt.datetime.now().isoformat(timespec="seconds")
                missing = missing_prerequisites(experiment, not args.no_prerequisite_check)
                display_command = " ".join(command)
                print(f"\n[runner] {experiment.name} (repeat {repeat}/{args.repeat})\n[runner] {display_command}\n", flush=True)

                if missing:
                    raw_relative = ""
                    base = {
                        "timestamp": timestamp_value,
                        "run_id": run_id,
                        "repeat": repeat,
                        "experiment": experiment.name,
                        "metric_name": "run.skipped",
                        "metric_value": "",
                        "metric_unit": "",
                        "context": ";".join(missing),
                        "status": "SKIPPED",
                        "returncode": "",
                        "wall_time_seconds": "",
                        "command": display_command,
                        "raw_output": "",
                    }
                    writer.writerow(base)
                    csv_file.flush()
                    print(f"[runner] skipped; missing: {', '.join(missing)}", flush=True)
                    continue

                raw_path = output_dir / f"{experiment.name}.r{repeat}.log"
                returncode, output, wall_time = run_process(command, args.timeout)
                raw_path.write_text(output, encoding="utf-8")
                try:
                    raw_relative = str(raw_path.relative_to(BENCHMARK_DIR))
                except ValueError:
                    raw_relative = str(raw_path)
                status = "PASS" if returncode == 0 else "FAIL"
                if returncode != 0:
                    failures += 1

                base = {
                    "timestamp": timestamp_value,
                    "run_id": run_id,
                    "repeat": repeat,
                    "experiment": experiment.name,
                    "status": status,
                    "returncode": returncode,
                    "wall_time_seconds": f"{wall_time:.9f}",
                    "command": display_command,
                    "raw_output": raw_relative,
                }
                metrics = parse_profiler_table(output) + parse_plain_metrics(output)
                metrics.append(
                    {
                        "metric_name": "run.wall_time",
                        "metric_value": f"{wall_time:.9f}",
                        "metric_unit": "s",
                        "context": "runner",
                    }
                )
                write_csv_row(writer, base, metrics)
                csv_file.flush()
                print(f"\n[runner] {status}; returncode={returncode}; wall_time={wall_time:.3f}s; log={raw_relative}\n", flush=True)

                if returncode != 0 and args.stop_on_failure:
                    break
            if args.stop_on_failure and failures:
                break

    print(f"[runner] metrics CSV: {csv_path}", flush=True)
    if args.plot:
        try:
            from summarize_benchmarks import summarize
        except ImportError as exc:
            print(f"[runner] could not import summarizer: {exc}", file=sys.stderr)
            return 2
        return summarize(csv_path=csv_path, output_dir=output_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
