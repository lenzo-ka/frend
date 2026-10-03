"""Benchmark frend lattice folds over externally supplied detection forests.

The parent reads the fixed JSONL corpus and starts one child at a time. Each
child receives one text on stdin, so ``--timeout`` bounds pathological texts
without mixing process start-up into any recorded stage. Detection is never run.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_DATA = Path("/Volumes/k02/processed/frend/fold-bench")
_MODES = (
    "lattice_build",
    "resolve_k1",
    "resolve_k8",
    "resolve_k64",
    "keep_all",
    "ranked_path_k64",
)


class _ModeTimeout(Exception):
    pass


def _elapsed(operation, timeout: float | None = None):
    def expired(_signum, _frame):
        raise _ModeTimeout(f"mode timeout after {timeout:g} seconds")

    previous = signal.signal(signal.SIGALRM, expired)
    if timeout is not None:
        signal.setitimer(signal.ITIMER_REAL, timeout)
    started = time.perf_counter_ns()
    try:
        value = operation()
        return value, time.perf_counter_ns() - started
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _worker(args: argparse.Namespace) -> int:
    from frend.fold_resolve import _count_top_level, _fold_covers, build_lattice
    from frend.lattice import resolve_choices, resolve_lattice

    case = json.load(sys.stdin)
    detections = case["detections"]
    locale = case["locale"]
    text = case["text"]
    resolve_lattice([], locale=locale, source_text="")
    result = {
        "id": case["id"],
        "bucket": case["bucket"],
        "locale": locale,
        "mode": case["mode"],
        "text_chars": len(text),
        "text_bytes": len(text.encode()),
        "candidates": len(detections),
        "distinct_spans": len({(item["start"], item["end"]) for item in detections}),
        "starts": len({item["start"] for item in detections}),
        "timings_ns": {},
        "errors": {},
    }

    def emit() -> None:
        if args.worker_output is not None:
            args.worker_output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")

    graph = roots = id_to_index = None
    try:
        (graph, roots, id_to_index), elapsed = _elapsed(
            lambda: build_lattice(detections), args.mode_timeout
        )
        result["timings_ns"]["lattice_build"] = elapsed
        result["lattice_position_nodes"] = len(graph.tiers[0].items)
        result["lattice_candidate_nodes"] = len(graph.tiers[1].items)
        result["lattice_nodes"] = sum(len(tier.items) for tier in graph.tiers)
        result["lattice_edges"] = len(graph.relations)
    except Exception as error:  # noqa: BLE001 - a benchmark records refusals
        result["errors"]["lattice_build"] = f"{type(error).__name__}: {error}"
    if graph is not None and roots is not None:
        try:
            result["top_level_size"], _elapsed_ns = _elapsed(
                lambda: _count_top_level(graph, roots), args.mode_timeout
            )
        except Exception as error:  # noqa: BLE001 - a benchmark records refusals
            result["errors"]["top_level_size"] = f"{type(error).__name__}: {error}"
    emit()

    operations = {
        "resolve_k1": lambda: resolve_lattice(
            detections, locale=locale, source_text=text, output_cap=1
        ),
        "resolve_k8": lambda: resolve_lattice(
            detections, locale=locale, source_text=text, output_cap=8
        ),
        "resolve_k64": lambda: resolve_lattice(
            detections, locale=locale, source_text=text, output_cap=64
        ),
        "keep_all": lambda: resolve_choices(detections, locale=locale, source_text=text),
    }
    if graph is not None and roots is not None and id_to_index is not None:
        operations["ranked_path_k64"] = lambda: _fold_covers(
            detections, graph, roots, id_to_index, 64
        )
    for name, operation in operations.items():
        try:
            _value, elapsed = _elapsed(operation, args.mode_timeout)
            result["timings_ns"][name] = elapsed
        except Exception as error:  # noqa: BLE001 - a benchmark records refusals
            result["errors"][name] = f"{type(error).__name__}: {error}"
        emit()
    if args.worker_output is None:
        json.dump(result, sys.stdout, sort_keys=True)
    return 0


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_cases(data_dir: Path) -> list[dict]:
    texts = {row["id"]: row for row in _read_jsonl(data_dir / "texts.jsonl")}
    cases = []
    for path in sorted(data_dir.glob("*.jsonl")):
        if path.name == "texts.jsonl":
            continue
        parts = path.name.removesuffix(".jsonl").split(".")
        if len(parts) != 3:
            continue
        bucket, locale, mode = parts
        by_id: dict[str, list[dict]] = defaultdict(list)
        for row in _read_jsonl(path):
            identifier = row.pop("id")
            by_id[identifier].append(row)
        for identifier, text_row in texts.items():
            if text_row["bucket"] == bucket and text_row["locale"] == locale:
                cases.append({**text_row, "mode": mode, "detections": by_id[identifier]})
    return cases


def _solve3(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            scale = augmented[row][column]
            augmented[row] = [
                left - scale * right
                for left, right in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[row][3] for row in range(3)]


def fit_scaling(rows: list[dict], mode: str) -> dict | None:
    """Fit log(time) ~ log(candidates) + log(position nodes)."""
    points = [
        (
            math.log(max(1, row["candidates"])),
            math.log(max(1, row["lattice_position_nodes"])),
            math.log(row["timings_ns"][mode]),
            row,
        )
        for row in rows
        if mode in row["timings_ns"] and "lattice_position_nodes" in row
    ]
    if len(points) < 3:
        return None
    design = [(1.0, candidates, positions) for candidates, positions, _time, _row in points]
    matrix = [
        [
            sum(left[i] * right[j] for left, right in zip(design, design, strict=True))
            for j in range(3)
        ]
        for i in range(3)
    ]
    vector = [
        sum(row[i] * point[2] for row, point in zip(design, points, strict=True))
        for i in range(3)
    ]
    coefficients = _solve3(matrix, vector)
    if coefficients is None:
        return None
    intercept, candidate_slope, position_slope = coefficients
    predictions = [
        intercept + candidate_slope * candidates + position_slope * positions
        for candidates, positions, _time, _row in points
    ]
    observed = [point[2] for point in points]
    mean = sum(observed) / len(observed)
    residuals = [
        actual - predicted
        for actual, predicted in zip(observed, predictions, strict=True)
    ]
    denominator = sum((actual - mean) ** 2 for actual in observed)
    r_squared = 1 - sum(value * value for value in residuals) / denominator if denominator else 1.0
    outliers = sorted(
        (
            {
                "id": point[3]["id"],
                "mode": point[3]["mode"],
                "locale": point[3]["locale"],
                "residual_log": residual,
                "actual_ms": math.exp(point[2]) / 1_000_000,
                "predicted_ms": math.exp(prediction) / 1_000_000,
            }
            for point, prediction, residual in zip(points, predictions, residuals, strict=True)
        ),
        key=lambda item: item["residual_log"],
        reverse=True,
    )[:5]
    return {
        "n": len(points),
        "candidate_slope": candidate_slope,
        "position_slope": position_slope,
        "r_squared": r_squared,
        "outliers": outliers,
    }


def _timeout_row(case: dict, seconds: float) -> dict:
    detections = case["detections"]
    return {
        "id": case["id"],
        "bucket": case["bucket"],
        "locale": case["locale"],
        "mode": case["mode"],
        "text_chars": len(case["text"]),
        "text_bytes": len(case["text"].encode()),
        "candidates": len(detections),
        "distinct_spans": len({(item["start"], item["end"]) for item in detections}),
        "starts": len({item["start"] for item in detections}),
        "timings_ns": {},
        "errors": {"worker": f"timeout after {seconds:g} seconds"},
    }


def _run(args: argparse.Namespace) -> int:
    data_dir = args.data_dir.resolve()
    expected_results = data_dir / "results"
    output = args.output.resolve()
    if output.parent != expected_results:
        raise SystemExit(f"--output must be directly under {expected_results}")
    expected_results.mkdir(parents=True, exist_ok=True)
    cases = load_cases(data_dir)
    rows = []
    worker_output = output.with_suffix(output.suffix + ".worker")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--worker-output",
        str(worker_output),
        "--mode-timeout",
        str(args.mode_timeout),
    ]
    for index, case in enumerate(cases, 1):
        worker_output.unlink(missing_ok=True)
        try:
            subprocess.run(
                command,
                input=json.dumps(case),
                text=True,
                capture_output=True,
                cwd=_REPO,
                timeout=args.timeout,
                check=True,
            )
            rows.append(json.loads(worker_output.read_text(encoding="utf-8")))
        except subprocess.TimeoutExpired:
            partial = (
                json.loads(worker_output.read_text(encoding="utf-8"))
                if worker_output.exists()
                else _timeout_row(case, args.timeout)
            )
            partial["errors"]["worker"] = f"timeout after {args.timeout:g} seconds"
            rows.append(partial)
        if index % 25 == 0 or index == len(cases):
            print(f"{index}/{len(cases)}", file=sys.stderr, flush=True)
    payload = {
        "schema_version": 1,
        "data_dir": str(data_dir),
        "case_count": len(cases),
        "ranked_path_output_cap": 64,
        "timeout_seconds_per_mode": args.mode_timeout,
        "timeout_seconds_per_text": args.timeout,
        "rows": rows,
        "scaling": {mode: fit_scaling(rows, mode) for mode in _MODES},
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    worker_output.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--mode-timeout", type=float, default=2.0)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return _worker(args)
    if args.output is None:
        parser.error("--output is required")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
