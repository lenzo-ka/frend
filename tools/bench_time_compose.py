"""Benchmark sourced clock composition in fresh processes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

from icukit.detectors import DateTimeValue, detect

from frend.normalize import _reading_detectors
from frend.time_rules import (
    GENERATED_TIME_LOCALES,
    generate_time_alternatives,
    load_time_rule_bundle,
)

REPO = Path(__file__).resolve().parents[1]
INPUTS = Path(__file__).with_name("time_latency_inputs.json")


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))]


def _detection(locale: str, text: str) -> dict:
    rows = [
        row
        for row in detect(text, _reading_detectors(locale))
        if row["start"] == 0
        and row["end"] == len(text)
        and isinstance(row.get("value"), DateTimeValue)
        and str(row.get("type", "")).startswith("time:")
    ]
    if not rows:
        raise ValueError(f"{locale} probe {text!r} has no full-span time detection")
    return rows[0]


def _one(locale: str, iterations: int) -> dict:
    text = json.loads(INPUTS.read_text(encoding="utf-8"))["locales"][locale]
    detection = _detection(locale, text)
    value = detection["value"]
    from frend.verbalize import _number_leaf

    for number in (1, 5, 14, 30):
        _number_leaf(Decimal(number), "cardinal", locale)
    started = time.perf_counter_ns()
    cold = generate_time_alternatives(value, detection, locale)
    cold_ns = time.perf_counter_ns() - started
    if not cold or not any(item.provenance.startswith(f"time-rule:{locale}+") for item in cold):
        raise ValueError(f"{locale} cold generation did not execute its time rule")
    samples = []
    for _ in range(iterations):
        before = time.perf_counter_ns()
        alternatives = generate_time_alternatives(value, detection, locale)
        samples.append(time.perf_counter_ns() - before)
        if not alternatives:
            raise ValueError(f"{locale} warm generation was incomplete")
    return {
        "locale": locale,
        "probe": text,
        "cold_ns": cold_ns,
        "p50_ns": percentile(samples, 0.50),
        "p95_ns": percentile(samples, 0.95),
        "iterations": iterations,
        "returned_count": len(cold),
        "bundle_count": load_time_rule_bundle.cache_info().currsize,
    }


def benchmark(out: Path, iterations: int, repeat: int) -> dict:
    rows = []
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for run in range(1, repeat + 1):
        for locale in GENERATED_TIME_LOCALES:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--child-locale",
                    locale,
                    "--iterations",
                    str(iterations),
                ],
                cwd=REPO,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            row = json.loads(completed.stdout)
            row["run"] = run
            rows.append(row)
    summary = {}
    failures = []
    for locale in GENERATED_TIME_LOCALES:
        selected = [row for row in rows if row["locale"] == locale]
        metrics = {
            name: int(statistics.median(row[name] for row in selected))
            for name in ("cold_ns", "p50_ns", "p95_ns")
        }
        summary[locale] = metrics
        for name, limit in (("cold_ns", 10_000_000), ("p50_ns", 300_000), ("p95_ns", 1_000_000)):
            if metrics[name] > limit:
                failures.append(
                    {"locale": locale, "metric": name, "value": metrics[name], "limit": limit}
                )
    report = {
        "schema": 1,
        "repeat": repeat,
        "iterations": iterations,
        "tracemalloc": False,
        "rows": rows,
        "summary": summary,
        "failures": failures,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--child-locale", choices=GENERATED_TIME_LOCALES)
    args = parser.parse_args()
    if args.child_locale:
        print(json.dumps(_one(args.child_locale, args.iterations), ensure_ascii=False))
        return 0
    if args.out is None:
        parser.error("--out is required")
    return int(bool(benchmark(args.out, args.iterations, args.repeat)["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
