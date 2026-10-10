"""Benchmark sourced tiergraph fraction and percent composition in fresh processes."""

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

from icukit.detectors import detect

from frend.fraction_rules import (
    GENERATED_FRACTION_LOCALES,
    generate_fraction_alternatives,
    generate_percent_alternatives,
    load_fraction_rule_bundle,
)
from frend.normalize import _reading_detectors

REPO = Path(__file__).resolve().parents[1]
INPUTS = Path(__file__).with_name("fraction_latency_inputs.json")


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))]


def _detection(locale: str, text: str, kind: str) -> dict:
    rows = [
        row
        for row in detect(text, _reading_detectors(locale))
        if row["start"] == 0
        and row["end"] == len(text)
        and (
            kind == "fraction"
            and (
                str(row.get("type", "")).startswith("fraction:")
                or str(row.get("type", "")).startswith("number:fraction")
            )
            or kind == "percent"
            and row.get("type") == "number:percent"
        )
    ]
    if not rows:
        raise ValueError(f"{locale} probe {text!r} has no full-span {kind} detection")
    return rows[0]


def _generate(locale: str, kind: str, detection: dict):
    if kind == "fraction":
        return generate_fraction_alternatives(detection, locale)
    value = Decimal(str(detection["value"].decimal)).scaleb(2)
    return generate_percent_alternatives(value, detection, locale)


def _one(locale: str, iterations: int) -> dict:
    probes = json.loads(INPUTS.read_text(encoding="utf-8"))["locales"][locale]
    detections = {kind: _detection(locale, text, kind) for kind, text in probes.items()}
    # Measure PR6's cold composition layer, not the immediate base's lazy ICU
    # formatter construction. The paired end-to-end gate measures both together.
    from frend.verbalize import _number_leaf, _spoken_decimal

    _number_leaf(Decimal(3), "cardinal", locale)
    _number_leaf(Decimal(7), "ordinal", locale)
    _spoken_decimal(Decimal(45), locale)
    started = time.perf_counter_ns()
    cold = {kind: _generate(locale, kind, detections[kind]) for kind in probes}
    cold_ns = time.perf_counter_ns() - started
    if any(not alternatives for alternatives in cold.values()):
        raise ValueError(f"{locale} cold generation was incomplete")
    samples = []
    for index in range(iterations):
        kind = "fraction" if index % 2 == 0 else "percent"
        before = time.perf_counter_ns()
        alternatives = _generate(locale, kind, detections[kind])
        samples.append(time.perf_counter_ns() - before)
        if not alternatives:
            raise ValueError(f"{locale} warm {kind} generation was incomplete")
    return {
        "locale": locale,
        "probes": probes,
        "cold_ns": cold_ns,
        "p50_ns": percentile(samples, 0.50),
        "p95_ns": percentile(samples, 0.95),
        "iterations": iterations,
        "returned_counts": {kind: len(values) for kind, values in cold.items()},
        "bundle_count": load_fraction_rule_bundle.cache_info().currsize,
    }


def benchmark(out: Path, iterations: int, repeat: int) -> dict:
    rows = []
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for run in range(1, repeat + 1):
        for locale in GENERATED_FRACTION_LOCALES:
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
    for locale in GENERATED_FRACTION_LOCALES:
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
    parser.add_argument("--child-locale", choices=GENERATED_FRACTION_LOCALES)
    args = parser.parse_args()
    if args.child_locale:
        print(json.dumps(_one(args.child_locale, args.iterations), ensure_ascii=False))
        return 0
    if args.out is None:
        parser.error("--out is required")
    return int(bool(benchmark(args.out, args.iterations, args.repeat)["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
