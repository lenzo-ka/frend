#!/usr/bin/env python3
"""Measure cold and warm direct composition for regional telephone rules."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from frend.telephone import TelephoneDetector
from frend.telephone_rules import generate_telephone_alternatives, regional_telephone_formats

PROBES = (
    ("fr_FR", "01 23 45 67 89"),
    ("pt_BR", "(11) 96123-4567"),
    ("pt_PT", "21 203 0405"),
)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * percentile))]


def benchmark() -> dict[str, object]:
    rows = []
    for locale, written in PROBES:
        detector = TelephoneDetector(locale)
        cold = []
        for _ in range(20):
            regional_telephone_formats.cache_clear()
            start = time.perf_counter_ns()
            detection = detector.detect(written)[0]
            generated = generate_telephone_alternatives(detection["value"], locale)
            cold.append((time.perf_counter_ns() - start) / 1_000_000)
            if not generated:
                raise ValueError(f"no generated reading for {locale}")
        detection = detector.detect(written)[0]
        warm = []
        for _ in range(500):
            start = time.perf_counter_ns()
            generate_telephone_alternatives(detection["value"], locale)
            warm.append((time.perf_counter_ns() - start) / 1_000_000)
        row = {
            "locale": locale,
            "cold_p50_ms": statistics.median(cold),
            "warm_p50_ms": statistics.median(warm),
            "warm_p95_ms": _percentile(warm, 0.95),
            "cold_limit_ms": 10.0,
            "warm_p50_limit_ms": 1.0,
            "warm_p95_limit_ms": 2.0,
        }
        row["passed"] = (
            row["cold_p50_ms"] <= row["cold_limit_ms"]
            and row["warm_p50_ms"] <= row["warm_p50_limit_ms"]
            and row["warm_p95_ms"] <= row["warm_p95_limit_ms"]
        )
        rows.append(row)
    return {"schema": 1, "rows": rows, "passed": all(row["passed"] for row in rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    document = benchmark()
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
