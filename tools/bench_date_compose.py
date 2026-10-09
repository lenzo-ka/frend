"""Benchmark CLDR-derived tiergraph date composition in fresh locale processes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from icukit.detectors import DateTimeValue, detect
from tiergraph import generate, recognize

from frend.date_rules import (
    GENERATED_DATE_LOCALES,
    date_grammar_input,
    exact_date_generation_count,
    generate_date_alternatives,
    load_date_rule_bundle,
    render_date_derivation,
)
from frend.normalize import _reading_detectors

REPO = Path(__file__).resolve().parents[1]
INPUTS = Path(__file__).with_name("date_latency_inputs.json")


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))]


def _date_detection(locale: str, text: str) -> dict:
    rows = [
        row
        for row in detect(text, _reading_detectors(locale))
        if str(row.get("type", "")).startswith("date:")
        and row["start"] == 0
        and row["end"] == len(text)
        and set(dict(row["value"].fields)) == {"y", "M", "d"}
    ]
    if not rows:
        raise ValueError(f"{locale} probe {text!r} has no full-span y/M/d date detection")
    return rows[0]


def _one(locale: str, iterations: int) -> dict:
    probes = json.loads(INPUTS.read_text(encoding="utf-8"))["locales"]
    text = probes[locale]["probe"]
    detection = _date_detection(locale, text)
    value = detection["value"]
    assert isinstance(value, DateTimeValue)

    started = time.perf_counter_ns()
    bundle = load_date_rule_bundle(locale)
    loaded = time.perf_counter_ns()
    grammar_input = date_grammar_input(value, detection, locale)
    forest = recognize(bundle.lowered, grammar_input, collapse_units=False)
    recognized = time.perf_counter_ns()
    exact = exact_date_generation_count(forest, grammar_input)
    result = generate(forest, count=exact)
    generated = time.perf_counter_ns()
    rendered = [render_date_derivation(item) for item in result.derivations]
    finished = time.perf_counter_ns()
    if result.truncated or len(rendered) != exact:
        raise ValueError(f"{locale} cold generation was incomplete")

    samples: list[int] = []
    for _index in range(iterations):
        before = time.perf_counter_ns()
        warm = generate_date_alternatives(value, detection, locale)
        samples.append(time.perf_counter_ns() - before)
        if len(warm) != exact:
            raise ValueError(f"{locale} warm generation was incomplete")
    return {
        "locale": locale,
        "probe": text,
        "cold_ns": finished - started,
        "phases_ns": {
            "load_lower": loaded - started,
            "recognize": recognized - loaded,
            "generate": generated - recognized,
            "render": finished - generated,
        },
        "p50_ns": percentile(samples, 0.50),
        "p95_ns": percentile(samples, 0.95),
        "iterations": iterations,
        "exact_target_count": exact,
        "returned_count": len(result.derivations),
        "truncated": result.truncated,
        "target_piece_count": sum(len(item.pieces) for item in result.derivations),
        "grammar_fingerprint": bundle.lowered.program.fingerprint(),
        "icu_version": bundle.icu_version,
        "bundle_count": load_date_rule_bundle.cache_info().currsize,
    }


def _child(locale: str, iterations: int) -> int:
    print(json.dumps(_one(locale, iterations), ensure_ascii=False))
    return 0


def _exercise_unseen_dates(values_per_locale: int = 2000) -> int:
    """Run complete production generation for values that cannot share a date key."""
    generated = 0
    for locale_index, locale in enumerate(GENERATED_DATE_LOCALES):
        start = date(2000, 1, 1) + timedelta(days=locale_index * values_per_locale)
        for index in range(values_per_locale):
            value = start + timedelta(days=index)
            structured = DateTimeValue(
                (("y", value.year), ("M", value.month), ("d", value.day)),
                "gregorian",
            )
            generate_date_alternatives(structured, {"value": structured}, locale)
            generated += 1
    return generated


def benchmark(out: Path, iterations: int, repeat: int) -> dict:
    rows = []
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for run in range(1, repeat + 1):
        for locale in GENERATED_DATE_LOCALES:
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
    for locale in GENERATED_DATE_LOCALES:
        selected = [row for row in rows if row["locale"] == locale]
        p50 = int(statistics.median(row["p50_ns"] for row in selected))
        p95 = int(statistics.median(row["p95_ns"] for row in selected))
        summary[locale] = {"p50_ns": p50, "p95_ns": p95}
        if p50 > 300_000:
            failures.append({"locale": locale, "metric": "p50_ns", "value": p50})
        if p95 > 1_000_000:
            failures.append({"locale": locale, "metric": "p95_ns", "value": p95})
    load_date_rule_bundle.cache_clear()
    for locale in GENERATED_DATE_LOCALES:
        load_date_rule_bundle(locale)
    peak_bundle_count = load_date_rule_bundle.cache_info().currsize
    unseen_generations = _exercise_unseen_dates()
    stable_bundle_count = load_date_rule_bundle.cache_info().currsize
    if peak_bundle_count != len(GENERATED_DATE_LOCALES):
        failures.append(
            {
                "metric": "peak_bundle_count",
                "value": peak_bundle_count,
                "expected": len(GENERATED_DATE_LOCALES),
            }
        )
    if stable_bundle_count != peak_bundle_count:
        failures.append(
            {
                "metric": "bundle_count_after_unseen_dates",
                "value": stable_bundle_count,
                "expected": peak_bundle_count,
            }
        )
    report = {
        "schema": 1,
        "repeat": repeat,
        "iterations": iterations,
        "tracemalloc": False,
        "rows": rows,
        "summary": summary,
        "peak_observed_bundle_count": peak_bundle_count,
        "bundle_count_after_unseen_dates": stable_bundle_count,
        "never_seen_date_values_per_locale": 2000,
        "never_seen_generation_calls": unseen_generations,
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
    parser.add_argument("--child-locale", choices=GENERATED_DATE_LOCALES)
    args = parser.parse_args()
    if args.child_locale:
        return _child(args.child_locale, args.iterations)
    if args.out is None:
        parser.error("--out is required")
    report = benchmark(args.out, args.iterations, args.repeat)
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
