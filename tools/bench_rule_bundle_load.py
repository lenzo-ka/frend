#!/usr/bin/env python3
"""Measure cold generated-rule bundle load time and resident-set size."""

from __future__ import annotations

import argparse
import importlib
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import frend

_FAMILIES = ("date", "time", "fraction", "money")


def _peak_kib() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak // 1024 if sys.platform == "darwin" else peak


def _probe(family: str, locale: str) -> dict[str, int]:
    module = importlib.import_module(f"frend.{family}_rules")
    loader = getattr(module, f"load_{family}_rule_bundle")
    loader.cache_clear()
    before = _peak_kib()
    started = time.perf_counter_ns()
    loader(locale)
    elapsed = time.perf_counter_ns() - started
    peak = _peak_kib()
    return {"elapsed_ns": elapsed, "peak_rss_kib": peak, "rss_delta_kib": peak - before}


def _locales(family: str) -> tuple[str, ...]:
    root = Path(frend.__file__).resolve().parent / "data" / f"{family}_rules"
    return tuple(path.stem for path in sorted(root.glob("*.json")))


def _measure(family: str, repeats: int) -> dict[str, object]:
    rows = {}
    for locale in _locales(family):
        samples = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            completed = subprocess.run(
                [sys.executable, __file__, "--probe", family, locale],
                check=True,
                capture_output=True,
                text=True,
            )
            sample = json.loads(completed.stdout)
            sample["cold_load_ns"] = time.perf_counter_ns() - started
            samples.append(sample)
        rows[locale] = {
            "median_cold_load_ns": int(statistics.median(item["cold_load_ns"] for item in samples)),
            "median_loader_ns": int(statistics.median(item["elapsed_ns"] for item in samples)),
            "median_peak_rss_kib": int(statistics.median(item["peak_rss_kib"] for item in samples)),
            "median_rss_delta_kib": int(
                statistics.median(item["rss_delta_kib"] for item in samples)
            ),
        }
    return {"family": family, "repeats": repeats, "locales": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=_FAMILIES)
    parser.add_argument("locale", nargs="?")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.probe:
        if args.locale is None:
            parser.error("--probe requires a locale")
        print(json.dumps(_probe(args.family, args.locale), sort_keys=True))
        return 0
    if args.locale is not None:
        parser.error("locale is only valid with --probe")
    print(json.dumps(_measure(args.family, args.repeats), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
