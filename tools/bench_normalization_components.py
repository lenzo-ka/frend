#!/usr/bin/env python3
"""Measure cold regional normalization loading and expansion."""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

LOCALES = ("es_ES", "es_MX", "pt_BR", "pt_PT")


def _rss_kib() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value / 1024) if sys.platform == "darwin" else int(value)


def _child(mode: str) -> int:
    loader = None
    if mode == "plain":
        import frend.normalization_schema  # noqa: F401
    else:
        from frend.normalization_components import load_normalization_bundle

        load_normalization_bundle.cache_clear()
        loader = load_normalization_bundle
    baseline_rss = _rss_kib()
    started = time.perf_counter_ns()
    if mode == "plain":
        root = Path.cwd() / "frend" / "data" / "normalization"
        documents = [json.loads((root / f"{locale}.json").read_text()) for locale in LOCALES]
    else:
        assert loader is not None
        documents = [loader(locale) for locale in LOCALES]
    elapsed = time.perf_counter_ns() - started
    if [document["locale"] for document in documents] != list(LOCALES):
        raise ValueError("normalization locale drift")
    peak_rss = _rss_kib()
    print(
        json.dumps(
            {
                "load_ns": elapsed,
                "peak_rss_kib": peak_rss,
                "rss_delta_kib": max(0, peak_rss - baseline_rss),
            }
        )
    )
    return 0


def benchmark(mode: str, repeat: int) -> dict[str, object]:
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(Path.cwd()), PYTHONDONTWRITEBYTECODE="1")
    rows = []
    for _index in range(repeat):
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--child", mode],
            cwd=Path.cwd(),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        rows.append(json.loads(completed.stdout))
    return {
        "mode": mode,
        "repeat": repeat,
        "median_load_ns": int(statistics.median(row["load_ns"] for row in rows)),
        "median_peak_rss_kib": int(statistics.median(row["peak_rss_kib"] for row in rows)),
        "median_rss_delta_kib": int(statistics.median(row["rss_delta_kib"] for row in rows)),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plain", "expanded"), default="expanded")
    parser.add_argument("--child", choices=("plain", "expanded"))
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.child:
        return _child(args.child)
    document = benchmark(args.mode, args.repeat)
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
