"""Shared fresh-process harness for the date, time, and fraction benchmarks."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = int(len(ordered) * fraction + 0.999999) - 1
    return ordered[min(len(ordered) - 1, max(0, index))]


def run_composition_benchmark(
    *,
    script: Path,
    repository: Path,
    locales: Iterable[str],
    output: Path,
    iterations: int,
    repeat: int,
    limits: Mapping[str, int],
    finish: Callable[[list[dict]], Mapping[str, object]] | None = None,
) -> dict:
    """Run locale workers, summarize their timing vectors, and write one receipt."""
    locale_list = tuple(locales)
    rows = []
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for run in range(1, repeat + 1):
        for locale in locale_list:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(script.resolve()),
                    "--child-locale",
                    locale,
                    "--iterations",
                    str(iterations),
                ],
                cwd=repository,
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
    for locale in locale_list:
        selected = [row for row in rows if row["locale"] == locale]
        metrics = {name: int(statistics.median(row[name] for row in selected)) for name in limits}
        summary[locale] = metrics
        for name, limit in limits.items():
            if metrics[name] > limit:
                failures.append(
                    {
                        "locale": locale,
                        "metric": name,
                        "value": metrics[name],
                        "limit": limit,
                    }
                )

    extra = dict(finish(failures)) if finish is not None else {}
    report = {
        "schema": 1,
        "repeat": repeat,
        "iterations": iterations,
        "tracemalloc": False,
        "rows": rows,
        "summary": summary,
        **extra,
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
