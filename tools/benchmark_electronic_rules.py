#!/usr/bin/env python3
"""Absolute latency and RSS gates for the sourced electronic rule path."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _rss_bytes() -> int:
    output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(output.strip()) * 1024


def _percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(quantile * len(ordered) - 1)))]


def _value(index: int):
    from frend.electronic import ElectronicValue, runs

    text = f"a{index}b-c_{index + 1}@d{index + 2}.de"
    return ElectronicValue("email", f"d{index + 2}.de", runs(text))


def _probe_value():
    from frend.electronic import ElectronicValue, runs

    return ElectronicValue("email", "c.de", runs("a-b@c.de"))


def _child() -> dict[str, object]:
    from frend.electronic_rules import (
        generate_electronic_alternatives,
        load_electronic_rule_bundle,
    )

    before = _rss_bytes()
    load_electronic_rule_bundle("de_DE")
    gc.collect()
    loaded = _rss_bytes()
    started = time.perf_counter_ns()
    first = generate_electronic_alternatives(_probe_value(), "de_DE")
    cold = time.perf_counter_ns() - started
    warm = []
    probe = _probe_value()
    for _index in range(200):
        started = time.perf_counter_ns()
        generate_electronic_alternatives(probe, "de_DE")
        warm.append(time.perf_counter_ns() - started)
    gc.collect()
    post_workload = _rss_bytes()
    rss_passes = []
    receipts = []
    for pass_index in range(3):
        selected = 0
        offset = pass_index * 2000 + 10_000
        for index in range(2000):
            alternatives = generate_electronic_alternatives(_value(offset + index), "de_DE")
            selected += any(
                "electronic-rule:de_DE:email" in item.provenance for item in alternatives
            )
        gc.collect()
        rss_passes.append(_rss_bytes())
        receipts.append(
            {
                "pass": pass_index + 1,
                "count": 2000,
                "first_index": offset,
                "last_index": offset + 1999,
                "selected_generated": selected,
            }
        )
    return {
        "cold_ns": cold,
        "warm_p50_ns": _percentile(warm, 0.50),
        "warm_p95_ns": _percentile(warm, 0.95),
        "alternatives": len(first),
        "rss_before_bytes": before,
        "rss_loaded_bytes": loaded,
        "rss_loaded_delta_bytes": loaded - before,
        "rss_post_workload_delta_bytes": post_workload - before,
        "rss_passes_bytes": rss_passes,
        "pass3_minus_pass1_bytes": rss_passes[2] - rss_passes[0],
        "receipts": receipts,
    }


def _run_child(repo: Path) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPYCACHEPREFIX"] = str(
        Path(tempfile.gettempdir()) / f"frend-electronic-empty-pycache-{os.getpid()}"
    )
    environment["PYTHONPATH"] = str(repo)
    result = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--child"],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        print(json.dumps(_child()))
        return 0
    if args.repo is None or args.out is None:
        parser.error("--repo and --out are required")
    runs = [_run_child(args.repo.resolve()) for _ in range(args.repeat)]
    runtime_bytes = (args.repo / "frend/data/electronic_rules/de_DE.json").stat().st_size
    loaded_limit = min(8 * 2**20, max(256 * 2**10, 4 * runtime_bytes))
    summary = {
        name: int(statistics.median(int(run[name]) for run in runs))
        for name in (
            "cold_ns",
            "warm_p50_ns",
            "warm_p95_ns",
            "rss_loaded_delta_bytes",
            "rss_post_workload_delta_bytes",
            "pass3_minus_pass1_bytes",
        )
    }
    failures = []
    limits = {
        "cold_ns": 10_000_000,
        "warm_p50_ns": 300_000,
        "warm_p95_ns": 1_000_000,
        "rss_loaded_delta_bytes": loaded_limit,
        "rss_post_workload_delta_bytes": loaded_limit + 2 * 2**20,
        "pass3_minus_pass1_bytes": 2 * 2**20,
    }
    for metric, limit in limits.items():
        if summary[metric] > limit:
            failures.append({"metric": metric, "observed": summary[metric], "limit": limit})
    if any(
        receipt["selected_generated"] != receipt["count"]
        for run in runs
        for receipt in run["receipts"]
    ):
        failures.append({"metric": "soak-receipts", "observed": "incomplete"})
    report = {
        "schema": 1,
        "repeat": args.repeat,
        "bundle_bytes": runtime_bytes,
        "limits": limits,
        "summary": summary,
        "runs": runs,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "failures": failures}, sort_keys=True))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
