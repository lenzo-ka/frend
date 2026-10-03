"""The latency benchmark preserves main's no-flag measurement receipt."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parents[1]
_TOOL = _REPO / "tools" / "benchmark_latency.py"
_BASELINE = "87a5e6af51135b257d71ec494239c8afe3fdc261"


def _tool():
    spec = importlib.util.spec_from_file_location("benchmark_latency", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deterministic_projection(payload: dict) -> dict:
    """Keep receipt data unaffected by clocks, GC event timing, or argv paths."""
    return {
        "top_level_keys": sorted(payload),
        "schema_version": payload["schema_version"],
        "subject": payload["subject"],
        "manifest_sha256": payload["manifest_sha256"],
        "parameters": {key: payload[key] for key in ("warmup", "runs", "seed", "max_words")},
        "environment": payload["environment"],
        "denominators": payload["denominators"],
        "vector_lengths": {
            locale: {name: len(samples) for name, samples in vectors.items()}
            for locale, vectors in payload["vectors"].items()
        },
        "summary_fields": {
            locale: {name: sorted(summary) for name, summary in boundaries.items()}
            for locale, boundaries in payload["summaries"].items()
        },
        "gc_enabled": payload["gc"]["enabled"],
        "errors": payload["errors"],
        "imports": payload["imports"],
    }


def _run(script: Path, output: Path) -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(_REPO), str(_REPO / "tools")))
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--subject-root",
            str(_REPO),
            "--expected-head",
            head,
            "--manifest",
            str(_REPO / "tools" / "latency_inputs-v1.json"),
            "--warmup",
            "0",
            "--runs",
            "1",
            "--seed",
            "1",
            "--output",
            str(output),
        ],
        cwd=_REPO,
        env=environment,
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_no_flag_full_measurement_matches_main(tmp_path):
    baseline_script = tmp_path / "benchmark_latency-main.py"
    baseline_script.write_bytes(
        subprocess.check_output(
            ["git", "show", f"{_BASELINE}:tools/benchmark_latency.py"], cwd=_REPO
        )
    )
    baseline = _run(baseline_script, tmp_path / "baseline.json")
    candidate = _run(_TOOL, tmp_path / "candidate.json")

    assert _deterministic_projection(candidate) == _deterministic_projection(baseline)


def test_output_inside_repository_refuses():
    with pytest.raises(SystemExit, match="must be outside"):
        _tool().main(
            [
                "--subject-root",
                str(_REPO),
                "--expected-head",
                "irrelevant",
                "--output",
                str(_REPO / "latency-result.json"),
            ]
        )
