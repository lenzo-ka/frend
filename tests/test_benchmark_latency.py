"""The latency benchmark measures the default fold before detection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).parents[1]
_TOOL = _REPO / "tools" / "benchmark_latency.py"
_GOLDEN = _REPO / "tests" / "data" / "benchmark_latency_no_flag_main-87a5e6a.json"


def _tool():
    spec = importlib.util.spec_from_file_location("benchmark_latency", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deterministic_projection(payload: dict) -> dict:
    """Keep stable receipt data while normalizing machine and checkout identity."""
    return {
        "top_level_keys": sorted(payload),
        "schema_version": payload["schema_version"],
        "fold": payload["fold"],
        "subject_fields": sorted(payload["subject"]),
        "manifest_sha256": payload["manifest_sha256"],
        "parameters": {key: payload[key] for key in ("warmup", "runs", "seed", "max_words")},
        "environment_fields": sorted(payload["environment"]),
        "denominators": payload["denominators"],
        "vector_lengths": {
            locale: {name: len(samples) for name, samples in vectors.items()}
            for locale, vectors in payload["vectors"].items()
        },
        "summary_fields": {
            locale: {name: sorted(summary) for name, summary in boundaries.items()}
            for locale, boundaries in payload["summaries"].items()
        },
        "gc": {
            "fields": sorted(payload["gc"]),
            "enabled": payload["gc"]["enabled"],
            "generation_fields": sorted(payload["gc"]["by_generation"]),
        },
        "errors": payload["errors"],
        "imports": {
            name: Path(path).resolve().relative_to(_REPO.resolve()).as_posix()
            for name, path in payload["imports"].items()
        },
    }


def _run(tool, output: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repository_state = iter(("golden-subject", ""))
    monkeypatch.setattr(tool, "_git", lambda *_args: next(repository_state))
    tool.main(
        [
            "--subject-root",
            str(_REPO),
            "--expected-head",
            "golden-subject",
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
        ]
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_no_flag_full_measurement_matches_declared_fold_receipt(tmp_path, monkeypatch):
    baseline = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    candidate = _run(_tool(), tmp_path / "candidate.json", monkeypatch)

    assert _deterministic_projection(candidate) == baseline


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


def test_profile_corpus_falls_back_to_corpora_root(tmp_path, monkeypatch):
    tool = _tool()
    seen = {}
    monkeypatch.setenv("FREND_CORPORA", str(tmp_path / "corpora"))
    monkeypatch.setattr(
        tool, "_profile_run", lambda args: seen.update(corpus_dir=args.corpus_dir) or 0
    )

    assert (
        tool.main(
            [
                "--profile",
                "google-tn",
                "--subject-root",
                str(_REPO),
                "--expected-head",
                "irrelevant",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
        == 0
    )
    assert seen["corpus_dir"] == (
        tmp_path / "corpora" / "google" / "tn-en_with_types" / "en_with_types"
    )


def test_profile_corpus_without_option_or_environment_refuses(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FREND_CORPORA", raising=False)

    with pytest.raises(SystemExit):
        _tool().main(
            [
                "--profile",
                "google-tn",
                "--subject-root",
                str(_REPO),
                "--expected-head",
                "irrelevant",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
    assert "--corpus-dir is required when FREND_CORPORA is not set" in capsys.readouterr().err
