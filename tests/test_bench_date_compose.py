"""The date composition benchmark measures the complete tiergraph path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _bench():
    name = "bench_date_compose_for_tests"
    spec = importlib.util.spec_from_file_location(name, REPO / "tools/bench_date_compose.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _bench()


def test_one_sample_covers_load_lower_recognize_generate_and_render():
    row = bench._one("pt_BR", 2)
    assert set(row["phases_ns"]) == {"load_lower", "recognize", "generate", "render"}
    assert all(value >= 0 for value in row["phases_ns"].values())
    assert row["exact_target_count"] == row["returned_count"] == 2
    assert row["truncated"] is False
    assert row["target_piece_count"] > 0


def test_latency_manifest_has_exactly_the_eleven_gate_locales():
    import json

    document = json.loads(bench.INPUTS.read_text(encoding="utf-8"))
    assert set(document["locales"]) == {*bench.GENERATED_DATE_LOCALES, "en_US"}


def test_unseen_cache_probe_runs_complete_generation(monkeypatch):
    calls = []

    def record(value, detection, locale):
        calls.append((value, detection, locale))
        return ()

    monkeypatch.setattr(bench, "generate_date_alternatives", record)
    generated = bench._exercise_unseen_dates(2)

    assert generated == 2 * len(bench.GENERATED_DATE_LOCALES)
    assert len(calls) == generated
    assert {locale for _value, _detection, locale in calls} == set(bench.GENERATED_DATE_LOCALES)
