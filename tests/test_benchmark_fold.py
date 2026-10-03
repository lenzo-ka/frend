"""The fold benchmark consumes fixed detections without running a detector."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _tool():
    path = Path(__file__).parents[1] / "tools" / "benchmark_fold.py"
    spec = importlib.util.spec_from_file_location("benchmark_fold", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_fixture_times_every_fold_mode(tmp_path):
    data = tmp_path / "fold-bench"
    data.mkdir()
    text = {"id": "short-en_US-000", "bucket": "short", "locale": "en_US", "text": "1/2"}
    detection = {
        "id": text["id"],
        "text": "1",
        "start": 0,
        "end": 1,
        "type": "number:decimal",
        "value": {"kind": "number", "decimal": "1", "currency": None},
        "captures": [],
        "spec": None,
    }
    (data / "texts.jsonl").write_text(json.dumps(text) + "\n", encoding="utf-8")
    (data / "short.en_US.plain.jsonl").write_text(
        json.dumps(detection) + "\n", encoding="utf-8"
    )
    output = data / "results" / "fixture.json"

    assert _tool().main(["--data-dir", str(data), "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["case_count"] == 1
    row = payload["rows"][0]
    assert set(row["timings_ns"]) == {
        "keep_all",
        "lattice_build",
        "ranked_path_k64",
        "resolve_k1",
        "resolve_k8",
        "resolve_k64",
    }
    assert row["candidates"] == row["distinct_spans"] == row["starts"] == 1
    assert row["lattice_nodes"] > 0
    assert row["lattice_edges"] > 0
    assert row["top_level_size"] == 1
