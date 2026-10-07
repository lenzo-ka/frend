"""The fold benchmark consumes fixed detections without running a detector."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _tool():
    path = Path(__file__).parents[1] / "tools" / "benchmark_fold.py"
    spec = importlib.util.spec_from_file_location("benchmark_fold", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path):
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
    (data / "short.en_US.plain.jsonl").write_text(json.dumps(detection) + "\n", encoding="utf-8")
    stats = [
        {
            "file": "short.en_US.plain.jsonl",
            "bucket": "short",
            "locale": "en_US",
            "mode": "plain",
            "texts": 1,
            "rows": 1,
            "per_text": [{"id": text["id"], "candidates": 1}],
        }
    ]
    (data / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return data, text, detection


def test_small_fixture_times_every_fold_mode(tmp_path):
    data, _text, _detection = _fixture(tmp_path)
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


def test_missing_expected_detection_id_refuses(tmp_path):
    data, _text, _detection = _fixture(tmp_path)
    (data / "short.en_US.plain.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="missing detection rows"):
        _tool().load_cases(data)


def test_zero_detection_id_declared_by_stats_is_allowed(tmp_path):
    data, text, _detection = _fixture(tmp_path)
    (data / "short.en_US.plain.jsonl").write_text("", encoding="utf-8")
    stats = json.loads((data / "stats.json").read_text(encoding="utf-8"))
    stats[0]["rows"] = 0
    stats[0]["per_text"] = [{"id": text["id"], "candidates": 0}]
    (data / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

    cases = _tool().load_cases(data)

    assert len(cases) == 1
    assert cases[0]["detections"] == []


def test_unexpected_detection_id_refuses(tmp_path):
    data, _text, detection = _fixture(tmp_path)
    detection["id"] = "short-en_US-999"
    (data / "short.en_US.plain.jsonl").write_text(json.dumps(detection) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="unexpected ID"):
        _tool().load_cases(data)


def test_output_inside_repository_refuses():
    repository = Path(__file__).parents[1]

    with pytest.raises(SystemExit, match="must be outside"):
        _tool().main(
            [
                "--data-dir",
                "/does/not/matter",
                "--output",
                str(repository / "fold-result.json"),
            ]
        )


def test_data_dir_falls_back_to_processed_root(tmp_path, monkeypatch):
    tool = _tool()
    seen = {}
    monkeypatch.setenv("FREND_PROCESSED", str(tmp_path / "processed"))
    monkeypatch.setattr(tool, "_run", lambda args: seen.update(data_dir=args.data_dir) or 0)

    assert tool.main(["--output", str(tmp_path / "result.json")]) == 0
    assert seen["data_dir"] == tmp_path / "processed" / "frend" / "fold-bench"


def test_data_dir_without_option_or_environment_refuses(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FREND_PROCESSED", raising=False)

    with pytest.raises(SystemExit):
        _tool().main(["--output", str(tmp_path / "result.json")])
    assert "--data-dir is required when FREND_PROCESSED is not set" in capsys.readouterr().err
