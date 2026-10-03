"""Seen/unseen evaluation strata stay private, exact-case, and training-only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _tool(name: str):
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_receipt(strata, path: Path, receipt: dict[str, object]) -> None:
    strata.metadata_path(path).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _receipt(strata, body: bytes) -> dict[str, object]:
    pins = _tool("corpus_inputs")._entry(strata.SOURCE_ID)["shards"]
    shards = [
        {"relative_path": name, "sha256": pins[name]} for name in sorted(strata.TRAINING_SHARDS)
    ]
    return {
        "schema_version": strata.SCHEMA_VERSION,
        "extractor_id": strata.EXTRACTOR_ID,
        "kind": "google-tn-written-form-vocabulary",
        "source_id": strata.SOURCE_ID,
        "locale": strata.LOCALE,
        "case_preserved": True,
        "source_shards": shards,
        "vocabulary_size": len(body.splitlines()),
        "artifact_sha256": hashlib.sha256(body).hexdigest(),
    }


def test_build_vocabulary_and_cli_refuse_a_non_training_shard(tmp_path, monkeypatch):
    strata = _tool("seen_strata")
    build = _tool("build_seen_strata")
    VerifiedInput = sys.modules["corpus_inputs"].VerifiedInput

    held_out = VerifiedInput(
        strata.SOURCE_ID,
        "output-00090-of-00100",
        "0" * 64,
        "shippable-share-alike",
    )
    with pytest.raises(ValueError, match="training shards 00-89"):
        strata.build_vocabulary(tmp_path / "direct.vocab", [held_out])

    monkeypatch.setattr(build, "TRAINING_SHARDS", frozenset({held_out.relative_path}))
    monkeypatch.setattr(build, "verified_inputs", lambda *_args, **_kwargs: (held_out,))
    with pytest.raises(ValueError, match="training shards 00-89"):
        build.main(
            [
                "--corpus-dir",
                str(tmp_path),
                "--out",
                str(tmp_path / "cli.vocab"),
            ]
        )


def test_builder_and_evaluator_extract_identical_seen_forms(tmp_path, monkeypatch):
    strata = _tool("seen_strata")
    evaluate = _tool("evaluate_google_tn")
    VerifiedInput = sys.modules["corpus_inputs"].VerifiedInput
    name = "output-00000-of-00100"
    body = (
        "PLAIN\tMixed\t<self>\n"
        "PLAIN\tmixed\t<self>\n"
        "PLAIN\tStraße\tstrasse\n"
        "PLAIN\t\t<self>\n"
        "<eos>\t<eos>\n"
    ).encode()
    shard = tmp_path / name
    shard.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    item = VerifiedInput(
        strata.SOURCE_ID,
        name,
        digest,
        "shippable-share-alike",
        shard,
    )
    monkeypatch.setattr(strata, "TRAINING_SHARDS", frozenset({name}))
    monkeypatch.setattr(strata, "_entry", lambda _source: {"shards": {name: digest}})
    path = tmp_path / "written.vocab"
    strata.build_vocabulary(path, [item])

    rows = evaluate._rows(item, None)
    evaluator_forms = {row[1] for sentence in rows for row in sentence}
    match = strata.load_vocabulary(path, evaluator_forms)
    assert evaluator_forms == match.seen == frozenset({"", "Mixed", "mixed", "Straße"})


def test_loader_is_case_preserving_and_receipt_bound(tmp_path):
    strata = _tool("seen_strata")
    path = tmp_path / "written.vocab"
    body = b"Seen\nseen\n"
    path.write_bytes(body)
    _write_receipt(strata, path, _receipt(strata, body))

    match = strata.load_vocabulary(path, ["Seen", "SEEN", "seen"])
    assert match.seen == frozenset({"Seen", "seen"})

    path.write_bytes(body + b"stale\n")
    with pytest.raises(ValueError, match="does not match its receipt"):
        strata.load_vocabulary(path, ["Seen"])


def test_loader_refuses_extraction_rule_mismatch(tmp_path):
    strata = _tool("seen_strata")
    path = tmp_path / "extractor.vocab"
    body = b"Seen\nseen\n"
    path.write_bytes(body)
    receipt = _receipt(strata, body)
    receipt["extractor_id"] = "obsolete-extractor"
    _write_receipt(strata, path, receipt)
    with pytest.raises(ValueError, match="wrong extraction rule"):
        strata.load_vocabulary(path, ["Seen"])


def test_loader_refuses_catalog_hash_mismatch(tmp_path):
    strata = _tool("seen_strata")
    path = tmp_path / "catalog.vocab"
    body = b"Seen\nseen\n"
    path.write_bytes(body)
    receipt = _receipt(strata, body)
    receipt["source_shards"][0]["sha256"] = "0" * 64
    _write_receipt(strata, path, receipt)
    with pytest.raises(ValueError, match="source digest mismatch"):
        strata.load_vocabulary(path, ["Seen"])


def test_loader_refuses_vocabulary_count_mismatch(tmp_path):
    strata = _tool("seen_strata")
    path = tmp_path / "count.vocab"
    body = b"Seen\nseen\n"
    path.write_bytes(body)
    receipt = _receipt(strata, body)
    receipt["vocabulary_size"] = 3
    _write_receipt(strata, path, receipt)
    with pytest.raises(ValueError, match="does not match its receipt"):
        strata.load_vocabulary(path, ["Seen"])
