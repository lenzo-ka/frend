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


def test_vocabulary_refuses_a_non_training_shard():
    strata = _tool("seen_strata")
    from corpus_inputs import VerifiedInput

    held_out = VerifiedInput(
        strata.SOURCE_ID,
        "output-00090-of-00100",
        "0" * 64,
        "shippable-share-alike",
    )
    with pytest.raises(ValueError, match="training shards 00-89"):
        strata.require_training_inputs([held_out])


def test_loader_is_case_preserving_and_receipt_bound(tmp_path):
    strata = _tool("seen_strata")
    corpus_inputs = _tool("corpus_inputs")
    path = tmp_path / "written.vocab"
    body = b"Seen\nseen\n"
    path.write_bytes(body)
    pins = corpus_inputs._entry(strata.SOURCE_ID)["shards"]
    receipt = {
        "schema_version": strata.SCHEMA_VERSION,
        "kind": "google-tn-written-form-vocabulary",
        "source_id": strata.SOURCE_ID,
        "locale": strata.LOCALE,
        "case_preserved": True,
        "source_shards": [
            {"relative_path": name, "sha256": pins[name]} for name in sorted(strata.TRAINING_SHARDS)
        ],
        "vocabulary_size": 2,
        "artifact_sha256": hashlib.sha256(body).hexdigest(),
    }
    strata.metadata_path(path).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    match = strata.load_vocabulary(path, ["Seen", "SEEN", "seen"])
    assert match.seen == frozenset({"Seen", "seen"})

    path.write_bytes(body + b"stale\n")
    with pytest.raises(ValueError, match="does not match its receipt"):
        strata.load_vocabulary(path, ["Seen"])
