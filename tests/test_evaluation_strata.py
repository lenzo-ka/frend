"""Exact aggregate fixtures for opt-in seen/unseen reporting."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _tool(name: str):
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluator_fixture_counts_tokens_classes_and_sentences(monkeypatch):
    evaluate = _tool("evaluate_google_tn")
    sentences = [[("PLAIN", "Seen", "self")], [("PLAIN", "Novel", "spoken")]]
    outcomes = [("PLAIN", True, True), ("PLAIN", False, False)]
    monkeypatch.setattr(evaluate, "_map", lambda *_args: outcomes)
    plain = evaluate._per_token(sentences, 1)
    assert "strata" not in plain
    rendered = json.dumps(plain, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(rendered).hexdigest() == (
        "f5cf979ce6fa1a5912908cfe88ee5d4cda9c730efefc98d6d6b53bd00476cd50"
    )
    assert all("strata" not in row for row in plain["classes"].values())
    report = evaluate._per_token(sentences, 1, strata_seen=frozenset({"Seen"}))
    assert report["strata"]["tokens"]["SEEN"]["tokens"] == 1
    assert report["strata"]["tokens"]["SEEN"]["first_choice_tokens"] == 1
    assert report["strata"]["tokens"]["UNSEEN"]["tokens"] == 1
    assert report["strata"]["tokens"]["UNSEEN"]["first_choice_tokens"] == 0
    assert report["strata"]["sentences"]["ALL_SEEN"]["sentences"] == 1
    assert report["strata"]["sentences"]["HAS_UNSEEN"]["sentences"] == 1
    assert report["classes"]["PLAIN"]["strata"]["SEEN"]["tokens"] == 1
    assert report["classes"]["PLAIN"]["strata"]["UNSEEN"]["tokens"] == 1


def test_evaluator_no_strata_json_and_text_are_byte_pinned(monkeypatch):
    evaluate = _tool("evaluate_google_tn")
    sentences = [[("PLAIN", "Seen", "self")], [("PLAIN", "Novel", "spoken")]]
    outcomes = [("PLAIN", True, True), ("PLAIN", False, False)]
    monkeypatch.setattr(evaluate, "_rows", lambda *_args: sentences)
    monkeypatch.setattr(evaluate, "_map", lambda *_args: outcomes)
    monkeypatch.setattr(evaluate, "_running_text_rows", lambda *_args: [])

    report = evaluate.evaluate({evaluate._TEST_FILE: object()}, 1)
    assert "strata" not in report
    blob = (
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\0" + evaluate._render(report)
    ).encode()
    assert hashlib.sha256(blob).hexdigest() == (
        "0183bf9a3bcf126c05ebac7524581c73eead11679ef24c262f90256099cba447"
    )


def test_triage_fixture_splits_every_reported_family(monkeypatch):
    triage = _tool("triage_misses")
    sentence = (("PLAIN", "Seen", "self"), ("PLAIN", "Novel", "spoken"))

    def score(row, _before, _after, _profile):
        if row[1] == "Seen":
            return True, True, False, None
        return False, False, False, triage.Classification("V", "other", "new entry")

    monkeypatch.setattr(triage, "_score_token", score)
    result = triage._score_sentence(("shard", 0, sentence, ("SEEN", "UNSEEN"), None))
    report = triage._aggregate([result], 1)
    assert report["strata"]["tokens"]["SEEN"]["accuracy"]["first_choice_tokens"] == 1
    unseen = report["strata"]["tokens"]["UNSEEN"]
    assert unseen["sample"] == {"tokens": 1, "sentences": 1}
    assert unseen["misses"]["tokens"] == 1
    assert unseen["by_class"]["V"]["tokens"] == 1
    assert unseen["v_by_family"]["other"]["tokens"] == 1
    assert unseen["v_by_fix_kind"]["new entry"]["tokens"] == 1
    assert report["strata"]["sentences"]["HAS_UNSEEN"]["sentences"] == 1

    plain_result = dict(result)
    plain_result.pop("strata")
    plain_result.pop("sentence_stratum")
    plain = triage._aggregate([plain_result], 1)
    assert "strata" not in plain
    rendered = json.dumps(plain, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(rendered).hexdigest() == (
        "1f2ad29e3d76b544fa309cfec45d31795c861f60ee61b82bffc6d1eb9947bc92"
    )
