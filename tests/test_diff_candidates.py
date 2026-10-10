"""The isolated candidate differ compares complete public alternative payloads."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _module():
    name = "diff_candidates_for_tests"
    spec = importlib.util.spec_from_file_location(name, REPO / "tools/diff_candidates.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


diff = _module()


def test_date_signature_includes_provenance_prior_and_exact_weight():
    found, signature = diff._offer_signature("March 3, 2020", "en_US")
    assert found
    alternatives = [
        alternative
        for sentence in signature["sentences"]
        for edge in sentence["offers"]
        for alternative in edge["alternatives"]
    ]
    assert alternatives
    assert all(
        set(alternative) == {"text", "provenance", "prior_provenance", "weight"}
        for alternative in alternatives
    )


def test_runtime_eval_shard_set_is_exactly_90_through_94():
    assert diff.SHARDS == tuple(f"output-{number:05d}-of-00100" for number in range(90, 95))


def test_error_is_part_of_candidate_signature(monkeypatch):
    monkeypatch.setattr(diff, "resolve_choices", lambda *args, **kwargs: 1 / 0)
    found, signature = diff._offer_signature("March 3, 2020", "en_US")

    assert found
    assert signature == {
        "error": {
            "module": "builtins",
            "type": "ZeroDivisionError",
            "message": "division by zero",
        }
    }


def test_signature_observes_public_sentence_handling(monkeypatch):
    original = diff._sentence_ranges
    _found, before = diff._offer_signature("March 3, 2020. Next.", "en_US")
    monkeypatch.setattr(diff, "_sentence_ranges", lambda text, locale: [(0, len(text))])
    _found, after = diff._offer_signature("March 3, 2020. Next.", "en_US")

    assert original("March 3, 2020. Next.", "en_US") != [(0, 20)]
    assert before != after


def test_fraction_family_selects_fraction_and_percent_only():
    for text in ("3/7", "45%"):
        found, signature = diff._offer_signature(text, "en_US", "fraction")
        assert found
        assert signature["sentences"]
    found, _signature = diff._offer_signature("March 3, 2020", "en_US", "fraction")
    assert not found


def test_child_refuses_report_shard_99(tmp_path):
    with pytest.raises(ValueError, match="refusing non-development shard"):
        diff._child(
            tmp_path,
            tmp_path / "out.json",
            None,
            "fraction",
            "output-00099-of-00100",
        )
