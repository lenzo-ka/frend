"""Sentence-cluster bootstrap and paired-run fixtures."""

from __future__ import annotations

import importlib.util
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


def _payload(first: int, *, tokens: int = 1, fingerprint: str = "sample-a") -> dict:
    return {
        "schema_version": 1,
        "unit": "sentence",
        "sample_fingerprint": fingerprint,
        "records": [
            {
                "id": "shard:0",
                "tokens": tokens,
                "first": first,
                "any": first,
                "sentence_first": first == tokens,
            }
        ],
    }


def _two_sentence_payload() -> dict:
    return {
        "schema_version": 1,
        "unit": "sentence",
        "sample_fingerprint": "sample-a",
        "records": [
            {
                "id": f"shard:{index}",
                "tokens": 1,
                "first": index,
                "any": index,
                "sentence_first": bool(index),
            }
            for index in (0, 1)
        ],
    }


def test_paired_two_sentence_identical_arms_have_exact_zero_interval():
    compare = _tool("compare_runs")
    a_payload = _two_sentence_payload()
    b_payload = {**a_payload, "records": list(reversed(a_payload["records"]))}
    report = compare.compare(a_payload, b_payload, 1000, 7)
    for row in report["metrics"].values():
        assert row["delta_ci95_pp"] == [0.0, 0.0]
        assert row["delta_ci95_defined_replicates"] == 1000
        assert row["delta_ci_excludes_zero"] is False


def test_compare_refuses_mismatched_sample_fingerprints():
    compare = _tool("compare_runs")
    with pytest.raises(ValueError, match="same sample fingerprint"):
        compare.compare(
            _payload(0, fingerprint="sample-a"),
            _payload(1, fingerprint="sample-b"),
            10,
            7,
        )


def test_paired_fixed_one_sentence_gain_has_exact_interval():
    compare = _tool("compare_runs")
    report = compare.compare(_payload(0), _payload(1), 100, 7)
    tokens = report["metrics"]["overall:first_choice_tokens"]
    sentences = report["metrics"]["overall:first_choice_sentences"]
    assert tokens["delta_ci95_pp"] == [100.0, 100.0]
    assert sentences["delta_ci95_pp"] == [100.0, 100.0]
    assert tokens["delta_ci_excludes_zero"] is True


def test_interval_seed_is_deterministic():
    bootstrap = _tool("bootstrap_intervals")
    metrics = {"accuracy": ([1, 0, 1], [1, 2, 3])}
    first = bootstrap.percentile_ratio_intervals(metrics, 137, 20261002)
    second = bootstrap.percentile_ratio_intervals(metrics, 137, 20261002)
    assert first == second


def test_resampling_clusters_on_sentences_not_tokens():
    bootstrap = _tool("bootstrap_intervals")
    metrics = {"accuracy": ([100, 0], [100, 1])}
    result = bootstrap.percentile_ratio_intervals(metrics, 1000, 20261002)["accuracy"]
    low, high = result["ci95"]
    assert low == 0.0
    assert high == 1.0
    assert result["defined_replicates"] == 1000


def test_ratio_intervals_drop_undefined_draws_and_apply_reliability_floor():
    bootstrap = _tool("bootstrap_intervals")
    mostly_defined = bootstrap.percentile_ratio_intervals(
        {"rare": ([1] * 4 + [0] * 96, [1] * 4 + [0] * 96)}, 1000, 7
    )["rare"]
    assert mostly_defined["ci95"] == [1.0, 1.0]
    assert 950 <= mostly_defined["defined_replicates"] < 1000

    unreliable = bootstrap.percentile_ratio_intervals(
        {"rare": ([1] + [0] * 99, [1] + [0] * 99)}, 1000, 7
    )["rare"]
    assert unreliable["ci95"] is None
    assert unreliable["defined_replicates"] < 950


def test_paired_intervals_drop_undefined_draws_and_apply_reliability_floor():
    bootstrap = _tool("bootstrap_intervals")
    vector = [1] * 4 + [0] * 96
    result = bootstrap.paired_delta_intervals({"rare": (vector, vector, vector, vector)}, 1000, 7)[
        "rare"
    ]
    assert result["ci95"] == [0.0, 0.0]
    assert 950 <= result["defined_replicates"] < 1000

    sparse = [1] + [0] * 99
    unreliable = bootstrap.paired_delta_intervals(
        {"rare": (sparse, sparse, sparse, sparse)}, 1000, 7
    )["rare"]
    assert unreliable["ci95"] is None
    assert unreliable["defined_replicates"] < 950


def test_sentence_payloads_carry_sample_identity():
    evaluate = _tool("evaluate_google_tn")
    triage = _tool("triage_misses")
    records = _two_sentence_payload()["records"]

    first = evaluate._per_sentence_payload(records, None, "typographic", "corpus-a")
    second = evaluate._per_sentence_payload(records, "google-tn", "typographic", "corpus-a")
    assert first["fold"] == "typographic"
    assert first["sample_fingerprint"] == second["sample_fingerprint"]
    assert (
        first["sample_fingerprint"]
        != evaluate._per_sentence_payload(records, None, "typographic", "corpus-b")[
            "sample_fingerprint"
        ]
    )
    assert (
        first["sample_fingerprint"]
        != evaluate._per_sentence_payload(records[:1], None, "typographic", "corpus-a")[
            "sample_fingerprint"
        ]
    )
    assert (
        triage._per_sentence_payload(records, None, "existing-fingerprint")["sample_fingerprint"]
        == "existing-fingerprint"
    )


def test_evaluator_adds_intervals_and_text_free_sentence_records(monkeypatch):
    evaluate = _tool("evaluate_google_tn")
    sentences = [[("PLAIN", "secret", "spoken")], [("PLAIN", "other", "spoken")]]
    monkeypatch.setattr(
        evaluate,
        "_map",
        lambda *_args: [("PLAIN", True, True), ("PLAIN", False, True)],
    )
    records = []
    report = evaluate._per_token(sentences, 1, intervals=100, record_sink=records)
    assert report["intervals"]["replicates"] == 100
    assert report["intervals"]["first_choice_accuracy"] == [0.0, 1.0]
    assert report["intervals"]["defined_replicates"]["first_choice_accuracy"] == 100
    assert records == [
        {
            "id": "sample:0",
            "fold": "typographic",
            "tokens": 1,
            "first": 1,
            "any": 1,
            "sentence_first": True,
        },
        {
            "id": "sample:1",
            "fold": "typographic",
            "tokens": 1,
            "first": 0,
            "any": 1,
            "sentence_first": False,
        },
    ]
    assert "secret" not in repr(records)

    without_fold = []
    evaluate._per_token(sentences, 1, fold=None, record_sink=without_fold)
    assert {record["fold"] for record in without_fold} == {None}


def test_triage_reports_each_class_interval_in_percentage_points(monkeypatch):
    triage = _tool("triage_misses")
    sentence = (("DATE", "1/2/03", "spoken"),)
    monkeypatch.setattr(
        triage,
        "_score_token",
        lambda *_args: (False, False, False, triage.Classification("D")),
    )
    result = triage._score_sentence(("shard", 0, sentence, None, None))
    report = triage._aggregate([result], 1, intervals=100)
    assert report["by_class"]["D"]["share_sampled_tokens_pp_ci95"] == [100.0, 100.0]
    assert report["by_class"]["D"]["share_sampled_tokens_pp_ci95_defined_replicates"] == 100
    assert report["by_class"]["S"]["share_sampled_tokens_pp_ci95"] == [0.0, 0.0]


def test_accuracy_only_skips_classification_but_keeps_intervals(monkeypatch):
    triage = _tool("triage_misses")
    calls = []

    def score(_row, _before, _after, _profile, classify_misses=True):
        calls.append(classify_misses)
        return False, True, False, None

    monkeypatch.setattr(triage, "_score_token", score)
    sentence = (("DATE", "1/2/03", "spoken"),)
    result = triage._score_sentence(("shard", 0, sentence, ("UNSEEN",), None, False))
    report = triage._aggregate([result], 1, intervals=100, classify_misses=False)
    assert calls == [False]
    assert report["accuracy"]["first_choice_ci95"] == [0.0, 0.0]
    assert report["accuracy"]["first_choice_ci95_defined_replicates"] == 100
    assert report["accuracy"]["any_reading_ci95"] == [1.0, 1.0]
    assert "by_class" not in report
    assert "by_class" not in report["strata"]["tokens"]["UNSEEN"]
