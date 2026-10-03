"""Sentence-cluster bootstrap and paired-run fixtures."""

from __future__ import annotations

import importlib.util
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


def _payload(first: int, *, tokens: int = 1) -> dict:
    return {
        "schema_version": 1,
        "unit": "sentence",
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


def test_paired_identical_arms_have_exact_zero_interval():
    compare = _tool("compare_runs")
    payload = _payload(1)
    report = compare.compare(payload, payload, 100, 7)
    for row in report["metrics"].values():
        assert row["delta_ci95_pp"] == [0.0, 0.0]
        assert row["delta_ci_excludes_zero"] is False


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
    low, high = bootstrap.percentile_ratio_intervals(metrics, 1000, 20261002)["accuracy"]
    assert low == 0.0
    assert high == 1.0


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
    assert records == [
        {"id": "sample:0", "tokens": 1, "first": 1, "any": 1, "sentence_first": True},
        {"id": "sample:1", "tokens": 1, "first": 0, "any": 1, "sentence_first": False},
    ]
    assert "secret" not in repr(records)


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
    assert report["accuracy"]["any_reading_ci95"] == [1.0, 1.0]
    assert "by_class" not in report
    assert "by_class" not in report["strata"]["tokens"]["UNSEEN"]
