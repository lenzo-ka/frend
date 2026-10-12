"""The isolated candidate differ compares complete public alternative payloads."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from icukit.detectors import DateTimeValue

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


def _assert_tiled_offsets(signature, source):
    output_spans = [unit["output_span"] for unit in signature["units"]]
    source_spans = [unit["source_span"] for unit in signature["units"]]
    assert output_spans[0][0] == source_spans[0][0] == 0
    assert all(
        left[1] == right[0] for left, right in zip(output_spans, output_spans[1:], strict=False)
    )
    assert all(
        left[1] == right[0] for left, right in zip(source_spans, source_spans[1:], strict=False)
    )
    assert output_spans[-1][1] == len(signature["text"])
    assert source_spans[-1][1] == len(source)


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


@pytest.mark.parametrize(
    "text",
    [
        "Meet me at 5pm, please.",
        "At 10:05 we leave.",
        "Call at 5pm, then again at 6:30.",
        "March 3, 2020 at 2:07 PM.",
        "The window is 18:00 UTC.",
    ],
)
def test_time_family_sentence_signature_keeps_prose_offsets_and_alternatives(text):
    found, signature = diff._offer_signature(text, "en_US", "time")

    assert found
    _assert_tiled_offsets(signature, text)
    assert signature["sentences"]
    assert any(
        str(edge["type"] or "").startswith(("time:", "date:"))
        for sentence in signature["sentences"]
        for edge in sentence["offers"]
    )


def test_time_signature_changes_when_only_a_typed_second_field_changes(monkeypatch):
    text = "03:04:05"
    before = diff._graph_signature(text, "en_US")
    original_detect = diff.detect

    def changed_seconds(source, detectors):
        for detection in original_detect(source, detectors):
            if str(detection.get("type", "")).startswith("time:"):
                detection = dict(detection)
                value = detection["value"]
                detection["value"] = DateTimeValue(
                    tuple((name, 6 if name == "s" else item) for name, item in value.fields),
                    value.calendar,
                )
            yield detection

    monkeypatch.setattr(diff, "detect", changed_seconds)
    after = diff._graph_signature(text, "en_US")

    assert before != after
    time_edges = [edge for edge in after if str(edge["type"] or "").startswith("time:")]
    assert time_edges[0]["value"]["fields"]["fields"][-1] == ["s", 6]


def test_time_prose_offset_guard_rejects_a_corrupt_unit_boundary():
    text = "Meet me at 5pm, please."
    found, signature = diff._offer_signature(text, "en_US", "time")
    assert found
    signature["units"][1]["output_span"][0] += 1

    with pytest.raises(AssertionError):
        _assert_tiled_offsets(signature, text)


def test_sentence_reader_joins_tokens_with_deterministic_spacing(tmp_path):
    shard = diff.SHARDS[0]
    (tmp_path / shard).write_text(
        "PLAIN\tMeet\t<self>\nTIME\t5pm\tfive p m\n<eos>\t<eos>\n",
        encoding="utf-8",
    )

    assert list(diff._sentences(tmp_path, None, (shard,))) == [(shard, 1, "Meet 5pm")]


def test_child_refuses_report_shard_99(tmp_path):
    with pytest.raises(ValueError, match="refusing non-development shard"):
        diff._child(
            tmp_path,
            tmp_path / "out.json",
            None,
            "fraction",
            "output-00099-of-00100",
        )
