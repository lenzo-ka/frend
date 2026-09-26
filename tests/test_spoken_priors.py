"""Spoken-alternative measurements and their corpus builder."""

from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest

from frend.spoken_priors import (
    SpokenPriorTable,
    load_spoken_prior_table,
    measurement_sub_key,
    source_prior,
)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "tools" / "build_spoken_priors.py"
_FIXTURE = Path(__file__).parent / "data" / "spoken_priors"
_NORMALIZATION_PROVENANCE = (
    "Written tokens: remove trailing spaces and commas before recognition. "
    "Spoken forms and alternatives: split at characters whose Unicode general category "
    "starts with P and at whitespace recognized by str.split(), then lowercase each token "
    "independently."
)


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_spoken_priors", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shipped_table_shape_and_denominators():
    document = json.loads((_REPO / "frend" / "data" / "spoken_priors.json").read_text())
    table = load_spoken_prior_table()
    assert table.kinds == tuple(sorted(document["kinds"]))
    assert document["provenance"]["sub_key_rules"] == {
        "cardinal": None,
        "date": "the written order of month and day: month-first, day-first, or other",
        "decimal": None,
        "digit": None,
        "electronic": None,
        "fraction": "the decimal integer value of the reading's denominator capture",
        "measure": "the reading's ICU unit identifier; percent for a percent",
        "money": None,
        "ordinal": None,
        "symbol": "the code point of a symbol, the script of a letter",
        "time": None,
    }
    for kind, record in document["kinds"].items():
        assert record["matched"] + record["unmatched"] == record["total"]
        assert sum(record["source_matched"].values()) == record["matched"]
        assert sum(record["unmatched_by_reason"].values()) == record["unmatched"]
        assert set(record["unmatched_by_reason"]) == {
            "unrecognized",
            "unverbalized",
            "no_alternative_matched",
        }
        for item in record["top_unmatched"]:
            assert sum(item["reasons"].values()) == item["count"]
        if document["provenance"]["sub_key_rules"][kind] is None:
            assert record["sub_keys"] == {}
        else:
            assert sum(item["matched"] for item in record["sub_keys"].values()) == record["matched"]
            assert all(
                sum(item["source_matched"].values()) == item["matched"]
                for item in record["sub_keys"].values()
            )


def test_shipped_normalization_provenance_pins_contract():
    document = json.loads((_REPO / "frend" / "data" / "spoken_priors.json").read_text())
    assert document["provenance"]["normalization"] == _NORMALIZATION_PROVENANCE
    assert _load_builder().build_document(_FIXTURE)["provenance"]["normalization"] == (
        _NORMALIZATION_PROVENANCE
    )


def test_lookup_returns_only_measured_count_and_matched_share():
    table = SpokenPriorTable(
        {
            "cardinal": {
                "total": 5,
                "matched": 4,
                "unmatched": 1,
                "source_matched": {"icu:test": 4},
                "sub_keys": {},
                "unmatched_by_reason": {
                    "unrecognized": 0,
                    "unverbalized": 0,
                    "no_alternative_matched": 1,
                },
            }
        },
        {"source": "fixture", "sub_key_rules": {"cardinal": None}},
    )
    assert table.lookup("cardinal", "icu:test").count == 4
    assert table.lookup("cardinal", "icu:test").share == Decimal(1)
    assert table.lookup("cardinal", "absent") is None
    assert source_prior("absent", "absent") is None


def test_sub_key_lookup_blends_every_source_toward_the_kind_share():
    table = SpokenPriorTable(
        {
            "fraction": {
                "total": 10,
                "matched": 10,
                "unmatched": 0,
                "source_matched": {"icu:ordinal": 6, "lexical:half": 4},
                "sub_keys": {
                    "2": {"matched": 4, "source_matched": {"lexical:half": 4}},
                    "3": {"matched": 6, "source_matched": {"icu:ordinal": 6}},
                },
                "unmatched_by_reason": {
                    "unrecognized": 0,
                    "unverbalized": 0,
                    "no_alternative_matched": 0,
                },
            }
        },
        {"sub_key_rules": {"fraction": "denominator"}},
    )

    # (sub-key count + 5 * kind share) / (sub-key matched + 5)
    half = table.lookup("fraction", "lexical:half", "2")
    ordinal = table.lookup("fraction", "icu:ordinal", "2")
    assert (half.count, half.share) == (4, Decimal(6) / Decimal(9))
    assert (ordinal.count, ordinal.share) == (0, Decimal(3) / Decimal(9))
    assert table.lookup("fraction", "icu:ordinal", "missing").share == Decimal("0.6")
    assert table.lookup("fraction", "unseen:source", "2") is None


def test_fraction_sub_key_comes_from_the_shared_denominator_capture():
    class Capture:
        name = "denominator"
        value = 32

    assert measurement_sub_key("fraction", {"captures": (Capture(),)}) == "32"
    assert measurement_sub_key("decimal", {"captures": (Capture(),)}) is None


def test_builder_keeps_each_alternative_with_its_reading_sub_key(monkeypatch):
    build = _load_builder()

    class Capture:
        name = "denominator"

        def __init__(self, value):
            self.value = value

    class Detector:
        def detect(self, written):
            return (
                {
                    "start": 0,
                    "end": len(written),
                    "type": "number:fraction",
                    "value": "same-value",
                    "captures": (Capture(3),),
                },
                {
                    "start": 0,
                    "end": len(written),
                    "type": "number:fraction",
                    "value": "same-value",
                    "captures": (Capture(4),),
                },
            )

    class Lattice:
        def __init__(self, detection):
            self.edges = (type("Edge", (), {"kind": "reading", "detection": detection})(),)

    def fake_verbalize(edge, **kwargs):
        sub_key = measurement_sub_key("fraction", edge.detection)
        alternative = type(
            "Alternative", (), {"text": f"reading {sub_key}", "provenance": f"source:{sub_key}"}
        )()
        return type("Unit", (), {"verbalized": True, "alternatives": (alternative,)})()

    monkeypatch.setattr(
        "frend.lattice.resolve_lattice", lambda detections, source_text: Lattice(detections[0])
    )
    monkeypatch.setattr("frend.verbalize.verbalize_edge", fake_verbalize)

    recognized, alternatives = build._alternatives("x", "fraction", {"fraction": (Detector(),)})

    assert recognized
    assert alternatives == [
        ("reading 3", "source:3", "3"),
        ("reading 4", "source:4", "4"),
    ]


def test_loader_rejects_zero_denominator_source_and_inconsistent_totals():
    reasons = {
        "unrecognized": 0,
        "unverbalized": 0,
        "no_alternative_matched": 0,
    }
    with pytest.raises(ValueError, match="source cannot be present"):
        SpokenPriorTable(
            {
                "decimal": {
                    "total": 0,
                    "matched": 0,
                    "unmatched": 0,
                    "source_matched": {"icu:test": 0},
                    "sub_keys": {},
                    "unmatched_by_reason": reasons,
                }
            },
            {"sub_key_rules": {"decimal": None}},
        )
    with pytest.raises(ValueError, match=r"matched \+ unmatched must equal total"):
        SpokenPriorTable(
            {
                "decimal": {
                    "total": 2,
                    "matched": 0,
                    "unmatched": 1,
                    "source_matched": {},
                    "sub_keys": {},
                    "unmatched_by_reason": {
                        **reasons,
                        "unrecognized": 1,
                    },
                }
            },
            {"sub_key_rules": {"decimal": None}},
        )


def test_fixture_matching_normalization_sentinels_and_unmatched():
    build = _load_builder()
    document = build.build_document(_FIXTURE)
    cardinal = document["kinds"]["cardinal"]
    assert cardinal["total"] == 7
    assert cardinal["matched"] == 6
    assert cardinal["unmatched"] == 1
    assert cardinal["source_matched"]["icu-rbnf:%spellout-numbering"] == 6
    assert cardinal["top_unmatched"] == [
        {
            "spoken": "banana",
            "count": 1,
            "reasons": {
                "no_alternative_matched": 1,
                "unrecognized": 0,
                "unverbalized": 0,
            },
            "written_forms": {"13": 1},
        }
    ]
    decimal = document["kinds"]["decimal"]
    assert decimal["matched"] == 1
    assert decimal["unmatched"] == 0
    assert decimal["unmatched_by_reason"]["unverbalized"] == 0
    fraction = document["kinds"]["fraction"]
    assert fraction["matched"] == 1
    assert fraction["unmatched"] == 0
    assert fraction["sub_keys"] == {
        "2": {
            "matched": 1,
            "source_matched": {"icu-rbnf:%spellout-numbering+lexical:en_US": 1},
        }
    }
    for record in document["kinds"].values():
        assert sum(record["unmatched_by_reason"].values()) == record["unmatched"]
    assert document["provenance"]["skipped_sentinel_rows"] == 2


def test_fixture_check_rederives_byte_identical_document(tmp_path):
    build = _load_builder()
    output = tmp_path / "spoken_priors.json"
    assert build.main([str(_FIXTURE), "--out", str(output)]) == 0
    assert build.main([str(_FIXTURE), "--check", "--out", str(output)]) == 0
    output.write_text(output.read_text().replace("}", "} ", 1))
    assert build.main([str(_FIXTURE), "--check", "--out", str(output)]) == 1


def test_recognition_profile_includes_compacts_currency_names_and_xcd():
    build = _load_builder()
    detectors = build._detectors()

    assert [type(detector).__name__ for detector in detectors["cardinal"]] == [
        "FlexibleNumberDetector",
        "FlexibleCompactDetector",
        "FlexibleCompactDetector",
        "LetterNameDetector",
        "SingleLetterWordDetector",
        "WrittenFormsDetector",
    ]
    assert sum(
        type(detector).__name__ == "FlexibleCurrencyNameDetector" for detector in detectors["money"]
    ) == len(build._CURRENCIES)
    assert "XCD" in build._CURRENCIES
    document = build.build_document(_FIXTURE)
    assert document["provenance"]["recognition_profile"]["compact_styles"] == ["long", "short"]


@pytest.mark.parametrize(
    ("kind", "written"),
    [
        ("decimal", "5 million"),
        ("decimal", "2 million"),
        ("money", "$1.3 billion"),
        ("money", "EC$1.3 billion"),
        ("money", "2016 USD"),
    ],
)
def test_extended_profile_recognizes_baseline_unmatched_forms(kind, written):
    build = _load_builder()
    recognized, _alternatives = build._alternatives(written, kind, build._detectors())

    assert recognized
