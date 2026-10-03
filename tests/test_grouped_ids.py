"""Checksum ISBN and training-supported grouped-ID readings."""

from __future__ import annotations

import importlib.util
import re
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from icukit.detectors import detect

from frend import grouped_ids as grouped_module
from frend import resolve_lattice
from frend.grouped_ids import GroupedDigitsDetector
from frend.spoken_priors import normalize_spoken
from frend.telephone import TelephoneDetector
from frend.verbalize import verbalize_edge
from frend.written_forms import WrittenFormsDetector


def _forms(text: str) -> tuple[str, list[str]]:
    detections = list(detect(text, (WrittenFormsDetector("en_US"),)))
    grouped = [item for item in detections if item["type"] == "identifier:grouped-digits"]
    assert len(grouped) == 1
    item = grouped[0]
    lattice = resolve_lattice(grouped, source_text=text)
    edge = next(edge for edge in lattice.edges if edge.kind == "reading")
    forms = [normalize_spoken(alt.text) for alt in verbalize_edge(edge).alternatives]
    return item["text"], forms


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("0-306-40615-2", "o sil three o six sil four o six one five sil two"),
        (
            "978-0-7524-4250-1",
            "nine seven eight sil o sil seven five two four sil four two five o sil one",
        ),
        ("0-8044-2957-X", "o sil eight o four four sil two nine five seven sil x"),
        ("1-85445-157", "one sil eight five four four five sil one five seven"),
    ],
)
def test_grouped_ids_span_the_token_and_read_each_character(written, spoken):
    surface, forms = _forms(written)
    assert surface == written
    assert forms[0] == spoken


def test_isbn_label_enables_the_checksum_case_telephone_rejects():
    text = "ISBN 123-456-7881"
    assert TelephoneDetector("en_US").detect(text) == []
    detection = GroupedDigitsDetector("en_US").detect(text)
    assert [(item["text"], item["start"], item["end"]) for item in detection] == [
        ("123-456-7881", 5, len(text))
    ]


@pytest.mark.parametrize(
    "written",
    [
        "2020-2024",  # years/range; its N4-N4 shape is shipped
        "4-6-2",  # score; its N1-N1-N1 shape is shipped
        "2024-01-01",  # date
        "$1-85445-157",  # money prefix on a shipped positive shape
        "$-1-85445-157",
        "$ 1-85445-157",
        "- 1-85445-157",
        "+ 1-85445-157",
        "1-85445-157 $",
        "1-85445-157 %",
        "12:34",  # time
        "12345-6789",  # ZIP+4; its N5-N4 shape is shipped
        "1.2.3",  # version
        "212-868-4444",  # #68 telephone
        "5-3",  # arithmetic
        "0-306-40615-3",  # invalid ISBN-10, on a dominant ISBN shape
        "978-0-7524-4250-2",  # invalid ISBN-13, on a dominant ISBN shape
        "0-306-40615-2.1",
        "0-306-40615-2-1",
        "0-306-40615-2 1",
        "0-306-40615-2a",
        "0-306-40615-21",
        "0–306-40615-2",
        "0-306-٤0615-2",
        "1-85445-157٢",
    ],
)
def test_ambiguous_numeric_forms_are_not_grouped_ids(written):
    assert GroupedDigitsDetector("en_US").detect(written) == []


def test_generic_width_guard_blocks_shipped_short_and_two_group_shapes(monkeypatch):
    detector = GroupedDigitsDetector("en_US")
    for written in ("4-6-2", "2020-2024", "12345-6789"):
        shape = grouped_module.grouped_id_shape(written)
        assert shape in grouped_module._prior("en_US")
        assert detector.detect(written) == []
    monkeypatch.setattr(grouped_module, "_generic_id_shape", lambda _groups: True)
    for written in ("4-6-2", "2020-2024", "12345-6789"):
        assert detector.detect(written)


def test_checksum_guard_rejects_an_invalid_isbn_on_a_shipped_shape(monkeypatch):
    written = "0-306-40615-3"
    detector = GroupedDigitsDetector("en_US")
    shape = grouped_module.grouped_id_shape(written)
    assert shape in grouped_module._prior("en_US")
    assert detector.detect(written) == []
    monkeypatch.setattr(grouped_module, "is_valid_isbn", lambda _text: True)
    assert detector.detect(written)


@pytest.mark.parametrize(
    "text",
    ["$1-85445-157", "$-1-85445-157", "$ 1-85445-157", "- 1-85445-157"],
)
def test_left_affix_boundary_blocks_money_and_signs(text, monkeypatch):
    detector = GroupedDigitsDetector("en_US")
    assert detector.detect(text) == []
    monkeypatch.setattr(grouped_module, "_valid_left_boundary", lambda _text, _start: True)
    assert detector.detect(text)


@pytest.mark.parametrize("text", ["1-85445-157 $", "1-85445-157 %"])
def test_right_affix_boundary_blocks_money_and_percent(text, monkeypatch):
    detector = GroupedDigitsDetector("en_US")
    assert detector.detect(text) == []
    monkeypatch.setattr(grouped_module, "_valid_right_boundary", lambda _text, _end: True)
    assert detector.detect(text)


@pytest.mark.parametrize(
    "text",
    [
        "0-306-40615-2.1",
        "0-306-40615-2-1",
        "0-306-40615-2 1",
        "0-306-40615-2a",
        "0-306-40615-21",
    ],
)
def test_right_token_boundary_blocks_partial_matches(text, monkeypatch):
    detector = GroupedDigitsDetector("en_US")
    assert detector.detect(text) == []
    monkeypatch.setattr(grouped_module, "_valid_right_boundary", lambda _text, _end: True)
    assert detector.detect(text)


def test_unicode_dash_blocks_an_ascii_suffix_match(monkeypatch):
    detector = GroupedDigitsDetector("en_US")
    text = "0–306-40615-2"
    assert grouped_module.grouped_id_shape("306-40615-2") in grouped_module._prior("en_US")
    assert detector.detect(text) == []
    monkeypatch.setattr(grouped_module, "_valid_left_boundary", lambda _text, _start: True)
    assert detector.detect(text)


def test_non_ascii_digit_blocks_an_adjacent_ascii_match(monkeypatch):
    detector = GroupedDigitsDetector("en_US")
    text = "1-85445-157٢"
    assert detector.detect(text) == []
    monkeypatch.setattr(grouped_module, "_valid_right_boundary", lambda _text, _end: True)
    assert detector.detect(text)


def test_measured_shape_gate_is_required_for_a_generic_id(monkeypatch):
    written = "1-85445-157"
    detector = GroupedDigitsDetector("en_US")
    assert detector.detect(written)
    monkeypatch.setattr(grouped_module, "_prior", lambda _locale: frozenset())
    assert detector.detect(written) == []


def test_existing_telephone_keeps_its_detector_and_reading():
    written = "212-868-4444"
    detections = list(detect(written, (WrittenFormsDetector("en_US"),)))
    types = [item["type"] for item in detections]
    assert types.count("telephone:number") == 1
    assert "identifier:grouped-digits" not in types


def test_same_span_telephone_precedence_blocks_a_checksum_collision(monkeypatch):
    written = "223-456-7890"  # valid NANP and, coincidentally, valid ISBN-10 checksum
    detector = WrittenFormsDetector("en_US")
    types = [item["type"] for item in detector.detect(written)]
    assert types.count("telephone:number") == 1
    assert "identifier:grouped-digits" not in types

    monkeypatch.setattr(detector.telephone, "detect", lambda _text: [])
    assert [item["type"] for item in detector.detect(written)] == ["identifier:grouped-digits"]


def test_locale_without_measured_counts_has_no_generic_id_opinion():
    assert GroupedDigitsDetector("ru_RU").detect("1-85445-157") == []


def _assert_grouped_prior_is_aggregate(document):
    assert set(document) == {"schema_version", "locale", "provenance", "shapes"}
    assert type(document["schema_version"]) is int and document["schema_version"] == 1
    assert document["locale"] == "en"
    provenance = document["provenance"]
    assert set(provenance) == {"builder", "corpus", "locale", "shards", "selection", "reading"}
    assert provenance["builder"] == "tools/build_grouped_id_priors.py"
    assert provenance["corpus"] == "google-tn:en_with_types"
    assert provenance["locale"] == "en"
    assert provenance["selection"] == (
        "exact ASCII digit-group shape with at least 3 grouped-reading rows "
        "and a strict grouped-reading majority over all corpus rows"
    )
    assert provenance["reading"] == (
        "one ICU digit name per written digit, locale lexical o for zero, "
        "and literal sil between written groups"
    )
    assert type(provenance["shards"]) is list and provenance["shards"]
    shard_pattern = re.compile(r"output-[0-9]{5}-of-[0-9]{5}\Z")
    assert all(
        type(shard) is str and shard_pattern.fullmatch(shard) for shard in provenance["shards"]
    )
    assert type(document["shapes"]) is dict and document["shapes"]
    shape_pattern = re.compile(r"N[1-9][0-9]*(?:[- ]N[1-9][0-9]*)+(?:[- ]X)?\Z")
    class_pattern = re.compile(r"[A-Z][A-Z_]*\Z")
    for shape, row in document["shapes"].items():
        assert shape_pattern.fullmatch(shape)
        assert set(row) == {"classes", "readings"}
        assert row["classes"]
        assert all(class_pattern.fullmatch(key) for key in row["classes"])
        assert set(row["readings"]) == {"grouped", "other"}
        for counts in row.values():
            assert all(type(count) is int for count in counts.values())


def test_shipped_table_contains_only_allowlisted_aggregate_fields():
    document = grouped_module.measured_table("grouped_id_priors", "en_US")
    assert document is not None
    _assert_grouped_prior_is_aggregate(document)


@pytest.mark.parametrize("path", [("excerpt",), ("provenance", "excerpt")])
def test_privacy_guard_rejects_unallowlisted_string_fields(path):
    document = deepcopy(grouped_module.measured_table("grouped_id_priors", "en_US"))
    assert document is not None
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "raw corpus surface"
    with pytest.raises(AssertionError):
        _assert_grouped_prior_is_aggregate(document)


def test_builder_counts_classes_and_grouped_readings_by_shape(tmp_path):
    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "build_grouped_id_priors", tools / "build_grouped_id_priors.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    shard = tmp_path / "output-00000-of-00001"
    grouped = "one sil eight five four four five sil one five seven\n"
    shard.write_text(
        "TELEPHONE\t1-85445-157\t"
        + grouped
        + "TELEPHONE\t1-85445-157\t"
        + grouped
        + "TELEPHONE\t1-85445-157\t"
        + grouped
        + "CARDINAL\t1-85445-157\tone hundred million\n"
        + "TELEPHONE\t4-6-2\tfour sil six sil two\n"
        + "TELEPHONE\t2-8-2\ttwo sil eight sil two\n",
        encoding="utf-8",
    )
    document = builder.build_document(tmp_path)
    assert document["shapes"]["N1-N5-N3"] == {
        "classes": {"CARDINAL": 1, "TELEPHONE": 3},
        "readings": {"grouped": 3, "other": 1},
    }
    assert "N1-N1-N1" not in document["shapes"]
