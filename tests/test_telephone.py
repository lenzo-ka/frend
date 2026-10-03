"""Training-derived telephone spans and grouped readings."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from icukit.detectors import detect

from frend import resolve_lattice
from frend import telephone as telephone_module
from frend.spoken_priors import normalize_spoken
from frend.telephone import TelephoneDetector
from frend.verbalize import verbalize_edge
from frend.written_forms import WrittenFormsDetector


def _forms(text: str) -> list[str]:
    detections = list(detect(text, (WrittenFormsDetector("en_US"),)))
    telephone = [item for item in detections if item["type"] == "telephone:number"]
    assert [(item["start"], item["end"]) for item in telephone] == [(0, len(text))]
    lattice = resolve_lattice(telephone, source_text=text)
    edge = next(item for item in lattice.edges if item.kind == "reading")
    return [normalize_spoken(item.text) for item in verbalize_edge(edge).alternatives]


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("(650) 696-1060", "six five o sil six nine six sil one o six o"),
        ("212-868-4444", "two one two sil eight six eight sil four four four four"),
        (
            "1-888-237-2834",
            "one sil eight eight eight sil two three seven sil two eight three four",
        ),
        ("212 868 4404", "two one two sil eight six eight sil four four o four"),
        ("212-868 4404", "two one two sil eight six eight sil four four o four"),
        ("(212)868-4404", "two one two sil eight six eight sil four four o four"),
    ],
)
def test_phone_shape_covers_the_whole_token_and_uses_measured_groups(written, spoken):
    assert _forms(written)[0] == spoken


@pytest.mark.parametrize(
    "written",
    [
        "2020-2024",  # years/range
        "100-2000",  # a numeric range that resembles a local phone
        "3-2",  # score
        "978-0-7524-4250-1",  # ISBN-13
        "0-521-56136-1",  # ISBN-10
        "12345-6789",  # ZIP+4
        "123-45-6789",  # identifier
    ],
)
def test_non_phone_numeric_shapes_are_not_telephone(written):
    assert TelephoneDetector("en_US").detect(written) == []


def test_learned_label_context_vetoes_a_shipped_shape(monkeypatch):
    detector = TelephoneDetector("en_US")
    assert detector.detect("223-456-7881")
    assert detector.detect("ISBN 223-456-7881") == []
    assert detector.detect("ISBN 123-456-7881") == []

    monkeypatch.setattr(telephone_module, "_left_context", lambda _text, _start: None)
    assert detector.detect("ISBN 223-456-7881")


@pytest.mark.parametrize(
    "written",
    [
        "000-000-0000",
        "112-868-4444",  # area code must begin 2--9
        "212-168-4444",  # exchange must begin 2--9
        "211-868-4444",  # N11 area codes are excluded
        "212-911-4444",  # N11 exchanges are excluded
    ],
)
def test_invalid_nanp_numbers_do_not_use_a_shipped_shape(written, monkeypatch):
    detector = TelephoneDetector("en_US")
    assert detector.detect(written) == []

    monkeypatch.setattr(telephone_module, "_valid_nanp", lambda _groups: True)
    assert detector.detect(written)


@pytest.mark.parametrize(
    "text",
    [
        "+1-888-237-2834",
        "$212-868-4444",
        "€212-868-4444",
        "±212-868-4444",
        "x212-868-4444",
    ],
)
def test_prefixed_number_does_not_use_a_shipped_shape(text, monkeypatch):
    detector = TelephoneDetector("en_US")
    assert detector.detect(text) == []

    monkeypatch.setattr(telephone_module, "_valid_left_boundary", lambda _text, _start: True)
    assert detector.detect(text)


def test_number_embedded_directly_after_a_digit_is_not_telephone():
    assert TelephoneDetector("en_US").detect("9(212) 868-4444") == []


@pytest.mark.parametrize("written", ["٢١٢-٨٦٨-٤٤٤٤", "212–868–4444"])
def test_untrained_unicode_digits_and_separators_are_not_telephone(written):
    assert TelephoneDetector("en_US").detect(written) == []


def test_locale_without_a_measured_table_has_no_telephone_opinion():
    assert TelephoneDetector("ru_RU").detect("212-868-4444") == []


def test_shipped_left_context_keys_use_only_the_fixed_shape_alphabet():
    document = telephone_module.measured_table("telephone_priors", "en_US")
    assert document is not None
    assert document["left_contexts"]
    assert all(
        telephone_module._is_left_context_shape(context) for context in document["left_contexts"]
    )
    assert not telephone_module._is_left_context_shape("isbn")


def test_builder_counts_shapes_readings_and_left_context_from_corpus_rows(tmp_path):
    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "build_telephone_priors", tools / "build_telephone_priors.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    shard = tmp_path / "output-00000-of-00001"
    shard.write_text(
        "LETTERS\tISBN\ti s b n\n"
        "TELEPHONE\t650-696-1060\tsix five o sil six nine six sil one o six o\n"
        "LETTERS\tISBN\ti s b n\n"
        "TELEPHONE\t570-966-1031\tfive seven o sil nine six six sil one o three one\n"
        "LETTERS\tISBN\ti s b n\n"
        "TELEPHONE\t205-556-1010\ttwo o five sil five five six sil one o one o\n"
        "CARDINAL\t105-222-3030\tone billion fifty two million two hundred twenty "
        "twenty three thousand three hundred thirty three\n"
        "TELEPHONE\t978-0-7524-4250-1\tnine seven eight sil o sil seven five two four sil "
        "four two five o sil one\n",
        encoding="utf-8",
    )
    document = builder.build_document(tmp_path)
    assert document["left_contexts"] == {
        "upper:4": {"classes": {"LETTERS": 3}, "prediction": "LETTERS"}
    }
    assert document["shapes"]["N3-N3-N4"] == {
        "classes": {"CARDINAL": 1, "TELEPHONE": 3},
        "patterns": {
            "z:n:z": {
                "readings": [{"modes": ["digits-o", "digits", "digits-o"], "count": 3}],
                "unclassified_telephone": 0,
            }
        },
    }
    assert "N3-N1-N4-N4-N1" not in document["shapes"]
