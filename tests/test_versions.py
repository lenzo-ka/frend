"""Dotted versions with an explicit ``v`` prefix."""

from __future__ import annotations

import pytest
from icukit.detectors import detect

import frend
from frend.spoken_priors import normalize_spoken
from frend.verbalize import _spoken_version, _version_separator
from frend.versions import VersionDetector, VersionValue


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("v2.3.1", "v two point three point one"),
        ("V10.0.2", "v ten point zero point two"),
    ],
)
def test_prefixed_dotted_version_is_one_span_with_spoken_separators(written, spoken):
    detections = VersionDetector("en_US").detect(f"Upgrade to {written} now.")

    assert [(item["text"], item["start"], item["end"]) for item in detections] == [
        (written, 11, 11 + len(written))
    ]
    normalized = frend.normalize(written, offsets=True)
    assert normalize_spoken(normalized.text) == spoken
    assert [(unit.source_span, unit.reader) for unit in normalized.units] == [
        ((0, len(written)), "version:dotted")
    ]


def test_language_version_remains_an_ordinary_decimal():
    assert VersionDetector("en_US").detect("Python 3.12") == []
    assert normalize_spoken(frend.normalize("Python 3.12")) == "python three point one two"


def test_version_components_preserve_written_leading_zeros():
    assert normalize_spoken(frend.normalize("v1.02.003")) == (
        "v one point zero two point zero zero three"
    )


def test_many_ambiguous_components_do_not_multiply_spoken_alternatives():
    value = VersionValue("v", ("105",) * 32)

    alternatives = _spoken_version(value, "en_US")

    assert len(alternatives) == 1
    assert alternatives[0].text.count("point") == 31


def test_version_separator_names_the_written_period_in_comma_decimal_locale():
    assert _version_separator("es_ES").text == "punto"


def test_sentence_period_after_version_remains_outside_the_version_span():
    assert [item["text"] for item in VersionDetector("en_US").detect("Use v2.3.1.")] == ["v2.3.1"]
    normalized = frend.normalize("Use v2.3.1.", offsets=True)
    assert normalized.text.endswith(" one .")
    assert (normalized.units[-1].source_span, normalized.units[-1].reader) == ((10, 11), None)


@pytest.mark.parametrize(
    "written",
    [
        "192.168.1.1",
        "example.com",
        "2024.01.31",
        "End.",
        "v2.3.1.example.com",
        "v2.3.1.рф",
    ],
)
def test_nonversion_dots_are_not_version_spans(written):
    assert list(detect(written, (VersionDetector("en_US"),))) == []
