"""Runs of capitals and initials, spelled or said as measured; Roman numerals left to icukit."""

from __future__ import annotations

import pytest
from icukit.abbreviation_recognize import AbbreviationDetector
from icukit.detectors import detect
from icukit.recognize import FlexibleNumberDetector

from frend import resolve_lattice
from frend.letters import LettersDetector, cv_pattern, is_roman, numeral_share
from frend.verbalize import verbalize_lattice

_DETECTORS = (FlexibleNumberDetector("en_US"), AbbreviationDetector("en_US"), LettersDetector())


def _read(text: str) -> list[list[str]]:
    lattice = resolve_lattice(list(detect(text, _DETECTORS)), source_text=text)
    return [
        [alternative.text for alternative in unit.alternatives]
        for unit in verbalize_lattice(lattice).best_path.units
        if unit.best.provenance != "surface:passthrough"
    ]


def _spans(text: str) -> list[tuple[str, str]]:
    return [(d["type"], d["text"]) for d in LettersDetector().detect(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the ATM", [("letters:run", "ATM")]),
        ("UFOs and AFI's", [("letters:run", "UFOs"), ("letters:run", "AFI's")]),
        ("Jane S. Smith", [("letters:initial", "S.")]),
        ("USA.", [("letters:run", "USA")]),
    ],
)
def test_detects_runs_initials_and_suffixes(text, expected):
    assert _spans(text) == expected


@pytest.mark.parametrize("text", ["A&I", "I saw it", "McDonald", "ATMs2", "U.S.A", "GRA-B", "II"])
def test_leaves_mixed_single_and_numeral_runs(text):
    assert _spans(text) == []


def test_a_consonant_run_spells_first():
    assert _read("at the GWR")[0][0] == "g w r"


def test_a_plural_rides_on_the_last_letter():
    assert _read("UFOs")[0] == ["u f o's", "ufos"]


def test_a_word_shaped_run_is_said_first():
    # "cvc" is said more than spelled in the corpus; "GUS" reads as written.
    assert cv_pattern("GUS") == "cvc"
    assert _read("GUS")[0][0] == "gus"


def test_an_initial_is_its_letter_not_an_abbreviation():
    assert _read("Jane S. Smith") == [["s"]]


def test_roman_numerals_follow_the_corpus_per_surface():
    assert is_roman("II") and is_roman("CD") and not is_roman("ATM")
    assert numeral_share("II") > 0.9 > 0.1 > numeral_share("CD")
    assert _read("World War II")[0][0] == "two"
    assert _read("a CD")[0][0] == "c d"


def test_a_lexicon_acronym_keeps_its_own_measure():
    assert _read("NASA")[0][0] == "nasa"
    assert _read("FBI")[0][0] == "f b i"
