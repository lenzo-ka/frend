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
        (
            "J.R.R. Tolkien",
            [("letters:initial", "J."), ("letters:initial", "R."), ("letters:initial", "R.")],
        ),
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


def test_capitals_the_lexicon_lists_without_expansion_are_spelled():
    # icukit's lexicon lists "J.R.R." (a sentence-break exception) with no expansion.
    assert _read("J.R.R. Tolkien") == [["j r r"]]


def test_a_chain_of_initials_reads_one_initial_at_a_time():
    # Without icukit's abbreviation reader, the chain is frend's own: each initial alone.
    text = "X.Q.Z. Smith"
    lattice = resolve_lattice(list(detect(text, (LettersDetector(),))), source_text=text)
    units = verbalize_lattice(lattice).best_path.units
    spoken = [u.best.text for u in units if u.best.provenance != "surface:passthrough"]
    assert spoken == ["x", "q", "z"]


# Capitals by Unicode general category Lu, in one ICU script (dualplan-ranges P3).


def test_a_cyrillic_capital_run_is_a_letter_run():
    assert _spans("СССР") == [("letters:run", "СССР")]


def test_a_latin_capital_with_accent_joins_a_run():
    assert _spans("ÉCU") == [("letters:run", "ÉCU")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ATM", [("letters:run", "ATM")]),
        ("UFOs", [("letters:run", "UFOs")]),
        ("S.", [("letters:initial", "S.")]),
        (
            "J.R.R.",
            [("letters:initial", "J."), ("letters:initial", "R."), ("letters:initial", "R.")],
        ),
    ],
)
def test_latin_runs_and_initials_are_unchanged(text, expected):
    assert _spans(text) == expected


@pytest.mark.parametrize("text", ["AΒC", "X.Β."])
def test_capitals_of_two_scripts_are_not_one_run(text):
    # Latin A and C around Greek Beta; a Latin and a Greek initial.
    assert _spans(text) == []


def _acronym_builder():
    import importlib.util
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "build_acronym_priors", tools / "build_acronym_priors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_acronym_builder_counts_what_the_reader_matches(tmp_path):
    """One predicate: the builder counts a token exactly when the letters reader matches
    it whole, so no count stands for a population the reader never ranks."""
    tokens = ["ÉCO", "СССР", "AΒ", "AB漢", "ATM", "Ab"]
    shard = tmp_path / "output-00000-of-00100"
    shard.write_text(
        "".join(f"LETTERS\t{token}\t{' '.join(token.lower())}\n" for token in tokens)
        + "<eos>\t<eos>\n",
        encoding="utf-8",
    )
    document = _acronym_builder().build_document(tmp_path)
    counted = document["keys"].get("*", {}).get("spelled", 0)
    matched = [token for token in tokens if _spans(token) == [("letters:run", token)]]
    assert counted == len(matched)
    assert matched == ["ÉCO", "СССР", "ATM"]


def test_the_vowels_are_the_locales():
    """``cv_pattern`` and ``letter_key`` read ``letter.vowels``; a locale whose table has
    none (ru's is empty) forms no vowel key."""
    from frend.electronic import letter_key

    assert cv_pattern("GUS") == cv_pattern("GUS", "en_US") == "cvc"
    assert letter_key("NASA") == "upper:4:v"
    assert letter_key("GWR") == "upper:3:nv"
    assert cv_pattern("GUS", "ru_RU") is None
    assert letter_key("NASA", "ru_RU") == "upper:4"


def test_a_changed_vowel_set_changes_the_keys(monkeypatch):
    """Verify by breaking: the keys follow the table, not a string in the code."""
    from frend import letters
    from frend.electronic import letter_key

    real = letters.lexical_forms

    def only_a(locale):
        forms = dict(real(locale))
        forms["letter.vowels"] = {"why": "test", "value": "a"}
        return forms

    monkeypatch.setattr(letters, "lexical_forms", only_a)
    letters._vowels_for.cache_clear()
    try:
        assert cv_pattern("GUS") == "ccc"
        assert letter_key("GUS") == "upper:3:nv"
    finally:
        letters._vowels_for.cache_clear()
