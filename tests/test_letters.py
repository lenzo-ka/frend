"""Runs of capitals and initials, spelled or said as measured; Roman numerals left to icukit."""

from __future__ import annotations

import pytest
from icukit.abbreviation_recognize import AbbreviationDetector
from icukit.detectors import detect
from icukit.recognize import FlexibleNumberDetector

from frend import resolve_lattice
from frend.letters import LettersDetector, cv_pattern, is_roman, is_spelled_token, numeral_share
from frend.verbalize import verbalize_lattice

_DETECTORS = (FlexibleNumberDetector("en_US"), AbbreviationDetector("en_US"), LettersDetector())


def _read(text: str) -> list[list[str]]:
    lattice = resolve_lattice(list(detect(text, _DETECTORS)), source_text=text)
    return [
        [alternative.text for alternative in unit.alternatives]
        for unit in verbalize_lattice(lattice).best_path.units
        if unit.best.provenance != "surface:passthrough"
        and not any("spelled-token" in item.provenance for item in unit.alternatives)
        and unit.best.provenance != "surface:word"
    ]


def _spans(text: str) -> list[tuple[str, str]]:
    return [
        (d["type"], d["text"])
        for d in LettersDetector().detect(text)
        if d["type"] != "letters:token"
    ]


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


@pytest.mark.parametrize("text", ["pdf", "pc", "fMRI", "Meic"])
def test_short_locale_script_tokens_offer_spelling(text):
    found = [d for d in LettersDetector().detect(text) if d["type"] == "letters:token"]
    assert [(d["start"], d["end"], d["text"]) for d in found] == [(0, len(text), text)]
    assert is_spelled_token(text)


@pytest.mark.parametrize("text", ["a", "toolong", "NASA", "AΒ"])
def test_short_token_rule_is_bounded(text):
    assert not is_spelled_token(text)


def test_unattested_rule_say_token_still_offers_both_readings():
    assert is_spelled_token("word")
    found = [d for d in LettersDetector().detect("word") if d["type"] == "letters:token"]
    lattice = resolve_lattice(found, source_text="word")
    unit = verbalize_lattice(lattice).best_path.units[0]
    assert [item.text for item in unit.alternatives] == ["word", "w o r d"]


def test_consonant_only_rule_spells_first():
    found = [d for d in LettersDetector().detect("xyz") if d["type"] == "letters:token"]
    lattice = resolve_lattice(found, source_text="xyz")
    unit = verbalize_lattice(lattice).best_path.units[0]
    assert [item.text for item in unit.alternatives] == ["x y z", "xyz"]


def test_exact_token_prior_ranks_but_keeps_both_readings(monkeypatch):
    from frend import verbalize

    monkeypatch.setattr(
        verbalize,
        "spelled_token_prior",
        lambda token, locale: {"shares": {"spell": 0.9, "say": 0.1}},
    )
    lattice = resolve_lattice(list(detect("pdf", (LettersDetector(),))), source_text="pdf")
    unit = verbalize_lattice(lattice).best_path.units[0]
    assert [item.text for item in unit.alternatives] == ["p d f", "pdf"]


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
    spoken = [
        u.best.text
        for u in units
        if u.best.provenance not in {"surface:passthrough", "surface:word"}
        and not any("spelled-token" in item.provenance for item in u.alternatives)
    ]
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


# Tokens the builder and the reader must agree on, one by one: decomposed and
# precomposed accents, a Common capital beside a Latin one, other scripts, and the
# refusals (two real scripts, a non-capital, a lower case letter).
_POPULATION = ["ÉCO", "E\u0301CO", "AB\u0301CD", "СССР", "ℂA", "ℂℍ", "AΒ", "AB漢", "ATM", "Ab"]


def _counted(tmp_path, token: str) -> bool:
    shard = tmp_path / token.encode("unicode_escape").decode("ascii").replace("\\", "_")
    shard.mkdir()
    (shard / "output-00000-of-00001").write_text(
        f"LETTERS\t{token}\t{' '.join(token.lower())}\n<eos>\t<eos>\n", encoding="utf-8"
    )
    return "*" in _acronym_builder().build_document(shard)["keys"]


def test_the_acronym_builder_counts_what_the_reader_matches(tmp_path):
    """One predicate: the builder counts a token exactly when the letters reader matches
    it whole, token by token, so no count stands for a population the reader never
    ranks. A builder that counted "AΒ" and dropped "СССР" keeps the total and fails here."""
    counted = {token for token in _POPULATION if _counted(tmp_path, token)}
    matched = {token for token in _POPULATION if _spans(token) == [("letters:run", token)]}
    assert counted == matched
    assert matched == {"ÉCO", "E\u0301CO", "AB\u0301CD", "СССР", "ℂA", "ℂℍ", "ATM"}


def test_the_builder_counts_the_numerals_the_reader_defers(tmp_path):
    """The one stated difference: a run the corpus reads as a Roman numeral ("II") is
    counted by the builder (under ``roman:II`` too) and left by the reader to icukit."""
    assert _counted(tmp_path, "II")
    assert _spans("II") == []


# Decomposed input is one run, with exact spans (fugu P3 review, finding 1).


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the E\u0301CO fund", [("letters:run", "E\u0301CO", 4, 8)]),
        ("AB\u0301CD", [("letters:run", "AB\u0301CD", 0, 5)]),
        ("E\u0301COs", [("letters:run", "E\u0301COs", 0, 5)]),
        ("Jane E\u0301. Smith", [("letters:initial", "E\u0301.", 5, 8)]),
        ("e\u0301AB", []),
        ("ABe\u0301", []),
    ],
)
def test_combining_marks_stay_with_their_capital(text, expected):
    found = [
        (d["type"], d["text"], d["start"], d["end"])
        for d in LettersDetector().detect(text)
        if d["type"] != "letters:token"
    ]
    assert found == expected
    assert all(text[start:end] == surface for _, surface, start, end in found)


def test_a_decomposed_run_reads_as_the_precomposed_one():
    assert _read("E\u0301CO") == _read("ÉCO")


# Each capital is spelled by its own lower case (fugu P3 review, finding 2).


def _letters_alternatives(text: str) -> list[str]:
    lattice = resolve_lattice(list(detect(text, (LettersDetector(),))), source_text=text)
    units = verbalize_lattice(lattice).best_path.units
    return [
        a.text for u in units if u.best.provenance != "surface:passthrough" for a in u.alternatives
    ]


def test_each_capital_is_spelled_by_its_own_lower_case():
    # Not "i ̇ b" (the full lowering of İ leaves a free dot) nor "ο ς" (final sigma).
    assert "i b" in _letters_alternatives("İB")
    assert "ο σ" in _letters_alternatives("ΟΣ")
    assert "e\u0301 c o".replace("e\u0301", "é") in _letters_alternatives("E\u0301CO")


def test_english_spelled_forms_are_unchanged():
    assert "a t m" in _letters_alternatives("ATM")
    assert _read("UFOs")[0] == ["u f o's", "ufos"]
    assert _read("J.R.R. Tolkien") == [["j r r"]]


# Script runs per UAX #24: Common capitals take their neighbors' script (finding 3).


@pytest.mark.parametrize("text", ["ℂA", "ℂℍ", "Aℂ"])
def test_a_common_capital_takes_its_neighbors_script(text):
    assert _spans(text) == [("letters:run", text)]


@pytest.mark.parametrize("text", ["AΒ", "AℂΒ"])
def test_two_real_scripts_are_still_not_one_run(text):
    assert _spans(text) == []


def test_the_vowels_are_the_locales():
    """``cv_pattern`` and ``letter_key`` read ``letter.vowels``; a locale whose table has
    none (ru's is empty) forms no vowel key."""
    from frend.electronic import letter_key

    assert cv_pattern("GUS") == cv_pattern("GUS", "en_US") == "cvc"
    assert letter_key("NASA") == "upper:4:v"
    assert letter_key("GWR") == "upper:3:nv"
    # English "y" is a vowel: "MYTH" has one.
    assert letter_key("MYTH") == "upper:4:v"
    assert cv_pattern("MYTH") == "cvcc"
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
