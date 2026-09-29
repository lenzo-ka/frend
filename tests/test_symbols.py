"""Standalone symbols and letters of other scripts, read by CLDR's and ICU's names."""

from __future__ import annotations

import pytest

from frend import resolve_lattice
from frend.spoken_priors import normalize_spoken
from frend.symbols import SymbolDetector
from frend.verbalize import verbalize_edge


def _forms(text):
    (detection,) = SymbolDetector().detect(text)
    lattice = resolve_lattice([detection], source_text=text)
    edge = next(e for e in lattice.edges if e.kind == "reading")
    # The measured order, before any context tree (the symbol stands alone here).
    unit = verbalize_edge(edge, source_text=text, rerank_by_context=False)
    assert unit.unspoken == ()
    return [normalize_spoken(a.text) for a in unit.alternatives]


@pytest.mark.parametrize(
    ("text", "first"),
    [("&", "and"), ("#", "number"), (".", ""), (",", ""), ("-", ""), ("α", "alpha"), ("風", "")],
)
def test_symbol_reads_as_the_corpus_measures_it(text, first):
    """CLDR's names and silence are offered; the corpus picks "and" for "&", nothing for
    punctuation and for a character of a script it does not read, a Greek letter's name."""
    assert _forms(text)[0] == first


def test_every_cldr_name_and_silence_are_offered():
    forms = _forms("&")
    assert {"ampersand", "and", ""} <= set(forms)


@pytest.mark.parametrize("text", ["R&D", "AT&T", "a-b", "x.y", "3.14", "αβ"])
def test_a_character_inside_a_word_is_left_alone(text):
    assert SymbolDetector().detect(text) == []


def test_a_standalone_symbol_in_running_text_is_read():
    detections = SymbolDetector().detect("Tom & Jerry")
    assert [(d["type"], d["text"]) for d in detections] == [("symbol:cldr", "&")]


def test_the_locale_scripts_come_from_icu():
    """Likely subtags' script first, the exemplars' scripts, then Common and Inherited."""
    from frend.symbols import locale_scripts

    assert locale_scripts("en_US") == ("Latn", "Zinh", "Zyyy")
    assert locale_scripts("ru")[0] == "Cyrl"
    assert {"Hani", "Hira", "Kana"} <= set(locale_scripts("ja_JP"))


def test_a_letter_of_the_locale_script_is_not_read_by_name():
    """Cyrillic is another script for en_US and the locale's own for ru."""
    assert [d["type"] for d in SymbolDetector("en_US").detect("Ж")] == ["symbol:letter"]
    assert SymbolDetector("ru").detect("Ж") == []


def test_spaced_letters_of_another_script_are_one_span():
    text = "« Т Е С Т - а б »."
    runs = [d for d in SymbolDetector().detect(text) if d["type"] == "symbol:script-run"]
    assert [(d["text"], d["start"], d["end"]) for d in runs] == [
        ("Т Е С Т", 2, 9),
        ("а б", 12, 15),
    ]
    assert not any(d["type"] == "symbol:letter" for d in SymbolDetector().detect(text))


def test_a_word_is_no_run_and_does_not_join_its_neighbors():
    detections = SymbolDetector().detect("Москва и Петербург")
    assert [(d["type"], d["text"]) for d in detections] == [("symbol:letter", "и")]
    detections = SymbolDetector().detect("а б Москва в г")
    assert [(d["type"], d["text"]) for d in detections] == [
        ("symbol:script-run", "а б"),
        ("symbol:script-run", "в г"),
    ]


def test_a_run_reads_by_transliteration_names_as_written_and_silence():
    text = "Θ Α"
    (detection,) = SymbolDetector().detect(text)
    assert detection["value"].transform == "Greek-Latin"
    lattice = resolve_lattice([detection], source_text=text)
    edge = next(e for e in lattice.edges if e.kind == "reading")
    unit = verbalize_edge(edge, source_text=text, rerank_by_context=False)
    readings = {(a.text, a.provenance) for a in unit.alternatives}
    assert readings == {
        ("th a", "icu-transliteration:Greek-Latin"),
        ("theta alpha", "icu-name:letter"),
        ("Θ Α", "surface:as-written"),
        ("", "surface:silence"),
    }
    # Greek is measured by name, so the names lead, as they do for a lone letter.
    assert unit.alternatives[0].text == "theta alpha"


def test_the_transform_is_one_icu_lists_else_any():
    from frend.symbols import transform_id

    assert transform_id("Cyrl", "en_US") in ("Cyrillic-Latin", "Cyrl-Latn")
    assert transform_id("Latn", "ru") in ("Latin-Cyrillic", "Latn-Cyrl")


def test_spaced_letters_resolve_as_one_choice():
    """Ninety-six spaced letters were 2^96 covers; as one run they are a handful."""
    from icukit.detectors import detect

    text = " ".join("абвгдежзиклмнопрстуфхцчшщэюя" * 4)
    lattice = resolve_lattice(list(detect(text, [SymbolDetector()])), source_text=text)
    readings = [e for e in lattice.edges if e.kind == "reading"]
    assert len(readings) == 1 and readings[0].end - readings[0].start == len(text)
