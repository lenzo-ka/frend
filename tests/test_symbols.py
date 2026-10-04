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


@pytest.mark.parametrize("text", ["ᵋ", "々"])
def test_measured_property_classes_are_silent_first_with_name_fallback(text):
    (detection,) = SymbolDetector().detect(text)
    assert detection["type"] == "symbol:property"
    lattice = resolve_lattice([detection], source_text=text)
    edge = next(edge for edge in lattice.edges if edge.kind == "reading")
    unit = verbalize_edge(edge, source_text=text)

    assert unit.alternatives[0].text == ""
    assert unit.alternatives[0].provenance == "surface:silence"
    assert unit.alternatives[1].text
    assert unit.alternatives[1].provenance == "icu-name:property"


def test_property_rule_uses_icu_class_even_inside_a_token():
    detections = SymbolDetector().detect("aᵋb")
    assert [(item["type"], item["start"], item["end"]) for item in detections] == [
        ("symbol:property", 1, 2)
    ]


def test_mixed_modifier_class_is_report_only():
    from frend.symbols import silent_property_class

    assert silent_property_class("ʻ") is None
    assert silent_property_class("ー") is None


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


def _run_unit(text, locale="en_US"):
    (detection,) = SymbolDetector(locale).detect(text)
    lattice = resolve_lattice([detection], source_text=text)
    edge = next(e for e in lattice.edges if e.kind == "reading")
    return detection, verbalize_edge(edge, source_text=text, rerank_by_context=False)


def test_a_run_reads_by_transliteration_names_as_written_and_silence():
    detection, unit = _run_unit("Θ Α")
    assert detection["value"].transform == "Greek-Latin; Latin-ASCII"
    readings = {(a.text, a.provenance) for a in unit.alternatives}
    assert readings == {
        ("th a", "icu-transliteration:Greek-Latin; Latin-ASCII"),
        ("theta alpha", "icu-name:letter"),
        ("Θ Α", "surface:as-written"),
        ("", "surface:silence"),
    }
    # Greek is measured by name, so the names lead, as they do for a lone letter.
    assert unit.alternatives[0].text == "theta alpha"


def test_a_cyrillic_run_is_silent_first_as_its_lone_letters_are():
    _, unit = _run_unit("Т Е С Т")
    assert unit.alternatives[0].text == ""
    assert ("t e s t", "icu-transliteration:Cyrillic-Latin; Latin-ASCII") in {
        (a.text, a.provenance) for a in unit.alternatives
    }


@pytest.mark.parametrize(
    ("text", "spoken"),
    [("ж з", "z z"), ("風 雨", "feng yu"), ("Ё Щ", "e s")],
)
def test_a_latin_transliteration_holds_only_ascii_letters(text, spoken):
    (detection,) = SymbolDetector().detect(text)
    readings = {source: name for name, source in detection["value"].names}
    transliterated = next(v for k, v in readings.items() if k.startswith("icu-transliteration"))
    assert transliterated == spoken
    assert transliterated.isascii()


def test_the_transform_is_one_icu_lists_else_any():
    from frend.symbols import transform_id

    assert transform_id("Cyrl", "en_US") == "Cyrillic-Latin; Latin-ASCII"
    assert transform_id("Latn", "ru") == "Latin-Cyrillic"


def test_a_run_joins_only_letters_of_one_script():
    detections = SymbolDetector().detect("Т Α")
    assert [(d["type"], d["text"]) for d in detections] == [
        ("symbol:letter", "Т"),
        ("symbol:letter", "Α"),
    ]
    detections = SymbolDetector().detect("а б Θ Α")
    assert [(d["type"], d["text"]) for d in detections] == [
        ("symbol:script-run", "а б"),
        ("symbol:script-run", "Θ Α"),
    ]


@pytest.mark.parametrize(
    ("text", "letters"),
    [("Т Е С Т5", ["Т", "Е", "С"]), ("xА Б В", ["Б", "В"])],
)
def test_a_run_touching_a_word_keeps_its_letters_one_by_one(text, letters):
    detections = SymbolDetector().detect(text)
    assert [(d["type"], d["text"]) for d in detections] == [
        ("symbol:letter", letter) for letter in letters
    ]


@pytest.mark.parametrize("text", ["Т\nЕ", "Т  Е", "Т\tЕ"])
def test_a_run_joins_across_one_space_only(text):
    detections = SymbolDetector().detect(text)
    assert [d["type"] for d in detections] == ["symbol:letter", "symbol:letter"]


@pytest.mark.parametrize(
    ("text", "names", "spoken"),
    [
        ("А́ Б", "a be", "a b"),
        ("й́ к", "short i ka", "j k"),
        ("ά β", "alpha with tonos beta", "a b"),
    ],
)
def test_a_run_names_its_letters_without_their_marks(text, names, spoken):
    (detection,) = SymbolDetector().detect(text)
    readings = {source.split(":")[0]: name for name, source in detection["value"].names}
    assert readings["icu-name"] == names
    assert readings["icu-transliteration"] == spoken


def test_a_locale_naming_no_script_writes_its_language_s_scripts():
    """Serbian is written in Cyrillic and Latin: "sr" reads neither as another script."""
    from frend.symbols import locale_scripts

    assert {"Cyrl", "Latn"} <= set(locale_scripts("sr"))
    assert "Latn" not in locale_scripts("sr_Cyrl")
    assert SymbolDetector("sr").detect("A B") == []
    assert SymbolDetector("sr").detect("А Б") == []


def test_a_lone_latin_letter_reads_as_written_in_any_locale():
    assert SymbolDetector("ru").detect("A") == []
    assert SymbolDetector("sr_Cyrl").detect("A") == []


def test_a_transliteration_outside_the_locale_scripts_is_dropped():
    """ICU's Any-Hira leaves Greek as Greek; ja reads no Greek, so no such reading."""
    (detection,) = SymbolDetector("ja_JP").detect("α β")
    sources = [source for _, source in detection["value"].names]
    assert not any(source.startswith("icu-transliteration") for source in sources)


def test_spaced_letters_resolve_as_one_choice():
    """Ninety-six spaced letters were 2^96 covers; as one run they are a handful."""
    from icukit.detectors import detect

    text = " ".join("абвгдежзиклмнопрстуфхцчшщэюя" * 4)
    lattice = resolve_lattice(list(detect(text, [SymbolDetector()])), source_text=text)
    readings = [e for e in lattice.edges if e.kind == "reading"]
    assert len(readings) == 1 and readings[0].end - readings[0].start == len(text)
