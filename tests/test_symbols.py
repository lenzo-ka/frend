"""Standalone symbols and letters of other scripts, read by CLDR's and ICU's names."""

from __future__ import annotations

import pytest

from frend import resolve_lattice
from frend.spoken_priors import normalize_spoken
from frend.symbols import SymbolDetector, _cldr_names
from frend.verbalize import verbalize_edge

pytestmark = pytest.mark.skipif(
    not _cldr_names("en_US"), reason="icukit before #136 lists no symbol names"
)


def _forms(text):
    (detection,) = SymbolDetector().detect(text)
    lattice = resolve_lattice([detection], source_text=text)
    edge = next(e for e in lattice.edges if e.kind == "reading")
    unit = verbalize_edge(edge, source_text=text)
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
