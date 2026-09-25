"""Forms ICU writes nowhere, read by frend itself (kal's ruling): "2: 13", "339 U.S.",
"6 3", and eras written with periods, attached, or first."""

from __future__ import annotations

import pytest

from frend import resolve_lattice
from frend.spoken_priors import normalize_spoken
from frend.verbalize import verbalize_edge
from frend.written_forms import WrittenFormsDetector


def _spans(text):
    return [(d["type"], d["text"]) for d in WrittenFormsDetector().detect(text)]


def _forms(text):
    (detection,) = WrittenFormsDetector().detect(text)
    lattice = resolve_lattice([detection], source_text=text)
    edge = next(e for e in lattice.edges if e.kind == "reading")
    unit = verbalize_edge(edge, source_text=text)
    assert unit.unspoken == ()
    return [normalize_spoken(a.text) for a in unit.alternatives]


@pytest.mark.parametrize(
    ("text", "type_", "spoken"),
    [
        ("7: 31", "time:spaced-colon", "seven thirty one"),
        ("339 U.S.", "number:cardinal:citation", "three hundred thirty nine"),
        ("6 3", "number:digits", "six three"),
        ("500 B.C.", "date:era", "five hundred b c"),
        ("2000 BC.", "date:era", "two thousand b c"),
        ("4AD", "date:era", "four a d"),
    ],
)
def test_form_icu_writes_nowhere_reads_as_the_corpus_says_it(text, type_, spoken):
    assert _spans(text) == [(type_, text)]
    assert spoken in _forms(text)


def test_citation_also_offers_the_reporter_and_zero_also_reads_o():
    assert "three hundred thirty nine u s" in _forms("339 U.S.")
    assert {"one zero two", "one o two"} <= set(_forms("1 0 2"))


def test_era_written_first_is_said_first():
    assert _forms("A.D. 1066")[0].startswith("a d ")
    assert _forms("500 B.C.")[0].endswith(" b c")


@pytest.mark.parametrize("text", ["500 BC", "3.14", "10: 75", "25: 00", "12:30", "U.S. 339"])
def test_leaves_icus_own_forms_and_non_matches_alone(text):
    """ "500 BC" is ICU's own form, icukit reads it; the rest are not these forms."""
    assert _spans(text) == []
