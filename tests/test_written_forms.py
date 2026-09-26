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
        ("6 3", "number:digits", "six three"),
        ("500 B.C.", "date:era", "five hundred b c"),
        ("4AD", "date:era", "four a d"),
    ],
)
def test_form_icu_writes_nowhere_reads_as_the_corpus_says_it(text, type_, spoken):
    assert _spans(text) == [(type_, text)]
    assert spoken in _forms(text)


def test_citation_reads_its_volume_only_where_a_page_follows():
    """ "339 U.S. 629" is a citation (the corpus says the volume alone; "u s" also offered);
    "The 339 U.S. troops" is prose, and no citation is read there."""
    text = "see 339 U.S. 629"
    (detection,) = WrittenFormsDetector().detect(text)
    assert detection["text"] == "339 U.S."
    lattice = resolve_lattice([detection], source_text=text)
    edge = next(e for e in lattice.edges if e.kind == "reading")
    forms = [normalize_spoken(a.text) for a in verbalize_edge(edge, source_text=text).alternatives]
    assert {"three hundred thirty nine", "three hundred thirty nine u s"} <= set(forms)
    assert _spans("The 339 U.S. troops returned.") == []


def test_zero_digit_also_reads_o():
    assert {"one zero two", "one o two"} <= set(_forms("1 0 2"))


def test_era_written_first_is_said_first():
    assert _forms("A.D. 1066")[0].startswith("a d ")
    assert _forms("500 B.C.")[0].endswith(" b c")


@pytest.mark.parametrize(
    "text", ["500 BC", "500 BC. Next", "3.14", "10: 75", "25: 00", "12:30", "U.S. 339"]
)
def test_leaves_icus_own_forms_and_non_matches_alone(text):
    """ "500 BC" is ICU's own form, icukit reads it (its period ends the sentence); the rest
    are not these forms."""
    assert _spans(text) == []


@pytest.mark.parametrize("text", ["2,026", "in 2,026", "March 5, 2,026", "2,026 BC"])
def test_a_grouped_number_is_never_a_year(text):
    """kal: "Dates won't have commas in like 2,026": no reader frend measures with reads a
    grouped number as a date, and its number reading gets no digit-by-digit form."""
    from icukit.detectors import detect

    from tests.test_spoken_priors import _load_builder

    profile = _load_builder()._detectors()
    detectors = [d for kind in ("date", "time", "cardinal", "digit") for d in profile[kind]]
    found = [d for d in detect(text, detectors) if "2,026" in text[d["start"] : d["end"]]]
    assert found and all(not d["type"].startswith(("date:", "time:")) for d in found)
    number = next(d for d in found if d["type"].startswith("number:"))
    lattice = resolve_lattice([number], source_text=text)
    edge = next(e for e in lattice.edges if e.kind == "reading")
    forms = [normalize_spoken(a.text) for a in verbalize_edge(edge, source_text=text).alternatives]
    assert "two thousand twenty six" in forms
    assert not any(form.startswith("two zero") or form.startswith("two o ") for form in forms)
