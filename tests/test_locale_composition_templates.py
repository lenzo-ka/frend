"""Locale-owned composition frames must not leak English into other locales."""

from __future__ import annotations

import pytest
from icukit.recognize import FlexibleCurrencyDetector, FlexibleFractionDetector

from frend import normalize, resolve_lattice, verbalize_edge
from frend import verbalize as verbalize_module


def _fraction_alternatives(locale: str, written: str):
    detection = next(
        item
        for item in FlexibleFractionDetector(locale).detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    lattice = resolve_lattice([detection], source_text=written)
    edge = next(item for item in lattice.edges if item.kind == "reading")
    return verbalize_edge(
        edge, source_text=written, locale=locale, rerank_by_context=False
    ).alternatives


def _fraction_forms(locale: str, written: str) -> set[str]:
    return {alternative.text for alternative in _fraction_alternatives(locale, written)}


@pytest.mark.parametrize(
    ("locale", "written", "connector"),
    [("de_DE", "4/2", "over"), ("es_MX", "1 1/2", "and"), ("fr_FR", "1/4", "over")],
    ids=["de_DE-4/2", "es_MX-1 1/2", "fr_FR-1/4"],
)
def test_no_english_connector_outside_english(locale, written, connector):
    assert all(connector not in form.split() for form in _fraction_forms(locale, written))


@pytest.mark.parametrize(
    ("locale", "written", "spoken"), [("de_DE", "2/3", "zwei drittes")], ids=["de_DE-2/3"]
)
def test_non_english_fraction_has_no_plural_s(locale, written, spoken):
    assert spoken not in _fraction_forms(locale, written)


@pytest.mark.parametrize(
    ("locale", "written", "spoken"),
    [
        ("es_ES", "2/7", "dos séptimos"),
        ("es_MX", "2/7", "dos séptimos"),
        ("fr_FR", "2/9", "deux neuvièmes"),
        ("pt_BR", "3/8", "três oitavos"),
    ],
)
def test_locale_owned_fraction_plural_restores_romance_reading(locale, written, spoken):
    alternatives = _fraction_alternatives(locale, written)
    restored = next(item for item in alternatives if item.text == spoken)
    assert f"lexical:{locale}" in restored.provenance.split("+")
    assert "lexical:en_US" not in restored.provenance.split("+")


def test_money_minor_joiner_comes_from_en_lexical(monkeypatch):
    written = "$20.50"
    detection = next(
        item
        for item in FlexibleCurrencyDetector("en_US", "USD").detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    lattice = resolve_lattice([detection], source_text=written)
    edge = next(item for item in lattice.edges if item.kind == "reading")

    def forms() -> set[str]:
        return {
            item.text
            for item in verbalize_edge(
                edge, source_text=written, locale="en_US", rerank_by_context=False
            ).alternatives
        }

    assert "twenty dollars and fifty cents" in forms()
    lexical = dict(verbalize_module._lexical_for("en_US"))
    lexical.pop("money.minor_joiner")
    monkeypatch.setattr(verbalize_module, "_lexical_for", lambda locale: lexical)
    assert all(" and " not in form for form in forms())


def test_money_minor_joiner_is_a_two_phrase_frame():
    assert verbalize_module._lexical("money.minor_joiner", "en_US") == "{0} and {1}"


@pytest.mark.parametrize(
    ("written", "lexical_text"),
    [("2/3", "two thirds"), ("1/2", "one over two"), ("$20.50", "twenty dollars and fifty cents")],
)
def test_lexical_composition_records_lexical_provenance(written, lexical_text):
    if written.startswith("$"):
        detectors = [FlexibleCurrencyDetector("en_US", "USD")]
    else:
        detectors = [FlexibleFractionDetector("en_US")]
    detection = next(
        item
        for detector in detectors
        for item in detector.detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    edge = next(
        item
        for item in resolve_lattice([detection], source_text=written).edges
        if item.kind == "reading"
    )
    alternative = next(
        item
        for item in verbalize_edge(
            edge, source_text=written, locale="en_US", rerank_by_context=False
        ).alternatives
        if item.text == lexical_text
    )
    assert "lexical:en_US" in alternative.provenance.split("+")


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("1/4", " one quarter "),
        ("1 1/2", " one and a half "),
        ("$20.50", " twenty dollars and fifty cents "),
        ("March 3, 2020", " March third, twenty twenty "),
    ],
)
def test_english_fraction_and_money_unchanged(written, spoken):
    assert normalize(written, locale="en_US") == spoken


def test_english_ambiguous_numeric_date_keeps_context_rank_after_provenance_move():
    assert normalize("1/3/2026", locale="en_US") == (" the third of January twenty twenty-six ")
