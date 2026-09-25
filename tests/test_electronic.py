"""URLs, email addresses and bare domains: frend's own recognition and measured speech."""

from __future__ import annotations

import json

import pytest
from icukit.detectors import detect
from icukit.recognize import FlexibleDateDetector, FlexibleNumberDetector

from frend import resolve, resolve_lattice
from frend.electronic import (
    ElectronicDetector,
    decode_letter_notation,
    load_electronic_priors,
    tld_version,
    top_level_domains,
)
from frend.spoken_priors import normalize_spoken
from frend.verbalize import verbalize_edge


def _spans(text):
    return [(d["type"], d["text"]) for d in ElectronicDetector().detect(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("boston.com", [("electronic:domain", "boston.com")]),
        ("BBC.co.uk", [("electronic:domain", "BBC.co.uk")]),
        (
            "at http://www.ucc.ie/celt/trotula.html today",
            [("electronic:url", "http://www.ucc.ie/celt/trotula.html")],
        ),
        ("See www.nps.gov/vafo).", [("electronic:url", "www.nps.gov/vafo")]),
        ("write jane.doe@example.org, please", [("electronic:email", "jane.doe@example.org")]),
        # The local part is itself a domain; only the whole address is a reading.
        ("jane.doe.com@example.org", [("electronic:email", "jane.doe.com@example.org")]),
    ],
)
def test_detector_finds_maximal_spans_without_trailing_punctuation(text, expected):
    assert _spans(text) == expected


@pytest.mark.parametrize(
    "text", ["e.g. this", "i.e. that", "3.14", "report.pdf", "under_score.com", "::"]
)
def test_detector_rejects_what_is_not_a_host_under_a_top_level_domain(text):
    """No TLD ("pdf", a single letter), no letters, or a host ICU's IDNA refuses."""
    assert _spans(text) == []


def _forms(text):
    (detection,) = ElectronicDetector().detect(text)
    edge = next(
        e for e in resolve_lattice([detection], source_text=text).edges if e.kind == "reading"
    )
    return verbalize_edge(edge, source_text=text)


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("boston.com", "boston dot com"),
        ("BBC.co.uk", "b b c dot co dot u k"),
        ("et.al", "et dot a l"),
    ],
)
def test_measured_speech_ranks_the_corpus_form_first(text, spoken):
    unit = _forms(text)
    assert normalize_spoken(unit.alternatives[0].text) == spoken
    assert unit.alternatives[0].weight is not None
    assert unit.unspoken == ()


def test_url_speech_keeps_the_corpus_form_among_its_readings():
    unit = _forms("http://www.ucc.ie/celt/trotula.html")
    corpus = (
        "h t t p colon slash slash w w w dot u c c dot i e slash celt slash trotula dot h t m l"
    )
    assert corpus in [normalize_spoken(a.text) for a in unit.alternatives]


def test_email_reads_at_as_its_one_lexical_name():
    """The corpus holds no email address, so only "@" is not measured."""
    unit = _forms("jane.doe@example.org")
    assert normalize_spoken(unit.alternatives[0].text) == "jane dot doe at example dot org"
    assert unit.alternatives[0].provenance == "measured:electronic+lexical:en_US"
    assert _forms("boston.com").alternatives[0].provenance == "measured:electronic"


def test_a_url_outranks_the_readings_icukit_finds_inside_it():
    """icukit reads "2004" and "03" inside a URL; the URL's span covers them in the 1-best."""
    text = "see www.example.com/2004/03 today"
    detections = [
        *detect(text, [FlexibleNumberDetector("en_US"), FlexibleDateDetector("en_US")]),
        *ElectronicDetector().detect(text),
    ]
    assert any(d["type"].startswith("number") for d in detections)
    best = resolve(detections, source_text=text).best
    assert [d["type"] for d in best] == ["electronic:url"]


def test_decode_letter_notation_joins_letters_into_words():
    assert (
        decode_letter_notation("b_letter o_letter  _letter x_letter dot c_letter o_letter m_letter")
        == "bo x dot com"
    )


def test_priors_table_records_its_sources_and_counts_only():
    table = load_electronic_priors()
    assert table["provenance"]["tld_list"] == tld_version()
    assert table["provenance"]["rows_aligned"] <= table["provenance"]["rows"]
    assert table["separators"]["."] and table["letters"]["tld:com"]
    counts = [
        n
        for section in ("letters", "digits")
        for row in table[section].values()
        for n in row.values()
    ]
    assert all(isinstance(n, int) and n >= 0 for n in counts)
    assert all(key == "*" or ":" in key for key in table["letters"])
    assert "com" in top_level_domains() and "pdf" not in top_level_domains()
    assert json.dumps(table)
