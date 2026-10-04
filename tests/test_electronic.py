"""URLs, email addresses and bare domains: frend's own recognition and measured speech."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from importlib.resources import files

import pytest
from icukit.detectors import detect
from icukit.recognize import FlexibleDateDetector, FlexibleNumberDetector

import frend.electronic as electronic
from frend import InputValidationError, resolve, resolve_lattice
from frend.electronic import (
    DEFAULT_MAX_EMAIL_CHARS,
    DEFAULT_MAX_URL_CHARS,
    ElectronicDetector,
    decode_letter_notation,
    load_electronic_priors,
    load_electronic_span_priors,
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
    ("text", "kind"),
    [
        ("/http://example.com", "electronic:url"),
        ("http://example.com;", "electronic:url"),
        ("http://example.com)", "electronic:url"),
        ("#LA003", "electronic:hashtag"),
        ("TV.comWallace", "electronic:domain"),
        ("/6.doc", "electronic:document"),
        ("archive/annual-report.doc", "electronic:document"),
        ("//www.example.com", "electronic:url"),
    ],
)
def test_training_supported_electronic_shapes_keep_the_whole_token(text, kind):
    assert _spans(text) == [(kind, text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see http://example.com.", [("electronic:url", "http://example.com")]),
        ("see (http://example.com)", [("electronic:url", "http://example.com")]),
        ("TV.comWallace.", [("electronic:domain", "TV.comWallace")]),
        ("/6.doc.", [("electronic:document", "/6.doc")]),
        ("# Heading", []),
        ("#1", []),
        ("version 1.2.3", []),
        ("annual-report.doc", []),
        ("report.doc", []),
        ("E.coli", []),
        ("e.g.", []),
        ("U.S.", []),
        ("i.e.", []),
    ],
)
def test_span_conventions_do_not_absorb_prose_or_versions(text, expected):
    assert _spans(text) == expected


@pytest.mark.parametrize(
    ("feature", "text"),
    [
        ("leading-slash-scheme", "/http://example.com"),
        ("trailing-semicolon-scheme", "http://example.com;"),
        ("trailing-parenthesis-scheme", "http://example.com)"),
        ("hashtag", "#LA003"),
        ("tld-uppercase-suffix", "TV.comWallace"),
        ("path-document", "/6.doc"),
        ("scheme-relative-url", "//www.example.com"),
    ],
)
def test_each_span_convention_depends_on_its_measured_guard(monkeypatch, feature, text):
    document = load_electronic_span_priors()
    assert document is not None
    without = {**document, "features": {**document["features"]}}
    without["features"].pop(feature)
    monkeypatch.setattr(electronic, "load_electronic_span_priors", lambda **_kwargs: without)
    detections = ElectronicDetector().detect(text)
    assert not any(d["start"] == 0 and d["end"] == len(text) for d in detections)


def test_detector_refuses_oversized_candidates_without_emitting_a_prefix():
    url_prefix = "http://example.com/"
    exact_url = url_prefix + "a" * (DEFAULT_MAX_URL_CHARS - len(url_prefix))
    assert _spans(exact_url) == [("electronic:url", exact_url)]
    assert _spans(exact_url + "a") == []

    email_suffix = "@example.org"
    exact_email = "a" * (DEFAULT_MAX_EMAIL_CHARS - len(email_suffix)) + email_suffix
    assert _spans(exact_email) == [("electronic:email", exact_email)]
    assert _spans("a" + exact_email) == []


def test_detector_caps_are_configurable_and_input_is_validated_first():
    assert ElectronicDetector(max_url_chars=11).detect("example.com")
    assert ElectronicDetector(max_url_chars=10).detect("example.com") == []
    assert ElectronicDetector(max_url_chars=5).detect("a@example.org")[0]["type"] == (
        "electronic:email"
    )
    with pytest.raises(InputValidationError, match="NUL"):
        ElectronicDetector().detect("example.com\x00")


def test_detector_validation_runs_before_candidate_scanning(monkeypatch):
    def unexpected_scan(*_args):
        raise AssertionError("recognition started before validation")

    monkeypatch.setattr(electronic, "_bounded_segments", unexpected_scan)
    with pytest.raises(InputValidationError, match="NUL"):
        ElectronicDetector().detect("example.com\x00")


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


def test_trained_closing_parenthesis_uses_the_locale_lexical_form():
    unit = _forms("http://example.com)")
    assert normalize_spoken(unit.alternatives[0].text).endswith("closing parenthesis")
    assert "lexical:en_US" in unit.alternatives[0].provenance


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


def _assert_span_prior_is_aggregate(document):
    assert set(document) == {"schema_version", "locale", "provenance", "features"}
    assert type(document["schema_version"]) is int and document["schema_version"] == 1
    assert document["locale"] == "en"
    provenance = document["provenance"]
    assert set(provenance) == {
        "builder",
        "corpus",
        "locale",
        "shards",
        "selection",
        "privacy",
    }
    assert provenance["builder"] == "tools/build_electronic_span_priors.py"
    assert provenance["corpus"] == "google-tn:en_with_types"
    assert provenance["locale"] == "en"
    assert provenance["selection"] == (
        "feature enabled with at least 3 ELECTRONIC rows and a strict ELECTRONIC "
        "majority over all corpus rows"
    )
    assert provenance["privacy"] == (
        "aggregate feature and corpus-class counts only; no corpus text"
    )
    assert type(provenance["shards"]) is list
    assert provenance["shards"] == [f"output-{index:05d}-of-00100" for index in range(90)]
    assert type(document["features"]) is dict
    assert set(document["features"]) == {
        "hashtag",
        "leading-slash-scheme",
        "path-document",
        "scheme-relative-url",
        "tld-uppercase-suffix",
        "trailing-parenthesis-scheme",
        "trailing-semicolon-scheme",
    }
    class_pattern = re.compile(r"[A-Z][A-Z_]*\Z")
    for feature, row in document["features"].items():
        assert feature.replace("-", "").isalpha()
        assert set(row) == {"classes"}
        assert type(row["classes"]) is dict and row["classes"]
        for name, count in row["classes"].items():
            assert class_pattern.fullmatch(name)
            assert type(count) is int and count > 0


def test_span_priors_table_contains_only_aggregate_counts():
    document = load_electronic_span_priors()
    assert document is not None
    _assert_span_prior_is_aggregate(document)


@pytest.mark.parametrize("path", [("excerpt",), ("provenance", "privacy")])
def test_span_prior_privacy_guard_rejects_a_corpus_excerpt(path):
    document = deepcopy(load_electronic_span_priors())
    assert document is not None
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "raw corpus surface"
    with pytest.raises(AssertionError):
        _assert_span_prior_is_aggregate(document)


def _assert_electronic_lexical_additions_are_forms(document):
    assert document["forms"]["separator.words"] == {
        "why": (
            "no CLDR spoken form for '@' in an address; every one of 1,791 training "
            "ELECTRONIC scheme URLs ending ')' says 'closing parenthesis'"
        ),
        "value": {"@": "at", ")": "closing parenthesis"},
    }


def test_electronic_lexical_additions_are_lexical_forms_only():
    lexical = json.loads(
        files("frend").joinpath("data", "en", "lexical.json").read_text(encoding="utf-8")
    )
    _assert_electronic_lexical_additions_are_forms(lexical)


def test_electronic_lexical_guard_rejects_a_corpus_surface():
    lexical = json.loads(
        files("frend").joinpath("data", "en", "lexical.json").read_text(encoding="utf-8")
    )
    lexical["forms"]["separator.words"]["value"]["raw corpus surface"] = "not a form"
    with pytest.raises(AssertionError):
        _assert_electronic_lexical_additions_are_forms(lexical)
