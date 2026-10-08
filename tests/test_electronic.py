"""URLs, email addresses and bare domains: frend's own recognition and measured speech."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from decimal import Decimal
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
from tools.google_tn_rows import GOOGLE_TN_CLASSES


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


@pytest.mark.parametrize(
    ("loader_name", "path"),
    [
        ("load_electronic_priors", ("letters", "*")),
        ("load_electronic_priors", ("digits", "*")),
        ("load_electronic_priors", ("separators", ".")),
        ("load_electronic_span_priors", ("features", "hashtag", "classes")),
    ],
)
@pytest.mark.parametrize("bad_count", [-1, True, 1.5, None])
def test_electronic_prior_rows_refuse_invalid_counts(monkeypatch, loader_name, path, bad_count):
    loader = getattr(electronic, loader_name)
    document = deepcopy(loader())
    assert document is not None
    row = document
    for key in path:
        row = row[key]
    if bad_count is None:
        row.clear()
        row["zero"] = 0
    else:
        row[next(iter(row))] = bad_count

    table_name = (
        "electronic_priors" if loader_name == "load_electronic_priors" else "electronic_span_priors"
    )
    cached_loader = (
        electronic._electronic_priors
        if loader_name == "load_electronic_priors"
        else electronic._electronic_span_priors
    )
    monkeypatch.setattr(
        electronic,
        "measured_table",
        lambda name, _locale: document if name == table_name else None,
    )
    cached_loader.cache_clear()
    try:
        with pytest.raises(ValueError, match="count row"):
            loader()
    finally:
        cached_loader.cache_clear()


def test_valid_electronic_prior_rows_stay_normalized():
    assert sum(electronic.letter_probabilities("abc", tld=False).values()) == Decimal(1)
    assert sum(electronic.digit_probabilities("2004").values()) == Decimal(1)
    assert sum(electronic.separator_names(".").values()) == Decimal(1)


def test_letter_probabilities_refuse_malformed_mock_evidence(monkeypatch):
    monkeypatch.setattr(
        electronic,
        "load_electronic_priors",
        lambda **_kwargs: {"letters": {"*": {"word": -1, "spelled": 2}}},
    )
    with pytest.raises(ValueError, match="count row"):
        electronic.letter_probabilities("abc", tld=False)


@pytest.mark.parametrize(
    ("document", "consume"),
    [
        (
            {"letters": {"*": {}}},
            lambda: electronic.letter_probabilities("abc", tld=False),
        ),
        (
            {"letters": {"*": {"word": 1}, "lower:3:v": {}}},
            lambda: electronic.letter_probabilities("abc", tld=False),
        ),
        (
            {"letters": {"*": {"word": 1}, "tld:com": {}}},
            lambda: electronic.letter_probabilities("com", tld=True),
        ),
        (
            {"digits": {"*": {"cardinal": 1}, "4:n": {}}},
            lambda: electronic.digit_probabilities("2004"),
        ),
        (
            {"separators": {".": {}}},
            lambda: electronic.separator_names("."),
        ),
    ],
)
def test_probability_consumers_refuse_present_empty_rows(monkeypatch, document, consume):
    monkeypatch.setattr(electronic, "load_electronic_priors", lambda **_kwargs: document)
    with pytest.raises(ValueError, match="count row"):
        consume()


def test_span_consumer_refuses_present_empty_class_row(monkeypatch):
    monkeypatch.setattr(
        electronic,
        "load_electronic_span_priors",
        lambda **_kwargs: {"features": {"hashtag": {"classes": {}}}},
    )
    with pytest.raises(ValueError, match=r"features\.hashtag\.classes"):
        electronic._supported_span_features("#abc", "en_US")


def test_probability_consumers_preserve_absent_optional_rows(monkeypatch):
    monkeypatch.setattr(
        electronic,
        "load_electronic_priors",
        lambda **_kwargs: {
            "letters": {"*": {"word": 1}},
            "digits": {"*": {"cardinal": 1}},
            "separators": {},
        },
    )
    assert electronic.letter_probabilities("abc", tld=False) == {"word": Decimal(1)}
    assert electronic.digit_probabilities("2004") == {"cardinal": Decimal(1)}
    assert electronic.separator_names(".") == {}


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
    for row in document["features"].values():
        assert set(row) == {"classes"}
        assert type(row["classes"]) is dict and row["classes"]
        for name, count in row["classes"].items():
            assert name in GOOGLE_TN_CLASSES
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


def test_span_prior_privacy_guard_rejects_an_unknown_corpus_class():
    document = deepcopy(load_electronic_span_priors())
    assert document is not None
    document["features"]["hashtag"]["classes"]["RAW_CORPUS_SURFACE"] = 1
    with pytest.raises(AssertionError):
        _assert_span_prior_is_aggregate(document)


# Main's complete lexical value shape, plus this branch's documented ")" entry in
# separator.words.  ``str`` leaves are lexical forms; all container keys and lengths
# are closed so a new path cannot silently turn this hand-written table into storage.
_FORM = str
_PAIR = [_FORM, _FORM]
_CURRENCY_UNIT = [_PAIR, _PAIR]
_RANGE_PATTERNS = [{"id": _FORM, "pattern": _FORM}] * 2
_LEXICAL_VALUE_SCHEMA = {
    "range.connector": {
        "range": _RANGE_PATTERNS,
        "ratio": _RANGE_PATTERNS,
        "dimension": _RANGE_PATTERNS,
    },
    "range.separator": {"range": ["-"], "ratio": [":"], "dimension": ["x", "×"]},
    "zero.words": [_FORM, _FORM, _FORM],
    "zero.digit": _FORM,
    "zero.minute": [_FORM, _FORM],
    "zero.year": {"oh": _FORM},
    "clock.oclock": _FORM,
    "clock.hundred": _FORM,
    "possessive.suffix": _FORM,
    "separator.words": {"@": _FORM, ")": _FORM},
    "symbol_run.line": _FORM,
    "symbol_run.repeated": _FORM,
    "symbol_run.mixed_emoji": _FORM,
    "symbol_run.mixed_symbols": _FORM,
    "letter.vowels": _FORM,
    "currency.units": {
        code: _CURRENCY_UNIT
        for code in ("USD", "EUR", "GBP", "JPY", "CNY", "INR", "CAD", "AUD", "KRW", "RUB")
    },
    "currency.region_names": {"USD": _PAIR},
    "fraction.denominators": {"2": _PAIR, "4": _PAIR},
    "fraction.one": _FORM,
    "ordinal.article": _FORM,
    "date.day_first": _FORM,
    "sign.plus": _FORM,
    "numeral.plural": [
        {
            "ends": [_FORM],
            "not_after": [_FORM, _FORM, _FORM, _FORM, _FORM],
            "strip": 1,
            "add": _FORM,
        },
        {"ends": [_FORM, _FORM, _FORM, _FORM], "add": _FORM},
        {"ends": [_FORM], "add": _FORM},
    ],
    "duration.milliseconds": _FORM,
    "measure.per_plural": True,
}
_SHORT_LEXICAL_FORM = re.compile(r"(?:[^\W\d_]|[ '\-]|\{(?:[01])?\})*\Z")


def _assert_lexical_value(value, schema, path):
    if schema is _FORM:
        assert type(value) is str, path
        assert len(value) <= 64, path
        assert _SHORT_LEXICAL_FORM.fullmatch(value), path
    elif isinstance(schema, dict):
        assert type(value) is dict and set(value) == set(schema), path
        for key, child_schema in schema.items():
            _assert_lexical_value(value[key], child_schema, (*path, key))
    elif isinstance(schema, list):
        assert type(value) is list and len(value) == len(schema), path
        for index, (child, child_schema) in enumerate(zip(value, schema, strict=True)):
            _assert_lexical_value(child, child_schema, (*path, index))
    else:
        assert type(value) is type(schema) and value == schema, path


def _assert_lexical_file_contains_only_forms(document):
    assert set(document) == {"locale", "schema_version", "source", "forms"}
    assert document["locale"] == "en"
    assert type(document["schema_version"]) is int and document["schema_version"] == 1
    assert document["source"] == "frend/curated"
    assert type(document["forms"]) is dict
    assert set(document["forms"]) == set(_LEXICAL_VALUE_SCHEMA)
    for key, schema in _LEXICAL_VALUE_SCHEMA.items():
        entry = document["forms"][key]
        assert type(entry) is dict and set(entry) == {"why", "value"}, key
        assert type(entry["why"]) is str and entry["why"].strip(), key
        _assert_lexical_value(entry["value"], schema, ("forms", key, "value"))


def _lexical_document():
    return json.loads(
        files("frend").joinpath("data", "en", "lexical.json").read_text(encoding="utf-8")
    )


def test_electronic_lexical_additions_are_lexical_forms_only():
    lexical = _lexical_document()
    _assert_lexical_file_contains_only_forms(lexical)
    assert lexical["forms"]["separator.words"] == {
        "why": (
            "no CLDR spoken form for '@' in an address; every one of 1,791 training "
            "ELECTRONIC scheme URLs ending ')' says 'closing parenthesis'"
        ),
        "value": {"@": "at", ")": "closing parenthesis"},
    }


@pytest.mark.parametrize(
    "path",
    [
        ("forms", "range.connector", "excerpt"),
        ("forms", "separator.words", "value", "raw corpus surface"),
    ],
)
def test_electronic_lexical_guard_rejects_a_corpus_surface(path):
    lexical = _lexical_document()
    target = lexical
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "not a form"
    with pytest.raises(AssertionError):
        _assert_lexical_file_contains_only_forms(lexical)


@pytest.mark.parametrize("surface", ["123", "https://example.org", "x" * 65])
def test_electronic_lexical_guard_rejects_nonlexical_values(surface):
    lexical = _lexical_document()
    lexical["forms"]["zero.digit"]["value"] = surface
    with pytest.raises(AssertionError):
        _assert_lexical_file_contains_only_forms(lexical)
