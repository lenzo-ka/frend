"""Reflective, all-path verbalization against real icukit detections."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from types import SimpleNamespace

import icu
import pytest
from icukit.abbreviation_recognize import AbbreviationDetector
from icukit.detectors import NumberValue, all_detectors, detect
from icukit.recognize import (
    AlphanumericRunsDetector,
    FlexibleCompactDetector,
    FlexibleCurrencyDetector,
    FlexibleCurrencyNameDetector,
    FlexibleDateDetector,
    FlexibleFractionDetector,
    FlexibleMeasureDetector,
    FlexibleMixedMeasureDetector,
    FlexibleNumberDetector,
    FlexibleNumericDurationDetector,
    FlexibleOrdinalDetector,
    FlexiblePercentDetector,
    FlexibleTextDateDetector,
    FlexibleTimeDetector,
    PluralNumeralDetector,
)

from frend import compose_choices, resolve, resolve_choices, resolve_lattice
from frend.electronic import ElectronicDetector
from frend.spoken_priors import normalize_spoken
from frend.verbalize import (
    SpokenAlternative,
    _weekday_name,
    register_curated_alternative,
    verbalize_edge,
    verbalize_lattice,
)

DETECTORS = all_detectors("en_US", ("yMd", "Md", "y"))


def _real(text: str, type_: str):
    return next(detection for detection in DETECTORS.detect(text) if detection["type"] == type_)


def _result(text: str, type_: str):
    detection = _real(text, type_)
    return verbalize_lattice(resolve_lattice([detection], source_text=text))


def _det(text, type_, value, *, start=0, end=None):
    return {
        "type": type_,
        "value": value,
        "start": start,
        "end": len(text) if end is None else end,
        "captures": (),
    }


def _rbnf():
    return icu.RuleBasedNumberFormat(icu.URBNFRuleSetTag.SPELLOUT, icu.Locale("en_US"))


def _rule_sets():
    rbnf = _rbnf()
    return [rbnf.getRuleSetName(index) for index in range(rbnf.getNumberOfRuleSetNames())]


def _by_text(alternatives, text):
    return next(item for item in alternatives if item.text.lower() == text.lower())


def _first_forms(rule_sets):
    forms = {}
    rbnf = _rbnf()
    for name in rule_sets:
        forms.setdefault(rbnf.format(2026, name), f"icu-rbnf:{name}")
    return set(forms.items())


def test_real_year_harvests_year_and_cardinal_rule_sets_reflectively():
    result = _result("2026", "date:y")
    alternatives = result.best_path.units[0].alternatives
    actual = {(item.text, item.provenance) for item in alternatives}
    applicable = [name for name in _rule_sets() if "ordinal" not in name]
    expected = _first_forms(applicable)

    assert actual == expected
    assert result.best_path.units[0].best == alternatives[0]


def test_real_plain_cardinal_excludes_year_and_ordinal_rules_reflectively():
    result = verbalize_lattice(
        resolve_lattice([_det("2026", "number:decimal", NumberValue("2026"))], source_text="2026")
    )
    alternatives = result.best_path.units[0].alternatives
    applicable = [
        name for name in _rule_sets() if "ordinal" not in name and "numbering-year" not in name
    ]

    assert {(item.text, item.provenance) for item in alternatives} == _first_forms(applicable)
    assert all("numbering-year" not in item.provenance for item in alternatives)


def test_real_full_date_composes_bounded_field_cross_product():
    result = _result("2/29/2024", "date:yMd")
    alternatives = result.best_path.units[0].alternatives

    assert 1 < len(alternatives) <= 8
    assert any(item.text.startswith("February ") for item in alternatives)
    assert any(item.text.startswith("the twenty ninth of February ") for item in alternatives)
    assert all(not any(character.isdigit() for character in item.text) for item in alternatives)
    assert all("icu-datetime:LLLL" in item.provenance for item in alternatives)
    assert any("ordinal" in item.provenance for item in alternatives)
    assert any("numbering-year" in item.provenance for item in alternatives)


def test_one_reading_can_produce_unboundedly_many_spoken_forms():
    counts = []
    for digits in (4, 8):
        text = "1." + "0" * digits
        detection = _det(text, "number:decimal", NumberValue(Decimal(text)))
        result = verbalize_lattice(resolve_lattice([detection], source_text=text))
        counts.append(len(result.best_path.units[0].alternatives))
        assert counts[-1] == 2**digits + 1
    assert counts[1] > counts[0] * 2


def test_date_alternatives_are_never_cut_to_a_prefix(monkeypatch):
    import frend.verbalize as verbalize

    real_leaf = verbalize._number_leaf
    extra = tuple(f"synthetic year {index}" for index in range(5))

    def wider_leaf(value, kind, locale):
        alternatives = real_leaf(value, kind, locale)
        if kind != "year":
            return alternatives
        return (*alternatives, *(SpokenAlternative(text, "test:year") for text in extra))

    monkeypatch.setattr(verbalize, "_number_leaf", wider_leaf)
    texts = [item.text for item in _result("2/29/2024", "date:yMd").best_path.units[0].alternatives]

    for year in extra:
        assert any(year in text and not text.startswith("the ") for text in texts), year
        assert any(year in text and text.startswith("the ") for text in texts), year


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("25 July 2011", "the twenty fifth of july twenty eleven"),
        ("15 February 2016", "the fifteenth of february twenty sixteen"),
        ("17 July 2012", "the seventeenth of july twenty twelve"),
        ("21 September 2014", "the twenty first of september twenty fourteen"),
        ("1 February 2010", "the first of february twenty ten"),
    ],
)
def test_real_dates_include_corpus_day_first_forms(written, spoken):
    _assert_corpus_date_form(written, spoken)


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("18 September", "the eighteenth of september"),
        ("1 July", "the first of july"),
        ("July 1", "july first"),
        ("500 BC", "five hundred b c"),
        ("2000 AD", "two thousand a d"),
        ("April 4, 1904", "april fourth nineteen o four"),
    ],
)
def test_year_less_era_and_zero_led_year_dates_speak_the_corpus_form(written, spoken):
    """icukit #97 reads day-month dates and era years; the corpus says them this way."""
    detection = next(
        item
        for item in FlexibleTextDateDetector("en_US").detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    unit = verbalize_lattice(resolve_lattice([detection], source_text=written)).best_path.units[0]
    assert spoken in {
        item.text.lower().replace(",", "").replace("-", " ") for item in unit.alternatives
    }
    assert unit.unspoken == ()


def test_era_also_reads_as_the_icu_wide_name_and_a_year_without_oh_gains_no_o():
    detection = next(
        item
        for item in FlexibleTextDateDetector("en_US").detect("500 BC")
        if item["start"] == 0 and item["end"] == len("500 BC")
    )
    unit = verbalize_lattice(resolve_lattice([detection], source_text="500 BC")).best_path.units[0]
    assert ("five hundred Before Christ", "icu-rbnf:%spellout-numbering+icu-datetime:GGGG") in {
        (item.text, item.provenance) for item in unit.alternatives
    }
    plain = _full_span_forms(
        "March 2, 2014", [FlexibleTextDateDetector("en_US")], "date:text-flexible"
    )
    assert not any(" o " in f" {item.text} " for item in plain.alternatives)


def _assert_corpus_date_form(written, spoken):
    detection = next(
        item
        for item in FlexibleTextDateDetector("en_US").detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text=written))
        .best_path.units[0]
        .alternatives
    )

    assert _by_text(alternatives, spoken).weight is not None
    cardinal_year = next(
        item for item in alternatives if item.provenance.endswith("icu-rbnf:%spellout-numbering")
    )
    assert cardinal_year.weight is None


def test_curated_supplement_is_additive_and_weighted_form_ranks_first():
    lattice = resolve_lattice(
        [_det("2026", "number:decimal", NumberValue("2026"))], source_text="2026"
    )
    curated = SpokenAlternative("a sourced house style", "curated:test-guide", Decimal("0.7"))

    result = verbalize_lattice(lattice, supplements={("number:decimal", "2026"): (curated,)})
    alternatives = result.best_path.units[0].alternatives

    assert alternatives[0] == curated
    assert any(item.provenance.startswith("icu-rbnf:") for item in alternatives[1:])


def test_measured_sources_order_by_share(monkeypatch):
    from frend.spoken_priors import SourceMeasurement

    shares = {
        "curated:lower": SourceMeasurement(3, Decimal("0.3")),
        "curated:higher": SourceMeasurement(7, Decimal("0.7")),
    }
    monkeypatch.setattr(
        "frend.verbalize.source_prior", lambda kind, source, sub_key: shares.get(source)
    )

    alternatives = (
        verbalize_lattice(
            resolve_lattice([_det("42", "number:decimal", NumberValue("42"))], source_text="42"),
            supplements={
                ("number:decimal", "42"): (
                    SpokenAlternative("lower", "curated:lower"),
                    SpokenAlternative("higher", "curated:higher"),
                )
            },
        )
        .best_path.units[0]
        .alternatives
    )

    measured = [item for item in alternatives if item.weight is not None]
    assert [(item.text, item.weight) for item in measured] == [
        ("higher", Decimal("0.7")),
        ("lower", Decimal("0.3")),
    ]


def test_unmeasured_lexical_form_outranks_unmeasured_icu(monkeypatch):
    monkeypatch.setattr("frend.verbalize.source_prior", lambda kind, source, sub_key: None)
    supplements = {
        ("number:decimal", "42"): (SpokenAlternative("house forty-two", "curated:test"),)
    }

    alternatives = (
        verbalize_lattice(
            resolve_lattice([_det("42", "number:decimal", NumberValue("42"))], source_text="42"),
            supplements=supplements,
        )
        .best_path.units[0]
        .alternatives
    )

    assert alternatives[0].provenance == "curated:test"
    assert alternatives[0].weight is None


def test_measured_icu_form_outranks_unmeasured_lexical(monkeypatch):
    from frend.spoken_priors import SourceMeasurement

    monkeypatch.setattr(
        "frend.verbalize.source_prior",
        lambda kind, source, sub_key: (
            SourceMeasurement(1, Decimal("0.1")) if source.startswith("icu-rbnf:") else None
        ),
    )
    supplements = {
        ("number:decimal", "42"): (SpokenAlternative("house forty-two", "curated:test"),)
    }

    alternatives = (
        verbalize_lattice(
            resolve_lattice([_det("42", "number:decimal", NumberValue("42"))], source_text="42"),
            supplements=supplements,
        )
        .best_path.units[0]
        .alternatives
    )

    assert alternatives[0].provenance.startswith("icu-rbnf:")
    assert alternatives[0].weight == Decimal("0.1")


def test_caller_supplied_weight_wins_over_measured_source(monkeypatch):
    from frend.spoken_priors import SourceMeasurement

    monkeypatch.setattr(
        "frend.verbalize.source_prior",
        lambda kind, source, sub_key: SourceMeasurement(1, Decimal("1")),
    )
    registry = {}
    monkeypatch.setattr("frend.verbalize._CURATED", registry)
    register_curated_alternative(
        "number:decimal", "42", "house forty-two", "test", weight=Decimal("0.01")
    )

    alternatives = (
        verbalize_lattice(
            resolve_lattice([_det("42", "number:decimal", NumberValue("42"))], source_text="42")
        )
        .best_path.units[0]
        .alternatives
    )

    assert alternatives[0] == SpokenAlternative("house forty-two", "curated:test", Decimal("0.01"))


def test_rules_without_decimal_separator_degrade_without_passing_float(monkeypatch):
    seen = []

    class RejectingFormatter:
        def format(self, value, ruleset):
            seen.append((value, ruleset))
            if isinstance(value, Decimal):
                raise TypeError("Decimal rejected")
            return str(value)

        def getRules(self):
            return ""

    monkeypatch.setattr("frend.verbalize._spellout_formatter", lambda locale: RejectingFormatter())
    result = verbalize_lattice(
        resolve_lattice(
            [_det("0.125", "number:decimal", NumberValue("0.125"))], source_text="0.125"
        )
    )

    assert result.best_path.spoken == "0.125"
    assert result.best_path.units[0].verbalized is False
    assert seen and seen[0][0] == 0
    assert not any(isinstance(value, float) for value, _ in seen)


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("3.5", "three point five"),
        ("2.5", "two point five"),
        ("1.5", "one point five"),
        ("4.5", "four point five"),
        ("1.6", "one point six"),
    ],
)
def test_real_decimals_include_corpus_spoken_forms(written, spoken):
    detection = next(
        item
        for item in FlexibleNumberDetector("en_US").detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text=written))
        .best_path.units[0]
        .alternatives
    )

    assert _by_text(alternatives, spoken).weight is not None


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        (".1", "point one"),
        (".2", "point two"),
        (".3", "point three"),
        (".0", "point o"),
        ("3.00", "three point o o"),
    ],
)
def test_real_decimals_include_leading_point_and_o_forms(written, spoken):
    detection = next(
        item
        for item in FlexibleNumberDetector("en_US").detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text=written))
        .best_path.units[0]
        .alternatives
    )

    assert _by_text(alternatives, spoken).weight is not None


def test_decimal_without_captures_keeps_zero_integer():
    result = verbalize_lattice(
        resolve_lattice(
            [_det(".5", "number:decimal", NumberValue("0.5"))],
            source_text=".5",
        )
    )

    assert "zero point five" in {item.text for item in result.best_path.units[0].alternatives}


def test_integral_decimal_keeps_spoken_digits_and_collapse():
    detection = next(iter(FlexibleNumberDetector("en_US").detect("2.0")))
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text="2.0"))
        .best_path.units[0]
        .alternatives
    )

    assert {"two point zero", "two"} <= {item.text for item in alternatives}


def test_negative_decimal_keeps_the_sign_without_float_conversion():
    detection = next(iter(FlexibleNumberDetector("en_US").detect("-2.5")))
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text="-2.5"))
        .best_path.units[0]
        .alternatives
    )

    assert "minus two point five" in {item.text for item in alternatives}


def test_real_flexible_fraction_composes_captured_numeric_leaves():
    detectors = DETECTORS.with_(FlexibleFractionDetector("en_US"))
    detection = next(d for d in detectors.detect("1/2") if d["type"] == "fraction:flexible")
    result = verbalize_lattice(resolve_lattice([detection], source_text="1/2"))

    assert len(result.best_path.units[0].alternatives) >= 1
    assert result.best_path.spoken == "one half"
    assert result.best_path.units[0].verbalized is True


def test_fraction_denominator_measurement_ranks_half_above_second():
    detection = next(
        item
        for item in FlexibleFractionDetector("en_US").detect("1/2")
        if item["start"] == 0 and item["end"] == 3
    )
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text="1/2"))
        .best_path.units[0]
        .alternatives
    )

    assert alternatives[0].text == "one half"
    assert alternatives[0].weight > Decimal("0.98")
    second = _by_text(alternatives, "one second")
    assert second.weight is None or second.weight < Decimal("0.02")


def test_unattested_denominator_uses_kind_level_shares_for_every_alternative(monkeypatch):
    from frend.spoken_priors import SpokenPriorTable

    ordinal = "icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-ordinal"
    over = "icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-numbering"
    table = SpokenPriorTable(
        {
            "fraction": {
                "total": 4,
                "matched": 4,
                "unmatched": 0,
                "source_matched": {ordinal: 3, over: 1},
                "sub_keys": {"2": {"matched": 4, "source_matched": {ordinal: 3, over: 1}}},
                "unmatched_by_reason": {
                    "unrecognized": 0,
                    "unverbalized": 0,
                    "no_alternative_matched": 0,
                },
            }
        },
        {"sub_key_rules": {"fraction": "denominator"}},
    )
    monkeypatch.setattr("frend.verbalize.source_prior", table.lookup)
    written = "1/999"
    detection = next(
        item
        for item in FlexibleFractionDetector("en_US").detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text=written))
        .best_path.units[0]
        .alternatives
    )

    assert [(item.provenance, item.weight) for item in alternatives[:2]] == [
        (ordinal, Decimal("0.75")),
        (over, Decimal("0.25")),
    ]
    assert all(item.weight is None for item in alternatives[2:])


def test_sparse_denominator_blends_its_one_row_toward_the_kind_share():
    """One observation is evidence, not a certainty (kal, 2026-09-23)."""
    from frend.spoken_priors import SUB_KEY_PRIOR_STRENGTH, load_spoken_prior_table, source_prior

    written = "1/103"
    ordinal = "icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-ordinal"
    table = load_spoken_prior_table()
    kind_share = source_prior("fraction", ordinal).share
    sub_matched, sub_sources = table._sub_keys["fraction"]["103"]
    measurement = source_prior("fraction", ordinal, "103")
    detection = next(
        item
        for item in FlexibleFractionDetector("en_US").detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text=written))
        .best_path.units[0]
        .alternatives
    )

    assert (sub_matched, sub_sources.get(ordinal)) == (1, 1)
    assert measurement.count == 1
    assert measurement.share == (1 + SUB_KEY_PRIOR_STRENGTH * kind_share) / (
        1 + SUB_KEY_PRIOR_STRENGTH
    )
    assert kind_share < measurement.share < 1
    assert alternatives[0].provenance == ordinal
    assert alternatives[0].weight == measurement.share
    # Sources the one row never saw are no longer unmeasured: they keep a kind-derived share.
    later = [item.weight for item in alternatives[1:] if item.weight is not None]
    assert later and all(weight < measurement.share for weight in later)


def test_well_attested_denominator_stays_close_to_its_own_share():
    from frend.spoken_priors import source_prior

    lexical = "icu-rbnf:%spellout-numbering+lexical:en_US"
    blended = source_prior("fraction", lexical, "2")

    assert blended.count == 283
    assert Decimal("0.98") < blended.share < Decimal(1)


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("1/2", "one half"),
        ("3/4", "three quarters"),
        ("24/7", "twenty four sevenths"),
        ("2/3", "two thirds"),
        ("2 1/2", "two and a half"),
        ("243/2014", "two hundred forty three two thousand fourteenths"),
        ("50/50", "fifty fiftieths"),
    ],
)
def test_real_fractions_include_corpus_spoken_forms(written, spoken):
    detection = next(
        item
        for item in FlexibleFractionDetector("en_US").detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    result = verbalize_lattice(resolve_lattice([detection], source_text=written))
    alternatives = result.best_path.units[0].alternatives

    corpus = _by_text(alternatives, spoken)
    over = next(item for item in alternatives if " over " in item.text)
    assert corpus.weight is not None
    assert over.weight is None or over.weight < corpus.weight


def test_fraction_keeps_regular_and_over_alternatives():
    detection = next(iter(FlexibleFractionDetector("en_US").detect("1/4")))
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text="1/4"))
        .best_path.units[0]
        .alternatives
    )

    assert {"one quarter", "one fourth", "one over four"} <= {item.text for item in alternatives}


@pytest.mark.parametrize(
    ("written", "spoken", "competitor"),
    [
        ("-1/2", "minus one half", "minus one over two"),
        ("-54/55", "minus fifty four fifty fifths", "minus fifty four over fifty-five"),
        ("-5/8", "minus five eighths", "minus five over eight"),
        (
            "-707/13",
            "minus seven hundred seven thirteenths",
            "minus seven hundred seven over thirteen",
        ),
        ("-15/1", "minus fifteen over one", "minus fifteen firsts"),
    ],
)
def test_real_fractions_include_corpus_negative_forms(written, spoken, competitor):
    detection = next(
        item
        for item in FlexibleFractionDetector("en_US").detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text=written))
        .best_path.units[0]
        .alternatives
    )

    corpus = _by_text(alternatives, spoken)
    rival = _by_text(alternatives, competitor)
    assert corpus.weight is not None
    assert rival.weight is None or rival.weight < corpus.weight


@pytest.mark.parametrize(
    ("written", "currency", "spoken"),
    [
        ("$100,000", "USD", "one hundred thousand dollars"),
        ("$10,000", "USD", "ten thousand dollars"),
        ("$500,000", "USD", "five hundred thousand dollars"),
        ("$1,000", "USD", "one thousand dollars"),
        ("$1.00", "USD", "one dollar"),
    ],
)
def test_real_money_includes_corpus_spoken_forms(written, currency, spoken):
    detection = next(
        item
        for item in FlexibleCurrencyDetector("en_US", currency).detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    result = verbalize_lattice(resolve_lattice([detection], source_text=written))
    alternatives = result.best_path.units[0].alternatives

    assert _by_text(alternatives, spoken).weight is not None


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("5 million", "five million"),
        ("$1.5 million", "one point five million dollars"),
        ("$1.3 billion", "one point three billion dollars"),
        ("$2.8 million", "two point eight million dollars"),
        ("$2.5M", "two point five million dollars"),
    ],
)
def test_real_compacts_include_corpus_spoken_forms(written, spoken):
    detectors = (
        FlexibleCurrencyDetector("en_US", "USD")
        if written.startswith("$")
        else FlexibleCompactDetector("en_US", "long")
    )
    detection = next(
        item
        for item in detectors.detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text=written))
        .best_path.units[0]
        .alternatives
    )

    expected = _by_text(alternatives, spoken)
    if written == "5 million":
        assert expected.weight is None
    else:
        assert expected.weight is not None
        wide_name = next(item for item in alternatives if item.text.endswith("US dollars"))
        assert wide_name.weight is None


@pytest.mark.parametrize(
    ("written", "corpus_spoken", "icu_spoken"),
    [
        (
            "EC$1.3 billion",
            "one point three billion dollars",
            "one point three billion East Caribbean dollars",
        ),
        (
            "EC$6.5 million",
            "six point five million dollars",
            "six point five million East Caribbean dollars",
        ),
    ],
)
def test_xcd_compacts_keep_icu_wide_name(written, corpus_spoken, icu_spoken):
    detection = next(
        item
        for item in FlexibleCurrencyDetector("en_US", "XCD").detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    unit = verbalize_lattice(resolve_lattice([detection], source_text=written)).best_path.units[0]

    assert unit.verbalized
    assert icu_spoken in {item.text for item in unit.alternatives}
    assert icu_spoken.startswith(corpus_spoken.removesuffix(" dollars"))


@pytest.mark.parametrize(
    ("written", "currency", "spoken"),
    [
        ("$9.50", "USD", "nine dollars and fifty cents"),
        ("$2.93", "USD", "two dollars and ninety three cents"),
        ("£1.05", "GBP", "one pound and five pence"),
        ("£13.15", "GBP", "thirteen pounds and fifteen pence"),
        ("€49.95", "EUR", "forty nine euros and ninety five cents"),
    ],
)
def test_real_money_includes_corpus_minor_unit_forms(written, currency, spoken):
    detection = next(
        item
        for item in FlexibleCurrencyDetector("en_US", currency).detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text=written))
        .best_path.units[0]
        .alternatives
    )

    assert spoken in {item.text for item in alternatives}


def test_unknown_lexical_currency_keeps_icu_wide_name():
    rendered = icu.MeasureFormat(icu.Locale("en_US"), icu.UMeasureFormatWidth.WIDE).formatMeasure(
        icu.Measure(2, icu.CurrencyUnit("CHF"))
    )
    expected = f"{_rbnf().format(2, '%spellout-numbering')} {rendered.split(' ', 1)[1]}"
    result = verbalize_lattice(
        resolve_lattice(
            [_det("CHF 2", "number:currency:CHF", NumberValue("2", currency="CHF"))],
            source_text="CHF 2",
        )
    )

    assert expected in {item.text for item in result.best_path.units[0].alternatives}


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("2016 USD", "two thousand sixteen united states dollars"),
        ("2003 USD", "two thousand three united states dollars"),
        ("1990 USD", "one thousand nine hundred ninety united states dollars"),
        ("2 USD", "two united states dollars"),
        ("1 USD", "one united states dollar"),
    ],
)
def test_real_usd_names_include_region_expanded_forms(written, spoken):
    detection = next(
        item
        for item in FlexibleCurrencyNameDetector("en_US", "USD").detect(written)
        if item["start"] == 0 and item["end"] == len(written)
    )
    alternatives = (
        verbalize_lattice(resolve_lattice([detection], source_text=written))
        .best_path.units[0]
        .alternatives
    )

    assert _by_text(alternatives, spoken).weight is not None


@pytest.mark.parametrize(
    ("written", "decimal", "currency", "expected", "truncated"),
    [
        ("¥8.5", "8.5", "JPY", "eight point five yen", "eight yen"),
        (
            "$1.505",
            "1.505",
            "USD",
            "one point five zero five dollars",
            "one dollar and fifty cents",
        ),
    ],
)
def test_money_beyond_currency_precision_uses_exact_decimal(
    written, decimal, currency, expected, truncated
):
    result = verbalize_lattice(
        resolve_lattice(
            [_det(written, f"number:currency:{currency}", NumberValue(decimal, currency=currency))],
            source_text=written,
        )
    )
    alternatives = {item.text for item in result.best_path.units[0].alternatives}

    assert expected in alternatives
    assert truncated not in alternatives


def test_negative_subunit_money_keeps_minus_prefix():
    result = verbalize_lattice(
        resolve_lattice(
            [_det("-$0.50", "number:currency:USD", NumberValue("-0.50", currency="USD"))],
            source_text="-$0.50",
        )
    )

    assert "minus zero dollars and fifty cents" in {
        item.text for item in result.best_path.units[0].alternatives
    }


def test_table_currency_with_zero_minor_units_still_collapses():
    result = verbalize_lattice(
        resolve_lattice(
            [_det("$1.00", "number:currency:USD", NumberValue("1.00", currency="USD"))],
            source_text="$1.00",
        )
    )

    assert "one dollar" in {item.text for item in result.best_path.units[0].alternatives}


def test_every_ranked_path_and_per_unit_alternatives_are_preserved():
    text = "12"
    detections = [_real(text, "number:decimal"), _det(text, "unsupported:reading", None)]
    lattice = resolve_lattice(detections, source_text=text, output_cap=2)
    result = verbalize_lattice(lattice)

    assert tuple(path.rank for path in result.paths) == tuple(path.rank for path in lattice.paths)
    assert tuple(path.edge_ids for path in result.paths) == tuple(
        path.edge_ids for path in lattice.paths
    )
    assert all(
        path.alternatives == tuple(unit.alternatives for unit in path.units)
        for path in result.paths
    )
    assert result.best_path.rank == lattice.best_path.rank
    assert result.ambiguous == lattice.ambiguous
    unsupported = next(unit for path in result.paths for unit in path.units if not unit.verbalized)
    assert unsupported.best.text == text


def test_primary_path_readout_spaces_units_instead_of_running_them_together():
    text = "2024-02"
    detections = [
        _det(text, "number:decimal", NumberValue("2024"), start=0, end=4),
        _det(text, "number:decimal", NumberValue("2"), start=5, end=7),
    ]
    result = verbalize_lattice(resolve_lattice(detections, source_text=text))

    assert "-0" not in result.best_path.spoken
    assert " - " in result.best_path.spoken
    assert result.best_path.spoken == " ".join(unit.best.text for unit in result.best_path.units)


def test_units_trace_to_edge_prior_provenance():
    detection = _real("7", "number:decimal")
    lattice = resolve_lattice(
        [detection],
        source_text="7",
        class_prior={"number:decimal": Decimal("1")},
        class_prior_source="test-prior",
    )
    result = verbalize_lattice(lattice)
    unit = result.best_path.units[0]
    edge = next(edge for edge in lattice.edges if edge.id == unit.edge_id)

    assert edge.prior is not None
    assert (unit.tier, unit.provenance) == (edge.prior.tier, edge.prior.provenance)


def test_verbalization_is_deterministic_and_frozen():
    lattice = resolve_lattice([_real("42", "number:decimal")], source_text="42")
    first = verbalize_lattice(lattice)
    second = verbalize_lattice(lattice)

    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.paths = ()


def _units(text, detectors):
    """Every non-passthrough unit of the keep-all graph over ``text``."""
    lattice = resolve_choices(list(detect(text, detectors)), source_text=text)
    return [
        unit
        for unit in compose_choices(lattice).units
        if unit.alternatives[0].provenance != "surface:passthrough"
    ]


def _widest_date(text, detectors):
    dates = [u for u in _units(text, detectors) if u.verbalized]
    return max(dates, key=lambda unit: len(unit.alternatives[0].text))


@pytest.mark.parametrize(
    ("written", "plain", "detectors", "day"),
    [
        (
            "Monday, March 16, 1908",
            "March 16, 1908",
            [FlexibleTextDateDetector("en_US")],
            "Monday",
        ),
        (
            "Wednesday, March 18, 1908",
            "March 18, 1908",
            list(all_detectors("en_US", ("yMMMMEEEEd", "yMMMMd")).detectors),
            "Wednesday",
        ),
    ],
)
def test_written_weekday_leads_every_date_form(written, plain, detectors, day):
    """Both icukit weekday encodings, ICU's number and a name, reach speech."""
    with_day = _widest_date(written, detectors)
    without = _widest_date(plain, detectors)

    assert [a.text for a in with_day.alternatives] == [
        f"{day}, {a.text}" for a in without.alternatives
    ]
    assert all(a.provenance.startswith("icu-datetime:EEEE+") for a in with_day.alternatives)
    assert with_day.unspoken == ()


def test_every_icu_weekday_number_and_name_is_spoken():
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    for number, name in enumerate(days, 1):
        assert _weekday_name(SimpleNamespace(value=number), "gregorian", "en_US").text == name
        assert _weekday_name(SimpleNamespace(value=name.lower()), "gregorian", "en_US").text == name
    for bad in (0, 8, True, "funday", None):
        with pytest.raises(ValueError, match="weekday"):
            _weekday_name(SimpleNamespace(value=bad), "gregorian", "en_US")


def test_written_plus_sign_leads_with_plus_and_keeps_the_unsigned_form():
    number = [FlexibleNumberDetector("en_US")]
    (plus,) = [u for u in _units("+5", number) if u.verbalized]
    (minus,) = [u for u in _units("-5", number) if u.verbalized]

    assert [a.text for a in plus.alternatives] == ["plus five", "five"]
    assert plus.alternatives[0].provenance.startswith("lexical:en_US+")
    assert plus.unspoken == ()
    assert [a.text for a in minus.alternatives] == ["minus five"]
    assert minus.unspoken == ()


def test_a_capture_no_verbalizer_speaks_is_recorded_not_dropped():
    (detection,) = FlexibleNumberDetector("en_US").detect("5")
    extra = replace(detection["captures"][0], name="era", text="5")
    marked = {**detection, "captures": (*detection["captures"], extra)}
    lattice = resolve_choices([marked], source_text="5")
    (unit,) = [u for u in compose_choices(lattice).units if u.verbalized and u.unspoken]

    assert unit.unspoken == (extra,)
    assert unit.alternatives[0].text == "five"


_COVERAGE = [
    "5",
    "-5",
    "+5",
    "3.14",
    "-2.5",
    ".5",
    "12%",
    "$1.50",
    "£13.15",
    "$5 million",
    "30B",
    "1/2",
    "-1/2",
    "3 2/4",
    "½",
    "XIV",
    "2/29/2024",
    "11/17",
    "1908",
    "March 16, 1908",
    "Monday, March 16, 1908",
    "Wednesday, March 18, 1908",
    "21 September 2014",
    "july 25 2012",
    "The 29th very foggy.",
    "the 1st and 22nd",
    "Dr. Smith on Elm Dr.",
    "etc.",
    "5pm",
    "9:00 p.m.",
    "10:05",
    "1990s",
    "'90s",
    "1990's",
    "3D",
    "2Q22",
    "II's",
    "10 PM ET",
    "18:00 UTC",
    "V.",
    "XIVth",
    "500 BC",
    "1 July",
    "18 September",
    "60 km",
    "578.3/km2",
    "5'10\"",
    "79.20%",
    "boston.com",
    "jane.doe@example.org",
    "http://www.ucc.ie/celt/trotula.html",
    "1:47.22",
    "1:02:03",
    "/km²",
]


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("0.75", ["zero point seven five", "point seven five"]),
        (".75", ["point seven five"]),
        ("-0.75", ["minus zero point seven five"]),
        ("10.5", ["ten point five"]),
    ],
)
def test_a_written_leading_zero_may_go_unsaid(written, expected):
    (unit,) = [
        u
        for u in _units(written, [FlexibleNumberDetector("en_US")])
        if u.verbalized and u.alternatives[0].text.endswith("five")
    ]

    assert [a.text for a in unit.alternatives] == expected


def test_written_ordinal_is_spoken_as_an_ordinal():
    (unit,) = [u for u in _units("29th", [FlexibleOrdinalDetector("en_US")]) if u.verbalized]

    assert [a.text for a in unit.alternatives] == ["twenty-ninth"]
    assert all("spellout-ordinal" in a.provenance for a in unit.alternatives)
    assert unit.unspoken == ()


@pytest.mark.parametrize(
    ("written", "expected"),
    [("V.", {"fifth", "the fifth"}), ("XIVth", {"fourteenth", "the fourteenth"})],
)
def test_roman_ordinal_also_reads_with_the(written, expected):
    """The corpus reads "V." as "the fifth"; a written arabic ordinal gets no "the"."""
    unit = _full_span_forms(written, [FlexibleOrdinalDetector("en_US")], "ordinal:flexible")
    assert {a.text for a in unit.alternatives} == expected
    assert unit.unspoken == ()


def test_dot_time_competes_with_the_decimal_and_loses_on_both_layers():
    """icukit reads "3.14" as a time too; the decimal wins by captures and by the prior.

    The time reading carries fewer captures, so geometry leaves it off the 1-best, and
    the corpus prior for the shape agrees (TIME is a fraction of a percent of "N.N").
    Should the captures ever tie, the prior still decides for the decimal.
    """
    text = "3.14"
    detections = detect(text, [FlexibleNumberDetector("en_US"), FlexibleTimeDetector("en_US")])
    priors = {
        edge.detection["type"]: edge.prior.p
        for edge in resolve_lattice(detections, source_text=text).edges
        if edge.kind == "reading"
    }

    assert set(priors) == {"number:decimal", "time:flexible"}
    assert priors["number:decimal"] > 100 * priors["time:flexible"]
    assert [d["type"] for d in resolve(detections, source_text=text).best] == ["number:decimal"]


def test_written_ordinal_is_the_only_reading_of_its_token():
    """kal: "29th" dominates "29". Recognition no longer reads digits inside a word,
    so the ordinal is the token's only reading and the 1-best; a bare "29" returning
    here would be the in-word reading icukit removed."""
    text = "The 29th very foggy."
    detections = detect(text, [FlexibleNumberDetector("en_US"), FlexibleOrdinalDetector("en_US")])

    assert [(d["type"], text[d["start"] : d["end"]]) for d in detections] == [
        ("ordinal:flexible", "29th")
    ]
    assert [d["type"] for d in resolve(detections, source_text=text).best] == ["ordinal:flexible"]


def test_abbreviation_keeps_every_expansion_with_its_sense():
    (unit,) = [u for u in _units("Dr.", [AbbreviationDetector("en_US")]) if u.verbalized]

    assert [(a.text, a.provenance) for a in unit.alternatives] == [
        ("Doctor", "icukit-abbreviation:title/precedes-name"),
        ("Drive", "icukit-abbreviation:thoroughfare/address"),
    ]
    assert unit.unspoken == ()


@pytest.mark.parametrize("text", ["John Smith, MD, said", "in my amateur MD way"])
def test_md_competes_as_spelled_letters_a_state_and_a_roman_numeral(text):
    """kal: MD is the doctor said "M D", Maryland, or 1500; all three stay in the lattice."""
    units = [
        u
        for u in _units(text, [AbbreviationDetector("en_US"), FlexibleNumberDetector("en_US")])
        if u.verbalized
    ]
    forms = {(a.text, a.provenance) for u in units for a in u.alternatives}

    assert ("M D", "icukit-spell-out:title/follows-name") in forms
    assert ("Maryland", "icukit-abbreviation:region/address") in forms
    assert ("one thousand five hundred", "icu-rbnf:%spellout-numbering") in forms
    assert all(u.unspoken == () for u in units)


def test_abbreviation_without_an_expansion_speaks_its_surface_unverbalized():
    (unit,) = _units("Ms.", [AbbreviationDetector("en_US")])

    assert unit.verbalized is False
    assert [a.text for a in unit.alternatives] == ["Ms."]


def test_no_capture_goes_unspoken_across_the_recognition_profile():
    """A tripwire: a capture that no verbalizer speaks fails here, by name."""
    detectors = [
        FlexibleNumberDetector("en_US"),
        FlexibleCompactDetector("en_US", "long"),
        FlexibleCompactDetector("en_US", "short"),
        FlexibleCurrencyDetector("en_US", "USD"),
        FlexibleCurrencyDetector("en_US", "GBP"),
        FlexibleCurrencyNameDetector("en_US", "USD"),
        FlexiblePercentDetector("en_US"),
        FlexibleFractionDetector("en_US"),
        FlexibleDateDetector("en_US"),
        FlexibleTextDateDetector("en_US"),
        FlexibleOrdinalDetector("en_US"),
        AbbreviationDetector("en_US"),
        FlexibleTimeDetector("en_US"),
        PluralNumeralDetector("en_US"),
        AlphanumericRunsDetector("en_US"),
        FlexibleMeasureDetector("en_US", "kilometer"),
        FlexibleMeasureDetector("en_US", "square-kilometer"),
        FlexibleMixedMeasureDetector("en_US", "foot-and-inch"),
        ElectronicDetector("en_US"),
        FlexibleNumericDurationDetector("en_US"),
        *all_detectors("en_US", ("yMd", "Md", "y", "yMMMMEEEEd", "yMMMMd")).detectors,
    ]
    unspoken = {
        (text, capture.name, capture.text)
        for text in _COVERAGE
        for unit in _units(text, detectors)
        if unit.verbalized
        for capture in unit.unspoken
    }
    assert unspoken == set()


def _full_span_forms(text, detectors, type_):
    lattice = resolve_choices(list(detect(text, detectors)), source_text=text)
    graph = compose_choices(lattice)
    edges = {edge.id: edge for edge in lattice.edges}
    (unit,) = [
        u
        for u in graph.units
        if edges[u.edge_id].kind == "reading"
        and edges[u.edge_id].detection["type"] == type_
        and (edges[u.edge_id].start, edges[u.edge_id].end) == (0, len(text))
    ]
    return unit


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("5pm", {"five p m", "five o'clock p m"}),
        ("12am", {"twelve a m", "twelve o'clock a m"}),
        ("9:00 p.m.", {"nine p m", "nine o'clock p m"}),
        ("10:05", {"ten oh five", "ten o five"}),
        ("20:00", {"twenty", "twenty o'clock", "twenty hundred"}),
        ("14:30", {"fourteen thirty"}),
        ("10 PM ET", {"ten p m e t", "ten o'clock p m e t"}),
        ("4:44pm EST", {"four forty-four p m e s t"}),
        ("18:00 UTC", {"eighteen u t c", "eighteen o'clock u t c", "eighteen hundred u t c"}),
    ],
)
def test_time_speaks_the_written_hour_minutes_and_period(written, expected):
    """The corpus reads "5pm" as "five p m", "20:00" as "twenty hundred", and a written
    time zone as its letters after the time ("10 PM ET" "ten p m e t")."""
    unit = _full_span_forms(written, [FlexibleTimeDetector("en_US")], "time:flexible")
    assert {a.text for a in unit.alternatives} == expected
    assert unit.unspoken == ()


def test_time_with_seconds_stays_unverbalized():
    unit = _full_span_forms("2:04:23", [FlexibleTimeDetector("en_US")], "time:flexible")
    assert unit.verbalized is False


def test_measured_time_ranks_the_corpus_form_first():
    """With TIME measured, "five p m" outranks the lexical "o'clock" form."""
    unit = _full_span_forms("5pm", [FlexibleTimeDetector("en_US")], "time:flexible")
    assert unit.alternatives[0].text == "five p m"
    assert unit.alternatives[0].weight is not None


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("1990s", "nineteen nineties"),
        ("1990's", "nineteen nineties"),
        ("'90s", "nineties"),
        ("1900s", "nineteen hundreds"),
        ("20s", "twenties"),
        ("100s", "one hundreds"),
    ],
)
def test_plural_numeral_speaks_the_corpus_decade(written, spoken):
    unit = _full_span_forms(written, [PluralNumeralDetector("en_US")], "number:plural")
    assert spoken in [a.text for a in unit.alternatives]
    assert unit.unspoken == ()


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("3D", "three d"),
        ("2Q22", "two q twenty-two"),
        ("MP3", "m p three"),
        ("1080p", "ten eighty p"),
        ("A007", "a zero zero seven"),
    ],
)
def test_letter_digit_token_speaks_as_its_runs(written, spoken):
    unit = _full_span_forms(written, [AlphanumericRunsDetector("en_US")], "alnum:runs")
    assert spoken in [a.text for a in unit.alternatives]
    assert unit.unspoken == ()


def test_roman_numeral_reads_as_cardinal_or_ordinal_and_keeps_a_possessive():
    number = [FlexibleNumberDetector("en_US")]
    plain = _full_span_forms("II", number, "number:cardinal:roman")
    possessive = _full_span_forms("II's", number, "number:cardinal:roman")

    assert {"two", "second", "the second"} <= {a.text for a in plain.alternatives}
    assert {"two's", "the second's"} <= {a.text for a in possessive.alternatives}
    assert possessive.unspoken == ()


def _measure_forms(written, detector, type_):
    unit = _full_span_forms(written, [detector], type_)
    return unit, [normalize_spoken(item.text) for item in unit.alternatives]


@pytest.mark.parametrize(
    ("written", "unit", "spoken"),
    [
        ("60 km", "kilometer", "sixty kilometers"),
        ("1 km", "kilometer", "one kilometer"),
        ("1GB", "gigabyte", "one gigabyte"),
        ("26.7 mi", "mile", "twenty six point seven miles"),
        ("60 km/h", "kilometer-per-hour", "sixty kilometers per hour"),
        (
            "578.3/km2",
            "square-kilometer",
            "five hundred seventy eight point three per square kilometers",
        ),
    ],
)
def test_measure_speaks_the_amount_and_icus_wide_unit(written, unit, spoken):
    """icukit #97 reads measures; ICU names the unit in the plural the amount selects."""
    detection_unit, forms = _measure_forms(
        written, FlexibleMeasureDetector("en_US", unit), f"measure:{unit}"
    )
    assert spoken in forms
    assert detection_unit.unspoken == ()


def test_measured_rate_ranks_the_corpus_plural_after_per_first():
    """The unit sub-key lets a rate learn "per square kilometers" without every unit taking it."""
    rate, rate_forms = _measure_forms(
        "578.3/km2",
        FlexibleMeasureDetector("en_US", "square-kilometer"),
        "measure:square-kilometer",
    )
    plain, plain_forms = _measure_forms(
        "60 km", FlexibleMeasureDetector("en_US", "kilometer"), "measure:kilometer"
    )
    assert rate_forms[0].endswith("per square kilometers")
    assert rate.alternatives[0].weight is not None
    assert plain_forms == ["sixty kilometers"]


def test_mixed_measure_speaks_each_component_joined_as_icu_joins_units():
    unit, forms = _measure_forms(
        "5'10\"", FlexibleMixedMeasureDetector("en_US", "foot-and-inch"), "measure:foot-and-inch"
    )
    assert forms == ["five feet ten inches"]
    assert unit.unspoken == ()


def test_percent_reads_its_written_fraction_digits():
    """The value 0.792 has lost the written zero of "79.20%"; the captures keep it."""
    unit, forms = _measure_forms("79.20%", FlexiblePercentDetector("en_US"), "number:percent")
    assert "seventy nine point two o percent" in forms
    assert unit.unspoken == ()


def _duration_forms(written):
    detections = [
        d
        for d in FlexibleNumericDurationDetector("en_US").detect(written)
        if d["start"] == 0 and d["end"] == len(written)
    ]
    units = {}
    for detection in detections:
        lattice = resolve_lattice([detection], source_text=written)
        edge = next(e for e in lattice.edges if e.kind == "reading")
        unit = verbalize_edge(edge, source_text=written)
        units[detection["value"].unit] = unit
    return units


def test_race_time_speaks_the_corpus_form_with_milliseconds():
    """The corpus reads "31:18.85" as minutes, seconds "and eighty five milliseconds"."""
    unit = _duration_forms("31:18.85")["second"]
    forms = [normalize_spoken(a.text) for a in unit.alternatives]
    assert "thirty one minutes eighteen seconds and eighty five milliseconds" in forms
    assert "thirty one minutes eighteen point eight five seconds" in forms
    assert unit.unspoken == ()


def test_clock_like_duration_reads_both_ways_and_minutes_take_no_milliseconds():
    """ "2:30" is hours and minutes or minutes and seconds; a fraction of a minute is a
    decimal, never milliseconds."""
    units = _duration_forms("2:30")
    assert "two hours thirty minutes" in [
        normalize_spoken(a.text) for a in units["minute"].alternatives
    ]
    assert "two minutes thirty seconds" in [
        normalize_spoken(a.text) for a in units["second"].alternatives
    ]
    minutes = [normalize_spoken(a.text) for a in _duration_forms("1:47.22")["minute"].alternatives]
    assert not any("milliseconds" in form for form in minutes)


def _era_unit(written, era, year, era_text, form):
    from icukit.detectors import Capture, DateTimeValue

    start = written.index(era_text)
    detection = {
        "text": written,
        "start": 0,
        "end": len(written),
        "type": "date:text-flexible",
        "value": DateTimeValue((("G", era), ("y", year)), "gregorian"),
        "captures": (
            Capture("y", 0, len(str(year)), str(year), year, "numeric"),
            Capture("era", start, start + len(era_text), era_text, None, form),
        ),
    }
    lattice = resolve_lattice([detection], source_text=written)
    edge = next(e for e in lattice.edges if e.kind == "reading")
    return verbalize_edge(edge, source_text=written)


@pytest.mark.parametrize(
    ("written", "era", "year", "era_text", "form", "expected"),
    [
        ("300 Before Christ", 0, 300, "Before Christ", "wide", ["three hundred before christ"]),
        ("5 Common Era", 1, 5, "Common Era", "wide", ["five common era"]),
        (
            "300 BCE",
            0,
            300,
            "BCE",
            "short",
            ["three hundred b c e", "three hundred before common era"],
        ),
        ("2000 CE", 1, 2000, "CE", "short", ["two thousand c e", "two thousand common era"]),
        ("500 BC", 0, 500, "BC", "short", ["five hundred b c", "five hundred before christ"]),
    ],
)
def test_era_reads_as_written_a_name_as_words_an_abbreviation_by_letters(
    written, era, year, era_text, form, expected
):
    """icukit #103 marks a written wide era ("Before Christ") with form ``wide``: it is said
    as its words, never spelled; an abbreviation's wide alternative is its own CLDR family
    ("CE" is "Common Era", not "Anno Domini")."""
    unit = _era_unit(written, era, year, era_text, form)
    forms = [normalize_spoken(a.text) for a in unit.alternatives]
    assert forms[: len(expected)] == expected or set(expected) <= set(forms)
    assert not any("b e f o r e" in f or "c o m m o n" in f for f in forms)
    assert unit.unspoken == ()


@pytest.mark.skipif(
    not hasattr(__import__("icukit"), "UnitValue"),
    reason="icukit before #102 reads no amount-less unit",
)
@pytest.mark.parametrize(
    ("written", "unit", "spoken"),
    [
        ("/km²", "square-kilometer", "per square kilometer"),
        ("/s", "second", "per second"),
        ("per second", "second", "per second"),
    ],
)
def test_unit_without_an_amount_speaks_icus_wide_form_for_one(written, unit, spoken):
    """icukit #102 reads "/km²" as a UnitValue; the corpus says "per square kilometer"."""
    unit_reading = _full_span_forms(
        written, [FlexibleMeasureDetector("en_US", unit)], f"measure:{unit}"
    )
    assert [normalize_spoken(a.text) for a in unit_reading.alternatives] == [spoken]
    assert unit_reading.unspoken == ()
