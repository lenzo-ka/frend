from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from icukit.detectors import detect

from frend import compose_choices, normalize, resolve_choices
from frend.normalize import _reading_detectors

REPO = Path(__file__).resolve().parents[1]


def _load_gate():
    name = "locale_gate_for_number_tests"
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "locale_gate.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _graph(written: str, locale: str):
    detections = detect(written, _reading_detectors(locale))
    return compose_choices(resolve_choices(detections, source_text=written, locale=locale))


def _offers(written: str, spoken: str, locale: str) -> bool:
    return gate.offers(_graph(written, locale), spoken, form=gate.strict_form)


# new: main rounds this int64-representable integer through binary64.
def test_integer_beyond_double_precision_is_exact():
    assert "nine quadrillion nine hundred ninety-nine trillion" in normalize(
        "9999999999999999", locale="en_US"
    )


# new: main attempts a magnitude reading beyond ICU's exact int64 boundary.
def test_integer_beyond_int64_reads_digits():
    written = "124444234854823834553"
    result = normalize(written, locale="en_US").strip()
    assert result.startswith("one two four")
    assert result.endswith("five five three")
    assert "quadrillion" not in result
    alternatives = [
        alternative for unit in _graph(written, "en_US").units for alternative in unit.alternatives
    ]
    selected = next(alternative for alternative in alternatives if alternative.text == result)
    assert selected.provenance == "icu-rbnf:digits-beyond-int64+icu-rbnf-fallback:en"
    assert selected.provenance != "surface:unsupported"


# regression: signed-int64 minimum is exact and must retain its sign rather than
# taking the unsigned beyond-int64 digit-reading path.
def test_signed_int64_minimum_is_exact_and_negative():
    result = normalize("-9223372036854775808", locale="en_US").strip()
    assert result.startswith("minus nine two two")
    assert "icu-rbnf:digits-beyond-int64+icu-rbnf-fallback:en" in {
        alternative.provenance
        for unit in _graph("-9223372036854775808", "en_US").units
        for alternative in unit.alternatives
        if alternative.text == result
    }


# new: main rounds both grouped and ungrouped values to the same magnitude reading.
def test_integer_beyond_int64_reads_written_groups():
    grouped = "124,444,234,854,823,834,553"
    spoken = (
        "one hundred twenty-four four hundred forty-four two hundred thirty-four "
        "eight hundred fifty-four eight hundred twenty-three eight hundred thirty-four "
        "five hundred fifty-three"
    )
    assert _offers(grouped, spoken, "en_US")
    assert not _offers(grouped.replace(",", ""), spoken, "en_US")


# new: main ranks ICU's lunar-day ruleset first.
def test_zh_cardinal_is_not_a_lunar_day():
    assert normalize("5", locale="zh_CN").strip() == "五"


# new: main offers a year ruleset containing Latin decimal digits.
def test_ja_year_has_no_latin_digits():
    assert not any(
        character.isascii() and character.isdigit()
        for character in normalize("2024年", locale="ja_JP")
    )


@pytest.mark.parametrize(
    ("locale", "written", "spoken"),
    [
        pytest.param("pt_BR", "0,1", "zero vírgula um", id="pt_BR"),
        pytest.param("pt_PT", "0,5", "zero vírgula cinco", id="pt_PT"),
        pytest.param("ko_KR", "1.5", "일점오", id="ko_KR"),
        pytest.param("zh_CN", "1.5", "一点五", id="zh_CN"),
        pytest.param("ja_JP", "2.5", "二点五", id="ja_JP"),
    ],
)
# new: main passes these locale decimals through instead of speaking them.
def test_decimal_layout_matches_icu(locale, written, spoken):
    assert normalize(written, locale=locale).strip() == spoken
    if locale == "ja_JP":
        matching = [
            alternative
            for unit in _graph(written, locale).units
            for alternative in unit.alternatives
            if alternative.text == spoken
        ]
        assert matching
        assert all("financial" not in alternative.provenance for alternative in matching)
        assert any(
            "lexical:ja_JP+icu-rbnf:%spellout-cardinal" in alternative.provenance
            for alternative in matching
        )


# regression: ICU's Japanese cardinal direct form uses the written interpunct
# ``・``; r4.1 E3 requires only the lexical spoken separator ``点``.
def test_ja_decimal_excludes_written_interpunct_offer():
    assert not _offers("2.5", "二・五", "ja_JP")


@pytest.mark.parametrize(
    ("spoken"),
    [pytest.param("영점공오", id="icu-direct"), pytest.param("공점공오", id="composed")],
)
# new: main offers neither ICU-attested Korean zero reading.
def test_decimal_layout_offers(spoken):
    assert _offers("0.05", spoken, "ko_KR")


# guard for en_US; new for ko_KR: English order must stay byte-identical.
def test_decimal_fraction_digits_spaced_only_where_icu_spaces():
    assert normalize("3.14", locale="en_US") == " three point one four "
    assert normalize("3.14", locale="ko_KR") == " 삼점일사 "


@pytest.mark.parametrize(
    ("locale", "written", "spoken"),
    [
        pytest.param("es_ES", "7,25", "siete coma dos cinco", id="es_ES-new"),
        pytest.param("es_MX", "7.25", "siete punto dos cinco", id="es_MX-guard"),
    ],
)
# new for es_ES; guard for es_MX: the separator follows ICU's locale symbol.
def test_es_decimal_separator_follows_locale_symbol(locale, written, spoken):
    assert normalize(written, locale=locale).strip() == spoken


# new: main does not offer Spain Spanish's sourced whole-fraction reading.
def test_es_es_fraction_as_number_offered():
    assert _offers("7,25", "siete coma veinticinco", "es_ES")


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        pytest.param("€5", "五 欧元", id="EUR"),
        pytest.param("£100", "一百 英镑", id="GBP"),
        pytest.param("₩100", "一百 韩元", id="KRW"),
    ],
)
# new: main raises IndexError when the wide currency name has no separating space.
def test_currency_name_without_space_keeps_currency_frame(written, spoken):
    assert normalize(written, locale="zh_CN").strip() == spoken


# new: main extracts singular Million from a plural compact frame.
def test_compact_scale_agrees_in_number():
    assert normalize("460 Millionen", locale="de_DE").strip() == "vierhundertsechzig Millionen"


# guard: the first English decimal reading is a byte-identity requirement.
def test_english_decimal_point_order_unchanged():
    assert normalize("3.14", locale="en_US") == " three point one four "
