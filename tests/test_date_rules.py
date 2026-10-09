"""CLDR-derived date rules retain their source, generation, and rendering contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from icukit.detectors import DateTimeValue

from frend import normalize, prewarm
from frend.date_rules import (
    GENERATED_DATE_LOCALES,
    date_grammar_input,
    exact_date_generation_count,
    generate_date_alternatives,
    load_date_rule_bundle,
    render_date_derivation,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("locale", "written", "spoken"),
    [
        ("es_MX", "15/03/2024", "quince de marzo de dos mil veinticuatro"),
        ("es_ES", "15/03/2024", "quince de marzo de dos mil veinticuatro"),
        ("fr_FR", "02.03.2003", "deux mars deux mille trois"),
        ("de_DE", "02.03.2003", "zweiter März zweitausenddrei"),
        ("pt_BR", "15/03/2024", "quinze de março de dois mil e vinte e quatro"),
        ("pt_PT", "01/01/2000", "primeiro de janeiro de dois mil"),
        ("it_IT", "02/03/2003", "due marzo duemilatre"),
        ("zh_CN", "2019/2/10", "二〇一九年二月十日"),
        ("ko_KR", "2024. 6. 30.", "이천이십사년 유월 삼십일"),
        ("ja_JP", "2024/01/30", "二〇二四年一月三十日"),
    ],
)
def test_generated_date_is_selected_by_public_normalize(locale, written, spoken):
    result = normalize(written, locale=locale, offsets=True)
    assert result.text == f" {spoken} "
    assert len(result.units) == 1
    assert result.units[0].provenance.startswith(f"date-rule:{locale}+")


def test_korean_sourced_month_is_unit_free_before_rule_attachment():
    value = DateTimeValue((("y", 2024), ("M", 6), ("d", 30)), "gregorian")
    alternatives = generate_date_alternatives(value, {"value": value}, "ko_KR")
    assert alternatives[0].text == "이천이십사년 유월 삼십일"
    assert alternatives[0].text.count("월") == 1
    assert "lexical:ko_KR" in alternatives[0].provenance.split("+")


def test_pt_br_generation_is_exhaustive_and_not_truncated():
    value = DateTimeValue((("y", 2024), ("M", 3), ("d", 15)), "gregorian")
    detection = {"value": value}
    grammar_input = date_grammar_input(value, detection, "pt_BR")
    bundle = load_date_rule_bundle("pt_BR")
    from tiergraph import recognize

    forest = recognize(bundle.lowered, grammar_input, collapse_units=False)
    assert exact_date_generation_count(forest, grammar_input) == 2
    assert {item.text for item in generate_date_alternatives(value, detection, "pt_BR")} == {
        "quinze de março de dois mil e vinte e quatro",
        "quinze de março de duas mil e vinte e quatro",
    }


def test_all_bundles_consume_and_emit_each_field_once():
    for locale in GENERATED_DATE_LOCALES:
        bundle = load_date_rule_bundle(locale)
        assert bundle.locale == locale
        assert bundle.icu_version == "78.3"
        assert len(bundle.declaration.rules) == 5


def test_builder_check_is_byte_identical_under_icu_78_3():
    import tools.build_date_rules as builder

    for locale in builder.LOCALES:
        builder.check_bundle(
            REPO / "frend" / "data" / "date_rules" / f"{locale}.json",
            builder.build_bundle(locale),
        )


def test_generation_refuses_above_cap_before_generate(monkeypatch):
    import frend.date_rules as date_rules

    value = DateTimeValue((("y", 2024), ("M", 3), ("d", 15)), "gregorian")
    original = date_rules.date_field_realizations

    def widened(field, item, *, calendar, locale):
        forms = original(field, item, calendar=calendar, locale=locale)
        if field == "D":
            return forms * 129
        return forms

    monkeypatch.setattr(date_rules, "date_field_realizations", widened)
    monkeypatch.setattr(
        date_rules,
        "generate",
        lambda *_args, **_kwargs: pytest.fail("generate called above the cap"),
    )
    with pytest.raises(ValueError, match="above cap 128"):
        generate_date_alternatives(value, {"value": value}, "es_MX")


def test_generation_refuses_tiergraph_truncation(monkeypatch):
    import frend.date_rules as date_rules

    value = DateTimeValue((("y", 2024), ("M", 3), ("d", 15)), "gregorian")
    monkeypatch.setattr(
        date_rules,
        "generate",
        lambda *_args, **_kwargs: SimpleNamespace(derivations=(), truncated=True),
    )
    date_rules._date_rule_recipes.cache_clear()
    with pytest.raises(ValueError, match="truncated=True"):
        generate_date_alternatives(value, {"value": value}, "es_MX")


def test_renderer_concatenates_tokens_instead_of_using_text():
    derivation = type("Derivation", (), {"tokens": ("二〇二四", "年", "一", "月")})()
    assert render_date_derivation(derivation) == "二〇二四年一月"


def test_german_month_keeps_icu_format_context_case():
    result = normalize("02.03.2003", locale="de_DE", offsets=True)
    assert result.text == " zweiter März zweitausenddrei "
    assert "icu-datetime:LLLL" in result.units[0].provenance.split("+")
    assert "icu-rbnf-fallback:de" in result.units[0].provenance.split("+")
    assert "zwei tausend drei" not in result.text


def test_bundles_name_only_shippable_icu_source():
    for path in sorted((REPO / "frend/data/date_rules").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["provenance"]["source"] == "icu/78.3/cldr-date-patterns"


def test_prewarm_makes_all_generated_date_bundles_resident():
    load_date_rule_bundle.cache_clear()
    prewarm(("en_US", *GENERATED_DATE_LOCALES))
    assert load_date_rule_bundle.cache_info().currsize == 10
    misses = load_date_rule_bundle.cache_info().misses
    for locale in GENERATED_DATE_LOCALES:
        load_date_rule_bundle(locale)
    assert load_date_rule_bundle.cache_info().misses == misses


def test_checked_manifests_pin_closed_scope_and_predicted_recovery_totals():
    changes = json.loads((REPO / "tools/date_expected_changes.json").read_text("utf-8"))
    recoveries = json.loads((REPO / "tools/date_expected_recoveries.json").read_text("utf-8"))
    assert len(changes["rows"]) == 150
    assert len({row["id"] for row in changes["rows"]}) == 150
    fields = {"strict", "presentation", "insensitive"}
    positive_changes = {
        row["id"]
        for row in changes["rows"]
        if any(not row["base"][field] and row["head"][field] for field in fields)
    }
    assert {row["id"] for row in recoveries["rows"]} == positive_changes
    assert recoveries["strict_total"] == 36
    assert recoveries["insensitive_total"] == 42
