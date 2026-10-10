"""Sourced fraction rules retain generation, provenance, and rendering contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from icukit.detectors import detect

from frend import normalize, prewarm
from frend.fraction_rules import (
    GENERATED_FRACTION_LOCALES,
    _denominator_realizations,
    _validate_compiled_recipes,
    generate_fraction_alternatives,
    load_fraction_rule_bundle,
    render_fraction_tokens,
)
from frend.normalization_schema import validate_bundle
from frend.normalize import _reading_detectors

REPO = Path(__file__).resolve().parents[1]


def _fraction_detection(locale: str, text: str) -> dict:
    return next(
        row
        for row in detect(text, _reading_detectors(locale))
        if row["start"] == 0
        and row["end"] == len(text)
        and (
            str(row.get("type", "")).startswith("fraction:")
            or str(row.get("type", "")).startswith("number:fraction")
        )
    )


@pytest.mark.parametrize(
    ("locale", "written", "spoken"),
    [
        ("es_MX", "3/17", "tres diecisieteavos"),
        ("es_ES", "1/30", "un trigésimo"),
        ("fr_FR", "2/8", "deux huitièmes"),
        ("de_DE", "3 1/2", "dreieinhalb"),
        ("pt_BR", "1/11", "um onze avos"),
        ("pt_PT", "3/8", "três oitavos"),
        ("it_IT", "2/8", "due ottavi"),
        ("zh_CN", "3 1/2", "三又二分之一"),
        ("ko_KR", "3 1/2", "삼과 이분의 일"),
        ("ja_JP", "3 1/2", "三と二分の一"),
    ],
)
def test_generated_fraction_is_selected_by_public_normalize(locale, written, spoken):
    result = normalize(written, locale=locale, offsets=True)
    assert result.text == f" {spoken} "
    assert len(result.units) == 1
    assert result.units[0].provenance.startswith(f"fraction-rule:{locale}+")


@pytest.mark.parametrize(
    ("locale", "spoken"),
    [
        ("es_MX", "cuarenta y cinco por ciento"),
        ("fr_FR", "quarante-cinq pour cent"),
        ("de_DE", "fünfundvierzig Prozent"),
        ("pt_BR", "quarenta e cinco por cento"),
        ("it_IT", "quarantacinque percento"),
        ("zh_CN", "百分之四十五"),
        ("ko_KR", "사십오 퍼센트"),
        ("ja_JP", "四十五 パーセント"),
    ],
)
def test_sourced_percent_frames(locale, spoken):
    assert normalize("45%", locale=locale) == f" {spoken} "


def test_spanish_and_portuguese_selector_boundaries():
    assert normalize("1/23", locale="es_MX") == " un veintitresavo "
    assert normalize("2/40", locale="es_MX") == " dos cuadragésimos "
    assert normalize("1/10", locale="pt_BR") == " um décimo "
    assert normalize("1/11", locale="pt_BR") == " um onze avos "
    assert normalize("1/100", locale="pt_BR") == " um centésimo "


@pytest.mark.parametrize("locale", ("es_MX", "es_ES"))
def test_spanish_tens_offer_avo_and_existing_prior_ordinal_forms(locale):
    expected = {
        20: ("veinteavos", "vigésimos"),
        30: ("treintavos", "trigésimos"),
        40: ("cuarentavos", "cuadragésimos"),
        50: ("cincuentavos", "quincuagésimos"),
        60: ("sesentavos", "sexagésimos"),
        70: ("setentavos", "septuagésimos"),
        80: ("ochentavos", "octogésimos"),
        90: ("noventavos", "nonagésimos"),
    }
    for denominator, (avo, ordinal) in expected.items():
        alternatives = generate_fraction_alternatives(
            _fraction_detection(locale, f"3/{denominator}"), locale
        )
        texts = tuple(item.text for item in alternatives)
        assert f"tres {avo}" in texts
        assert f"tres {ordinal}" in texts

    alternatives = generate_fraction_alternatives(_fraction_detection(locale, "3/20"), locale)
    assert alternatives[0].text == "tres vigésimos"
    assert normalize("3/20", locale=locale) == " tres vigésimos "


def test_french_plural_is_not_appended_twice_and_italian_avoids_display_name():
    assert normalize("2/8", locale="fr_FR") == " deux huitièmes "
    assert "huitièmess" not in normalize("2/8", locale="fr_FR")
    assert normalize("45%", locale="it_IT") == " quarantacinque percento "
    assert "percentuale" not in normalize("45%", locale="it_IT")


def test_signs_and_denominator_first_order_are_exact():
    assert normalize("-3/7", locale="zh_CN") == " 负七分之三 "
    assert normalize("-3/7", locale="ko_KR") == " 마이너스 칠분의 삼 "
    assert normalize("-3/7", locale="ja_JP") == " マイナス七分の三 "


@pytest.mark.parametrize(
    ("locale", "written", "spoken"),
    [
        ("de_DE", "3/7", "drei Siebtel"),
        ("de_DE", "5/40", "fünf Vierzigstel"),
        ("es_ES", "3/17", "tres diecisieteavos"),
        ("es_MX", "5/19", "cinco diecinueveavos"),
        ("pt_BR", "7/23", "sete vinte e três avos"),
        ("pt_PT", "7/23", "sete vinte e três avos"),
        ("fr_FR", "3/7", "trois septièmes"),
        ("it_IT", "3/7", "tre settimi"),
        ("zh_CN", "3/7", "七分之三"),
        ("ko_KR", "3/7", "칠분의 삼"),
        ("ja_JP", "3/7", "七分の三"),
    ],
)
def test_unseen_productive_fraction_probes_use_the_sourced_rule(locale, written, spoken):
    result = normalize(written, locale=locale, offsets=True)
    assert result.text == f" {spoken} "
    assert len(result.units) == 1
    assert result.units[0].provenance.startswith(f"fraction-rule:{locale}+")


@pytest.mark.parametrize(
    ("locale", "written", "spoken"),
    [
        ("de_DE", "1/101", "ein Hunderteintel"),
        ("es_ES", "1/3000", "un tresmilésimo"),
        ("es_MX", "1/100000", "un cienmilésimo"),
        ("es_ES", "1/1000000", "un millonésimo"),
        ("fr_FR", "2 1/2", "deux et demi"),
        ("it_IT", "2 1/2", "due e mezzo"),
        ("pt_BR", "2 1/2", "dois e meio"),
        ("pt_PT", "2 1/2", "dois e meio"),
        ("ko_KR", "2 1/2", "이와 이분의 일"),
        ("ko_KR", "3 1/2", "삼과 이분의 일"),
    ],
)
def test_source_review_speech_corrections(locale, written, spoken):
    assert normalize(written, locale=locale) == f" {spoken} "


def test_italian_cardinal_denominator_forms_are_general_lower_ranked_alternatives():
    one_431 = generate_fraction_alternatives(_fraction_detection("it_IT", "1/431"), "it_IT")
    three_432 = generate_fraction_alternatives(_fraction_detection("it_IT", "3/432"), "it_IT")

    assert "un su quattrocentotrentuno" in {item.text for item in one_431}
    assert "tre su quattrocentotrentadue" in {item.text for item in three_432}
    assert normalize("1/431", locale="it_IT") == " un quattrocentotrentunesimo "
    assert normalize("3/432", locale="it_IT") == " tre quattrocentotrentaduesimi "
    assert normalize("1/20", locale="it_IT") == " un ventesimo "


def test_missing_icu_plural_ordinal_uses_sourced_italian_plural_rule():
    alternatives = generate_fraction_alternatives(_fraction_detection("it_IT", "3/432"), "it_IT")
    derived = next(item for item in alternatives if item.text == "tre quattrocentotrentaduesimi")
    assert "normalization-record:denominator-plural-rule" in derived.provenance


@pytest.mark.parametrize("locale", (*GENERATED_FRACTION_LOCALES, "en_US"))
def test_fraction_percent_exception_sweep_representative_subset(locale):
    denominators = (1, 2, 3, 7, 20, 100, 431, 432, 999, 1200)
    written = [
        *(f"{numerator}/{denominator}" for denominator in denominators for numerator in (1, 3, 12)),
        *(f"2 3/{denominator}" for denominator in denominators),
        *(f"{amount}%" for amount in (0, 1, 73, 100, 1000)),
        "-3/7",
    ]
    for text in written:
        normalize(text, locale=locale)


@pytest.mark.parametrize(
    ("locale", "spoken"),
    [
        ("es_MX", "setenta y tres por ciento"),
        ("es_ES", "setenta y tres por ciento"),
        ("fr_FR", "soixante-treize pour cent"),
        ("de_DE", "dreiundsiebzig Prozent"),
        ("pt_BR", "setenta e três por cento"),
        ("pt_PT", "setenta e três por cento"),
        ("it_IT", "settantatré percento"),
        ("zh_CN", "百分之七十三"),
        ("ko_KR", "칠십삼 퍼센트"),
        ("ja_JP", "七十三 パーセント"),
    ],
)
def test_unseen_productive_percent_probes_use_the_sourced_rule(locale, spoken):
    result = normalize("73%", locale=locale, offsets=True)
    assert result.text == f" {spoken} "
    assert len(result.units) == 1
    assert result.units[0].provenance.startswith(f"fraction-rule:{locale}+")


def test_all_bundles_validate_closed_non_ldc_ancestry_and_field_rules():
    import tools.build_fraction_rules as builder

    for locale in GENERATED_FRACTION_LOCALES:
        bundle = load_fraction_rule_bundle(locale)
        assert bundle.locale == locale
        document = json.loads(
            (REPO / "frend/data/normalization" / f"{locale}.json").read_text("utf-8")
        )
        assert "ldc" not in json.dumps(document).casefold()
        validate_bundle(document, receipt_index=builder._receipt_index(document))


def test_builder_check_is_byte_identical_and_repeatable():
    import tools.build_fraction_rules as builder

    before = {
        locale: (REPO / "frend/data/fraction_rules" / f"{locale}.json").read_bytes()
        for locale in GENERATED_FRACTION_LOCALES
    }
    assert builder.main.__name__ == "main"
    for locale in GENERATED_FRACTION_LOCALES:
        expected = (
            json.dumps(builder.build_bundle(locale), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode()
        assert expected == before[locale]
        assert (
            expected
            == (
                json.dumps(
                    builder.build_bundle(locale), ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n"
            ).encode()
        )


def test_runtime_uses_precompiled_recipes_without_tiergraph_generation():
    import frend.fraction_rules as rules

    rules.load_fraction_rule_bundle.cache_clear()
    rules._fraction_rule_recipes.cache_clear()
    rules._EVENTS.update(load=0, lower=0, generate=0)
    assert rules._fraction_rule_recipes("es_ES", "fraction")
    assert rules.fraction_rule_events() == {"load": 1, "lower": 0, "generate": 0}


@pytest.mark.parametrize("field", ["tokens", "provenance"])
def test_runtime_rejects_compiled_recipe_drift_from_grammar(field):
    bundle = load_fraction_rule_bundle("fr_FR")
    recipes = {kind: list(items) for kind, items in bundle.recipes.items()}
    original = recipes["fraction"][0]
    recipes["fraction"][0] = replace(
        original,
        **{field: (*getattr(original, field), "unbound-drift")},
    )
    with pytest.raises(ValueError, match="compiled recipes do not match grammar"):
        _validate_compiled_recipes(bundle.declaration, recipes, "fr_FR")


def test_generation_refuses_above_cap_before_generate(monkeypatch):
    import frend.fraction_rules as rules

    detection = _fraction_detection("es_MX", "3/7")
    original = rules._cardinal_realizations

    def widened(value, locale, *, masculine_one=False):
        return original(value, locale, masculine_one=masculine_one) * 129

    monkeypatch.setattr(rules, "_cardinal_realizations", widened)
    with pytest.raises(ValueError, match="above cap 128"):
        generate_fraction_alternatives(detection, "es_MX")


def test_generation_refuses_tiergraph_truncation(monkeypatch):
    import tools.build_fraction_rules as builder

    monkeypatch.setattr(
        builder,
        "generate",
        lambda *_args, **_kwargs: SimpleNamespace(derivations=(), truncated=True),
    )
    with pytest.raises(ValueError, match="truncated=True"):
        builder.build_bundle("es_MX")


def test_unknown_selector_is_rejected():
    bundle = load_fraction_rule_bundle("es_MX")
    records = dict(bundle.records)
    records["denominator-strategy"] = MappingProxyType(
        {**records["denominator-strategy"], "forms": {"default": "unknown"}}
    )
    broken = replace(bundle, records=MappingProxyType(records))
    with pytest.raises(ValueError, match="unknown denominator strategy"):
        _denominator_realizations(17, 3, broken)


def test_rule_provenance_never_becomes_the_measurement_key():
    detection = _fraction_detection("fr_FR", "3/7")
    [alternative] = generate_fraction_alternatives(detection, "fr_FR")
    assert alternative.provenance.startswith("fraction-rule:fr_FR+")
    assert alternative.prior_provenance
    assert "fraction-rule:" not in alternative.prior_provenance


def test_unchanged_spanish_reading_preserves_the_legacy_measurement_key():
    detection = _fraction_detection("es_ES", "2/7")
    [alternative] = generate_fraction_alternatives(detection, "es_ES")
    assert alternative.text == "dos séptimos"
    assert alternative.prior_provenance == (
        "icu-rbnf:%spellout-numbering+icu-rbnf-fallback:es+"
        "icu-rbnf:%spellout-ordinal-masculine-adjective+icu-rbnf-fallback:es"
    )


def test_production_uses_the_exact_token_renderer(monkeypatch):
    import frend.fraction_rules as rules

    detection = _fraction_detection("zh_CN", "3/7")
    monkeypatch.setattr(rules, "render_fraction_tokens", lambda tokens: "|".join(tokens))
    alternatives = generate_fraction_alternatives(detection, "zh_CN")
    assert alternatives[0].text == "|七|分之|三"


def test_renderer_concatenates_tokens_exactly():
    assert render_fraction_tokens(("三", "又", "二分之一")) == "三又二分之一"


def test_chinese_fraction_and_mixed_frames_have_specific_receipts():
    document = json.loads((REPO / "frend/data/normalization/zh_CN.json").read_text("utf-8"))
    receipts = {source["receipt"] for source in document["sources"]}
    assert {"chinese-fractions.json", "chinese-mixed-fractions.json"} <= receipts
    assert document["rules"][0]["derived_from"] == ["jixi/education-cloud#zh-fractions"]
    assert document["rules"][1]["derived_from"] == ["life-education/journal#zh-mixed-fractions"]


def test_manifest_builder_cannot_select_new_recoveries_from_head(tmp_path):
    import tools.build_fraction_manifests as builder

    fields = ("strict", "presentation", "insensitive", "first_strict", "first_presentation")

    def row(identifier, recovered):
        return {
            "id": identifier,
            "locale": "fr_FR",
            "written": "3/7",
            "targets": ["trois septièmes"],
            "target_hits": {
                name: [recovered] for name in ("strict", "presentation", "insensitive")
            },
            "strict": recovered,
            "presentation": recovered,
            "insensitive": recovered,
            "first_strict": recovered,
            "first_presentation": recovered,
            "offer_signature": [
                {
                    "type": "number:fraction",
                    "alternatives": [
                        {
                            "provenance": (
                                "fraction-rule:fr_FR+"
                                "normalization-record:frend/curated#fr-fractions"
                            )
                        }
                    ],
                }
            ],
        }

    base_rows = [row("first", False), row("new-head-flip", False)]
    head_rows = [row("first", True), row("new-head-flip", True)]
    base_path = tmp_path / "base.json"
    head_path = tmp_path / "head.json"
    base_path.write_text(json.dumps({"cases": base_rows}), encoding="utf-8")
    head_path.write_text(json.dumps({"cases": head_rows}), encoding="utf-8")
    witness = {
        "rows": [
            {
                "id": "first",
                "base": {name: base_rows[0][name] for name in fields},
                "expected_speech": "trois septièmes",
                "source_record_id": "frend/curated#fr-fractions",
            }
        ]
    }
    _changes, recoveries = builder.build(base_path, head_path, witness)
    assert [item["id"] for item in recoveries["rows"]] == ["first"]


def test_prewarm_makes_all_generated_fraction_bundles_resident():
    load_fraction_rule_bundle.cache_clear()
    prewarm(("en_US", *GENERATED_FRACTION_LOCALES))
    assert load_fraction_rule_bundle.cache_info().currsize == 10
