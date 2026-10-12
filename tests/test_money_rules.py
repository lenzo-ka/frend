from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from icukit.detectors import NumberValue
from tiergraph import Realization

from frend import money_rules as money_rules_module
from frend import resolve_lattice, verbalize_edge
from frend.money_rules import (
    GENERATED_MONEY_LOCALES,
    generate_money_alternatives,
    load_money_rule_bundle,
    money_grammar_input,
    money_rule_events,
)
from frend.normalization_schema import validate_bundle
from frend.normalize import _currencies, _reading_detectors

REPO = Path(__file__).resolve().parents[1]


def _money_detection(text: str, locale: str):
    matches = [
        item
        for item in _reading_detectors(locale).detect(text)
        if str(item.get("type", "")).startswith("number:currency")
        and item["start"] == 0
        and item["end"] == len(text)
    ]
    assert matches
    return matches[0]


def _public_money_alternatives(text: str, locale: str):
    detection = _money_detection(text, locale)
    lattice = resolve_lattice([detection], source_text=text, locale=locale)
    edge = next(item for item in lattice.edges if item.kind == "reading")
    return verbalize_edge(edge, source_text=text, locale=locale).alternatives


@pytest.mark.parametrize(
    ("locale", "text", "expected"),
    [
        ("es_MX", "MXN 2", "dos pesos"),
        ("fr_FR", "2 EUR", "deux euros"),
        ("it_IT", "2 EUR", "due euro"),
        ("ko_KR", "₩2000", "이천원"),
    ],
)
def test_sourced_major_rule_joins_the_candidate_set(locale, text, expected):
    detection = _money_detection(text, locale)
    alternatives = generate_money_alternatives(detection["value"], detection, locale)

    routed = [item for item in alternatives if f"money-rule:{locale}" in item.provenance]
    assert routed
    assert expected in {item.text for item in routed}


@pytest.mark.parametrize(
    ("text", "expected"),
    [("R$ 1,09", "um real e nove centavos"), ("R$ 0,53", "cinquenta e três centavos")],
)
def test_pt_br_sourced_minor_shapes(text, expected):
    detection = _money_detection(text, "pt_BR")
    alternatives = generate_money_alternatives(detection["value"], detection, "pt_BR")
    assert expected in {item.text for item in alternatives if "money-rule:pt_BR" in item.provenance}


def test_public_verbalization_routes_sourced_money_rule():
    alternatives = _public_money_alternatives("MXN 2", "es_MX")
    assert any(
        item.text == "dos pesos" and "money-rule:es_MX" in item.provenance.split("+")
        for item in alternatives
    )


def test_negative_major_minor_amount_is_negated_once():
    alternatives = generate_money_alternatives(
        NumberValue("-1.09", currency="BRL"), {"captures": ()}, "pt_BR"
    )
    routed = [item for item in alternatives if "money-rule:pt_BR" in item.provenance]
    assert "menos um real e nove centavos" in {item.text for item in routed}
    assert all(item.text.count("menos") == 1 for item in routed)


def test_locale_currency_inventory_is_narrow_and_stably_deduped():
    assert _currencies("es_MX").count("MXN") == 1
    assert _currencies("pt_BR").count("BRL") == 1
    assert "MXN" not in _currencies("en_US")
    assert "BRL" not in _currencies("en_US")


def test_money_input_and_projection_consume_each_field_once():
    detection = _money_detection("R$ 1,09", "pt_BR")
    grammar_input = money_grammar_input(detection["value"], detection, "pt_BR")
    assert [item.symbol for item in grammar_input.tokens] == [
        "FIELD:MAJOR",
        "FIELD:CURRENCY",
        "FIELD:MINOR",
        "FIELD:MINOR_CURRENCY",
    ]


def test_direct_generator_refuses_129_derivations(monkeypatch):
    detection = _money_detection("2 EUR", "fr_FR")
    realizations = tuple(
        Realization((f"number-{index}",), ("test:number",), None) for index in range(129)
    )
    monkeypatch.setattr(money_rules_module, "_number_realizations", lambda *_args: realizations)
    with pytest.raises(ValueError, match="129 derivations"):
        generate_money_alternatives(detection["value"], detection, "fr_FR")


def test_runtime_loads_only_requested_locale_and_never_lowers_or_generates():
    load_money_rule_bundle.cache_clear()
    before = dict(money_rule_events())
    load_money_rule_bundle("es_MX")
    after = dict(money_rule_events())
    assert after["load"] == before["load"] + 1
    assert after["lower"] == before["lower"] == 0
    assert after["generate"] == before["generate"] == 0
    assert load_money_rule_bundle.cache_info().currsize == 1


def test_all_records_validate_and_runtime_projection_keeps_no_source_text():
    import tools.build_money_rules as builder

    for locale in GENERATED_MONEY_LOCALES:
        document = json.loads(
            (REPO / "frend/data/normalization" / f"money_{locale}.json").read_text("utf-8")
        )
        validate_bundle(document, receipt_index=builder._receipt_index(document))
        runtime = json.loads(
            (REPO / "frend/data/money_rules" / f"{locale}.json").read_text("utf-8")
        )
        lowered = json.dumps(runtime, ensure_ascii=False)
        assert "quoted_passage" not in lowered
        assert '"receipt"' not in lowered
        assert "source_text" not in lowered


def test_projection_hash_and_normalization_hash_are_exact():
    for locale in GENERATED_MONEY_LOCALES:
        normalization = json.loads(
            (REPO / "frend/data/normalization" / f"money_{locale}.json").read_text("utf-8")
        )
        runtime = json.loads(
            (REPO / "frend/data/money_rules" / f"{locale}.json").read_text("utf-8")
        )
        projection = {key: runtime[key] for key in ("grammar", "recipes", "records")}
        canonical = json.dumps(
            projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        assert runtime["projection_sha256"] == hashlib.sha256(canonical).hexdigest()
        assert (
            runtime["normalization_sha256"]
            == hashlib.sha256(
                json.dumps(
                    normalization, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        )


def test_runtime_refuses_normalization_hash_drift(tmp_path, monkeypatch):
    locale = "es_MX"
    runtime_dir = tmp_path / "data/money_rules"
    normalization_dir = tmp_path / "data/normalization"
    runtime_dir.mkdir(parents=True)
    normalization_dir.mkdir(parents=True)
    runtime = json.loads((REPO / "frend/data/money_rules" / f"{locale}.json").read_text("utf-8"))
    runtime["normalization_sha256"] = "0" * 64
    (runtime_dir / f"{locale}.json").write_text(json.dumps(runtime), encoding="utf-8")
    source = REPO / "frend/data/normalization" / f"money_{locale}.json"
    (normalization_dir / f"money_{locale}.json").write_bytes(source.read_bytes())
    monkeypatch.setattr(money_rules_module, "files", lambda _package: tmp_path)
    load_money_rule_bundle.cache_clear()
    with pytest.raises(ValueError, match="normalization hash mismatch"):
        load_money_rule_bundle(locale)
    load_money_rule_bundle.cache_clear()


def test_decimal_beyond_precision_uses_decimal_major_not_rounding():
    bundle = load_money_rule_bundle("pt_BR")
    detection = {"captures": ()}
    alternatives = generate_money_alternatives(
        NumberValue("1.009", currency="BRL"), detection, bundle.locale
    )
    assert any(
        item.text == "um vírgula zero zero nove real"
        and "money-rule:pt_BR" in item.provenance.split("+")
        for item in alternatives
    )
