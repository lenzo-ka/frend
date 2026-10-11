"""Sourced generated clock composition and its runtime contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from icukit.detectors import Capture, DateTimeValue
from tiergraph import GrammarHole, grammar_loads

from frend import resolve_lattice, verbalize_edge
from frend.normalization_schema import validate_bundle
from frend.runtime import prewarm
from frend.time_rules import (
    GENERATED_TIME_LOCALES,
    generate_time_alternatives,
    load_time_rule_bundle,
    time_rule_events,
)

REPO = Path(__file__).resolve().parents[1]


def _time_unit(
    written: str,
    fields: tuple[tuple[str, int], ...],
    captures: tuple[tuple[str, int, int, str, object, str | None], ...],
    locale: str,
    *,
    type_: str = "time:flexible",
):
    detection = {
        "text": written,
        "start": 0,
        "end": len(written),
        "type": type_,
        "value": DateTimeValue(fields, "gregorian"),
        "captures": tuple(Capture(*capture) for capture in captures),
    }
    lattice = resolve_lattice([detection], source_text=written)
    edge = next(item for item in lattice.edges if item.kind == "reading")
    return verbalize_edge(edge, source_text=written, locale=locale)


@pytest.mark.parametrize("locale", GENERATED_TIME_LOCALES)
def test_generated_declaration_has_six_exact_unique_hole_paths(locale):
    document = json.loads(
        (REPO / "frend/data/time_rules" / f"{locale}.json").read_text(encoding="utf-8")
    )
    declaration = grammar_loads(json.dumps(document["grammar"], ensure_ascii=False))
    rules = [rule for rule in declaration.rules if rule.left == declaration.start]

    expected_shapes = {
        "de_DE": {"H", "HM", "PH"},
        "it_IT": {"H", "HM", "PH", "PHM"},
    }.get(locale, {"H", "HM", "HMS", "PH", "PHM", "PHMS"})
    assert len(rules) == len(expected_shapes)
    assert {
        "".join(
            item.nonterminal.local_name
            for item in rule.source
            if isinstance(item, GrammarHole) and item.nonterminal.local_name in {"P", "H", "M", "S"}
        )
        for rule in rules
    } == expected_shapes
    for rule in rules:
        source = [
            item.nonterminal.local_name for item in rule.source if isinstance(item, GrammarHole)
        ]
        target = [
            item.nonterminal.local_name for item in rule.target if isinstance(item, GrammarHole)
        ]
        assert len(source) == len(set(source))
        assert sorted(source) == sorted(target)


@pytest.mark.parametrize(
    ("locale", "written", "fields", "captures", "spoken"),
    [
        (
            "es_ES",
            "3:02:32",
            (("H", 3), ("m", 2), ("s", 32)),
            (
                ("H", 0, 1, "3", 3, "numeric"),
                ("m", 2, 4, "02", 2, "numeric"),
                ("s", 5, 7, "32", 32, "numeric"),
            ),
            "tres horas dos minutos y treinta y dos segundos",
        ),
        (
            "fr_FR",
            "03:04:05",
            (("H", 3), ("m", 4), ("s", 5)),
            (
                ("H", 0, 2, "03", 3, "numeric"),
                ("m", 3, 5, "04", 4, "numeric"),
                ("s", 6, 8, "05", 5, "numeric"),
            ),
            "trois heures quatre minutes cinq secondes",
        ),
        (
            "pt_BR",
            "14:30:05",
            (("H", 14), ("m", 30), ("s", 5)),
            (
                ("H", 0, 2, "14", 14, "numeric"),
                ("m", 3, 5, "30", 30, "numeric"),
                ("s", 6, 8, "05", 5, "numeric"),
            ),
            "catorze horas e trinta minutos e cinco segundos",
        ),
        (
            "zh_CN",
            "03:04:05",
            (("H", 3), ("m", 4), ("s", 5)),
            (
                ("H", 0, 2, "03", 3, "numeric"),
                ("m", 3, 5, "04", 4, "numeric"),
                ("s", 6, 8, "05", 5, "numeric"),
            ),
            "三点四分五秒",
        ),
        (
            "ja_JP",
            "03:04:05",
            (("H", 3), ("m", 4), ("s", 5)),
            (
                ("H", 0, 2, "03", 3, "numeric"),
                ("m", 3, 5, "04", 4, "numeric"),
                ("s", 6, 8, "05", 5, "numeric"),
            ),
            "三時四分五秒",
        ),
    ],
)
def test_seconds_rules_realize_every_field_and_mark_s_spoken(
    locale, written, fields, captures, spoken
):
    unit = _time_unit(written, fields, captures, locale)

    assert unit.verbalized is True
    assert unit.alternatives[0].text == spoken
    assert unit.alternatives[0].provenance.startswith(f"time-rule:{locale}+")
    assert "s" not in {capture.name for capture in unit.unspoken}


def test_italian_clock_minutes_use_the_attested_elliptical_form():
    unit = _time_unit(
        "3:40",
        (("H", 3), ("m", 40)),
        (
            ("H", 0, 1, "3", 3, "numeric"),
            ("m", 2, 4, "40", 40, "numeric"),
        ),
        "it_IT",
    )

    assert "tre e quaranta" in {
        alternative.text
        for alternative in unit.alternatives
        if alternative.provenance.startswith("time-rule:it_IT+")
    }


@pytest.mark.parametrize("locale", ["pt_BR", "pt_PT"])
def test_portuguese_clock_minutes_keep_the_attested_unit(locale):
    unit = _time_unit(
        "6:30",
        (("H", 6), ("m", 30)),
        (
            ("H", 0, 1, "6", 6, "numeric"),
            ("m", 2, 4, "30", 30, "numeric"),
        ),
        locale,
    )

    assert "seis horas e trinta minutos" in {
        alternative.text
        for alternative in unit.alternatives
        if alternative.provenance.startswith(f"time-rule:{locale}+")
    }


@pytest.mark.parametrize(
    ("written", "semantic_hour", "written_hour", "period"),
    [("3:30 p.m.", 15, 3, "de la tarde"), ("11:30 p.m.", 23, 11, "de la noche")],
)
def test_spanish_period_uses_semantic_hour_but_speaks_written_hour(
    written, semantic_hour, written_hour, period
):
    unit = _time_unit(
        written,
        (("H", semantic_hour), ("m", 30)),
        (
            ("H", 0, len(str(written_hour)), str(written_hour), written_hour, "numeric"),
            ("m", len(str(written_hour)) + 1, len(str(written_hour)) + 3, "30", 30, "numeric"),
            (
                "day-period",
                len(str(written_hour)) + 3,
                len(written),
                written[len(str(written_hour)) + 3 :],
                None,
                "symbol",
            ),
        ),
        "es_ES",
    )

    assert f"{'tres' if written_hour == 3 else 'once'} y treinta {period}" in {
        item.text for item in unit.alternatives
    }


def test_korean_period_selects_native_hour_but_period_free_uses_sino_korean():
    period = _time_unit(
        "오전 10:30:05",
        (("H", 10), ("m", 30), ("s", 5)),
        (
            ("day-period", 0, 2, "오전", None, "symbol"),
            ("H", 3, 5, "10", 10, "numeric"),
            ("m", 6, 8, "30", 30, "numeric"),
            ("s", 9, 11, "05", 5, "numeric"),
        ),
        "ko_KR",
    )
    clock24 = _time_unit(
        "14:05",
        (("H", 14), ("m", 5)),
        (("H", 0, 2, "14", 14, "numeric"), ("m", 3, 5, "05", 5, "numeric")),
        "ko_KR",
    )

    assert "오전 열시 삼십분 오초" in {item.text for item in period.alternatives}
    assert "십사시 오분" in {item.text for item in clock24.alternatives}


def test_inconsistent_period_hour_falls_back_and_keeps_seconds_unspoken():
    unit = _time_unit(
        "2:07:09 PM",
        (("H", 15), ("m", 7), ("s", 9)),
        (
            ("H", 0, 1, "2", 2, "numeric"),
            ("m", 2, 4, "07", 7, "numeric"),
            ("s", 5, 7, "09", 9, "numeric"),
            ("day-period", 7, 10, " PM", None, "symbol"),
        ),
        "es_ES",
    )

    assert unit.verbalized is False
    assert {capture.name for capture in unit.unspoken} == {"s"}


def test_italian_seconds_have_no_unsupported_spoken_recipe():
    unit = _time_unit(
        "17:15:26",
        (("H", 17), ("m", 15), ("s", 26)),
        (
            ("H", 0, 2, "17", 17, "numeric"),
            ("m", 3, 5, "15", 15, "numeric"),
            ("s", 6, 8, "26", 26, "numeric"),
        ),
        "it_IT",
    )

    assert unit.verbalized is False
    assert unit.alternatives[0].text == "17:15:26"
    assert {capture.name for capture in unit.unspoken} == {"s"}


def test_hour_and_seconds_without_minutes_falls_back_instead_of_raising_key_error():
    unit = _time_unit(
        "17:26",
        (("H", 17), ("s", 26)),
        (("H", 0, 2, "17", 17, "numeric"), ("s", 3, 5, "26", 26, "numeric")),
        "es_ES",
    )

    assert unit.verbalized is False
    assert unit.alternatives[0].text == "17:26"
    assert {capture.name for capture in unit.unspoken} == {"s"}


def test_datetime_preserves_and_realizes_seconds_with_written_hour_and_semantic_period():
    written = "15/03/2024, 2:07:09 p.m."
    unit = _time_unit(
        written,
        (("y", 2024), ("M", 3), ("d", 15), ("H", 14), ("m", 7), ("s", 9)),
        (
            ("d", 0, 2, "15", 15, "numeric"),
            ("M", 3, 5, "03", 3, "numeric"),
            ("y", 6, 10, "2024", 2024, "numeric"),
            ("datetime-glue", 10, 12, ", ", None, None),
            ("H", 12, 13, "2", 2, "numeric"),
            ("m", 14, 16, "07", 7, "numeric"),
            ("s", 17, 19, "09", 9, "numeric"),
            ("day-period", 19, len(written), " p.m.", None, "symbol"),
        ),
        "es_ES",
        type_="date:datetime-flexible",
    )

    assert "dos horas siete minutos y nueve segundos de la tarde" in unit.alternatives[0].text
    assert "time-rule:es_ES" in unit.alternatives[0].provenance
    assert "s" not in {capture.name for capture in unit.unspoken}


def test_runtime_loads_only_the_requested_locale_and_never_lowers_or_generates():
    load_time_rule_bundle.cache_clear()
    before = dict(time_rule_events())

    load_time_rule_bundle("es_ES")

    after = dict(time_rule_events())
    assert after["load"] == before["load"] + 1
    assert after["lower"] == before["lower"] == 0
    assert after["generate"] == before["generate"] == 0


def test_all_typed_records_validate_and_runtime_projection_keeps_no_source_text():
    import tools.build_time_rules as builder

    for locale in GENERATED_TIME_LOCALES:
        document = json.loads(
            (REPO / "frend/data/normalization" / f"time_{locale}.json").read_text("utf-8")
        )
        validate_bundle(document, receipt_index=builder._receipt_index(document))
        runtime = json.loads((REPO / "frend/data/time_rules" / f"{locale}.json").read_text("utf-8"))
        lowered = json.dumps(runtime, ensure_ascii=False)
        assert "quoted_passage" not in lowered
        assert '"receipt"' not in lowered
        assert "source_text" not in lowered


def test_time_bundles_are_lazy_by_locale_and_prewarm_loads_each_once():
    load_time_rule_bundle.cache_clear()
    _time_unit(
        "14:30:05",
        (("H", 14), ("m", 30), ("s", 5)),
        (
            ("H", 0, 2, "14", 14, "numeric"),
            ("m", 3, 5, "30", 30, "numeric"),
            ("s", 6, 8, "05", 5, "numeric"),
        ),
        "fr_FR",
    )
    assert load_time_rule_bundle.cache_info().currsize == 1

    load_time_rule_bundle.cache_clear()
    prewarm(GENERATED_TIME_LOCALES)
    assert load_time_rule_bundle.cache_info().currsize == len(GENERATED_TIME_LOCALES)


def test_english_prewarm_loads_no_time_bundle():
    load_time_rule_bundle.cache_clear()
    before = time_rule_events()["load"]

    assert prewarm(["en_US"]) == ("en_US",)

    assert time_rule_events()["load"] == before


def test_direct_generator_refuses_more_than_the_declared_cap():
    detection = {
        "captures": (
            Capture("H", 0, 2, "14", 14, "numeric"),
            Capture("m", 3, 5, "30", 30, "numeric"),
        )
    }
    with pytest.raises(ValueError, match="1 derivations"):
        generate_time_alternatives(
            DateTimeValue((("H", 14), ("m", 30)), "gregorian"),
            detection,
            "pt_BR",
            max_derivations=0,
        )
