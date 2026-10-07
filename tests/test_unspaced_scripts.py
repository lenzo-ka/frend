from __future__ import annotations

import pytest
from icukit.detectors import detect

from frend import NormalizedText, compose_choices, normalize, resolve_choices
from frend.normalize import _reading_detectors


def test_ja_reading_between_kana_has_no_spaces():
    assert normalize("そこに鳥1羽がいます", locale="ja_JP") == "そこに鳥一羽がいます"


def test_zh_reading_inside_han_has_no_spaces():
    result = normalize("我今天买了5个苹果", locale="zh_CN")
    assert "五" in result
    assert "5" not in result
    assert " " not in result


@pytest.mark.parametrize(
    ("locale", "written"),
    [
        pytest.param("de_DE", "23", id="de_DE-23"),
        pytest.param("it_IT", "155", id="it_IT-155"),
        pytest.param("es_ES", "21", id="es_ES-21"),
        pytest.param("es_ES", "200", id="es_ES-200"),
    ],
)
def test_no_soft_hyphen_in_output(locale, written):
    assert "\u00ad" not in normalize(written, locale=locale)
    detections = detect(written, _reading_detectors(locale))
    graph = compose_choices(resolve_choices(detections, source_text=written, locale=locale))
    assert all(
        "\u00ad" not in alternative.text
        for unit in graph.units
        for alternative in unit.alternatives
    )


def test_icu_rbnf_reading_strips_soft_hyphens():
    written = "23"
    detections = detect(written, _reading_detectors("de_DE"))
    graph = compose_choices(resolve_choices(detections, source_text=written, locale="de_DE"))
    icu_readings = [
        alternative
        for unit in graph.units
        for alternative in unit.alternatives
        if alternative.provenance.startswith("icu-rbnf:")
    ]
    assert icu_readings
    assert all("\u00ad" not in alternative.text for alternative in icu_readings)


def test_hangul_keeps_icu_spacing():
    assert normalize("13000", locale="ko_KR").strip() == "만 삼천"


def test_hangul_keeps_gaps_between_spoken_units():
    assert normalize("한1글", locale="ko_KR") == "한 일 글"


def test_offsets_tile_output_in_unspaced_text():
    source = "そこに鳥1羽がいます"
    result = normalize(source, locale="ja_JP", offsets=True)
    assert isinstance(result, NormalizedText)
    output_at = 0
    source_at = 0
    for unit in result.units:
        assert unit.output_span[0] == output_at
        output_at = unit.output_span[1]
        assert unit.source_span[0] == source_at
        assert source[unit.source_span[0] : unit.source_span[1]]
        source_at = unit.source_span[1]
    assert output_at == len(result.text)
    assert source_at == len(source)
    number = next(unit for unit in result.units if unit.reader == "number:decimal")
    assert source[number.source_span[0] : number.source_span[1]] == "1"
