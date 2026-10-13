from functools import cache

import icu
import pytest

from frend.context import _segments
from frend.generated_rule_locales import locales_for
from frend.grouped_ids import grouped_id_shape
from frend.lexical_boundaries import letter_bearing_hyphenated_alnum_before
from frend.ranges import range_separator_classes
from frend.telephone import telephone_shape
from frend.verbalize import range_separators

_SPAN_TEXTS = (
    "alpha beta",
    "中文分词",
    "한국어 낱말",
    "e\u0301lan déjà",
    "A😀B",
    "👩\u200d🔬 works",
    "٣٤ 56",
    "word—word",
    " tabs\tand\nlines ",
    "can't 5.5",
    "x" * 95 + " 中 😀",
)


@cache
def _breaker(locale: str) -> icu.BreakIterator:
    return icu.BreakIterator.createWordInstance(icu.Locale(locale))


def _legacy_segments(text: str, locale: str) -> list[tuple[int, int]]:
    breaker = _breaker(locale)
    breaker.setText(text)
    boundaries = [breaker.first(), *breaker]
    if any(ord(character) > 0xFFFF for character in text):
        to_code_point = {}
        unit = 0
        for index, character in enumerate(text):
            to_code_point[unit] = index
            unit += 2 if ord(character) > 0xFFFF else 1
        to_code_point[unit] = len(text)
        boundaries = [to_code_point[boundary] for boundary in boundaries]
    return [
        (start, end)
        for start, end in zip(boundaries, boundaries[1:], strict=False)
        if text[start:end].strip()
    ]


@pytest.mark.parametrize("locale", locales_for("time"))
@pytest.mark.parametrize("text", _SPAN_TEXTS)
def test_icukit_word_spans_are_exactly_the_legacy_110_case_bridge(locale, text):
    assert _segments(text, locale) == tuple(_legacy_segments(text, locale))


@pytest.mark.parametrize("locale", locales_for("time"))
def test_runtime_and_recognizer_derive_the_same_normal_range_separators(locale):
    assert range_separators(locale) == range_separator_classes(locale).get("range", frozenset())


@pytest.mark.parametrize(
    ("text", "end", "expected"),
    (
        ("A-5", 1, True),
        ("A-5", 3, True),
        ("5-10", 1, False),
        ("5-10", 4, False),
        ("é-5", 1, True),
        ("٣-5", 1, False),
        ("xx A-5 yy", 6, True),
        ("xx 5-10 yy", 7, False),
    ),
)
def test_hyphenated_alphanumeric_boundary_is_unicode_and_offset_exact(text, end, expected):
    assert letter_bearing_hyphenated_alnum_before(text, end) is expected


def _legacy_width_shape(text: str, transform) -> str:
    result = []
    index = 0
    while index < len(text):
        if text[index] in "0123456789":
            end = index + 1
            while end < len(text) and text[end] in "0123456789":
                end += 1
            result.append(f"N{end - index}")
            index = end
        else:
            result.append(transform(text[index]))
            index += 1
    return "".join(result)


def test_width_shapes_keep_every_ascii_width_separator_and_terminal_x_key_exact():
    for width in range(1, 33):
        digits = "7" * width
        for separator in ("-", " ", ".", "/", "(", ")"):
            for terminal in ("", "x", "X"):
                surface = f"{digits}{separator}{digits}{terminal}"
                assert telephone_shape(surface) == _legacy_width_shape(surface, lambda item: item)
                assert grouped_id_shape(surface) == _legacy_width_shape(surface, str.upper)
