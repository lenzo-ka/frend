from functools import cache

import icu
import pytest

from frend.context import _segments, _word_breaker
from frend.generated_rule_locales import locales_for
from frend.grouped_ids import grouped_id_shape
from frend.lexical_boundaries import letter_bearing_hyphenated_alnum_before
from frend.ranges import range_separator_classes
from frend.telephone import telephone_shape
from frend.verbalize import range_separators
from tools import bench_dry_helpers

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
def test_native_word_spans_keep_every_mixed_script_and_astral_offset_exact(locale, text):
    assert _segments(text, locale) == _legacy_segments(text, locale)


def test_word_segmentation_reuses_one_native_iterator_per_locale():
    assert isinstance(_word_breaker("en_US"), icu.BreakIterator)
    assert _word_breaker("en_US") is _word_breaker("en_US")


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


def test_native_span_benchmark_measures_distinct_text_cache_misses(monkeypatch):
    seen = []

    def record(text, start, end, *, locale):
        seen.append((text, start, end, locale))
        return (), ()

    monkeypatch.setattr(bench_dry_helpers, "neighbor_words", record)

    result = bench_dry_helpers.benchmark(32)

    assert len({text for text, _start, _end, _locale in seen}) == 32
    assert "neighbor_words_cache_miss" in result
