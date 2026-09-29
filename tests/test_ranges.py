"""Ranges ICU writes: icukit's range and date-interval readers in the spoken profile,
each read as one range, its ends read as frend reads each value alone and joined by the
locale's connector patterns (``frend.ranges``, ``frend.verbalize._spoken_range``)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from icukit.detectors import NumberValue, detect  # noqa: E402

from frend import resolve_lattice  # noqa: E402
from frend.context import TextContext  # noqa: E402
from frend.ranges import RangeValue, has_icu_ranges  # noqa: E402
from frend.spoken_priors import normalize_spoken  # noqa: E402
from frend.verbalize import _fill_slot, _spoken_range, verbalize_lattice  # noqa: E402

needs_icu_ranges = pytest.mark.skipif(
    not has_icu_ranges(),
    reason="this icukit builds no range readers (icukit.engine.range_detectors)",
)


def _readings(text: str, count: int = 2) -> list[tuple[str, str]]:
    """The first ``count`` readings of ``text`` as the evaluator reads a token (alone, its
    own context), each with its source: a span read as one unit gives that unit's
    readings; a span read as several gives its one first reading, sourced "(units)"."""
    from reading_profile import reading_detectors

    detections = list(detect(text, reading_detectors("en_US")))
    verbalized = verbalize_lattice(
        resolve_lattice(detections, source_text=text), context=TextContext(text)
    )
    units = verbalized.best_path.units
    if len(units) == 1:
        return [
            (normalize_spoken(item.text), item.provenance) for item in units[0].alternatives[:count]
        ]
    said = "".join(
        u.best.text if u.best.provenance == "surface:passthrough" else f" {u.best.text} "
        for u in units
    )
    return [(normalize_spoken(said), "(units)")]


def _assert_range_reads(text: str, said: str) -> None:
    readings = _readings(text)
    assert said in [reading for reading, _ in readings], readings
    assert all(source.startswith("range:") for _, source in readings), readings


# ------------------------------------------------------------------ ICU's own ranges


@needs_icu_ranges
def test_icu_en_dash_years():
    """CLDR's range separator between two years reads as one range of years (frend
    #43 already says "to" for the en dash between the three units it read)."""
    _assert_range_reads("1990–1995", "nineteen ninety to nineteen ninety five")


@needs_icu_ranges
def test_icu_spaced_interval():
    """CLDR's interval fallback, spaced ("{0} – {1}"), reads as the same range."""
    _assert_range_reads("1990 – 1995", "nineteen ninety to nineteen ninety five")


@needs_icu_ranges
def test_icu_measure_range_moves_the_unit():
    """A unit written once, after the right end, is said once, after it."""
    _assert_range_reads("5–10 km", "five to ten kilometers")


@needs_icu_ranges
def test_short_second_year_after_en_dash():
    """A four-digit left end reads as a year first, as the hyphen's "1990-95" does (the
    interim four-digit-year rule; the ranges plan's P6 measures and replaces both)."""
    (first, source), *_ = _readings("1990–95", 1)
    assert first == "nineteen ninety to ninety five", (first, source)
    assert source.startswith("range:"), source


@needs_icu_ranges
def test_money_range_offers_the_currency_in_place_and_at_the_end():
    """ "$5–10": the currency is written on the left only; both placements are offered,
    unmeasured (the ranges plan's (R))."""
    said = [reading for reading, _ in _readings("$5–10", 16)]
    assert "five dollars to ten" in said
    assert "five to ten dollars" in said


@needs_icu_ranges
def test_date_interval_ends_are_cut_where_icu_writes_them():
    """ICU writes "May 3 – 5, 2020" with the month on the left end and the year on the
    right: each end says what it writes."""
    assert _readings("May 3–5, 2020", 1)[0][0] == "may third to fifth twenty twenty"
    assert _readings("10:00 AM – 2:00 PM", 1)[0][0] == "ten a m to two p m"


def test_approximately_is_not_in_the_profile():
    """icukit builds an approximately reader beside each range reader; "~3" is no range,
    and none of them is read (filtering "number:approximately" alone would leave
    "measure:approximately" on "~3 km")."""
    from reading_profile import reading_detectors

    detectors = reading_detectors("en_US")
    assert not any(str(getattr(d, "type", "")).endswith(":approximately") for d in detectors)
    for text in ("~3", "~3 km"):
        types = [str(d["type"]) for d in detect(text, detectors)]
        assert not any(t.endswith(":approximately") for t in types), (text, types)


# ------------------------------------------------------------------ connector slots

_OT_DO = {
    "id": "ot-do",
    "pattern": "от {0} до {1}",
    "slots": {
        "0": {"case": "genitive", "gender": "masculine"},
        "1": {"case": "genitive", "gender": "masculine"},
    },
}


def _numbers(left: str, right: str) -> RangeValue:
    def end(decimal: str) -> dict:
        return {"type": "number:decimal", "text": decimal, "value": NumberValue(decimal=decimal)}

    return RangeValue((end(left),), "–", "range", (end(right),))


def test_russian_genitive_slot_fills_from_rbnf():
    """A slot's case and gender pick the locale's RBNF rule set (fixture pattern: ru has
    no range connector in its lexical table yet)."""
    said = _spoken_range(_numbers("5", "10"), "ru_RU", apply_source_priors=False, patterns=[_OT_DO])
    assert [item.text for item in said] == ["от пяти до десяти"]


def test_unfillable_slot_skips_the_pattern():
    """A slot the locale has no rule set for is not said in another case: the pattern is
    left out."""
    wrong = {**_OT_DO, "slots": {"0": {"case": "genitive", "gender": "xx"}, "1": {}}}
    assert _fill_slot({"value": NumberValue(decimal="5")}, wrong["slots"]["0"], "ru_RU") is None
    said = _spoken_range(_numbers("5", "10"), "ru_RU", apply_source_priors=False, patterns=[wrong])
    assert said == ()
