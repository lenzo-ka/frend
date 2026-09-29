"""Ranges ICU writes: icukit's range and interval readings, as frend speaks them.

icukit reads a range where ICU writes one: CLDR's number-range separator between two
amounts ("5–10", "$5–10", "10–15 kg"; ``number:range`` and ``measure:range``) and
CLDR's date-interval patterns ("1990–1995", "May 3 – 5, 2020", "10:30 – 11:45";
``date-interval:<skeleton>``). ICU writes the separator only; how a range's two ends
are joined when spoken (the connector, "to") is the locale's lexical form
(``data/<locale>/lexical.json`` ``range.connector``), and each end is read the way frend
reads that value alone.

:func:`from_icukit` turns such a reading into one frend speaks, ``range:icu``, whose
value is a :class:`RangeValue`: each end as a detection of its own (a number, a
currency amount, a measure, a date or a time), with the captures that say where each
field is written. A date interval's ends are cut the way ICU cuts the interval
(``icu.DateIntervalFormat``'s own spans), so "May 3 – 5, 2020" has "May 3" on the left
and "5, 2020" on the right.

The approximately readings icukit builds beside the ranges ("~3", "~3 km") are not
ranges and are never read here (:func:`icu_range_readers` leaves them out).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import icu
from icukit.detectors import Capture, DateTimeValue, NumberValue

__all__ = [
    "ICU_RANGE_TYPE",
    "RangeValue",
    "date_interval_readers",
    "from_icukit",
    "has_icu_ranges",
    "icu_range_readers",
]

ICU_RANGE_TYPE = "range:icu"

# The interval fields frend speaks on one end: a year, a month and a day, or an hour, its
# minutes and a day period. A skeleton that writes any other field (a weekday, a second,
# a time zone, an era) is not read as a range here, so no reading takes a span frend
# cannot say.
_SPOKEN_INTERVAL_FIELDS = frozenset("yMdLHhKkma")


def has_icu_ranges() -> bool:
    """Whether the installed icukit builds range readers (``icukit.engine.range_detectors``;
    icukit main after 0.8.0)."""
    import icukit.engine

    return hasattr(icukit.engine, "range_detectors")


def icu_range_readers(locale: str, readers: Iterable[object]) -> tuple[object, ...]:
    """icukit's range readers over the amount readers in ``readers`` (``number:range``
    over numbers, percents and currencies; ``measure:range`` over measures), each
    ``*:approximately`` reader left out. Empty where icukit builds none."""
    if not has_icu_ranges():
        return ()
    from icukit.detectors import DetectorSet
    from icukit.engine import range_detectors

    return tuple(
        reader
        for reader in range_detectors(locale, DetectorSet(tuple(readers))).detectors
        if not str(reader.type).endswith(":approximately")
    )


def date_interval_readers(locale: str) -> tuple[object, ...]:
    """icukit's generated date-interval readers for ``locale`` whose skeleton writes only
    fields frend speaks on a range's end (a year, month and day, or an hour, minutes and
    day period). Empty where icukit builds no range readers."""
    if not has_icu_ranges():
        return ()
    from icukit.engine import DATE_INTERVAL_FAMILY, generated_detectors

    return tuple(
        reader
        for reader in generated_detectors(locale, [DATE_INTERVAL_FAMILY]).detectors
        if set(str(reader.type).removeprefix("date-interval:")) <= _SPOKEN_INTERVAL_FIELDS
    )


@dataclass(frozen=True)
class RangeValue:
    """A written range: its ends, each a detection of its own, and how they are joined.

    ``left`` and ``right`` hold each end's readings (one each for an ICU range: the
    reading icukit parsed); each is a detection mapping (``type``, ``text``, ``value``,
    ``captures``), its captures placed within the end's own text.
    """

    left: tuple[Mapping, ...]
    separator: str  # "-", "–", ":", "x", "×"
    separator_class: Literal["range", "ratio", "dimension"]
    right: tuple[Mapping, ...]


def from_icukit(detection: Mapping) -> Mapping | None:
    """``detection`` as a ``range:icu`` reading when it is icukit's ``number:range``,
    ``measure:range`` or ``date-interval:*``; ``None`` for any other reading (an
    ``*:approximately`` reading among them)."""
    type_ = str(detection.get("type", ""))
    if type_ in ("number:range", "measure:range"):
        ends = _amount_ends(detection)
    elif type_.startswith("date-interval:") and not type_.startswith("date-interval:short-year"):
        ends = _interval_ends(detection, type_.removeprefix("date-interval:"))
    else:
        return None
    if ends is None:
        return None
    left, right = ends
    separator = str(getattr(_capture(detection, "separator"), "text", "")).strip()
    return {
        "type": ICU_RANGE_TYPE,
        "text": detection.get("text"),
        "start": detection.get("start"),
        "end": detection.get("end"),
        "value": RangeValue((left,), separator, "range", (right,)),
        "captures": tuple(detection.get("captures", ())),
        "source_type": type_,
    }


def _capture(detection: Mapping, name: str) -> Capture | None:
    return next(
        (c for c in detection.get("captures", ()) if getattr(c, "name", None) == name), None
    )


def _amount_ends(detection: Mapping) -> tuple[Mapping, Mapping] | None:
    start, end = _capture(detection, "start"), _capture(detection, "end")
    if start is None or end is None or start.value is None or end.value is None:
        return None
    unit_type = "measure" if str(detection["type"]).startswith("measure:") else None
    percent = not unit_type and any(_writes_percent(c.text, detection) for c in (start, end))
    return tuple(  # type: ignore[return-value]
        {
            "type": _amount_type(capture.value, unit_type, percent),
            "text": capture.text,
            "value": capture.value,
            "captures": (),
            "writes_unit": _writes_unit(capture.text),
        }
        for capture in (start, end)
    )


def _writes_percent(text: str, detection: Mapping) -> bool:
    spec = detection.get("spec")
    locale = str(getattr(spec, "locale", "en_US"))
    symbols = icu.DecimalFormatSymbols(icu.Locale(locale))
    return symbols.getSymbol(icu.DecimalFormatSymbols.kPercentSymbol) in text


def _amount_type(value: object, unit_type: str | None, percent: bool) -> str:
    if unit_type is not None:
        return f"measure:{getattr(value, 'unit', '')}"
    if isinstance(value, NumberValue) and value.currency is not None:
        return "number:currency"
    if percent:
        return "number:percent"
    return "number:decimal"


_AMOUNT_CHARS = frozenset("0123456789.,'+-−   ")


def _writes_unit(text: str) -> bool:
    """Whether an end's text writes more than its amount: a currency sign, a percent
    sign or a unit ("$5", "10%", "10 km"; not "5" or "1,000")."""
    return any(ch not in _AMOUNT_CHARS and not ch.isdecimal() for ch in text)


# ICU's date format fields, as the letters frend's date and time values carry.
_ICU_FIELDS = {
    icu.DateFormat.YEAR_FIELD: "y",
    icu.DateFormat.MONTH_FIELD: "M",
    icu.DateFormat.DATE_FIELD: "d",
    icu.DateFormat.HOUR_OF_DAY0_FIELD: "H",
    icu.DateFormat.HOUR_OF_DAY1_FIELD: "H",
    icu.DateFormat.HOUR0_FIELD: "H",
    icu.DateFormat.HOUR1_FIELD: "H",
    icu.DateFormat.MINUTE_FIELD: "m",
    icu.DateFormat.AM_PM_FIELD: "a",
}
_STANDALONE_MONTH_FIELD = 26  # UDAT_STANDALONE_MONTH_FIELD ("LLL")
_ICU_FIELDS[_STANDALONE_MONTH_FIELD] = "M"
_SPAN_CATEGORY = 0x1005  # UFIELD_CATEGORY_DATE_INTERVAL_SPAN
_DATE_CATEGORY = 1  # UFIELD_CATEGORY_DATE


def _interval_ends(detection: Mapping, skeleton: str) -> tuple[Mapping, Mapping] | None:
    """Each end of a date interval as a date or time detection of its own, holding the
    fields ICU writes on that end: ICU's interval format of the same two values says
    which fields each end writes (a field written outside both ends, "2020" in "May 3 –
    5, 2020", goes with the end it is written beside)."""
    value = detection.get("value")
    spec = detection.get("spec")
    locale = str(getattr(spec, "locale", "en_US"))
    start, end = getattr(value, "start", None), getattr(value, "end", None)
    if not isinstance(start, DateTimeValue) or not isinstance(end, DateTimeValue):
        return None
    placed = _placed_fields(start, end, skeleton, locale)
    if placed is None:
        return None
    twelve = any(letter in skeleton for letter in "hK")
    ends = []
    for side, point in ((0, start), (1, end)):
        written = {letter: place for (letter, where), place in placed.items() if where == side}
        if not written:
            return None
        ends.append(_end_detection(point, written, twelve, skeleton))
    return ends[0], ends[1]


def _placed_fields(
    start: DateTimeValue, end: DateTimeValue, skeleton: str, locale: str
) -> dict[tuple[str, int], tuple[int, str]] | None:
    """Where ICU writes each field of the interval of ``start`` and ``end``: for each
    field letter and end (0, 1) it is written on, its first position in ICU's text and
    that text. ICU's spans cover what differs between the two dates; a field written
    outside both ("May" in "May 3 – 5, 2020", "2020" after it) goes with the end it is
    written beside."""
    # The formatter writes in the process's time zone, so the two dates are made there
    # (the fields are what counts, not the instant).
    icu_locale = icu.Locale(locale)
    dates = []
    for point in (start, end):
        fields = dict(point.fields)
        calendar = icu.Calendar.createInstance(icu_locale)
        calendar.clear()
        calendar.set(
            fields.get("y", 2000), fields.get("M", 1) - 1, fields.get("d", 1),
            fields.get("H", 0), fields.get("m", 0), fields.get("s", 0),
        )  # fmt: skip
        dates.append(calendar.getTime())
    formatter = icu.DateIntervalFormat.createInstance(skeleton, icu_locale)
    formatted = formatter.formatToValue(icu.DateInterval(*dates))
    text = str(formatted)
    spans: dict[int, tuple[int, int]] = {}
    found: list[tuple[str, int, int]] = []
    position = icu.ConstrainedFieldPosition()
    while formatted.nextPosition(position):
        category, field = position.getCategory(), position.getField()
        if category == _SPAN_CATEGORY:
            spans[field] = (position.getStart(), position.getLimit())
        elif category == _DATE_CATEGORY and field in _ICU_FIELDS:
            found.append((_ICU_FIELDS[field], position.getStart(), position.getLimit()))
    if set(spans) != {0, 1}:
        return None  # ICU wrote one date, not an interval
    placed: dict[tuple[str, int], tuple[int, str]] = {}
    for letter, at, limit in found:
        side = next(
            (s for s, (a, b) in spans.items() if a <= at < b),
            0 if at < spans[0][0] else 1,
        )
        placed.setdefault((letter, side), (at, text[at:limit]))
    return placed


def _end_detection(
    point: DateTimeValue, written: Mapping[str, tuple[int, str]], twelve: bool, skeleton: str
) -> Mapping:
    """One end as a date or a time: the fields written on it, each a capture at the place
    ICU writes it (so a date's written order is known)."""
    fields = tuple((key, item) for key, item in point.fields if key in written)
    captures = [
        Capture(letter, written[letter][0], written[letter][0] + 1, written[letter][1], item)
        for letter, item in fields
    ]
    if not {"H", "m"} & set(written):
        return {
            "type": f"date:{skeleton}",
            "text": "",
            "value": DateTimeValue(fields, point.calendar),
            "captures": tuple(captures),
        }
    if twelve:
        hour = dict(point.fields).get("H", 0) % 12 or 12
        if "a" in written:
            # The day period is written on this end: the time reads its written 12-hour
            # hour and the period ("four p m").
            captures = [
                Capture("H", c.start, c.end, c.text, hour) if c.name == "H" else c for c in captures
            ]
            at, period = written["a"]
            captures.append(Capture("day-period", at, at + 1, period))
        else:
            # The period is written on the other end only: this end's hour is still the
            # 12-hour one written ("2:00 – 4:00 PM" is "two to four p m").
            fields = tuple((key, hour if key == "H" else item) for key, item in fields)
    return {
        "type": f"time:{skeleton}",
        "text": "",
        "value": DateTimeValue(fields, point.calendar),
        "captures": tuple(captures),
    }
