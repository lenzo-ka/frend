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

**Ranges written in running text** ("5-10", "5 - 10", "16:79", "3x4", "5-10 kg",
"$15,000-$25,000", "10:30-11:45"), which ICU writes nowhere: :class:`RangeDetector` reads
each as one span, ``range:dash``, ``range:ratio`` or ``range:dimension``, whose value is a
:class:`RangeValue`. What is a rule and what is trained is kept apart:

* **Rules** (stated, tested, never trained): the span's shape (R1: a ``range.separator``
  of the lexical table, or CLDR's en dash, with an ASCII digit before it and an ASCII
  digit, or a currency sign and one, after it), its spacing (R2: joined or spaced alike
  on both sides), no chain (R3: three or more digit groups are an identifier), the
  clock (R4a: a ratio that icukit's ``time:flexible`` reads whole is ``ratio:clock``),
  the sub-key (R4b, :func:`range_sub_key`), R6's refined training filter
  (:func:`punctuation_dash`: only a silent dash whose ends the real range verbalizer
  misreads is dropped; silence between correct ends is valid), and the end types (R7:
  numbers, percents, currency amounts, measures other than durations, and times, each
  the whole side: not glued to a word or, through punctuation, to another number).
* **Trained**: whether a span is emitted at all (E, :meth:`RangePriorTable.emits`: a
  sub-key's range triples against the single corpus tokens of its written shape, over
  the 90 training shards) and how its readings are ordered (J,
  :meth:`RangePriorTable.lookup`: the joint source the corpus says, per sub-key, blended
  toward its class). Both are ``data/<locale>/range_priors.json``
  (``tools/build_range_priors.py``); a locale without it emits no span, and the hyphen
  path of frend #43 reads what it always read.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Literal

import icu
from icukit.detectors import Capture, DateTimeValue, DetectorSet, NumberValue

from frend.abbreviation_variants import abbreviation_expansions
from frend.durations import fractional_duration_shape
from frend.locale_data import LOCALE_CACHE, canonical_locale, lexical_forms, measured_table

__all__ = [
    "EMIT_RATIO",
    "ICU_RANGE_TYPE",
    "RANGE_SUB_KEY_STRENGTH",
    "RANGE_TYPES",
    "RangeDetector",
    "RangePriorTable",
    "RangeValue",
    "date_interval_readers",
    "emit_relevant",
    "from_icukit",
    "has_icu_ranges",
    "icu_range_readers",
    "load_range_priors",
    "punctuation_dash",
    "range_class_key",
    "range_separator_classes",
    "range_sub_key",
    "written_sub_key",
]

ICU_RANGE_TYPE = "range:icu"

# The interval fields frend speaks on one end: a year, a month and a day, or an hour, its
# minutes and a day period. A skeleton that writes any other field (a weekday, a second,
# a time zone, an era) is not read as a range here, so no reading takes a span frend
# cannot say.
_SPOKEN_INTERVAL_FIELDS = frozenset("yMdLHhKkma")

try:
    from icukit.engine import DATE_INTERVAL_FAMILY, Family
except ImportError:  # icukit before generated date-interval readers
    SPOKEN_DATE_INTERVALS = None
else:
    # icukit memoizes generated families by identity.  Keep this subset family at
    # module scope so every locale can reuse that memo entry, and enumerate only
    # skeletons whose fields frend can actually speak.
    SPOKEN_DATE_INTERVALS = Family(
        "date-interval",
        lambda locale: [
            skeleton
            for skeleton in DATE_INTERVAL_FAMILY.enumerate(locale)
            if set(str(skeleton)) <= _SPOKEN_INTERVAL_FIELDS
        ],
        DATE_INTERVAL_FAMILY.invert,
        DATE_INTERVAL_FAMILY.skip_reason,
    )


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
    if not has_icu_ranges() or SPOKEN_DATE_INTERVALS is None:
        return ()
    from icukit.engine import generated_detectors

    return tuple(
        SpeakableDateIntervalDetector(reader)
        for reader in generated_detectors(locale, [SPOKEN_DATE_INTERVALS]).detectors
    )


@lru_cache(maxsize=LOCALE_CACHE)
def _date_interval_gang(locale: str) -> DetectorSet:
    """Retain the filtered interval readers as one compiled gang per locale."""
    return DetectorSet(tuple(date_interval_readers(locale)))


class SpeakableDateIntervalDetector:
    """One of icukit's date-interval readers, keeping only the readings frend can say as
    a range (:func:`from_icukit` reads them). ICU writes more fields on an end than the
    skeleton names where the two dates differ in a larger field ("5/1/2020, 10 AM –
    5/2/2020, 10 AM" in the ``h`` interval writes each end's date); such a reading is
    dropped here, so the span keeps the readings frend reads without it."""

    def __init__(self, reader: object) -> None:
        self.reader = reader
        self.type = reader.type
        self.locale = getattr(reader, "locale", None)
        self.skeleton = getattr(reader, "skeleton", None)
        self.group = getattr(reader, "group", "date-interval")

    def detect(self, text: str) -> list:
        return [
            found
            for found in self.reader.detect(text)
            if from_icukit(found, self.locale) is not None
        ]


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


def from_icukit(detection: Mapping, locale: str) -> Mapping | None:
    """``detection`` as a ``range:icu`` reading when it is icukit's ``number:range``,
    ``measure:range`` or ``date-interval:*``; ``None`` for any other reading (an
    ``*:approximately`` reading among them)."""
    type_ = str(detection.get("type", ""))
    if type_ in ("number:range", "measure:range"):
        ends = _amount_ends(detection, locale)
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


def _amount_ends(detection: Mapping, locale: str) -> tuple[Mapping, Mapping] | None:
    start, end = _capture(detection, "start"), _capture(detection, "end")
    if start is None or end is None or start.value is None or end.value is None:
        return None
    unit_type = "measure" if str(detection["type"]).startswith("measure:") else None
    percent = not unit_type and any(_writes_percent(c.text, locale) for c in (start, end))
    ends = []
    for capture in (start, end):
        type_ = _amount_type(capture.value, unit_type, percent)
        whole = _read_whole(capture.text, type_, locale)
        ends.append(
            {
                "type": type_,
                "text": capture.text,
                "value": capture.value if whole is None else whole["value"],
                "captures": () if whole is None else tuple(whole.get("captures", ())),
                "writes_unit": _writes_unit(capture.text),
                "number": _read_whole(_written_amount(capture.text), "number:decimal", locale),
            }
        )
    return ends[0], ends[1]


_AMOUNT = re.compile(r"[-\u2212+]?\d(?:[\d.,'\u00a0\u202f ]*\d)?")


def _written_amount(text: str) -> str:
    """The number an end writes, without its unit or currency ("79.20" in "79.20%",
    "5" in "$5", "10" in "10 km")."""
    found = _AMOUNT.search(text)
    return found.group(0) if found else ""


def _read_whole(text: str, type_: str, locale: str) -> Mapping | None:
    """``text`` read whole by the locale's number reader (``number:decimal``) or percent
    reader (``number:percent``): the reading and its captures in the end's own text
    ("79.20%": integer "79", fraction "20"), so the end is said as written. ``None`` for
    another type, or where the reader does not read the whole text."""
    if not text or type_ not in ("number:decimal", "number:percent"):
        return None
    from icukit.recognize import FlexibleNumberDetector, FlexiblePercentDetector

    reader = (FlexibleNumberDetector if type_ == "number:decimal" else FlexiblePercentDetector)(
        locale
    )
    return next(
        (
            found
            for found in reader.detect(text)
            if found.get("start") == 0
            and found.get("end") == len(text)
            and str(found.get("type")) == type_
        ),
        None,
    )


def _writes_percent(text: str, locale: str) -> bool:
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


_SKELETON_FIELDS = {"h": "H", "K": "H", "k": "H", "L": "M"}


def _utf16_index(text: str, at: int) -> int:
    """The code point index of UTF-16 offset ``at`` in ``text`` (ICU's positions are
    UTF-16; a character outside the BMP, as Adlam's digits are, takes two units)."""
    return len(text.encode("utf-16-le")[: 2 * at].decode("utf-16-le"))


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
    # The fields the skeleton names (an hour of any cycle is "H", a standalone month "M"),
    # and the day period ICU writes beside a 12-hour hour.
    named = {_SKELETON_FIELDS.get(letter, letter) for letter in skeleton} | {"a"}
    ends = []
    for side, point in ((0, start), (1, end)):
        written = {letter: place for (letter, where), place in placed.items() if where == side}
        if not written or not set(written) <= named:
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
        first, last = _utf16_index(text, at), _utf16_index(text, limit)
        placed.setdefault((letter, side), (first, text[first:last]))
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


# --------------------------------------------------------------------------------------
# Ranges written in running text (the ranges plan's P6).

# The span type of each separator class.
RANGE_TYPES = MappingProxyType(
    {"range": "range:dash", "ratio": "range:ratio", "dimension": "range:dimension"}
)
# A class as the range table keys it: the range class ("-" and the en dash pooled) is
# "dash".
_CLASS_KEYS = MappingProxyType({"range": "dash", "ratio": "ratio", "dimension": "dimension"})
_CLASS_OF_KEY = MappingProxyType({key: cls for cls, key in _CLASS_KEYS.items()})

# E: a sub-key is emitted as a span when its range triples are at least this many times
# its single corpus tokens of the same written shape. Stage 2 (the runtime-eval shards
# 90-94) found no value better than this beyond noise (0.5: +2 of 65,392 triples).
EMIT_RATIO = Decimal(1)
# E: a sub-key's own row decides only with at least this many observations (range
# triples and single tokens); a sparser one reads its class row.
RANGE_MIN_N = 50
# J: how many credited rows the class row's share is worth when a sub-key's share is
# blended toward it (as the spoken priors' SUB_KEY_PRIOR_STRENGTH). Stage 2 found no
# better value beyond noise (1: +7 of 65,392).
RANGE_SUB_KEY_STRENGTH = Decimal(5)

# R7: the readings a range's end may be, each the whole side. A duration ("10:30") is no
# end, and neither is another range or an approximately reading.
_END_TYPES = (
    "number:cardinal",
    "number:int",
    "number:decimal",
    "number:percent",
    "number:currency:",
    "measure:",
    "time:",
)
_ASCII_DIGITS = frozenset("0123456789")
# How far each side is read for an end (cut back to white space, so no word is split).
_END_REACH = 48


def range_class_key(separator_class: str) -> str:
    """The range table's key for a separator class: ``dash`` for the range class (the
    hyphen and CLDR's en dash pooled), else the class itself."""
    return _CLASS_KEYS[separator_class]


def range_separator_classes(locale: str) -> Mapping[str, frozenset[str]]:
    """R1's separators by class: the lexical table's ``range.separator`` (``range``:
    "-", ``ratio``: ":", ``dimension``: "x", "×"), plus CLDR's own number-range
    separator (the en dash) in the range class. Empty for a locale whose table has no
    ``range.connector`` (nothing to say a range with)."""
    return _separator_classes(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _separator_classes(locale: str) -> Mapping[str, frozenset[str]]:
    from frend.context import cldr_range_separator

    forms = lexical_forms(locale)
    connectors = (forms.get("range.connector") or {}).get("value") or {}
    written = (forms.get("range.separator") or {}).get("value") or {}
    out: dict[str, frozenset[str]] = {}
    for cls in RANGE_TYPES:
        separators = set(written.get(cls, ()))
        if cls == "range":
            cldr = cldr_range_separator(locale)
            if cldr:
                separators.add(cldr)
        if separators and connectors.get(cls):
            out[cls] = frozenset(separators)
    return MappingProxyType(out)


def _class_of(separator: str, classes: Mapping[str, frozenset[str]]) -> str | None:
    return next((cls for cls, members in classes.items() if separator in members), None)


def digit_groups(text: str, start: int, end: int, separators: frozenset[str]) -> list[int]:
    """The lengths of the ASCII digit groups joined, with no white space, by separators
    through ``text[start:end]``, left to right ("978-1-234" at its first "-": [3, 1, 3])."""
    left: list[int] = []
    at = start
    while True:
        first = at
        while first > 0 and text[first - 1] in _ASCII_DIGITS:
            first -= 1
        if first == at:
            break
        left.append(at - first)
        if first == 0 or text[first - 1] not in separators:
            break
        at = first - 1
    right: list[int] = []
    at = end
    while True:
        last = at
        while last < len(text) and text[last] in _ASCII_DIGITS:
            last += 1
        if last == at:
            break
        right.append(last - at)
        if last == len(text) or text[last] not in separators:
            break
        at = last + 1
    return [*reversed(left), *right]


def chain_length(text: str, at: int, separators: frozenset[str]) -> int:
    """R3: how many ASCII digit groups the separator at ``text[at]`` chains, each link a
    separator of the class spaced as this one is (joined, or spaced on both sides):
    "978-1-234" 3, "1 - 2 - 3" 3, "5-10" 2."""
    spaced = text[at - 1 : at].isspace()

    def link(sep_at: int) -> bool:
        before = text[sep_at - 1 : sep_at].isspace()
        after = text[sep_at + 1 : sep_at + 2].isspace()
        return before == after == spaced

    def digits_left(end: int) -> int:
        start = end
        while start > 0 and text[start - 1] in _ASCII_DIGITS:
            start -= 1
        return start

    def digits_right(start: int) -> int:
        end = start
        while end < len(text) and text[end] in _ASCII_DIGITS:
            end += 1
        return end

    count = 0
    sep = at
    while True:  # leftward
        end = sep
        while end > 0 and text[end - 1].isspace():
            end -= 1
        start = digits_left(end)
        if start == end:
            break
        count += 1
        prev = start
        while prev > 0 and text[prev - 1].isspace():
            prev -= 1
        if prev == 0 or text[prev - 1] not in separators or not link(prev - 1):
            break
        sep = prev - 1
    sep = at
    while True:  # rightward
        start = sep + 1
        while start < len(text) and text[start].isspace():
            start += 1
        end = digits_right(start)
        if start == end:
            break
        count += 1
        nxt = end
        while nxt < len(text) and text[nxt].isspace():
            nxt += 1
        if nxt >= len(text) or text[nxt] not in separators or not link(nxt):
            break
        sep = nxt
    return count


def _starts_an_amount(text: str) -> bool:
    """R1's right side: an ASCII digit, or a currency sign (General_Category ``Sc``)
    followed by one ("$10")."""
    if text[:1] in _ASCII_DIGITS:
        return True
    return len(text) > 1 and unicodedata.category(text[0]) == "Sc" and text[1] in _ASCII_DIGITS


def emit_relevant(
    left_written: str, separator: str, right_written: str, locale: str = "en_US"
) -> bool:
    """R1-R3 on a corpus triple (its ends as written tokens, joined as running text
    writes them): the separator is one of the locale's, the left token ends in an ASCII
    digit, the right starts with one (or a currency sign and one), and the joined text
    is no chain of three or more digit groups. Relevance for E and J: a triple these
    rules cannot emit on is no evidence about either."""
    classes = range_separator_classes(locale)
    cls = _class_of(separator, classes)
    if cls is None or not left_written or left_written[-1] not in _ASCII_DIGITS:
        return False
    if not _starts_an_amount(right_written):
        return False
    joined = left_written + separator + right_written
    return chain_length(joined, len(left_written), classes[cls]) < 3


def punctuation_dash(
    left: Sequence[str], middle: Sequence[str], right: Sequence[str], *, ends_match: bool
) -> bool:
    """R6 refined: a silent dash is dropped only when its ends are misread.

    Each row is ``(class, written, spoken)``.  Silence between ends that a range
    candidate says correctly is a valid silent range reading.  The builders determine
    ``ends_match`` with the real range verbalizer; this predicate retains only the
    stated punctuation shape and the result of that check.
    """
    del left, right
    return (
        middle[1] in ("-", "–")
        and middle[0] in ("VERBATIM", "PUNCT")
        and middle[2] == "sil"
        and not ends_match
    )


_GROUPED = re.compile(r"[0-9]{1,3}(?:,[0-9]{3})+")


def _ungrouped(text: str) -> str:
    """ASCII digits grouped by commas in threes, without the commas ("1,000" "1000")."""
    return text.replace(",", "") if _GROUPED.fullmatch(text) else text


def _sub_key(cls: str, left: str, right: str, clock: bool) -> str:
    """R4a/R4b: ``<class>:clock`` for a clock, ``<class>:<L>+<R>`` (ASCII digit lengths;
    ``:0`` added for a right end of two or more digits written with a leading zero)
    where both ends are ASCII digits only (grouping commas not counted), else
    ``<class>:other``."""
    key = _CLASS_KEYS[cls]
    if clock:
        return f"{key}:clock"
    # A number grouped by commas counts its digits ("1,000-2,000" is dash:4+4).
    left, right = _ungrouped(left), _ungrouped(right)
    if left and right and set(left) <= _ASCII_DIGITS and set(right) <= _ASCII_DIGITS:
        lead0 = ":0" if len(right) > 1 and right.startswith("0") else ""
        return f"{key}:{len(left)}+{len(right)}{lead0}"
    return f"{key}:other"


@lru_cache(maxsize=LOCALE_CACHE)
def _time_reader(locale: str):
    from icukit.recognize import FlexibleTimeDetector

    return FlexibleTimeDetector(locale)


@lru_cache(maxsize=1 << 14)
def is_clock(text: str, locale: str = "en_US") -> bool:
    """R4a: whether icukit's ``time:flexible`` reads ``text`` whole ("10:30"; not "16:79").
    Which strings are clocks is ICU's knowledge, not the range table's."""
    return any(
        found["start"] == 0 and found["end"] == len(text)
        for found in _time_reader(canonical_locale(locale)).detect(text)
    )


def written_sub_key(cls: str, left: str, separator: str, right: str, locale: str = "en_US") -> str:
    """R4a/R4b for a range written ``left``, ``separator``, ``right`` of class ``cls``
    (``range``, ``ratio``, ``dimension``): a ratio icukit reads as a clock is
    ``ratio:clock``."""
    clock = cls == "ratio" and is_clock(f"{left}{separator}{right}", locale)
    return _sub_key(cls, left, right, clock)


def range_sub_key(detection: Mapping) -> str:
    """The range table's sub-key for a range reading (R4a, R4b): the one its detector
    recorded, else derived from its ends' written text (``range:icu``: the range class,
    pooled with the hyphen as ``dash``)."""
    recorded = detection.get("sub_key")
    if recorded:
        return str(recorded)
    value = detection["value"]
    left = str(value.left[0].get("text", "")) if value.left else ""
    right = str(value.right[0].get("text", "")) if value.right else ""
    return _sub_key(value.separator_class, left, right, clock=False)


def emit_key(sub_key: str) -> str:
    """The key E is counted and read by: the sub-key without R4b's leading-zero mark (a
    single corpus token's shape is its digit lengths)."""
    return sub_key.removesuffix(":0")


# --------------------------------------------------------------------------------------
# The range table.


@dataclass(frozen=True)
class _Joint:
    matched: int
    sources: Mapping[str, int]


def _prior_count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _prior_count_mapping(value: object, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return {str(key): _prior_count(count, f"{field}.{key}") for key, count in value.items()}


class RangePriorTable:
    """``data/<locale>/range_priors.json``: E (``readings``: range triples against single
    tokens, per emit key and per class) and J (``kinds``: the credited joint sources per
    class and sub-key)."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        readings = document.get("readings")
        kinds = document.get("kinds")
        if not isinstance(readings, Mapping) or not isinstance(kinds, Mapping):
            raise ValueError("a range table needs readings and kinds")
        rows: dict[str, tuple[int, int]] = {}
        for key, row in readings.items():
            if not isinstance(row, Mapping):
                raise ValueError(f"readings.{key} must be a mapping")
            positives = _prior_count(row.get("range"), f"readings.{key}.range")
            singles = _prior_count_mapping(row.get("single_token"), f"readings.{key}.single_token")
            rows[str(key)] = (positives, sum(singles.values()))
        self._readings = MappingProxyType(rows)
        joint: dict[str, tuple[_Joint, Mapping[str, _Joint]]] = {}
        for cls, record in kinds.items():
            if not isinstance(record, Mapping):
                raise ValueError(f"kinds.{cls} must be a mapping")
            sources = _prior_count_mapping(
                record.get("source_matched", {}), f"kinds.{cls}.source_matched"
            )
            matched = _prior_count(record.get("matched"), f"kinds.{cls}.matched")
            if matched != sum(sources.values()):
                raise ValueError(f"kinds.{cls}: source counts must sum to matched")
            sub_key_rows = record.get("sub_keys", {})
            if not isinstance(sub_key_rows, Mapping):
                raise ValueError(f"kinds.{cls}.sub_keys must be a mapping")
            sub_keys = {}
            for sub_key, sub in sub_key_rows.items():
                if not isinstance(sub, Mapping):
                    raise ValueError(f"kinds.{cls}.sub_keys.{sub_key} must be a mapping")
                sub_sources = _prior_count_mapping(
                    sub.get("source_matched", {}),
                    f"kinds.{cls}.sub_keys.{sub_key}.source_matched",
                )
                sub_matched = _prior_count(
                    sub.get("matched"), f"kinds.{cls}.sub_keys.{sub_key}.matched"
                )
                if sub_matched != sum(sub_sources.values()):
                    raise ValueError(f"kinds.{cls}.sub_keys.{sub_key}: counts must sum")
                sub_keys[f"{cls}:{sub_key}"] = _Joint(sub_matched, MappingProxyType(sub_sources))
            joint[str(cls)] = (
                _Joint(matched, MappingProxyType(sources)),
                MappingProxyType(sub_keys),
            )
        self._kinds = MappingProxyType(joint)
        self.provenance = MappingProxyType(dict(document.get("provenance", {})))

    def emission(self, sub_key: str) -> tuple[int, int] | None:
        """E's row for ``sub_key``: (range triples, single tokens) of its emit key, or of
        its class where the key has fewer than ``RANGE_MIN_N`` observations; ``None``
        without either."""
        key = emit_key(sub_key)
        row = self._readings.get(key)
        if row is None or sum(row) < RANGE_MIN_N:
            # A key never observed, or seen too rarely to decide ("dash:4+10", 4 and 0),
            # reads its class row.
            row = self._readings.get(key.split(":", 1)[0])
        return row

    def emits(self, sub_key: str) -> bool:
        """E: whether a span is emitted for ``sub_key``: its range triples are at least
        ``EMIT_RATIO`` times its single tokens (a key never observed reads its class)."""
        row = self.emission(sub_key)
        if row is None:
            return False
        positives, negatives = row
        return positives > 0 and Decimal(positives) >= EMIT_RATIO * Decimal(negatives)

    def reading_prior(self, detection: Mapping):
        """The range span's reading prior: E's ``p = range / (range + single)`` over
        ``n = range + single`` for its sub-key (``None`` for any other reading)."""
        from frend.type_priors import ReadingPrior

        if not str(detection.get("type", "")).startswith("range:"):
            return None
        sub_key = range_sub_key(detection)
        row = self.emission(sub_key)
        if row is None:
            return ReadingPrior("range", sub_key, None, None, False, "unsupported", "unsupported")
        positives, negatives = row
        n = positives + negatives
        if n == 0 or positives == 0:
            return ReadingPrior("range", sub_key, None, n or None, False, "measured", "measured")
        return ReadingPrior(
            "range", sub_key, Decimal(positives) / Decimal(n), n, True, "measured", "measured"
        )

    def lookup(self, cls: str, source: str, sub_key: str | None = None):
        """J: ``source``'s share of the credited rows of class ``cls`` (``dash``,
        ``ratio``, ``dimension``), blended at ``sub_key`` toward the class share by
        ``RANGE_SUB_KEY_STRENGTH`` (the spoken priors' blend); ``None`` when measured at
        neither level. A sub-key never observed reads its class row."""
        from frend.spoken_priors import SourceMeasurement

        entry = self._kinds.get(cls)
        if entry is None:
            return None
        whole, sub_keys = entry
        kind_share = (
            Decimal(whole.sources[source]) / Decimal(whole.matched)
            if source in whole.sources and whole.matched
            else None
        )
        sub = sub_keys.get(sub_key) if sub_key is not None else None
        if sub is None:
            return (
                None if kind_share is None else SourceMeasurement(whole.sources[source], kind_share)
            )
        count = sub.sources.get(source, 0)
        if count == 0 and kind_share is None:
            return None
        strength = RANGE_SUB_KEY_STRENGTH
        share = (Decimal(count) + strength * (kind_share or Decimal(0))) / (
            Decimal(sub.matched) + strength
        )
        return SourceMeasurement(count, share)

    def connector_share(self, cls: str, connector: str) -> Decimal | None:
        """The class row's share of the sources joined by ``connector`` ("to", "silent"):
        R12's order for a lone ":" between numbers."""
        entry = self._kinds.get(cls)
        if entry is None or not entry[0].matched:
            return None
        whole = entry[0]
        said = sum(n for source, n in whole.sources.items() if f"+{connector}+" in source)
        return Decimal(said) / Decimal(whole.matched)


def load_range_priors(locale: str = "en_US") -> RangePriorTable | None:
    """``locale``'s range table (``data/<locale>/range_priors.json``, along the chain,
    never root), or ``None`` when it has none: no span is emitted, and ranges read as
    frend #43 reads them."""
    return _range_priors(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _range_priors(locale: str) -> RangePriorTable | None:
    document = measured_table("range_priors", locale)
    return None if document is None else RangePriorTable(document)


# --------------------------------------------------------------------------------------
# The detector.


class _Omitted:
    pass


_OMITTED = _Omitted()


def _span_captures(
    text: str, start: int, left: Mapping, at: int, separator: str, right_at: int, right: Mapping
) -> tuple[Capture, ...]:
    """A range span's captures, in the span's own text: each end's own captures (or the
    end whole), named ``start`` and ``end``, and the separator. A range is as structured
    as its ends and its separator (a cover's capture count ranks it against a reading
    of the same extent, "3x4" as letter-and-digit runs)."""

    def side(end: Mapping, name: str, base: int) -> list[Capture]:
        own = [
            Capture(name, base + c.start, base + c.end, c.text)
            for c in end.get("captures", ())
            if isinstance(getattr(c, "start", None), int)
        ]
        written = str(end.get("text", ""))
        return own or [Capture(name, base, base + len(written), written)]

    return (
        *side(left, "start", 0),
        Capture("separator", at - start, at - start + len(separator), separator),
        *side(right, "end", right_at - start),
    )


def _whole_before(text: str, at: int) -> bool:
    """R7's whole side, on the left: an end starting at ``at`` is not the tail of a word
    or of another number ("x5", "1/4" before ":81")."""
    if at == 0:
        return True
    before = text[at - 1]
    if before.isalnum():
        return False
    return before.isspace() or not text[at - 2 : at - 1].isdigit()


def _whole_after(text: str, at: int) -> bool:
    """R7's whole side, on the right: an end stopping at ``at`` is not the head of a word
    or of another number ("5th", "19" before "/28")."""
    if at == len(text):
        return True
    after = text[at]
    if after.isalnum():
        return False
    return after.isspace() or not text[at + 1 : at + 2].isdigit()


def _is_end_type(type_: str) -> bool:
    if type_.startswith("measure:duration") or type_.endswith((":range", ":approximately")):
        return False
    return type_.startswith(_END_TYPES)


@lru_cache(maxsize=LOCALE_CACHE)
def _month_numbers(locale: str) -> Mapping[str, int]:
    """ICU's wide and abbreviated month names, case-folded, to month numbers."""
    symbols = icu.DateFormatSymbols(icu.Locale(locale))
    names: dict[str, int] = {}
    for collection in (symbols.getMonths(), symbols.getShortMonths()):
        for month, name in enumerate(collection, 1):
            if name:
                names.setdefault(str(name).casefold(), month)
    return MappingProxyType(names)


def _month_number(text: str, locale: str) -> int | None:
    """A CLDR month name, or an existing lexicon abbreviation for one."""
    months = _month_numbers(locale)
    month = months.get(text.casefold())
    if month is not None:
        return month
    for expansion in abbreviation_expansions(text, locale):
        month = months.get(str(getattr(expansion, "text", "")).casefold())
        if month is not None:
            return month
    return None


def _month_detection(text: str, locale: str) -> Mapping | None:
    month = _month_number(text, locale)
    if month is None:
        return None
    return {
        "type": "date:M",
        "text": text,
        "start": 0,
        "end": len(text),
        "value": DateTimeValue((("M", month),), "gregorian"),
        "captures": (Capture("M", 0, len(text), text, month, "text"),),
    }


class RangeDetector:
    """A range written in running text, both ends in the text ("5-10", "5 - 10", "16:79",
    "3x4", "5-10 kg", "$15,000-$25,000"): one span, typed by its separator's class
    (``range:dash``, ``range:ratio``, ``range:dimension``), its value a
    :class:`RangeValue` of one end each, read by ``endpoints`` (the reading profile's
    own readers). R1-R4 and R7 are its rules; E (the locale's range table) decides
    whether a span is emitted. A locale without a table emits nothing."""

    type = "range"
    group = "range"

    def __init__(
        self,
        locale: str = "en_US",
        endpoints: Sequence[object] = (),
        *,
        table: RangePriorTable | None | _Omitted = _OMITTED,
    ) -> None:
        self.locale = canonical_locale(locale)
        self.endpoints = tuple(endpoints)
        self._endpoint_gang = DetectorSet(self.endpoints)
        self._table = table
        self._ends: dict[str, Mapping | None] = {}
        self._date_ends: dict[str, Mapping | None] = {}
        self._ordinal_ends: dict[str, Mapping | None] = {}

    @property
    def table(self) -> RangePriorTable | None:
        if isinstance(self._table, _Omitted):
            return load_range_priors(self.locale)
        return self._table

    # -- ends

    def _read(self, text: str) -> list:
        return self._endpoint_gang.detect(text) if text.strip() else []

    def end(self, text: str) -> Mapping | None:
        """``text`` read whole as a range's end (R7): the first reading, in icukit's
        order, that covers it and is an end type; ``None`` where none does. A plain
        number also carries its written number read whole (``number``), so its written
        digits are said ("05": "o five")."""
        if text in self._ends:
            return self._ends[text]
        found = None
        for reading in self._read(text):
            type_ = str(reading["type"])
            if reading["start"] == 0 and reading["end"] == len(text) and _is_end_type(type_):
                found = {
                    "type": type_,
                    "text": text,
                    "start": 0,
                    "end": len(text),
                    "value": reading["value"],
                    "captures": tuple(reading.get("captures", ())),
                    "writes_unit": _writes_unit(text),
                }
                if type_ == "number:decimal":
                    found["number"] = _read_whole(text, "number:decimal", self.locale)
                break
        self._ends[text] = found
        return found

    def _left_end(self, text: str, at: int) -> tuple[int, Mapping] | None:
        """The longest end reading of ``text`` that ends at ``at`` and is its whole side
        (not the tail of a longer word)."""
        low = max(0, at - _END_REACH)
        window = text[low:at]
        if low > 0:
            space = re.search(r"\s", window)
            if space is None:
                return None
            low += space.end()
            window = text[low:at]
        starts = sorted(
            {
                low + r["start"]
                for r in self._read(window)
                if r["end"] == len(window) and _is_end_type(str(r["type"]))
            }
        )
        for start in starts:
            if not _whole_before(text, start):
                continue
            end = self.end(text[start:at])
            if end is not None:
                return start, end
        return None

    def _right_end(self, text: str, at: int) -> tuple[int, Mapping] | None:
        """The longest end reading of ``text`` that starts at ``at`` and is its whole side
        (not the head of a longer word)."""
        high = min(len(text), at + _END_REACH)
        window = text[at:high]
        if high < len(text):
            spaces = [m.start() for m in re.finditer(r"\s", window)]
            if not spaces:
                return None
            window = window[: spaces[-1]]
        ends = sorted(
            {
                at + r["end"]
                for r in self._read(window)
                if r["start"] == 0 and _is_end_type(str(r["type"]))
            },
            reverse=True,
        )
        for stop in ends:
            if not _whole_after(text, stop):
                continue
            end = self.end(text[at:stop])
            if end is not None:
                return stop, end
        return None

    def date_end(self, text: str) -> Mapping | None:
        """A structured date/month written whole, excluding an otherwise bare year."""
        if text in self._date_ends:
            return self._date_ends[text]
        found = None
        for reading in self._read(text):
            value = reading.get("value")
            if reading["start"] != 0 or reading["end"] != len(text):
                continue
            if not isinstance(value, DateTimeValue) or not str(reading["type"]).startswith("date:"):
                continue
            fields = set(dict(value.fields))
            if fields == {"y"} and reading["type"] != "date:elided-year":
                continue
            found = {
                "type": reading["type"],
                "text": text,
                "start": 0,
                "end": len(text),
                "value": value,
                "captures": tuple(reading.get("captures", ())),
            }
            break
        if found is None:
            found = _month_detection(text, self.locale)
        self._date_ends[text] = found
        return found

    def _left_date_end(self, text: str, at: int) -> tuple[int, Mapping] | None:
        """The longest structured date/month ending immediately before ``at``."""
        low = max(0, at - _END_REACH)
        for start in range(low, at):
            if not _whole_before(text, start):
                continue
            found = self.date_end(text[start:at])
            if found is not None:
                return start, found
        return None

    def _right_date_end(self, text: str, at: int) -> tuple[int, Mapping] | None:
        """The longest structured date/month starting immediately after ``at``."""
        high = min(len(text), at + _END_REACH)
        for stop in range(high, at, -1):
            if not _whole_after(text, stop):
                continue
            found = self.date_end(text[at:stop])
            if found is not None:
                return stop, found
        return None

    def ordinal_end(self, text: str) -> Mapping | None:
        """A locale-valid written ordinal covering ``text`` whole."""
        if text in self._ordinal_ends:
            return self._ordinal_ends[text]
        found = None
        for reading in self._read(text):
            if (
                reading["start"] == 0
                and reading["end"] == len(text)
                and str(reading["type"]).startswith("ordinal:")
            ):
                found = {
                    "type": reading["type"],
                    "text": text,
                    "start": 0,
                    "end": len(text),
                    "value": reading["value"],
                    "captures": tuple(reading.get("captures", ())),
                }
                break
        self._ordinal_ends[text] = found
        return found

    def _left_ordinal_end(self, text: str, at: int) -> tuple[int, Mapping] | None:
        """The longest complete ordinal ending immediately before ``at``."""
        low = max(0, at - _END_REACH)
        for start in range(low, at):
            if not _whole_before(text, start):
                continue
            found = self.ordinal_end(text[start:at])
            if found is not None:
                return start, found
        return None

    def _right_ordinal_end(self, text: str, at: int) -> tuple[int, Mapping] | None:
        """The longest complete ordinal starting immediately after ``at``."""
        high = min(len(text), at + _END_REACH)
        for stop in range(high, at, -1):
            if not _whole_after(text, stop):
                continue
            found = self.ordinal_end(text[at:stop])
            if found is not None:
                return stop, found
        return None

    def _ordinal_chain_length(self, text: str, at: int, separators: frozenset[str]) -> int:
        """The ordinal groups joined through ``text[at]`` with matching spacing."""
        spans = {
            (reading["start"], reading["end"])
            for reading in self._read(text)
            if str(reading["type"]).startswith("ordinal:")
        }
        spaced = text[at - 1 : at].isspace()

        def link(separator_at: int) -> bool:
            before = text[separator_at - 1 : separator_at].isspace()
            after = text[separator_at + 1 : separator_at + 2].isspace()
            return before == after == spaced

        count = 0
        separator_at = at
        while True:  # leftward
            end = separator_at
            while end > 0 and text[end - 1].isspace():
                end -= 1
            starts = [start for start, stop in spans if stop == end]
            if not starts:
                break
            count += 1
            start = min(starts)
            previous = start
            while previous > 0 and text[previous - 1].isspace():
                previous -= 1
            if previous == 0 or text[previous - 1] not in separators or not link(previous - 1):
                break
            separator_at = previous - 1

        separator_at = at
        while True:  # rightward
            start = separator_at + 1
            while start < len(text) and text[start].isspace():
                start += 1
            stops = [stop for ordinal_start, stop in spans if ordinal_start == start]
            if not stops:
                break
            count += 1
            stop = max(stops)
            following = stop
            while following < len(text) and text[following].isspace():
                following += 1
            if following == len(text) or text[following] not in separators or not link(following):
                break
            separator_at = following
        return count

    def _day_end(self, left: Mapping, right: Mapping) -> Mapping | None:
        """A numeric right end as the left date's day, completing an abbreviated day."""
        value = left.get("value")
        if not isinstance(value, DateTimeValue):
            return None
        fields = dict(value.fields)
        if not {"M", "d"} <= set(fields):
            return None
        written = str(right.get("text", ""))
        if not written.isascii() or not written.isdigit() or len(written) > 2:
            return None
        left_day = int(fields["d"])
        day = int(written)
        left_capture = _capture(left, "d")
        left_written = str(getattr(left_capture, "text", left_day))
        if len(written) < len(left_written):
            day = int(left_written[: len(left_written) - len(written)] + written)
            if day <= left_day:
                return None
        if not 1 <= day <= 31:
            return None
        return {
            "type": "date:d",
            "text": written,
            "start": 0,
            "end": len(written),
            "value": DateTimeValue((("d", day),), value.calendar),
            "captures": (Capture("d", 0, len(written), written, day, "numeric"),),
        }

    def _date_candidate(
        self, text: str, at: int, left_edge: int, right_edge: int, separator: str
    ) -> dict | None:
        """A hyphen/dash joining structured dates, months, or a date and its day."""
        window = text[max(0, left_edge - _END_REACH) : left_edge]
        if not any(char.isalpha() or char in "'’" for char in window):
            return None
        words = re.findall(r"[^\W\d_]+\.?", window, re.UNICODE)
        plausible_month = any(_month_number(word, self.locale) is not None for word in words)
        plausible_elision = re.search(r"['’][0-9]{2}\s*$", window) is not None
        if not plausible_month and not plausible_elision:
            return None
        left = self._left_date_end(text, left_edge)
        if left is None:
            return None
        start, left_end = left
        right = self._right_date_end(text, right_edge)
        if right is None:
            numeric = self._right_end(text, right_edge)
            if numeric is None:
                return None
            stop, numeric_end = numeric
            right_end = self._day_end(left_end, numeric_end)
            if right_end is None:
                return None
        else:
            stop, right_end = right
        return {
            "type": "range:date",
            "start": start,
            "end": stop,
            "text": text[start:stop],
            "value": RangeValue((left_end,), separator, "range", (right_end,)),
            "captures": _span_captures(text, start, left_end, at, separator, right_edge, right_end),
            "sub_key": "dash:date",
            "rule": "date-structure",
        }

    def _ordinal_candidate(
        self, text: str, at: int, left_edge: int, right_edge: int, separator: str
    ) -> dict | None:
        """A range dash joining two complete locale-valid ordinal surfaces."""
        # Match the ordinary dash rule's spacing: joined, or spaced on both sides.
        if (left_edge < at) != (right_edge > at + 1):
            return None
        left = self._left_ordinal_end(text, left_edge)
        right = self._right_ordinal_end(text, right_edge)
        if left is None or right is None:
            return None
        (start, left_end), (stop, right_end) = left, right
        return {
            "type": "range:dash",
            "start": start,
            "end": stop,
            "text": text[start:stop],
            "value": RangeValue((left_end,), separator, "range", (right_end,)),
            "captures": _span_captures(text, start, left_end, at, separator, right_edge, right_end),
            "sub_key": "dash:ordinal",
            "rule": "ordinal-structure",
        }

    def sub_key(self, cls: str, left: str, separator: str, right: str) -> str:
        """R4a/R4b for a span's written ends."""
        return written_sub_key(cls, left, separator, right, self.locale)

    def _fractional_duration(self, text: str) -> bool:
        """Whether the measured duration reader resolves the whole token to seconds."""
        if not fractional_duration_shape(text, self.locale):
            return False
        units = {
            str(getattr(reading.get("value"), "unit", ""))
            for reading in self._read(text)
            if reading["start"] == 0
            and reading["end"] == len(text)
            and reading["type"] == "measure:duration:numeric"
        }
        return units == {"second"}

    # -- spans

    def candidates(self, text: str) -> list[dict]:
        """Every span R1-R3 and R7 allow in ``text``, before E: each with its sub-key."""
        classes = range_separator_classes(self.locale)
        if not classes:
            return []
        found = []
        for at, char in enumerate(text):
            cls = _class_of(char, classes)
            if cls is None:
                continue
            left_edge = at
            while left_edge > 0 and text[left_edge - 1].isspace():
                left_edge -= 1
            right_edge = at + 1
            while right_edge < len(text) and text[right_edge].isspace():
                right_edge += 1
            if cls == "range":
                date_candidate = self._date_candidate(text, at, left_edge, right_edge, char)
                if date_candidate is not None:
                    found.append(date_candidate)
                ordinal_candidate = self._ordinal_candidate(text, at, left_edge, right_edge, char)
                if (
                    ordinal_candidate is not None
                    and self._ordinal_chain_length(text, at, classes[cls]) < 3
                ):
                    found.append(ordinal_candidate)
            # R1: an ASCII digit before, an ASCII digit (or a currency sign and one) after.
            if left_edge == 0 or text[left_edge - 1] not in _ASCII_DIGITS:
                continue
            if not _starts_an_amount(text[right_edge : right_edge + 2]):
                continue
            # R2: joined or spaced alike on both sides ("10 -5" writes a signed number).
            if (left_edge < at) != (right_edge > at + 1):
                continue
            # R3: a chain of three or more digit groups is an identifier, joined or
            # spaced ("1-2-3", "1 - 2 - 3", "2008 - 09 - 30").
            if chain_length(text, at, classes[cls]) >= 3:
                continue
            left = self._left_end(text, left_edge)
            right = self._right_end(text, right_edge)
            if left is None or right is None:
                continue
            (start, left_end), (stop, right_end) = left, right
            # A fractional colon form that ICU recognizes as elapsed minutes and
            # seconds is not a ratio.  Unfractioned and invalid clocks retain the
            # range table's existing ratio behavior.
            if (
                cls == "ratio"
                and start == 0
                and stop == len(text)
                and self._fractional_duration(text)
            ):
                continue
            left_end, right_end = self.carried(left_end, right_end)
            sub_key = self.sub_key(cls, left_end["text"], char, right_end["text"])
            found.append(
                {
                    "type": RANGE_TYPES[cls],
                    "start": start,
                    "end": stop,
                    "text": text[start:stop],
                    "value": RangeValue((left_end,), char, cls, (right_end,)),
                    "captures": _span_captures(
                        text, start, left_end, at, char, right_edge, right_end
                    ),
                    "sub_key": sub_key,
                }
            )
        return found

    def carried(self, left: Mapping, right: Mapping) -> tuple[Mapping, Mapping]:
        """R10: a currency written on the left end only ("$5-10") is the right end's too,
        so the range can say it in place and at the end ("five dollars to ten", "five
        to ten dollars"): the right end, still written "10", is read as the currency
        amount it is. Any other pair is unchanged."""
        if not str(left["type"]).startswith("number:currency") or right["type"] != "number:decimal":
            return left, right
        sign = re.match(r"\D*", str(left["text"])).group(0)
        if not sign or not any(unicodedata.category(char) == "Sc" for char in sign):
            return left, right
        amount = self.end(sign + str(right["text"]))
        if amount is None or not str(amount["type"]).startswith("number:currency"):
            return left, right
        offset = len(sign)
        return left, {
            **amount,
            "text": right["text"],
            "end": len(str(right["text"])),
            "captures": tuple(
                Capture(c.name, c.start - offset, c.end - offset, c.text, c.value)
                for c in amount["captures"]
                if c.start >= offset
            ),
            "writes_unit": False,
            "number": right.get("number"),
        }

    def ends(self, left: str, right: str) -> tuple[Mapping, Mapping] | None:
        """A corpus triple's ends read as the detector reads them (the range builders)."""
        left_end, right_end = self.end(left), self.end(right)
        if left_end is None or right_end is None:
            return None
        return self.carried(left_end, right_end)

    def detect(self, text: str) -> list[dict]:
        table = self.table
        candidates = self.candidates(text)
        date_candidates = [found for found in candidates if found.get("rule") == "date-structure"]
        if date_candidates:
            # Where icukit reads the same span as an interval, keep its structure and
            # do not build a parallel frend reading.
            icu_spans = {
                (found["start"], found["end"])
                for found in _date_interval_gang(self.locale).detect(text)
            }
            date_candidates = [
                found
                for found in date_candidates
                if (found["start"], found["end"]) not in icu_spans
            ]
        ordinal_candidates = [
            found for found in candidates if found.get("rule") == "ordinal-structure"
        ]
        measured = (
            []
            if table is None
            else [
                found
                for found in candidates
                if found.get("rule") not in {"date-structure", "ordinal-structure"}
                and table.emits(found["sub_key"])
            ]
        )
        return [*date_candidates, *ordinal_candidates, *measured]
