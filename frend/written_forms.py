"""Recognize written forms ICU writes nowhere, which icukit leaves to frend.

kal ruled these frend's, as URLs are: no locale pattern produces them, so icukit does
not read them. Each is read into a value frend already speaks where one fits, so its
speech stays ICU's (numbers, era names) or the corpus's:

- a clock time with a space after the colon ("2: 13"), as ``time:spaced-colon``;
- a case citation's volume ("339 U.S. 629"; a page must follow), as
  ``number:cardinal:citation`` with the reporter as a capture;
- spaced single digits ("6 3"), as ``number:digits``, said digit by digit;
- numeric runs with two or more dots ("3.12.1"), as ``number:dotted-run``. Bare
  versions use the version reader's component and separator readings; structural
  IPv4 addresses use the corpus-measured electronic digit and dot readings. Valid
  English Y.M.D and unambiguous M.D.Y calendar dates fill icukit's recognition gap
  as ``date:dotted`` and use frend's existing date composition;
- an era written with dotted letters, attached to the year, or before it ("500
  B.C.", "4AD", "A.D. 1066"), as ``date:era``; "500 BC." is ICU's "500 BC" and a
  sentence-final period, left to icukit.
- an apostrophe-elided two-digit year (``'94`` or ``’94``), as
  ``date:elided-year``; ranges of them are joined by frend's range reader.

The shapes are icukit's census families (``coverage/gap_families.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from icukit.detectors import Capture, DateTimeValue, NumberValue

from frend.grouped_ids import GroupedDigitsDetector
from frend.locale_data import canonical_locale
from frend.telephone import TelephoneDetector

__all__ = ["DigitsValue", "DottedNumericValue", "WrittenFormsDetector"]

_SPACED_COLON = re.compile(r"(?<![\w:.])(\d{1,2}): (\d{2})(?![\w:])")
# A citation names its page too ("339 U.S. 629"); without one, "339 U.S. troops" is prose.
_CITATION = re.compile(r"(?<![\w.,])(\d+) (U\. ?S\.)(?= \d)")
_SPACED_DIGITS = re.compile(r"(?<![\w.,:/-])\d(?: \d)+(?![\w.,:/-])")
_DOTTED_NUMERIC_RUN = re.compile(r"(?<![\w.])\d+(?:\.\d+){2,}(?![\w.])")
# Dotted letters only: a period after "BC" or "AD" ends the sentence, not the era.
_ERA_WRITTEN = r"B\.C\.E\.|B\.C\.|A\.D\.|C\.E\."
_ERA_AFTER = re.compile(rf"(?<![\w.])(\d{{1,4}})( ?)({_ERA_WRITTEN}|AD|BC)(?!\w)")
_ERA_BEFORE = re.compile(r"(?<![\w.])(A\.D\.|AD)( ?)(\d{1,4})(?![\w.])")
_ELIDED_YEAR = re.compile(r"(?<!\w)['’](\d{2})(?!\d)")
_BEFORE_ERA = ("B",)


@dataclass(frozen=True)
class DigitsValue:
    """Spaced single digits, said one by one ("6 3")."""

    digits: str


@dataclass(frozen=True)
class DottedNumericValue:
    """Numeric groups separated by dots, with structural IPv4 and date classification."""

    groups: tuple[str, ...]

    @property
    def is_ipv4(self) -> bool:
        """Whether this is exactly four ASCII-decimal groups, each in 0--255."""
        return len(self.groups) == 4 and all(
            group.isascii() and group.isdecimal() and int(group) <= 255 for group in self.groups
        )

    @property
    def is_date(self) -> bool:
        """Whether three groups form a valid Y.M.D, M.D.Y, or D.M.Y date."""
        if len(self.groups) != 3:
            return False

        first, second, third = self.groups
        candidates: list[tuple[str, str, str]] = []
        if len(first) == 4 and 1 <= len(second) <= 2 and 1 <= len(third) <= 2:
            candidates.append((first, second, third))
        if len(third) in (2, 4) and 1 <= len(first) <= 2 and 1 <= len(second) <= 2:
            candidates.extend(((third, first, second), (third, second, first)))

        for year_text, month_text, day_text in candidates:
            year = int(year_text)
            if len(year_text) == 2:
                year += 2000
            try:
                date(year, int(month_text), int(day_text))
            except ValueError:
                continue
            return True
        return False

    @property
    def date_fields(self) -> tuple[tuple[str, int], ...] | None:
        """A calendar-valid English dotted date outside icukit's D.M.Y coverage.

        A four-digit first group makes Y.M.D unambiguous. M.D.Y is claimed only when
        its day is above 12: otherwise the same surface is a valid D.M.Y form already
        read by icukit. Two-digit final years follow icukit's 2000-based interpretation.
        """
        if len(self.groups) != 3:
            return None

        first, second, third = self.groups
        if len(first) == 4 and 1 <= len(second) <= 2 and 1 <= len(third) <= 2:
            year, month, day = int(first), int(second), int(third)
        elif (
            len(third) in (2, 4)
            and 1 <= len(first) <= 2
            and 1 <= len(second) <= 2
            and int(second) > 12
        ):
            year, month, day = int(third), int(first), int(second)
            if len(third) == 2:
                year += 2000
        else:
            return None

        try:
            date(year, month, day)
        except ValueError:
            return None
        return (("y", year), ("M", month), ("d", day))


def _era_index(text: str) -> int:
    return 0 if text.lstrip().upper().startswith(_BEFORE_ERA) else 1


def _detection(text: str, start: int, end: int, type_: str, value: object, captures) -> dict:
    return {
        "text": text[start:end],
        "start": start,
        "end": end,
        "type": type_,
        "value": value,
        "captures": tuple(captures),
    }


class WrittenFormsDetector:
    """Detect the written forms ICU writes nowhere that frend reads."""

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = canonical_locale(locale)
        self.telephone = TelephoneDetector(self.locale)
        self.grouped_digits = GroupedDigitsDetector(self.locale)

    def detect(self, text: str) -> list[dict]:
        found = self.telephone.detect(text)
        telephone_spans = {(item["start"], item["end"]) for item in found}
        found.extend(
            item
            for item in self.grouped_digits.detect(text)
            if (item["start"], item["end"]) not in telephone_spans
        )
        for match in _SPACED_COLON.finditer(text):
            hour, minute = int(match.group(1)), int(match.group(2))
            if hour > 23 or minute > 59:
                continue
            found.append(
                _detection(
                    text,
                    match.start(),
                    match.end(),
                    "time:spaced-colon",
                    DateTimeValue((("H", hour), ("m", minute)), "gregorian"),
                    (
                        Capture("H", match.start(1), match.end(1), match.group(1), hour, "numeric"),
                        Capture(
                            "m", match.start(2), match.end(2), match.group(2), minute, "numeric"
                        ),
                    ),
                )
            )
        if self.locale.partition("_")[0] == "en":
            for match in _CITATION.finditer(text):
                found.append(
                    _detection(
                        text,
                        match.start(),
                        match.end(),
                        "number:cardinal:citation",
                        NumberValue(match.group(1), None),
                        (
                            Capture(
                                "integer",
                                match.start(1),
                                match.end(1),
                                match.group(1),
                                match.group(1),
                                "numeric",
                            ),
                            Capture(
                                "reporter",
                                match.start(2),
                                match.end(2),
                                match.group(2),
                                None,
                                "symbol",
                            ),
                        ),
                    )
                )
        for match in _SPACED_DIGITS.finditer(text):
            digits = match.group().replace(" ", "")
            found.append(
                _detection(
                    text,
                    match.start(),
                    match.end(),
                    "number:digits",
                    DigitsValue(digits),
                    (
                        Capture(
                            "digits", match.start(), match.end(), match.group(), digits, "numeric"
                        ),
                    ),
                )
            )
        for match in _DOTTED_NUMERIC_RUN.finditer(text):
            groups = tuple(match.group().split("."))
            value = DottedNumericValue(groups)
            date_fields = value.date_fields
            if self.locale.partition("_")[0] == "en" and date_fields is not None:
                positions = tuple(re.finditer(r"\d+", match.group()))
                names = ("y", "M", "d") if len(groups[0]) == 4 else ("M", "d", "y")
                fields = dict(date_fields)
                found.append(
                    _detection(
                        text,
                        match.start(),
                        match.end(),
                        "date:dotted",
                        DateTimeValue(date_fields, "gregorian"),
                        tuple(
                            Capture(
                                name,
                                match.start() + position.start(),
                                match.start() + position.end(),
                                position.group(),
                                fields[name],
                                "numeric",
                            )
                            for name, position in zip(names, positions, strict=True)
                        ),
                    )
                )
                continue
            if value.is_date:
                continue
            found.append(
                _detection(
                    text,
                    match.start(),
                    match.end(),
                    "number:dotted-run",
                    value,
                    (
                        Capture(
                            "groups",
                            match.start(),
                            match.end(),
                            match.group(),
                            groups,
                            "numeric",
                        ),
                    ),
                )
            )
        if self.locale.partition("_")[0] == "en":
            for match in _ELIDED_YEAR.finditer(text):
                year = int(match.group(1))
                found.append(
                    _detection(
                        text,
                        match.start(),
                        match.end(),
                        "date:elided-year",
                        DateTimeValue((("y", year),), "gregorian"),
                        (
                            Capture(
                                "y",
                                match.start(1),
                                match.end(1),
                                match.group(1),
                                year,
                                "numeric",
                            ),
                        ),
                    )
                )
            for match in _ERA_AFTER.finditer(text):
                year, era = int(match.group(1)), match.group(3)
                if era in ("AD", "BC") and match.group(2) == " ":
                    continue  # "500 BC" is ICU's own form; icukit reads it
                found.append(
                    _detection(
                        text,
                        match.start(),
                        match.end(),
                        "date:era",
                        DateTimeValue((("G", _era_index(era)), ("y", year)), "gregorian"),
                        (
                            Capture(
                                "y", match.start(1), match.end(1), match.group(1), year, "numeric"
                            ),
                            Capture("era", match.start(3), match.end(3), era, None, "short"),
                        ),
                    )
                )
            for match in _ERA_BEFORE.finditer(text):
                year, era = int(match.group(3)), match.group(1)
                found.append(
                    _detection(
                        text,
                        match.start(),
                        match.end(),
                        "date:era",
                        DateTimeValue((("G", 1), ("y", year)), "gregorian"),
                        (
                            Capture("era", match.start(1), match.end(1), era, None, "short"),
                            Capture(
                                "y", match.start(3), match.end(3), match.group(3), year, "numeric"
                            ),
                        ),
                    )
                )
        return sorted(found, key=lambda d: (d["start"], d["end"]))
