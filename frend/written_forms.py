"""Recognize written forms ICU writes nowhere, which icukit leaves to frend.

kal ruled these frend's, as URLs are: no locale pattern produces them, so icukit does
not read them. Each is read into a value frend already speaks where one fits, so its
speech stays ICU's (numbers, era names) or the corpus's:

- a clock time with a space after the colon ("2: 13"), as ``time:spaced-colon``;
- a case citation's volume ("339 U.S. 629"; a page must follow), as
  ``number:cardinal:citation`` with the reporter as a capture;
- spaced single digits ("6 3"), as ``number:digits``, said digit by digit;
- an era written with dotted letters, attached to the year, or before it ("500
  B.C.", "4AD", "A.D. 1066"), as ``date:era``; "500 BC." is ICU's "500 BC" and a
  sentence-final period, left to icukit.

The shapes are icukit's census families (``coverage/gap_families.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from icukit.detectors import Capture, DateTimeValue, NumberValue

__all__ = ["DigitsValue", "WrittenFormsDetector"]

_SPACED_COLON = re.compile(r"(?<![\w:.])(\d{1,2}): (\d{2})(?![\w:])")
# A citation names its page too ("339 U.S. 629"); without one, "339 U.S. troops" is prose.
_CITATION = re.compile(r"(?<![\w.,])(\d+) (U\. ?S\.)(?= \d)")
_SPACED_DIGITS = re.compile(r"(?<![\w.,:/-])\d(?: \d)+(?![\w.,:/-])")
# Dotted letters only: a period after "BC" or "AD" ends the sentence, not the era.
_ERA_WRITTEN = r"B\.C\.E\.|B\.C\.|A\.D\.|C\.E\."
_ERA_AFTER = re.compile(rf"(?<![\w.])(\d{{1,4}})( ?)({_ERA_WRITTEN}|AD|BC)(?!\w)")
_ERA_BEFORE = re.compile(r"(?<![\w.])(A\.D\.|AD)( ?)(\d{1,4})(?![\w.])")
_BEFORE_ERA = ("B",)


@dataclass(frozen=True)
class DigitsValue:
    """Spaced single digits, said one by one ("6 3")."""

    digits: str


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
        self.locale = locale

    def detect(self, text: str) -> list[dict]:
        found = []
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
                            "reporter", match.start(2), match.end(2), match.group(2), None, "symbol"
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
                        Capture("y", match.start(1), match.end(1), match.group(1), year, "numeric"),
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
                        Capture("y", match.start(3), match.end(3), match.group(3), year, "numeric"),
                    ),
                )
            )
        return sorted(found, key=lambda d: (d["start"], d["end"]))
