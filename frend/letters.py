"""Recognize a run of capital letters, which frend spells out or reads as a word.

"ATM", "GWR" and "ESPN" are spelled in the corpus far more often than read; icukit's
abbreviation lexicon covers the acronyms it knows ("FBI", "NASA"), and this covers the
rest. A run is two or more Latin capitals standing alone, with a plural or possessive
("UFOs", "AFI's"), or one capital written as an initial ("S." in "Jane S. Smith"). A
run leaves a following period as written; an initial takes its period, as icukit's
abbreviations do, so "S." ties "South" on span and the corpus decides (a letter, 26,596
of 26,792 times in shard 0). Which reading comes
first is measured (``data/acronym_priors.json``), by the run's length and vowels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

from icukit.detectors import Capture

__all__ = ["LettersDetector", "LettersValue", "cv_pattern", "is_roman", "numeral_share"]

_RUN = re.compile(r"(?<![\w&'’.-])([A-Z]{2,})(s|['’]s)?(?![\w&'’]|-\w|\.\w)")
_INITIAL = re.compile(r"(?<![\w&'’.-])([A-Z])(\.)(?!\w)")


@dataclass(frozen=True)
class LettersValue:
    """A run of capitals and what follows it: "", a plural or possessive "s", or an
    initial's period."""

    surface: str
    letters: str
    suffix: str


def cv_pattern(letters: str) -> str | None:
    """The run's consonant-vowel pattern ("GUS" -> "cvc"), with ``letter_key``'s vowels;
    none past seven letters. The corpus says "cvc" as a word and spells "ccc"."""
    if len(letters) > 7:
        return None
    return "".join("v" if ch in "aeiouyAEIOUY" else "c" for ch in letters)


@cache
def _roman_reader(locale: str):
    from icukit.recognize import FlexibleNumberDetector

    return FlexibleNumberDetector(locale)


def is_roman(surface: str, locale: str = "en_US") -> bool:
    """Whether icukit reads the whole surface as a Roman numeral ("II", "CD", not "ATM")."""
    return any(
        item["type"] == "number:cardinal:roman"
        and (item["start"], item["end"]) == (0, len(surface))
        for item in _roman_reader(locale).detect(surface)
    )


def numeral_share(surface: str) -> float:
    """How often the corpus reads a Roman-valid run as a number rather than as letters or
    a word: the surface's own counts, blended toward all numerals' (``roman:*``)."""
    from frend.verbalize import _acronym_priors

    table = _acronym_priors()
    pooled = table.get("roman:*", {})
    parent = pooled.get("numeral", 0) / max(sum(pooled.values()), 1)
    own = table.get(f"roman:{surface}", {})
    return (own.get("numeral", 0) + 5 * parent) / (sum(own.values()) + 5)


def _captures(start: int, letters: str, suffix: str) -> tuple[Capture, ...]:
    middle = start + len(letters)
    captures = [Capture("letters", start, middle, letters, letters, None)]
    if suffix:
        name = "period" if suffix == "." else "suffix"
        captures.append(Capture(name, middle, middle + len(suffix), suffix, suffix, None))
    return tuple(captures)


class LettersDetector:
    """Detect runs of capital letters and single initials."""

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = locale

    def detect(self, text: str) -> list[dict]:
        detections = []
        for pattern, kind in ((_RUN, "letters:run"), (_INITIAL, "letters:initial")):
            for match in pattern.finditer(text):
                letters, suffix = match.group(1), match.group(2)
                if not suffix and is_roman(letters, self.locale) and numeral_share(letters) > 0.5:
                    # The corpus reads "II" as a number: icukit's Roman reading stands alone.
                    continue
                start, end = match.start(), match.end()
                detections.append(
                    {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "type": kind,
                        "value": LettersValue(text[start:end], letters, suffix or ""),
                        "captures": _captures(start, letters, suffix or ""),
                    }
                )
        return sorted(detections, key=lambda detection: detection["start"])
