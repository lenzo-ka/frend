"""Recognize a run of capital letters, which frend spells out or reads as a word.

"ATM", "GWR" and "ESPN" are spelled in the corpus far more often than read; icukit's
abbreviation lexicon covers the acronyms it knows ("FBI", "NASA"), and this covers the
rest. A run is two or more capitals of one script standing alone ("ATM", "ÉCU",
"СССР"), with a plural or possessive ("UFOs", "AFI's"), or one capital written as an
initial ("S." in "Jane S. Smith"). A capital is a letter of Unicode general category
``Lu``, and its script is ICU's; the acronym builder counts a token as a run by the
same predicate (:func:`is_letter_run`), so it counts only what this reader matches. A
run leaves a following period as written; an initial takes its period, as icukit's
abbreviations do, so "S." ties "South" on span and the corpus decides (a letter, 26,596
of 26,792 times in shard 0). Which reading comes
first is measured (``data/en/acronym_priors.json``), by the run's length and vowels; the
vowels are the locale's (``lexical.json``'s ``letter.vowels``), and a locale without
them has no vowel key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache, lru_cache

import icu
from icukit.detectors import Capture

from frend.locale_data import LOCALE_CACHE, canonical_locale, lexical_forms

__all__ = [
    "LettersDetector",
    "LettersValue",
    "capital_script",
    "cv_pattern",
    "is_letter_run",
    "is_roman",
    "letter_vowels",
    "numeral_share",
]

# A capital is a letter of general category Lu, as ICU's Unicode data has it; the
# character class below is built from that set, so the patterns match exactly its
# members, and a match is a run only if its capitals share one ICU script.
_CAPITALS = icu.UnicodeSet("[:Lu:]")
_CAPITALS.freeze()


def _character_class(unicode_set: icu.UnicodeSet) -> str:
    ranges = []
    for index in range(unicode_set.getRangeCount()):
        first, last = unicode_set.getRangeStart(index), unicode_set.getRangeEnd(index)
        span = re.escape(first) if first == last else f"{re.escape(first)}-{re.escape(last)}"
        ranges.append(span)
    return "[" + "".join(ranges) + "]"


_LU = _character_class(_CAPITALS)
_RUN = re.compile(rf"(?<![\w&'’.-])({_LU}{{2,}})(s|['’]s)?(?![\w&'’]|-\w|\.\w)")
_INITIALS = re.compile(rf"(?<![\w&'’.-])((?:{_LU}\.)+)(?!\w)")


def capital_script(letters: str) -> int | None:
    """The ICU script code every character of ``letters`` shares when each is a capital
    (general category ``Lu``); ``None`` when one is not, the scripts differ, or it is
    empty. "ÉCU" is Latin, "СССР" Cyrillic; "AΒ" (Latin A, Greek Beta) is neither."""
    scripts = set()
    for ch in letters:
        if not _CAPITALS.contains(ch):
            return None
        scripts.add(icu.Script.getScript(ord(ch)).getScriptCode())
    return scripts.pop() if len(scripts) == 1 else None


def is_letter_run(token: str) -> bool:
    """Whether ``token`` is, whole, a run of capitals this reader matches: two or more
    capitals of one script. The acronym builder counts by this, so what it counts and
    what the reader reads are one population."""
    return len(token) >= 2 and capital_script(token) is not None


def letter_vowels(locale: str = "en_US") -> frozenset[str] | None:
    """The locale's vowel letters in both cases (``lexical.json``'s ``letter.vowels``,
    cased by ICU for the locale), or ``None`` when its table has none: then no key that
    needs vowels is formed."""
    return _vowels_for(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _vowels_for(locale: str) -> frozenset[str] | None:
    vowels = lexical_forms(locale).get("letter.vowels")
    if not vowels:
        return None
    lower = str(vowels["value"])
    cased = icu.Locale(locale)
    upper = str(icu.UnicodeString(lower).toUpper(cased))
    return frozenset(lower) | frozenset(upper)


@dataclass(frozen=True)
class LettersValue:
    """A run of capitals and what follows it: "", a plural or possessive "s", or an
    initial's period."""

    surface: str
    letters: str
    suffix: str


def cv_pattern(letters: str, locale: str = "en_US") -> str | None:
    """The run's consonant-vowel pattern ("GUS" -> "cvc"), with the locale's vowels
    (:func:`letter_vowels`, as ``letter_key`` reads them); none past seven letters, and
    none for a locale with no vowels. The corpus says "cvc" as a word and spells "ccc"."""
    vowels = letter_vowels(locale)
    if vowels is None or len(letters) > 7:
        return None
    return "".join("v" if ch in vowels else "c" for ch in letters)


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


def numeral_share(surface: str, locale: str = "en_US") -> float:
    """How often the locale's corpus reads a Roman-valid run as a number rather than as
    letters or a word: the surface's own counts, blended toward all numerals'
    (``roman:*``); 0 for a locale with no acronym table."""
    from frend.verbalize import _acronym_priors

    table = _acronym_priors(locale=locale)
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
        for chain in _INITIALS.finditer(text):
            # "S." or a chain of initials ("J.R.R."): each letter with its period, the
            # chain's letters in one script.
            if capital_script(chain.group(1)[::2]) is None:
                continue
            for at in range(chain.start(), chain.end(), 2):
                letter = text[at]
                detections.append(
                    {
                        "text": text[at : at + 2],
                        "start": at,
                        "end": at + 2,
                        "type": "letters:initial",
                        "value": LettersValue(text[at : at + 2], letter, "."),
                        "captures": _captures(at, letter, "."),
                    }
                )
        for match in _RUN.finditer(text):
            letters, suffix = match.group(1), match.group(2) or ""
            if not is_letter_run(letters):
                # Capitals of two scripts ("AΒ") are not one run.
                continue
            if (
                not suffix
                and is_roman(letters, self.locale)
                and numeral_share(letters, self.locale) > 0.5
            ):
                # The corpus reads "II" as a number: icukit's Roman reading stands alone.
                continue
            start, end = match.start(), match.end()
            detections.append(
                {
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "type": "letters:run",
                    "value": LettersValue(text[start:end], letters, suffix),
                    "captures": _captures(start, letters, suffix),
                }
            )
        return sorted(detections, key=lambda detection: detection["start"])
