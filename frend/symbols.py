"""Recognize a standalone symbol or letter of another script, which frend speaks by name.

The names come from icukit: CLDR's per-locale spoken names for symbols
(``icu_abbreviations(locale, kinds=["symbol"])``: "&" ampersand, and; "#" hash sign,
number) and ICU's formal character names for everything else ("GREEK SMALL LETTER
ALPHA" gives "alpha"). A character reads only when it stands alone -- between spaces,
punctuation, or the ends of the text -- so "R&D" and "AT&T" are left as written. Every
reading also offers silence: the corpus says most punctuation, a lone dash and a
character of a script it does not read ("風") as nothing, and which name or silence
comes first is measured, per character for a symbol and per script for a letter.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import icu
from icukit.detectors import Capture

__all__ = ["SymbolDetector", "SymbolValue", "symbol_names"]

_COMMON = ("Zyyy", "Zinh", "Latn")


@dataclass(frozen=True)
class SymbolValue:
    """One standalone character: its script's short name and the names it can be read by."""

    char: str
    script: str
    names: tuple[tuple[str, str], ...]


def _script(char: str) -> str:
    return icu.Script.getScript(ord(char)).getShortName()


@cache
def _cldr_names(locale: str) -> dict[str, tuple[str, ...]]:
    """CLDR's names per symbol, or none where the installed icukit does not list symbols
    (before icukit #136): frend then reads no symbol, rather than failing."""
    from icukit import icu_abbreviations

    try:
        rows = icu_abbreviations(locale, kinds=["symbol"], locales=())
    except (TypeError, ValueError, KeyError):
        return {}
    names: dict[str, list[str]] = {}
    for row in rows:
        for expansion in row.expansions:
            if expansion not in names.setdefault(row.surface, []):
                names[row.surface].append(expansion)
    return {surface: tuple(items) for surface, items in names.items()}


def _letter_name(char: str) -> str | None:
    """The letter's own name from ICU's formal name ("GREEK SMALL LETTER ALPHA" -> "alpha")."""
    try:
        from icukit import get_char_name
    except ImportError:  # before icukit #136
        return None
    name = get_char_name(char) or ""
    if " LETTER " not in name:
        return None
    return name.split(" LETTER ", 1)[1].lower()


def symbol_names(char: str, locale: str = "en_US") -> tuple[tuple[str, str], ...]:
    """(name, source) pairs a character can be read by, CLDR's first."""
    cldr = _cldr_names(locale).get(char, ())
    if cldr:
        return tuple((name, f"cldr-symbol:{name}") for name in cldr)
    letter = _letter_name(char) if icu.Char.isalpha(char) and _script(char) not in _COMMON else None
    return ((letter, "icu-name:letter"),) if letter else ()


def _speakable(char: str, locale: str) -> bool:
    if char.isspace() or char.isdigit():
        return False
    if char in _cldr_names(locale):
        return True
    return icu.Char.isalpha(char) and _script(char) not in _COMMON


def _standalone(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not (before.isalnum() or after.isalnum())


class SymbolDetector:
    """Detect standalone symbols and letters of scripts frend reads by name."""

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = locale

    def detect(self, text: str) -> list[dict]:
        detections = []
        for index, char in enumerate(text):
            if not _speakable(char, self.locale) or not _standalone(text, index, index + 1):
                continue
            script = _script(char)
            kind = "symbol:cldr" if char in _cldr_names(self.locale) else "symbol:letter"
            detections.append(
                {
                    "text": char,
                    "start": index,
                    "end": index + 1,
                    "type": kind,
                    "value": SymbolValue(char, script, symbol_names(char, self.locale)),
                    "captures": (Capture("symbol", index, index + 1, char, char, None),),
                }
            )
        return detections
