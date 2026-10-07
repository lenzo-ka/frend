"""Locale-independent output spacing derived from ICU script properties."""

from __future__ import annotations

import icu

_UNSPACED_SCRIPTS: frozenset[int] = frozenset(
    code
    for code in {
        getattr(icu.UScriptCode, name)
        for name in dir(icu.UScriptCode)
        if name.isupper() and isinstance(getattr(icu.UScriptCode, name), int)
    }
    if icu.Script(code).breaksBetweenLetters()
)


def unspaced(char: str) -> bool:
    """Return whether ICU says letters in ``char``'s script omit word spaces."""
    return icu.Script.getScript(char).getScriptCode() in _UNSPACED_SCRIPTS


def unit_gap(left: str, right: str) -> str:
    """Return the spoken-unit gap required at this script boundary."""
    if left and right and unspaced(left[-1]) and unspaced(right[0]):
        return ""
    return " "


def strip_soft_hyphens(text: str) -> str:
    """Remove ICU's discretionary hyphenation hints from spoken output."""
    return text.replace("\u00ad", "")
