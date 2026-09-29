"""Recognize a standalone symbol or letter of another script, which frend speaks by name.

The names come from icukit: CLDR's per-locale spoken names for symbols
(``icu_abbreviations(locale, kinds=["symbol"])``: "&" ampersand, and; "#" hash sign,
number) and ICU's formal character names for everything else ("GREEK SMALL LETTER
ALPHA" gives "alpha"). A character reads only when it stands alone -- between spaces,
punctuation, or the ends of the text -- so "R&D" and "AT&T" are left as written. Every
reading also offers silence: the corpus says most punctuation, a lone dash and a
character of a script it does not read ("風") as nothing, and which name or silence
comes first is measured, per character for a symbol and per script for a letter.

A letter is "of another script" when its ICU script is none of the locale's: the script
ICU's likely subtags give the locale (``en_US`` -> ``en_Latn_US``), the scripts of its
exemplar characters (icukit's ``get_locale_scripts``), and Common and Inherited. Single
out-of-script letters set apart by single spaces ("Т О Д Н") are one span,
:class:`ScriptRunValue`, read as a unit: by ICU's
transliteration into the locale's script (a transform ICU lists for the two scripts,
else ``Any-<script>``), read as the locale reads its own letters and words; by the
letters' names; as written; or as nothing. One reading of the run replaces a free choice
per letter, which multiplied a sentence of spaced letters into 2^n covers. A lone
out-of-script letter reads as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import icu
from icukit.detectors import Capture

__all__ = [
    "ScriptRunValue",
    "SymbolDetector",
    "SymbolValue",
    "locale_scripts",
    "run_readings",
    "symbol_names",
    "transform_id",
]

# ICU's Common and Inherited scripts belong to every locale.
_NEUTRAL = frozenset(
    icu.Script(code).getShortName() for code in (icu.UScriptCode.COMMON, icu.UScriptCode.INHERITED)
)
_MARKS = icu.UnicodeSet("[:M:]")
_MARKS.freeze()
_NFC = icu.Normalizer2.getNFCInstance()
TRANSLITERATION_SOURCE = "icu-transliteration"
LETTER_NAME_SOURCE = "icu-name:letter"


@dataclass(frozen=True)
class SymbolValue:
    """One standalone character: its script's short name and the names it can be read by."""

    char: str
    script: str
    names: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ScriptRunValue:
    """A run of letters outside the locale's scripts, read as one unit.

    ``script`` is the run's ICU short script name ("Cyrl"), ``transform`` the ICU
    transform id that writes it in the locale's script (``None`` when ICU has none), and
    ``names`` the (reading, source) pairs it can be read by, silence aside.
    """

    text: str
    script: str
    transform: str | None
    names: tuple[tuple[str, str], ...]


def _script(char: str) -> str:
    return icu.Script.getScript(ord(char)).getShortName()


def _short(name: str) -> str | None:
    """ICU's short code for a script named either way ("Latin" or "Latn")."""
    try:
        codes = icu.Script.getCode(name)
    except icu.ICUError:
        return None
    for code in codes:
        script = icu.Script(code)
        if name in (script.getShortName(), script.getName()):
            return script.getShortName()
    return None


@cache
def locale_scripts(locale: str) -> tuple[str, ...]:
    """The locale's scripts as ICU short names, its likely script first: likely subtags'
    script, then its exemplar characters' (icukit), then Common and Inherited."""
    from icukit.locale import add_likely_subtags, get_locale_scripts

    scripts: list[str] = []
    likely = icu.Locale(add_likely_subtags(locale)).getScript()
    for name in (likely, *get_locale_scripts(locale)):
        code = _short(name) if name else None
        if code and code not in scripts:
            scripts.append(code)
    return (*scripts, *sorted(_NEUTRAL - set(scripts)))


def _lettered(char: str) -> bool:
    """A letter of a script of its own (not a Common modifier such as "ʹ")."""
    return icu.Char.isalpha(char) and _script(char) not in _NEUTRAL


def _foreign(char: str, locale: str) -> bool:
    """A letter of a script the locale does not write."""
    return icu.Char.isalpha(char) and _script(char) not in locale_scripts(locale)


@cache
def _transforms() -> dict[tuple[str, str], tuple[str, ...]]:
    """ICU's available transform ids without a variant, by (source, target) short script."""
    from icukit.transliterator import list_transliterators

    table: dict[tuple[str, str], list[str]] = {}
    for transform in list_transliterators():
        if "/" in transform or transform.count("-") != 1:
            continue
        source, target = transform.split("-")
        target_code = _short(target)
        source_code = "Any" if source == "Any" else _short(source)
        if target_code and source_code:
            table.setdefault((source_code, target_code), []).append(transform)
    return {key: tuple(sorted(ids)) for key, ids in table.items()}


@cache
def transform_id(script: str, locale: str) -> str | None:
    """The ICU transform writing ``script`` in the locale's script: one ICU lists from that
    script to the locale's (likely script first), else ICU's ``Any-<script>``."""
    targets = [code for code in locale_scripts(locale) if code not in _NEUTRAL]
    for source in (script, "Any"):
        for target in targets:
            ids = _transforms().get((source, target))
            if ids:
                return ids[0]
    return None


@cache
def _transliterator(transform: str):
    from icukit.transliterator import Transliterator

    return Transliterator(transform)


@cache
def _cldr_names(locale: str) -> dict[str, tuple[str, ...]]:
    """CLDR's names per symbol, from icukit's symbol rows."""
    from icukit import icu_abbreviations

    rows = icu_abbreviations(locale, kinds=["symbol"], locales=())
    names: dict[str, list[str]] = {}
    for row in rows:
        for expansion in row.expansions:
            if expansion not in names.setdefault(row.surface, []):
                names[row.surface].append(expansion)
    return {surface: tuple(items) for surface, items in names.items()}


def _letter_name(char: str) -> str | None:
    """The letter's own name from ICU's formal name ("GREEK SMALL LETTER ALPHA" -> "alpha")."""
    from icukit import get_char_name

    name = get_char_name(char) or ""
    if " LETTER " not in name:
        return None
    return name.split(" LETTER ", 1)[1].lower()


def symbol_names(char: str, locale: str = "en_US") -> tuple[tuple[str, str], ...]:
    """(name, source) pairs a character can be read by, CLDR's first."""
    cldr = _cldr_names(locale).get(char, ())
    if cldr:
        return tuple((name, f"cldr-symbol:{name}") for name in cldr)
    letter = _letter_name(char) if _foreign(char, locale) else None
    return ((letter, LETTER_NAME_SOURCE),) if letter else ()


def _speakable(char: str, locale: str) -> bool:
    if char.isspace() or char.isdigit():
        return False
    if char in _cldr_names(locale):
        return True
    return _foreign(char, locale)


def _standalone(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not (before.isalnum() or after.isalnum())


def _groups(text: str, locale: str) -> list[tuple[int, int]]:
    """Maximal stretches of out-of-script letters, each with the marks written on it."""
    groups: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if not _foreign(text[index], locale):
            index += 1
            continue
        start = index
        while index < len(text) and (_foreign(text[index], locale) or _MARKS.contains(text[index])):
            index += 1
        groups.append((start, index))
    return groups


def _runs(text: str, locale: str) -> list[tuple[int, int]]:
    """(start, end) per run of two or more single out-of-script letters one space apart
    ("Т Е С Т"). A word of another script ("αβ", "Москва") is no run: it stays as written."""
    runs: list[tuple[int, int]] = []
    chain: list[tuple[int, int]] = []
    for start, end in [*_groups(text, locale), (len(text) + 2, len(text) + 2)]:
        single = sum(not _MARKS.contains(ch) for ch in text[start:end]) == 1
        if single and chain and start == chain[-1][1] + 1 and text[chain[-1][1]].isspace():
            chain.append((start, end))
            continue
        if len(chain) > 1:
            runs.append((chain[0][0], chain[-1][1]))
        chain = [(start, end)] if single else []
    return runs


def run_readings(
    text: str, script: str, locale: str
) -> tuple[str | None, tuple[tuple[str, str], ...]]:
    """The run's transform and its readings besides silence: the locale's reading of its
    transliteration, the letters' names, and the run as written."""
    from frend.abbreviation_variants import AS_WRITTEN_SOURCE
    from frend.letters import spelled

    readings: list[tuple[str, str]] = []
    transform = transform_id(script, locale)
    if transform is not None:
        written = _transliterator(transform).transliterate(text)
        # Letters set apart are said one by one, each as ICU writes it in the locale's
        # script ("Θ" -> "th"), lower-cased as the locale spells; a word is said whole.
        units = [spelled(unit).replace(" ", "") for unit in written.split()]
        spoken = " ".join(unit for unit in units if any(_lettered(ch) for ch in unit))
        if spoken.strip():
            readings.append((spoken, f"{TRANSLITERATION_SOURCE}:{transform}"))
    names = [_letter_name(char) for char in _NFC.normalize(text) if not char.isspace()]
    if names and all(names):
        readings.append((" ".join(names), LETTER_NAME_SOURCE))  # type: ignore[arg-type]
    readings.append((text, AS_WRITTEN_SOURCE))
    return transform, tuple(readings)


class SymbolDetector:
    """Detect standalone symbols and letters of scripts frend reads by name."""

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = locale

    def detect(self, text: str) -> list[dict]:
        detections = []
        in_runs: set[int] = set()
        for start, end in _runs(text, self.locale):
            in_runs.update(range(start, end))
            if not _standalone(text, start, end):
                continue
            surface = text[start:end]
            script = _script(surface[0])
            transform, names = run_readings(surface, script, self.locale)
            detections.append(
                {
                    "text": surface,
                    "start": start,
                    "end": end,
                    "type": "symbol:script-run",
                    "value": ScriptRunValue(surface, script, transform, names),
                    "captures": (Capture("symbol", start, end, surface, surface, None),),
                }
            )
        for index, char in enumerate(text):
            if index in in_runs:
                continue
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
        detections.sort(key=lambda detection: detection["start"])
        return detections
