"""Recognize a run of capital letters, which frend spells out or reads as a word.

"ATM", "GWR" and "ESPN" are spelled in the corpus far more often than read; icukit's
abbreviation lexicon covers the acronyms it knows ("FBI", "NASA"), and this covers the
rest. A run is two or more capitals of one script standing alone ("ATM", "ÉCU",
"СССР"), with a plural or possessive ("UFOs", "AFI's"), or one capital written as an
initial ("S." in "Jane S. Smith"). A capital is a letter of Unicode general category
``Lu`` with any combining marks written after it, so a decomposed run reads as its NFC
form; its script is ICU's, resolved as UAX #24 resolves a run (a Common capital such as
"ℂ" takes its neighbors'). The acronym builder counts a token as a run by the same
predicate (:func:`is_letter_run`), so it counts what this reader matches, except the
Roman numerals the reader leaves to icukit (see :func:`is_letter_run`). A
run leaves a following period as written; an initial takes its period, as icukit's
abbreviations do, so "S." ties "South" on span and the corpus decides (a letter, 26,596
of 26,792 times in shard 0). Which reading comes first is taken from the case-preserved
spell-out dictionary. Optional unanimous case-variant lookup is caller-selected and
off by default. A missing bounded-token row uses the vowel rule. The older acronym
prior remains the fallback for unattested capital runs because the vowel rule regressed
held-out UNSEEN tokens; it also handles Roman numerals and the explicit Google-TN profile.

A non-uppercase token of two through six letters in the locale's script is also an
ambiguous spell-or-say candidate. Training counts rank the word and letter-name
readings when the token is an exception to the cheap rule; neither reading is removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import TYPE_CHECKING

import icu
from icukit import LetterNameDetector
from icukit.detectors import Capture

from frend.locale_data import LOCALE_CACHE, canonical_locale, lexical_forms

__all__ = [
    "LetterNames",
    "LettersDetector",
    "LettersValue",
    "capital_script",
    "capitals",
    "cv_pattern",
    "split_acronym_surface",
    "is_letter_run",
    "is_spelled_token",
    "is_roman",
    "letter_vowels",
    "letter_names",
    "numeral_share",
    "spelled",
    "spellout_dictionary_entry",
    "spelled_token_prior",
    "spelled_token_rule",
]

if TYPE_CHECKING:
    from frend.verbalize import SpokenAlternative

# A capital is a letter of general category Lu, as ICU's Unicode data has it; the
# character classes below are built from ICU's sets, so the patterns match exactly their
# members. A capital carries the combining marks (general category M) written after it,
# so a decomposed "E\u0301CO" is one run, as its NFC form "ÉCO" is, and the match keeps
# the text's own spans; a match is a run only if its capitals share one script.
_CAPITALS = icu.UnicodeSet("[:Lu:]")
_CAPITALS.freeze()
_LETTERS = icu.UnicodeSet("[:L:]")
_LETTERS.freeze()
_MARKS = icu.UnicodeSet("[:M:]")
_MARKS.freeze()
_NFC = icu.Normalizer2.getNFCInstance()
# UAX #24: a Common or Inherited character takes the script of its neighbors.
_NEUTRAL_SCRIPTS = frozenset({icu.UScriptCode.COMMON, icu.UScriptCode.INHERITED})


def _character_class(unicode_set: icu.UnicodeSet, *, bracket: bool = True) -> str:
    ranges = []
    for index in range(unicode_set.getRangeCount()):
        first, last = unicode_set.getRangeStart(index), unicode_set.getRangeEnd(index)
        span = re.escape(first) if first == last else f"{re.escape(first)}-{re.escape(last)}"
        ranges.append(span)
    body = "".join(ranges)
    return f"[{body}]" if bracket else body


_LU = _character_class(_CAPITALS)
_L = _character_class(_LETTERS)
_M = _character_class(_MARKS)
_MARK_RANGES = _character_class(_MARKS, bracket=False)
_CAPITAL = f"{_LU}{_M}*"
_BEFORE = rf"(?<![\w&'’.\-{_MARK_RANGES}])"
_RUN = re.compile(
    rf"{_BEFORE}((?:{_CAPITAL}){{2,}})(s|['’][sS])?(?![\w&'’{_MARK_RANGES}]|-\w|\.\w)"
)
_INITIALS = re.compile(rf"{_BEFORE}((?:{_CAPITAL}\.)+)(?![\w{_MARK_RANGES}])")
_INITIAL = re.compile(rf"({_CAPITAL})\.")
_SHORT_TOKEN = re.compile(rf"{_BEFORE}((?:{_L}{_M}*){{2,6}})(?![\w&'’{_MARK_RANGES}]|-\w|\.\w)")
# Uppercase apostrophe-S is ambiguous with an uppercase English contraction. Preserve
# the ordinary closed-class contractions that the previous suffix grammar left alone.
_EN_UPPERCASE_S_CONTRACTION_STEMS = frozenset(
    {
        "HE",
        "SHE",
        "IT",
        "THAT",
        "WHAT",
        "WHO",
        "WHERE",
        "WHEN",
        "WHY",
        "HOW",
        "THERE",
        "HERE",
        "LET",
    }
)


def capitals(letters: str) -> tuple[str, ...] | None:
    """The capitals of ``letters`` after NFC (ICU's Normalizer2), each with the combining
    marks written after it ("E\u0301CO" -> "É", "C", "O"); ``None`` when a character is
    neither a capital nor a mark on one, or it is empty."""
    units = _letters(_NFC.normalize(letters))
    if not units or not all(_CAPITALS.contains(unit[0]) for unit in units):
        return None
    return tuple(units)


def _letters(text: str) -> list[str]:
    # Each character with the combining marks written after it.
    units: list[str] = []
    for ch in text:
        if _MARKS.contains(ch) and units:
            units[-1] += ch
        else:
            units.append(ch)
    return units


def capital_script(letters: str) -> int | None:
    """The ICU script code of a run of capitals (:func:`capitals`), resolved as UAX #24
    resolves a run: a Common or Inherited capital ("ℂ") takes its neighbors' script, so
    "ℂA" is Latin; "ℂℍ", all Common, is Common. ``None`` when a character is not a
    capital or a mark on one, when two real scripts mix ("AΒ", Latin A and Greek Beta,
    and "AℂΒ"), or it is empty. "ÉCU" is Latin, "СССР" Cyrillic."""
    units = capitals(letters)
    if units is None:
        return None
    scripts = {icu.Script.getScript(ord(unit[0])).getScriptCode() for unit in units}
    real = scripts - _NEUTRAL_SCRIPTS
    if len(real) > 1:
        return None
    return real.pop() if real else min(scripts)


def is_letter_run(token: str) -> bool:
    """Whether ``token`` is, whole, a run of capitals this reader matches: two or more
    capitals (after NFC, each with its combining marks) of one script, as
    :func:`capital_script` resolves it. The acronym builder counts by this, so what it
    counts and what the reader matches are one population, with one exception: a bare
    run icukit reads as a Roman numeral that the corpus reads as a number more often
    than not ("II") is left by the reader to icukit's Roman reading, while the builder
    counts it (under its shape keys, and under ``roman:`` keys with its numeral
    readings)."""
    units = capitals(token)
    return units is not None and len(units) >= 2 and capital_script(token) is not None


def split_acronym_surface(token: str) -> tuple[str, str] | None:
    """Return an acronym's NFC base and suffix subkey, or ``None``.

    The suffix grammar is the detector's: a lower-case ``s`` is plural and either
    apostrophe followed by ``s`` or ``S`` is possessive. An upper-case final ``S``
    without an apostrophe remains part of the case-preserved base (``AIDS`` is bare,
    while ``AIDS'S`` is possessive).
    """
    token = _NFC.normalize(token)
    if len(token) >= 2 and token[-2] in "'’" and token[-1] in "sS":
        letters, subkey = token[:-2], "possessive"
    elif token.endswith("s"):
        letters, subkey = token[:-1], "plural"
    else:
        letters, subkey = token, "bare"
    return (letters, subkey) if is_letter_run(letters) else None


@lru_cache(maxsize=4096)
def is_spelled_token(token: str, locale: str = "en_US") -> bool:
    """Whether ``token`` is a bounded spell-or-say candidate for ``locale``."""
    from frend.symbols import locale_scripts

    units = _letters(_NFC.normalize(token))
    if not 2 <= len(units) <= 6 or token.isupper():
        return False
    if not all(_LETTERS.contains(unit[0]) for unit in units):
        return False
    scripts = {
        icu.Script.getScript(ord(unit[0])).getShortName()
        for unit in units
        if icu.Script.getScript(ord(unit[0])).getScriptCode() not in _NEUTRAL_SCRIPTS
    }
    return bool(scripts) and scripts <= set(locale_scripts(canonical_locale(locale)))


@lru_cache(maxsize=LOCALE_CACHE)
def _spelled_token_priors(locale: str) -> dict[str, dict[str, object]] | None:
    """Load a locale's exception mapping once, or ``None`` when none is measured."""
    from frend.locale_data import measured_table

    table = measured_table("spelled_token_priors", locale)
    return None if table is None else table["tokens"]


def spelled_token_prior(token: str, locale: str = "en_US") -> dict[str, object] | None:
    """The measured exception row for ``token``, or ``None`` when the rule decides."""
    priors = _spelled_token_priors(canonical_locale(locale))
    return None if priors is None else priors.get(_NFC.normalize(token))


@lru_cache(maxsize=LOCALE_CACHE)
def _spellout_dictionary(
    locale: str,
) -> tuple[dict[str, tuple[str, int, int]], dict[str, str], str] | None:
    """Load the compact exact rows and optional case-variant aliases for one locale."""
    from frend.locale_data import measured_table

    table = measured_table("spellout_dictionary", locale)
    if table is None:
        return None
    rows = {
        surface: (decision, say_count, spell_count)
        for surface, decision, say_count, spell_count in table["tokens"]
    }
    return rows, dict(table.get("casefold", ())), table["provenance"]["source"]


def spellout_dictionary_entry(
    token: str,
    locale: str = "en_US",
    *,
    case_variant_lookup: bool = False,
) -> dict[str, object] | None:
    """Return a valid exact row, or an opted-in unanimous case-variant row."""
    table = _spellout_dictionary(canonical_locale(locale))
    if table is None:
        return None
    token = _NFC.normalize(token)
    exact, aliases, source = table
    row = exact.get(token)
    if row is None and len(token) >= 2 and token[-2] in "'’" and token[-1] == "S":
        # The detector treats uppercase and lowercase possessive suffixes alike. Their
        # case is not part of the acronym, so reuse the measured lowercase-suffix row.
        row = exact.get(f"{token[:-1]}s")
    if row is None and case_variant_lookup:
        target = aliases.get(token.casefold())
        row = None if target is None else exact.get(target)
    if row is None:
        return None
    decision, say_count, spell_count = row
    if decision not in ("say", "spell"):
        raise ValueError(f"spell-out dictionary decision must be 'say' or 'spell': {decision!r}")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in (say_count, spell_count)
    ):
        raise ValueError("spell-out dictionary counts must be nonnegative integers")
    total = say_count + spell_count
    if total == 0:
        raise ValueError("spell-out dictionary counts must have a positive total")
    counts = {"say": say_count, "spell": spell_count}
    return {
        "counts": counts,
        "decision": decision,
        "source": source,
        "spell_share": spell_count / total,
    }


def spelled_token_rule(token: str) -> str:
    """Default spell-or-say decision: AEIOU means say; y remains a consonant."""
    return "say" if any(letter in "aeiou" for letter in token.casefold()) else "spell"


@dataclass(frozen=True)
class LetterNames:
    spoken: tuple[str, ...]
    provenance: str


@lru_cache(maxsize=4096)
def letter_names(letters: str, locale: str) -> LetterNames | None:
    """Return locale-authoritative names for every letter, or ``None``.

    icukit's letter-name facility establishes whether the requested locale has a
    complete inventory.  English retains frend's evaluator-stable surface spelling;
    an unsupported locale never falls back to Unicode lower-case speech.
    """
    units = _letters(_NFC.normalize(letters))
    if not units:
        return None
    canonical = canonical_locale(locale)
    detector = LetterNameDetector(canonical)
    # icukit's English inventory is authoritative for the locale.  Its detector is
    # intentionally ASCII-only, while frend's established English run spelling also
    # covers same-script accented and non-Latin capitals.  Preserve that evaluator
    # behavior only for the supported English inventory; never generalize it to a
    # locale for which icukit reports no names.
    if canonical.split("_", 1)[0] == "en" and detector.detect("A"):
        spoken = tuple("".join(icu.Char.tolower(ch) for ch in unit) for unit in units)
        return LetterNames(spoken, "icu:letter-name")
    for unit in units:
        detections = detector.detect(unit)
        if not any((item["start"], item["end"]) == (0, len(unit)) for item in detections):
            return None
    spoken = tuple("".join(icu.Char.tolower(ch) for ch in unit) for unit in units)
    return LetterNames(spoken, "icu:letter-name")


def spelled(letters: str, locale: str) -> SpokenAlternative | None:
    names = letter_names(letters, locale)
    if names is None:
        return None
    from frend.verbalize import SpokenAlternative

    return SpokenAlternative(" ".join(names.spoken), names.provenance)


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
    """Detect capital runs, initials, and bounded spell-or-say tokens."""

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = locale

    def detect(self, text: str) -> list[dict]:
        detections = []
        occupied: list[tuple[int, int]] = []
        for chain in _INITIALS.finditer(text):
            # "S." or a chain of initials ("J.R.R."): each letter, with its marks, and its
            # period, the chain's letters in one script.
            initials = list(_INITIAL.finditer(text, chain.start(), chain.end()))
            if capital_script("".join(initial.group(1) for initial in initials)) is None:
                continue
            for initial in initials:
                at, end, letter = initial.start(), initial.end(), initial.group(1)
                detections.append(
                    {
                        "text": text[at:end],
                        "start": at,
                        "end": end,
                        "type": "letters:initial",
                        "value": LettersValue(text[at:end], letter, "."),
                        "captures": _captures(at, letter, "."),
                    }
                )
        for match in _RUN.finditer(text):
            letters, suffix = match.group(1), match.group(2) or ""
            if not is_letter_run(letters):
                # Capitals of two scripts ("AΒ") are not one run.
                continue
            if (
                suffix.endswith("S")
                and canonical_locale(self.locale).split("_", 1)[0] == "en"
                and letters in _EN_UPPERCASE_S_CONTRACTION_STEMS
            ):
                continue
            if (
                not suffix
                and is_roman(letters, self.locale)
                and numeral_share(letters, self.locale) > 0.5
            ):
                # The corpus reads "II" as a number: icukit's Roman reading stands alone.
                continue
            start, end = match.start(), match.end()
            occupied.append((start, end))
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
        for match in _SHORT_TOKEN.finditer(text):
            letters = match.group(1)
            if (
                not is_spelled_token(letters, self.locale)
                or _spelled_token_priors(canonical_locale(self.locale)) is None
                or letter_names(letters, self.locale) is None
                or any(match.start() < end and match.end() > start for start, end in occupied)
            ):
                continue
            start, end = match.start(), match.end()
            detections.append(
                {
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "type": "letters:token",
                    "value": LettersValue(text[start:end], letters, ""),
                    "captures": _captures(start, letters, ""),
                }
            )
        return sorted(detections, key=lambda detection: detection["start"])
