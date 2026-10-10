"""Abbreviations as running text writes them: without the period, and in another case.

icukit's lexicon lists each abbreviation once, as it is conventionally written ("Mr.",
"St.", "vol.", "e.g."). Running text, and the Google TN corpus above all, writes the
same abbreviation without its period and in other cases ("Mr", "st", "Vol", "E.G.").
This module derives those written variants from the lexicon, reads them, and ranks
every reading of an abbreviation by how the corpus says it
(``data/<locale>/abbreviation_priors.json``, ``tools/build_abbreviation_priors.py``).

**Sources.** A lexicon surface is a variant source when it ends in ".", holds a
lowercase letter, has no inner "." and folds to more than one letter. So state codes
and acronyms ("IN", "OR", "OK": capitals, no period) never become lowercase words, and
a single letter ("p.", "c.", "v.") is left to the letter readers, where the corpus
mostly says the letter.

**Variants.** Each source gives its period-less form, its ICU lower case and its ICU
title case, each with and without the period, and these are read by
:class:`AbbreviationVariantDetector`. A variant that is itself a lexicon surface is
dropped (that surface is read as the lexicon has it), and so is the source. The ICU
upper case is not detected here: an all-capitals variant ("LT", "DR") is a letter run,
which the letters reader spells; the run is also offered the source's expansions ("MR"
Mister, :func:`upper_variant_expansions`), and where the corpus has a capitals row for
its key (:func:`measured_case`), that row ranks the expansions and the letters ("MR":
mister 562 of 611). A variant the corpus spells in its own case's row, or in any row
of its key when that case has none (:func:`measures_spelled`: "Lt", "ch", "Rt"), also
reads its letters; "no" does not (spelled only in capitals).

**Dotted chains.** A chain of single letters each with its period ("e.g.", "E.G.",
"j.r.r.") is spelled whatever its case; one whose ICU lower case is a lexicon chain
with expansions also offers them ("E.G." for example, from "e.g.";
:func:`chain_expansions`).

**Measure.** A written token's key is its ICU lower case with one trailing period
removed; its case is lower, title or upper. The corpus writes no variant key with a
period, so period is not measured; and it never writes some keys in title case ("mr",
"st", "dr"), so a title-case surface of those reads the key's share (a case with no
row reads the key). A case row is blended toward the key's share with the spoken
priors' strength (``SUB_KEY_PRIOR_STRENGTH``). What the corpus says is sorted into
categories by :func:`category`: the token as written, spelled, a lexicon expansion, or
other.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from functools import lru_cache

import icu

from frend.locale_data import LOCALE_CACHE, canonical_locale, measured_table
from frend.spoken_priors import SUB_KEY_PRIOR_STRENGTH, normalize_spoken

__all__ = [
    "AS_WRITTEN",
    "AS_WRITTEN_SOURCE",
    "OTHER",
    "SPELLED",
    "SPELLED_SOURCE",
    "VARIANT_TYPE",
    "AbbreviationVariantDetector",
    "abbreviation_expansions",
    "abbreviation_priors",
    "abbreviation_weights",
    "category",
    "chain_expansions",
    "chain_sources",
    "fold_key",
    "is_chain",
    "measured_case",
    "measured_keys",
    "measures_spelled",
    "upper_variant_expansions",
    "variant_sources",
    "variant_surfaces",
    "written_case",
]

VARIANT_TYPE = "abbreviation:variant"
# The categories a reading is measured under, beside a lexicon expansion's own words.
AS_WRITTEN = "as-written"
SPELLED = "spelled"
OTHER = "other"
# The source of a variant's reading as it is written, and of a dotted chain spelled.
AS_WRITTEN_SOURCE = "surface:as-written"
SPELLED_SOURCE = "surface:spelled"

# A chain of single letters, each with its period ("e.g.", "J.R.R.").
_CHAIN = re.compile(r"(?:[^\W\d_]\.){2,}")


def _language(locale: str) -> icu.Locale:
    return icu.Locale(canonical_locale(locale))


def _lower(text: str, locale: str) -> str:
    return str(icu.UnicodeString(text).toLower(_language(locale)))


def _upper(text: str, locale: str) -> str:
    return str(icu.UnicodeString(text).toUpper(_language(locale)))


def _title(text: str, locale: str) -> str:
    language = _language(locale)
    words = icu.BreakIterator.createWordInstance(language)
    return str(icu.UnicodeString(text).toTitle(words, language))


def _stem(written: str) -> str:
    """The written token less one trailing period."""
    return written[:-1] if written.endswith(".") else written


def is_chain(written: str) -> bool:
    """Whether ``written`` is a chain of two or more single letters, each with its period."""
    return _CHAIN.fullmatch(written) is not None


def fold_key(written: str, locale: str = "en_US") -> str:
    """The measure's key: ICU lower case of the written token, one trailing period removed."""
    return _lower(_stem(written), locale)


def written_case(written: str, locale: str = "en_US") -> str | None:
    """``lower``, ``title`` or ``upper`` as ICU maps the written token (less one trailing
    period); ``None`` for a token in none of the three ("mR"). A token with no cased
    letter is none."""
    stem = _stem(written)
    lower, upper = _lower(stem, locale), _upper(stem, locale)
    if lower == upper:
        return None
    if stem == lower:
        return "lower"
    if stem == upper:
        return "upper"
    if stem == _title(stem, locale):
        return "title"
    return None


def _spelled(written: str, locale: str) -> str:
    """The token's letters, lower-cased and one by one ("E.G." "e g", "Mr" "m r")."""
    return " ".join(_lower(ch, locale) for ch in written if ch.isalpha())


def category(written: str, said: str, expansions: frozenset[str], locale: str = "en_US") -> str:
    """How a reading ``said`` of ``written`` is measured (both sides normalized as the
    spoken priors normalize): ``spelled`` for its letters one by one, ``as-written`` for
    the token itself, a lexicon expansion (in ``expansions``) for its own words, else
    ``other``. A dotted chain said as written is spelled ("e.g." is "e g" either way)."""
    said = normalize_spoken(said)
    if is_chain(written) and said == _spelled(written, locale):
        return SPELLED
    if said == normalize_spoken(written):
        return AS_WRITTEN
    if said in expansions:
        return said
    if said == _spelled(written, locale):
        return SPELLED
    return OTHER


def _is_source(surface: str, locale: str) -> bool:
    stem = surface[:-1]
    return (
        surface.endswith(".")
        and "." not in stem
        and any(_lower(ch, locale) == ch and _upper(ch, locale) != ch for ch in stem)
        and len(fold_key(surface, locale)) > 1
    )


def variant_sources(entries: Mapping[str, object], locale: str = "en_US") -> tuple[str, ...]:
    """The lexicon surfaces variants are made from, in the lexicon's order."""
    return tuple(surface for surface in entries if _is_source(surface, locale))


def chain_sources(entries: Mapping[str, object], locale: str = "en_US") -> tuple[str, ...]:
    """The lexicon's dotted chains with an expansion ("e.g.", "U.S."), in its order."""
    return tuple(
        surface
        for surface, entry in entries.items()
        if is_chain(surface) and getattr(entry, "expansions", ())
    )


def variant_surfaces(entries: Mapping[str, object], locale: str = "en_US") -> dict[str, str]:
    """Each written variant, mapped to the lexicon surface it reads as.

    A variant two sources would both give is dropped: it is not one abbreviation.
    """
    found: dict[str, set[str]] = {}
    for source in variant_sources(entries, locale):
        stem = source[:-1]
        for form in {stem, _lower(stem, locale), _title(stem, locale)}:
            for variant in (form, f"{form}."):
                if variant not in entries:
                    found.setdefault(variant, set()).add(source)
    return {variant: next(iter(sources)) for variant, sources in found.items() if len(sources) == 1}


def _compiled(locale: str):
    from icukit.abbreviation_compile import compile_lexicon

    return compile_lexicon(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _lexicon(locale: str) -> Mapping[str, object]:
    compiled = _compiled(locale)
    return {} if compiled is None else dict(compiled.entries)


@lru_cache(maxsize=LOCALE_CACHE)
def _uppers(locale: str) -> Mapping[str, str]:
    """Each source's ICU upper case, not itself a lexicon surface, to its source."""
    entries = _lexicon(locale)
    found: dict[str, set[str]] = {}
    for source in variant_sources(entries, locale):
        upper = _upper(source[:-1], locale)
        if upper not in entries and f"{upper}." not in entries:
            found.setdefault(upper, set()).add(source)
    return {upper: next(iter(s)) for upper, s in found.items() if len(s) == 1}


@lru_cache(maxsize=LOCALE_CACHE)
def _chains(locale: str) -> Mapping[str, str]:
    """Each lexicon dotted chain with an expansion, by its ICU lower case."""
    entries = _lexicon(locale)
    return {_lower(surface, locale): surface for surface in chain_sources(entries, locale)}


def upper_variant_expansions(run: str, locale: str = "en_US") -> tuple[object, ...]:
    """The lexicon expansions of a source whose ICU upper case is ``run`` ("MR" Mister,
    "DR" Doctor and Drive); none for a run that is its own lexicon surface ("PA")."""
    locale = canonical_locale(locale)
    source = _uppers(locale).get(run)
    return () if source is None else _expansions(_lexicon(locale)[source])


def chain_expansions(written: str, locale: str = "en_US") -> tuple[object, ...]:
    """The lexicon expansions of the dotted chain ``written`` folds to ("E.G." for
    example, from "e.g."); none when the lexicon lists no such chain."""
    if not is_chain(written):
        return ()
    locale = canonical_locale(locale)
    source = _chains(locale).get(_lower(written, locale))
    return () if source is None else _expansions(_lexicon(locale)[source])


@lru_cache(maxsize=LOCALE_CACHE * 16)
def abbreviation_expansions(written: str, locale: str = "en_US") -> tuple[object, ...]:
    """The lexicon expansions of ``written``, including a derived written variant.

    Thus both a lexicon surface (``"Sept."``) and the period-less variant frend derives
    from it (``"Sept"``) return the same ``"September"`` month expansion.
    """
    locale = canonical_locale(locale)
    entries = _lexicon(locale)
    source = written if written in entries else variant_surfaces(entries, locale).get(written)
    return () if source is None else _expansions(entries[source])


def _expansions(entry) -> tuple[object, ...]:
    """A lexicon entry's expansions as icukit's detections carry them."""
    from icukit.abbreviation_recognize import AbbreviationExpansion

    return tuple(
        AbbreviationExpansion(item.value, item.sense, item.cue, item.type)
        for item in entry.expansions
    )


def measured_keys(locale: str = "en_US") -> Mapping[str, frozenset[str]]:
    """Every key the measure counts, with its lexicon expansions normalized: the variant
    sources' and the dotted chains'."""
    return _measured_keys(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _measured_keys(locale: str) -> Mapping[str, frozenset[str]]:
    entries = _lexicon(locale)
    keys: dict[str, set[str]] = {}
    for source in (*variant_sources(entries, locale), *chain_sources(entries, locale)):
        keys.setdefault(fold_key(source, locale), set()).update(
            normalize_spoken(item.value) for item in entries[source].expansions
        )
    return {key: frozenset(values) for key, values in keys.items()}


class AbbreviationVariantDetector:
    """Detect written variants of the locale's lexicon abbreviations ("Mr", "st", "Vol").

    Each detection's value is the lexicon entry's :class:`AbbreviationValue` (its
    surface is the lexicon's, "Mr."); the written text is the detection's own. A locale
    icukit has no lexicon for detects nothing.
    """

    group = "abbreviation"
    type = VARIANT_TYPE

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = locale
        self.entries = _lexicon(canonical_locale(locale))
        self.variants = variant_surfaces(self.entries, locale) if self.entries else {}
        self._surfaces = tuple(sorted(self.variants, key=lambda s: (-len(s), s)))

    def _word_starts(self, text: str) -> set[int]:
        words = icu.BreakIterator.createWordInstance(_language(self.locale))
        words.setText(text)
        return {0, *words}

    @staticmethod
    def _bounded(text: str, start: int, end: int) -> bool:
        """No letter, digit, "_" or "." just before ``start`` or just after ``end``."""
        before = start == 0 or not (text[start - 1].isalnum() or text[start - 1] in "_.")
        after = end == len(text) or not (text[end].isalnum() or text[end] in "_.")
        return before and after

    def detect(self, text: str) -> list[dict]:
        if not self.variants:
            return []
        from icukit.abbreviation_recognize import AbbreviationValue
        from icukit.detectors import Capture

        starts = self._word_starts(text)
        detections = []
        for start in range(len(text)):
            if start not in starts or not text[start].isalpha():
                continue
            surface = next(
                (
                    surface
                    for surface in self._surfaces
                    if text.startswith(surface, start)
                    and self._bounded(text, start, start + len(surface))
                ),
                None,
            )
            if surface is None:
                continue
            entry = self.entries[self.variants[surface]]
            end = start + len(surface)
            value = AbbreviationValue(
                entry.surface, _expansions(entry), entry.also, entry.break_behavior
            )
            detections.append(
                {
                    "text": surface,
                    "start": start,
                    "end": end,
                    "type": VARIANT_TYPE,
                    "value": value,
                    "captures": (Capture("surface", start, end, surface),),
                }
            )
        return detections


def abbreviation_priors(*, locale: str = "en_US") -> Mapping[str, Mapping[str, Mapping[str, int]]]:
    """The locale's measured abbreviation readings, ``key -> case -> category -> count``
    (``data/<locale>/abbreviation_priors.json``); empty when the locale has none."""
    return _priors_for(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _priors_for(locale: str) -> Mapping[str, Mapping[str, Mapping[str, int]]]:
    table = measured_table("abbreviation_priors", locale)
    return {} if table is None else table["keys"]


def measures_spelled(written: str, *, locale: str = "en_US") -> bool:
    """Whether the corpus says ``written`` spelled in its own case's row ("Lt": 1,848 of
    1,848 title rows; "ft" 42 lower), so its letters are a reading. A case the corpus
    has no row for reads the key's rows ("Mr": title is never written, and mr is spelled
    135 times of 8,756); a case row with no spelled count offers none, whatever another
    case says ("no": spelled only in capitals, never in its 69,124 lower-case rows)."""
    rows = abbreviation_priors(locale=locale).get(fold_key(written, locale), {})
    case = rows.get(written_case(written, locale) or "")
    if case:
        return bool(case.get(SPELLED))
    return any(counts.get(SPELLED) for counts in rows.values())


def measured_case(written: str, *, locale: str = "en_US") -> bool:
    """Whether the corpus has a row for ``written``'s key in ``written``'s own case
    ("MR": the upper row, mister 562 of 611)."""
    case = written_case(written, locale)
    rows = abbreviation_priors(locale=locale).get(fold_key(written, locale), {})
    return case is not None and bool(rows.get(case))


def abbreviation_weights(
    written: str, said: Sequence[str], *, locale: str = "en_US"
) -> tuple[Decimal | None, ...] | None:
    """Each reading's share of how the corpus says ``written``, or ``None`` when the
    corpus has no row for its key.

    Each reading in ``said`` is sorted into its :func:`category`. A share is of every
    row of the key (or of its case), "other" readings included. With a row for the
    token's case, it is that row's count blended toward the key's share by
    ``SUB_KEY_PRIOR_STRENGTH``; with none ("Mr": the corpus writes "mr" and "MR", never
    "Mr"), it is the key's share. A category the key never shows is unmeasured (``None``).
    """
    key = fold_key(written, locale)
    rows = abbreviation_priors(locale=locale).get(key)
    if not rows:
        return None
    expansions = measured_keys(locale).get(key, frozenset())
    key_counts: dict[str, int] = {}
    for case, counts in rows.items():
        case_total = 0
        for name, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    f"abbreviation prior count for {key!r}/{case!r}/{name!r} "
                    "must be a non-negative integer"
                )
            case_total += count
            key_counts[name] = key_counts.get(name, 0) + count
        if case_total == 0:
            raise ValueError(
                f"abbreviation prior row for {key!r}/{case!r} must have a positive total"
            )
    key_total = sum(key_counts.values())
    case_counts = rows.get(written_case(written, locale) or "")
    case_total = sum(case_counts.values()) if case_counts else 0
    weights: list[Decimal | None] = []
    for text in said:
        name = category(written, text, expansions, locale)
        if name == OTHER or name not in key_counts:
            weights.append(None)
            continue
        key_share = Decimal(key_counts[name]) / Decimal(key_total)
        if case_counts is None:
            weights.append(key_share)
            continue
        weights.append(
            (Decimal(case_counts.get(name, 0)) + SUB_KEY_PRIOR_STRENGTH * key_share)
            / (Decimal(case_total) + SUB_KEY_PRIOR_STRENGTH)
        )
    return tuple(weights)
