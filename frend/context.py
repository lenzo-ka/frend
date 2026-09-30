"""Context: which of a reading's spoken forms the text around it favors.

frend reads a span by itself; the text around it often settles how it is said ("NASA"
said as a word, "1995" as a year, a lone "-" between two numbers said "to"). This module
reads that context from running text by character offsets -- the span of a lattice
edge in the text it was read from -- and asks a small trained decision tree, one per
reading-choice problem, which spoken form the context favors.

**Features** (``features``), all computed from the text and the span, never from a
corpus's own tokens:

* ``F`` -- Flite's number-tree features (``us_nums_cart.c``, ``us_ffeatures.c``), as
  learned features, not Flite's trained tree: ``num_digits`` (the span's length),
  ``month_range`` (a number 1..31), and a coarse class of the words at -2..+2 (the
  span itself at 0): numeric, number, a month or weekday name (ICU's, for the locale),
  one of the frequent words (a list learned from the training data and shipped with
  the trees), or other. Words are ICU word-break segments that are not white space.
* ``W`` -- character classes: ICU's Word_Break, General_Category and Script of up to
  three characters before and after the span (white space next to the span skipped),
  frend-local until icukit's class-like feature package exists; ``char_classes`` and
  ``class_windows`` are the interface to move onto it.
* ``C`` -- Festival's hand-curated classes (``festival/lib/tokenpos.scm``: regnal names,
  king-like titles, section words), read from ``data/<locale>/context/festival_classes.json``.
* ``B`` -- frend's own first choice (its label) and that choice's weight.
* ``R`` -- a range's own shape, for the range problems only (``range_features``): its
  separator, each end's ASCII digit count, whether the right end is written with a
  leading zero, and whether the separator is spaced.

**Ranking** (``rerank``): the tree's top reading replaces frend's first choice only
when the tree gives it probability of at least :data:`CONTEXT_THRESHOLD`; otherwise the
order is frend's. Nothing else changes: weights stay what they were (a measured share
stays ranking metadata, and the alignment graph keeps every spoken form at unit weight).

**Ranges** (``connector_words``, ``cldr_range_separator``, ``between_numbers``): a
range separator between two numbers, spaced alike on both sides ("5 - 10", "5-10",
"10:30-11:45"; not "10 -5"), is also offered
as the locale's spoken range connector, from the lexical table's ``range.connector``
pattern ("{0} to {1}"; ``frend.verbalize`` reads the table); a locale without the form
gets no such reading.

**Range problems** (``rerank_range``): a range span's readings are one problem per
separator class (``range:range``, ``range:ratio``, ``range:dimension``), labeled by their
joint sources; the tree's top source replaces frend's first only at
:data:`RANGE_CONTEXT_THRESHOLD` or more.

The trees and their data are measured per locale (``data/<locale>/context/``); a locale
with none has no context ranking, and every reading keeps frend's own order.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import cache, lru_cache
from typing import Any

import icu

from frend.locale_data import LOCALE_CACHE, canonical_locale, measured_directory
from frend.spoken_priors import normalize_spoken

__all__ = [
    "CONTEXT_THRESHOLD",
    "RANGE_CONTEXT_THRESHOLD",
    "ContextChoice",
    "ContextModel",
    "TextContext",
    "between_numbers",
    "char_classes",
    "class_windows",
    "context_model",
    "features",
    "neighbor_words",
    "problem_labels",
    "cldr_range_separator",
    "connector_words",
    "range_features",
    "range_problem",
    "rerank",
    "rerank_range",
]

# A tree's top reading overrides frend's first choice only at this probability or more
# (P7 stage A, swept 0.6 to 0.9 on held-out shard 95: at 0.7 losses fall from 36 to 11
# while most gains stay).
CONTEXT_THRESHOLD = 0.7
# A range tree's top joint source overrides frend's first only at this probability or
# more: its own constant. Stage 2 (the runtime-eval shards 90-94) found no value better
# than this beyond noise (0.6: +16 of 65,392 triples).
RANGE_CONTEXT_THRESHOLD = 0.7

# Characters of running text read on either side of a span: enough for three words and
# three characters; a cut is moved to white space so no word is split by it.
_REACH = 96
_CLASS_WINDOW = 3
_WORDS_LEFT = 3  # Festival's title feature looks three words back
_WORDS_RIGHT = 2

BOS, EOS, PAD = "<BOS>", "<EOS>", "<PAD>"

# The feature families and the numeric features among them (the rest are categories).
FAMILIES = ("B", "F", "W", "C", "R")
NUMERIC_FEATURES = frozenset({"b_w0", "f_ndig", "r_ldig", "r_rdig"})


@dataclass(frozen=True)
class ContextChoice:
    """What the context tree said about one span's readings.

    ``problem`` is the reading-choice problem (the sorted labels of the span's
    readings), ``label`` the tree's top reading and ``probability`` its probability;
    ``applied`` is whether it replaced frend's first choice.
    """

    problem: str
    label: str
    probability: float
    applied: bool


@dataclass(frozen=True)
class TextContext:
    """The running text a lattice was read from, for its context.

    A lattice's own ``source_text`` is its context by default. A caller that reads a
    piece of a longer text alone (the evaluator reads one corpus token at a time) passes
    the longer ``text`` and the ``offset`` at which the piece starts in it. ``bos`` and
    ``eos`` say whether the text's ends are real ends (a sentence's), or cuts from a
    longer text (a training window), which read as padding.
    """

    text: str
    offset: int = 0
    bos: bool = True
    eos: bool = True


# --------------------------------------------------------------------------------------
# Character classes (W). The class alphabet is ICU's: Word_Break, General_Category and
# Script short names. This is the interface to move onto icukit's class-like package.


@lru_cache(maxsize=4096)
def char_classes(char: str) -> tuple[str, str, str]:
    """ICU's Word_Break, General_Category and Script short names for one character."""
    code = ord(char)
    word_break = icu.Char.getPropertyValueName(
        icu.UProperty.WORD_BREAK,
        icu.Char.getIntPropertyValue(code, icu.UProperty.WORD_BREAK),
        icu.UPropertyNameChoice.SHORT_PROPERTY_NAME,
    )
    category = icu.Char.getPropertyValueName(
        icu.UProperty.GENERAL_CATEGORY,
        icu.Char.getIntPropertyValue(code, icu.UProperty.GENERAL_CATEGORY),
        icu.UPropertyNameChoice.SHORT_PROPERTY_NAME,
    )
    return word_break, category, icu.Script.getScript(code).getShortName()


def class_windows(
    text: str, start: int, end: int, *, k: int = _CLASS_WINDOW, bos: bool = True, eos: bool = True
) -> dict[str, str]:
    """The classes of up to ``k`` characters before ``start`` and after ``end``.

    White space next to the span is skipped, so "5 - 10" and "5-10" see the same
    neighbors; white space further out is a character like any other. Past the end of
    the text the class is ``<BOS>``/``<EOS>`` when that end is the text's own
    (``bos``/``eos``), else ``<PAD>`` (a window cut from longer text).
    """
    before = text[max(0, start - _REACH) : start].rstrip()[::-1]  # nearest first
    after = text[end : end + _REACH].lstrip()
    out: dict[str, str] = {}
    for side, chars, marker in (
        ("-", before, BOS if bos else PAD),
        ("+", after, EOS if eos else PAD),
    ):
        for i in range(1, k + 1):
            classes = char_classes(chars[i - 1]) if len(chars) >= i else (marker,) * 3
            for name, value in zip(("wb", "gc", "sc"), classes, strict=True):
                out[f"w_{name}{side}{i}"] = value
    return out


# --------------------------------------------------------------------------------------
# Words around a span (F, C): ICU word-break segments that are not white space.


@cache
def _word_breaker(locale: str) -> icu.BreakIterator:
    return icu.BreakIterator.createWordInstance(icu.Locale(locale))


def _segments(text: str, locale: str) -> list[tuple[int, int]]:
    """The word-break segments of ``text`` that are not white space, as code-point
    offsets (ICU's boundaries are UTF-16 offsets, mapped back here)."""
    breaker = _word_breaker(locale)
    breaker.setText(text)
    boundaries = [breaker.first(), *breaker]
    if any(ord(char) > 0xFFFF for char in text):
        to_code_point = {}
        unit = 0
        for index, char in enumerate(text):
            to_code_point[unit] = index
            unit += 2 if ord(char) > 0xFFFF else 1
        to_code_point[unit] = len(text)
        boundaries = [to_code_point[b] for b in boundaries]
    return [(a, b) for a, b in zip(boundaries, boundaries[1:], strict=False) if text[a:b].strip()]


def neighbor_words(
    text: str,
    start: int,
    end: int,
    *,
    locale: str = "en_US",
    left: int = _WORDS_LEFT,
    right: int = _WORDS_RIGHT,
    bos: bool = True,
    eos: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The ``left`` words before the span (nearest first) and ``right`` words after.

    A word is an ICU word-break segment that is not white space ("Nov", ".", "5.5");
    one that overlaps the span is neither. Where the text runs out the slot is
    ``<BOS>``/``<EOS>`` at the text's own ends (``bos``/``eos``) and ``<PAD>`` where a
    window was cut from longer text.
    """
    lo = max(0, start - _REACH)
    before = text[lo:start]
    cut_left = lo > 0
    if cut_left:
        # Cut at white space, so no word is split by the window's edge.
        space = re.search(r"\s", before)
        if space is not None:
            before = before[space.end() :]
    hi = min(len(text), end + _REACH)
    after = text[end:hi]
    cut_right = hi < len(text)
    if cut_right:
        spaces = [match.start() for match in re.finditer(r"\s", after)]
        if spaces:
            after = after[: spaces[-1]]
    words_before = [before[a:b] for a, b in _segments(before, locale)][::-1]
    words_after = [after[a:b] for a, b in _segments(after, locale)]
    left_edge = BOS if bos and not cut_left else PAD
    right_edge = EOS if eos and not cut_right else PAD

    def pad(words: list[str], count: int, edge: str) -> tuple[str, ...]:
        words = words[:count]
        if len(words) < count:
            words = [*words, edge, *([PAD] * (count - len(words) - 1))]
        return tuple(words)

    return pad(words_before, left, left_edge), pad(words_after, right, right_edge)


@lru_cache(maxsize=LOCALE_CACHE)
def _calendar_names(locale: str) -> tuple[frozenset[str], frozenset[str]]:
    """ICU's month and weekday names for ``locale``, wide and abbreviated, lower-cased."""
    symbols = icu.DateFormatSymbols(icu.Locale(locale))
    lower = icu.Locale(locale)

    def fold(names: Sequence[str]) -> frozenset[str]:
        return frozenset(str(icu.UnicodeString(name).toLower(lower)) for name in names if name)

    months = fold([*symbols.getMonths(), *symbols.getShortMonths()])
    days = fold([*symbols.getWeekdays(), *symbols.getShortWeekdays()])
    return months, days


_NUMERIC = re.compile(r"\d+")
_NUMBER = re.compile(r"-?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?")


def _lower(word: str, locale: str) -> str:
    return str(icu.UnicodeString(word).toLower(icu.Locale(locale)))


def _word_class(word: str, locale: str, frequent: frozenset[str]) -> str:
    """Flite's ``token_pos_guess``, with ICU's names and a learned frequent-word list."""
    if word in (BOS, EOS, PAD):
        return word
    if _NUMERIC.fullmatch(word):
        return "numeric"
    if _NUMBER.fullmatch(word):
        return "number"
    lower = _lower(word, locale)
    months, days = _calendar_names(locale)
    if lower in months:
        return "month"
    if lower in days:
        return "day"
    if lower in frequent:
        return f"w:{lower}"
    return "other"


# --------------------------------------------------------------------------------------
# The features of one span.


def features(
    text: str,
    start: int,
    end: int,
    *,
    first: str,
    first_weight: Decimal | float | None,
    locale: str = "en_US",
    frequent: frozenset[str] = frozenset(),
    curated: Mapping[str, frozenset[str]] | None = None,
    bos: bool = True,
    eos: bool = True,
) -> dict[str, Any]:
    """Every feature of the span ``text[start:end]``, by name (families B, F, W, C).

    ``first`` and ``first_weight`` are frend's first choice's label and weight;
    ``frequent`` is the learned frequent-word list and ``curated`` Festival's classes
    (``regnal_names``, ``titles``, ``sections``), both from the locale's context data.
    """
    surface = text[start:end]
    before, after = neighbor_words(text, start, end, locale=locale, bos=bos, eos=eos)
    out: dict[str, Any] = {
        "b_first": first,
        "b_w0": -1.0 if first_weight is None else round(float(first_weight), 3),
        "f_ndig": float(len(surface)),
        "f_month": "1" if _NUMERIC.fullmatch(surface) and 0 < int(surface) < 32 else "0",
    }
    words = {-2: before[1], -1: before[0], 0: surface, 1: after[0], 2: after[1]}
    for offset, word in words.items():
        out[f"f_tpg{offset:+d}"] = _word_class(word, locale, frequent)
    out.update(class_windows(text, start, end, bos=bos, eos=eos))
    curated = curated or {}
    lowered = [_lower(word, locale) for word in before]
    next_word = _lower(after[0], locale)
    names = curated.get("regnal_names", frozenset())
    titles = curated.get("titles", frozenset())
    sections = curated.get("sections", frozenset())
    out["c_rexname-1"] = "1" if lowered[0] in names else "0"
    out["c_rextitle"] = (
        "1" if any(word in titles for word in lowered) or next_word in titles else "0"
    )
    out["c_section-1"] = "1" if lowered[0] in sections else "0"
    return out


# --------------------------------------------------------------------------------------
# The problem a span's readings pose, and the trees that answer it.


def problem_labels(provenances: Sequence[str]) -> tuple[str, ...]:
    """Each reading's label: its provenance, a repeat suffixed ``#2``, ``#3``, ..."""
    seen: dict[str, int] = {}
    labels = []
    for provenance in provenances:
        seen[provenance] = seen.get(provenance, 0) + 1
        count = seen[provenance]
        labels.append(provenance if count == 1 else f"{provenance}#{count}")
    return tuple(labels)


def distinct_readings(alternatives: Sequence[Any]) -> list[int]:
    """Indices of the alternatives that say something new (normalized spoken text),
    in frend's order: two that are said alike are one reading."""
    seen: set[str] = set()
    out = []
    for index, alternative in enumerate(alternatives):
        key = normalize_spoken(alternative.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(index)
    return out


class _Tree:
    def __init__(self, blob: bytes):
        from cartlet import Predictor

        self._predictor = Predictor(blob)
        self.names = list(self._predictor.feature_names)

    def distribution(self, values: Mapping[str, Any]) -> dict[str, float]:
        vector = [values.get(name) for name in self.names]
        dist = self._predictor.predict(vector, return_dist=True, missing="right")
        if not isinstance(dist, dict):  # a tree stored without distributions
            return {str(dist): 1.0}
        return {str(label): float(p) for label, p in dist.items()}


class ContextModel:
    """One locale's context trees (``data/<locale>/context/``), loaded as they are used."""

    def __init__(self, directory: Any, index: Mapping[str, Any], curated: Mapping[str, Any]):
        self._directory = directory
        self.index = index
        self.frequent = frozenset(index["frequent_words"])
        self.curated = {name: frozenset(words) for name, words in curated["classes"].items()}
        self._trees: dict[str, _Tree | None] = {}

    def tree(self, problem: str) -> _Tree | None:
        if problem not in self._trees:
            entry = self.index["trees"].get(problem)
            self._trees[problem] = (
                None
                if entry is None
                else _Tree(self._directory.joinpath(entry["file"]).read_bytes())
            )
        return self._trees[problem]

    def connector(self, separator: str) -> Mapping[str, Any] | None:
        """The standalone separator's problem, its "to" label and its constant first
        choice, used for a separator inside a written range ("5-10")."""
        return self.index.get("connectors", {}).get(separator)


def context_model(locale: str) -> ContextModel | None:
    """The locale's context trees, or ``None`` when its chain has none (no ranking)."""
    return _context_model(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _context_model(locale: str) -> ContextModel | None:
    directory = measured_directory("context", locale)
    if directory is None:
        return None
    index = json.loads(directory.joinpath("index.json").read_text(encoding="utf-8"))
    curated = json.loads(directory.joinpath("festival_classes.json").read_text(encoding="utf-8"))
    return ContextModel(directory, index, curated)


def _top(distribution: Mapping[str, float], labels: Sequence[str]) -> tuple[str, float] | None:
    """The most probable of ``labels`` (ties to frend's order), or ``None``."""
    ranked = sorted(
        ((distribution.get(label, 0.0), -order, label) for order, label in enumerate(labels)),
        reverse=True,
    )
    probability, _, label = ranked[0]
    return (label, probability) if probability > 0 else None


def rerank(
    alternatives: Sequence[Any],
    text: str,
    start: int,
    end: int,
    *,
    locale: str = "en_US",
    bos: bool = True,
    eos: bool = True,
    threshold: float = CONTEXT_THRESHOLD,
) -> tuple[tuple[Any, ...], ContextChoice | None]:
    """``alternatives`` with the context tree's top reading first, when the tree for
    their problem gives it at least ``threshold``; otherwise unchanged. Returns the
    (possibly reordered) alternatives and what the tree said (``None`` without a tree).
    """
    alternatives = tuple(alternatives)
    model = context_model(locale)
    if model is None:
        return alternatives, None
    distinct = distinct_readings(alternatives)
    if len(distinct) < 2:
        return alternatives, None
    labels = problem_labels([alternatives[index].provenance for index in distinct])
    problem = "\t".join(sorted(labels))
    tree = model.tree(problem)
    if tree is None:
        return alternatives, None
    first = alternatives[distinct[0]]
    values = features(
        text,
        start,
        end,
        first=labels[0],
        first_weight=first.weight,
        locale=locale,
        frequent=model.frequent,
        curated=model.curated,
        bos=bos,
        eos=eos,
    )
    top = _top(tree.distribution(values), labels)
    if top is None:
        return alternatives, None
    label, probability = top
    applied = label != labels[0] and probability >= threshold
    choice = ContextChoice(problem, label, probability, applied)
    if not applied:
        return alternatives, choice
    chosen = distinct[labels.index(label)]
    return (alternatives[chosen], *alternatives[:chosen], *alternatives[chosen + 1 :]), choice


def range_problem(separator_class: str) -> str:
    """The problem a range span's readings pose: one per separator class."""
    return f"range:{separator_class}"


_ASCII = frozenset("0123456789")


def range_features(separator: str, left: str, right: str) -> dict[str, Any]:
    """Family R: a range's separator (``r_sep``), each end's ASCII digit count
    (``r_ldig``, ``r_rdig``; -1 for an end that is not ASCII digits only), and whether
    the right end is written with a leading zero and two or more digits (``r_lead0``:
    "05"). Spacing is no feature: the corpus's triples record none."""

    def digits(text: str) -> float:
        return float(len(text)) if text and set(text) <= _ASCII else -1.0

    return {
        "r_sep": separator,
        "r_ldig": digits(left),
        "r_rdig": digits(right),
        "r_lead0": "1" if len(right) > 1 and set(right) <= _ASCII and right[0] == "0" else "0",
    }


def _range_shape(value: Any) -> tuple[str, str, str]:
    left = str(value.left[0].get("text", "")) if value.left else ""
    right = str(value.right[0].get("text", "")) if value.right else ""
    return value.separator, left, right


def distinct_sources(alternatives: Sequence[Any]) -> list[int]:
    """Indices of the first alternative of each provenance, in frend's order: a range
    tree's labels are joint sources, not texts."""
    seen: set[str] = set()
    out = []
    for index, alternative in enumerate(alternatives):
        if alternative.provenance in seen:
            continue
        seen.add(alternative.provenance)
        out.append(index)
    return out


def rerank_range(
    alternatives: Sequence[Any],
    text: str,
    start: int,
    end: int,
    value: Any,
    *,
    locale: str = "en_US",
    bos: bool = True,
    eos: bool = True,
    threshold: float | None = None,
) -> tuple[tuple[Any, ...], ContextChoice | None]:
    """A range span's readings (``value`` a ``RangeValue``) with its class's tree's top
    joint source first, when the tree gives it at least ``threshold``
    (:data:`RANGE_CONTEXT_THRESHOLD` by default); otherwise unchanged. The problem is
    ``range:<separator class>``; its labels are the span's distinct sources in frend's
    order."""
    alternatives = tuple(alternatives)
    if threshold is None:
        threshold = RANGE_CONTEXT_THRESHOLD
    model = context_model(locale)
    if model is None:
        return alternatives, None
    distinct = distinct_sources(alternatives)
    if len(distinct) < 2:
        return alternatives, None
    problem = range_problem(value.separator_class)
    tree = model.tree(problem)
    if tree is None:
        return alternatives, None
    labels = [alternatives[index].provenance for index in distinct]
    first = alternatives[distinct[0]]
    values = range_example_features(
        text, start, end, value, first=labels[0], first_weight=first.weight,
        locale=locale, model=model, bos=bos, eos=eos,
    )  # fmt: skip
    top = _top(tree.distribution(values), labels)
    if top is None:
        return alternatives, None
    label, probability = top
    applied = label != labels[0] and probability >= threshold
    choice = ContextChoice(problem, label, probability, applied)
    if not applied:
        return alternatives, choice
    chosen = distinct[labels.index(label)]
    return (alternatives[chosen], *alternatives[:chosen], *alternatives[chosen + 1 :]), choice


def range_example_features(
    text: str,
    start: int,
    end: int,
    value: Any,
    *,
    first: str,
    first_weight: Decimal | float | None,
    locale: str = "en_US",
    model: ContextModel | None = None,
    frequent: frozenset[str] | None = None,
    curated: Mapping[str, frozenset[str]] | None = None,
    bos: bool = True,
    eos: bool = True,
) -> dict[str, Any]:
    """Every feature of a range span (families B, F, W, C and R)."""
    if model is not None:
        frequent, curated = model.frequent, model.curated
    out = features(
        text, start, end, first=first, first_weight=first_weight, locale=locale,
        frequent=frequent or frozenset(), curated=curated, bos=bos, eos=eos,
    )  # fmt: skip
    out.update(range_features(*_range_shape(value)))
    return out


def connector_probability(
    text: str,
    start: int,
    end: int,
    *,
    locale: str = "en_US",
    bos: bool = True,
    eos: bool = True,
) -> tuple[str, float] | None:
    """For a range separator ``text[start:end]`` written inside a range ("5-10"): the
    standalone separator's tree ("5 - 10") read at the same place, as (its "to" label,
    the probability it gives "to"); ``None`` without that tree."""
    model = context_model(locale)
    if model is None:
        return None
    connector = model.connector(text[start:end])
    if connector is None:
        return None
    tree = model.tree(connector["problem"])
    if tree is None:
        return None
    values = features(
        text,
        start,
        end,
        first=connector["first"],
        first_weight=connector["first_weight"],
        locale=locale,
        frequent=model.frequent,
        curated=model.curated,
        bos=bos,
        eos=eos,
    )
    label = connector.get("connector_source")
    if not isinstance(label, str):
        return None
    return label, tree.distribution(values).get(label, 0.0)


# --------------------------------------------------------------------------------------
# Ranges: the spoken connector and the written separators.


def connector_words(pattern: str) -> str | None:
    """The words a range's two ends are joined by in a ``range.connector`` pattern
    ("{0} to {1}" gives "to"); ``None`` when the pattern says something outside the two
    ends (Russian's "от {0} до {1}" cannot be said by the separator alone)."""
    if not (pattern.startswith("{0}") and pattern.endswith("{1}")):
        return None
    return pattern[len("{0}") : -len("{1}")].strip() or None


def cldr_range_separator(locale: str) -> str | None:
    """CLDR's own number-range separator for ``locale`` (root ``miscPatterns/range``
    "{0}–{1}": the en dash), found along the locale's chain."""
    return _cldr_range_separator(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _cldr_range_separator(locale: str) -> str | None:
    from frend.locale_data import locale_chain

    for tag in locale_chain(locale):
        bundle = icu.ResourceBundle("", icu.Locale(tag))
        try:
            pattern = (
                bundle.get("NumberElements").get("latn").get("miscPatterns").getStringEx("range")
            )
        except icu.ICUError:
            continue
        pattern = str(pattern)
        if pattern.startswith("{0}") and pattern.endswith("{1}"):
            return pattern[len("{0}") : -len("{1}")].strip() or None
    return None


_ASCII_DIGITS = frozenset("0123456789")


def between_numbers(text: str, start: int, end: int) -> bool:
    """Whether the nearest characters before and after ``text[start:end]`` that are not
    white space are both ASCII digits, with the separator spaced alike on both sides:
    joined ("5-10", "10:30-11:45") or spaced ("5 - 10"). A sign spaced on one side only
    ("10 -5", "5 -10") writes a signed number after a number, not a range. Digits of
    another script are not read here, so they are no range ends (for now)."""
    before = text[:start]
    after = text[end:]
    left = before.rstrip()
    right = after.lstrip()
    if not (left and right and left[-1] in _ASCII_DIGITS and right[0] in _ASCII_DIGITS):
        return False
    return (len(left) == len(before)) == (len(right) == len(after))
