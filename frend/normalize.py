"""Plain text-in, spoken text-out normalization."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, overload

from icukit import break_sentence_spans
from icukit.detectors import DetectorSet, date_detectors
from icukit.recognize import (
    AlphanumericRunsDetector,
    FlexibleCompactDetector,
    FlexibleCurrencyDetector,
    FlexibleCurrencyNameDetector,
    FlexibleDateDetector,
    FlexibleFractionDetector,
    FlexibleMeasureDetector,
    FlexibleMixedMeasureDetector,
    FlexibleNumberDetector,
    FlexibleOrdinalDetector,
    FlexiblePercentDetector,
    FlexibleTextDateDetector,
    FlexibleTimeDetector,
    LetterNameDetector,
    PluralNumeralDetector,
    SingleLetterWordDetector,
)

from frend.abbreviation_variants import AbbreviationVariantDetector
from frend.durations import NumericDurationDetector
from frend.electronic import _ValidatedElectronicDetector
from frend.input_folds import InputFold, apply_input_fold
from frend.input_limits import (
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_UNIT_CHARS,
    InputValidationError,
    validate_input,
    validate_unit_length,
)
from frend.lattice import ReadingEdge, _resolve_lattice_validated
from frend.letters import (
    LettersDetector,
    _dotted_acronym_suffix_spans,
    _dotted_plural_acronym_period_spans,
    _uppercase_abbreviation_period_spans,
)
from frend.locale_data import LOCALE_CACHE, canonical_locale
from frend.profiles import GroupOrders, validate_groups, validate_profile
from frend.ranges import RangeDetector, _date_interval_gang, icu_range_readers
from frend.spacing import unit_gap
from frend.symbols import (
    DEFAULT_SYMBOL_RUN_THRESHOLD,
    SymbolDetector,
    _period_ending_abbreviation_ends,
)
from frend.symbols import (
    _code_ranges as _symbol_code_ranges,
)
from frend.symbols import (
    _overlaps as _symbol_overlaps,
)
from frend.verbalize import SpokenAlternative, VerbalizedUnit, _verbalize_lattice_validated
from frend.work_budgets import (
    DEFAULT_TIERGRAPH_WORK_BUDGET,
    TiergraphWorkBudget,
    work_meter,
)
from frend.written_forms import WrittenFormsDetector

__all__ = ["NormalizedText", "NormalizedUnit", "normalize"]

_DATE_SKELETONS = ("yMd", "Md", "y")
_CURRENCIES = ("USD", "EUR", "GBP", "JPY", "CNY", "INR", "CAD", "AUD", "KRW", "RUB", "XCD")
_MEASURE_UNITS = (
    "kilometer",
    "meter",
    "millimeter",
    "centimeter",
    "micrometer",
    "nanometer",
    "foot",
    "inch",
    "mile",
    "yard",
    "kilometer-per-hour",
    "mile-per-hour",
    "kilogram",
    "gram",
    "milligram",
    "pound",
    "ounce",
    "ton",
    "hectare",
    "acre",
    "square-kilometer",
    "square-mile",
    "square-meter",
    "kilowatt",
    "megawatt",
    "watt",
    "horsepower",
    "volt",
    "hertz",
    "kilohertz",
    "megahertz",
    "gigahertz",
    "kilobyte",
    "megabyte",
    "gigabyte",
    "terabyte",
    "liter",
    "milliliter",
    "degree",
    "celsius",
    "fahrenheit",
    "second",
    "minute",
    "hour",
    "gigawatt",
    "meter-per-second",
    "square-foot",
    "millibar",
    "cubic-meter",
)
_MIXED_MEASURES = ("foot-and-inch", "pound-and-ounce")


class _DetectorRegistry(Sequence[object]):
    """One retained icukit gang plus optional families unavailable for its locale.

    The sequence methods preserve the detector-profile interface used by offline tools.
    Production recognition goes through :meth:`detect`, so icukit can compile the gang
    once and retain its shared scan plan and tables for later texts.
    """

    def __init__(self, detectors: list[object], missing_families: list[str]) -> None:
        self.gang = DetectorSet(tuple(detectors))
        self.missing_families = tuple(missing_families)

    def __getitem__(self, index):
        return self.gang.detectors[index]

    def __iter__(self) -> Iterator[object]:
        return iter(self.gang.detectors)

    def __len__(self) -> int:
        return len(self.gang.detectors)

    def detect(self, text: str) -> list:
        return self.gang.detect(text)


@lru_cache(maxsize=LOCALE_CACHE)
def _date_detector_gang(locale: str) -> DetectorSet:
    """The retained strict date gang also used to protect sentence boundaries."""
    return date_detectors(locale, _DATE_SKELETONS)


@dataclass(frozen=True)
class NormalizedUnit:
    """One output unit and the original source span that produced it."""

    output_span: tuple[int, int]
    source_span: tuple[int, int]
    reader: str | None
    provenance: str


@dataclass(frozen=True)
class NormalizedText:
    """Normalized text, source alignment, and unavailable optional detector families."""

    text: str
    units: list[NormalizedUnit]
    fold: InputFold | None = None
    missing_detector_families: tuple[str, ...] = field(default=(), repr=False)


@lru_cache(maxsize=LOCALE_CACHE)
def _reading_detectors(
    locale: str, symbol_run_threshold: int = DEFAULT_SYMBOL_RUN_THRESHOLD
) -> _DetectorRegistry:
    """The evaluator's recognition profile, packaged for the public API."""
    from icukit.abbreviation_recognize import AbbreviationDetector

    runs = AlphanumericRunsDetector(locale)
    written = WrittenFormsDetector(locale)
    dates = _date_detector_gang(locale).with_(
        FlexibleDateDetector(locale),
        FlexibleTextDateDetector(locale),
        PluralNumeralDetector(locale),
        runs,
    )
    number = FlexibleNumberDetector(locale)
    numbers = (
        number,
        FlexibleCompactDetector(locale, "long"),
        FlexibleCompactDetector(locale, "short"),
    )
    cardinals = numbers + (LetterNameDetector(locale), SingleLetterWordDetector(locale))
    missing_families: list[str] = []
    mixed_measures: list[object] = []
    for mixed in _MIXED_MEASURES:
        family = f"measure:{mixed}"
        try:
            mixed_measures.append(FlexibleMixedMeasureDetector(locale, mixed))
        except ValueError:
            missing_families.append(family)
    measures = (
        FlexiblePercentDetector(locale),
        *(FlexibleMeasureDetector(locale, unit) for unit in _MEASURE_UNITS),
        *mixed_measures,
    )
    money = tuple(
        detector
        for code in _CURRENCIES
        for detector in (
            FlexibleCurrencyDetector(locale, code),
            FlexibleCurrencyNameDetector(locale, code),
        )
    )
    by_kind = (
        (*cardinals, written, *icu_range_readers(locale, cardinals)),
        (number, written),
        (SymbolDetector(locale, run_threshold=symbol_run_threshold),),
        numbers,
        (*dates.detectors, *_date_interval_gang(locale).detectors, written),
        (FlexibleFractionDetector(locale),),
        (FlexibleOrdinalDetector(locale), number, runs),
        (FlexibleTimeDetector(locale), NumericDurationDetector(locale), runs, written),
        (_ValidatedElectronicDetector(locale),),
        (*measures, *icu_range_readers(locale, measures)),
        (*money, *icu_range_readers(locale, money)),
    )
    seen: set[int] = set()
    detectors: list[object] = []
    for group in by_kind:
        for detector in group:
            if id(detector) not in seen:
                seen.add(id(detector))
                detectors.append(detector)
    detectors.extend(
        (
            AbbreviationDetector(locale),
            LettersDetector(locale),
            AbbreviationVariantDetector(locale),
        )
    )
    detectors.append(RangeDetector(locale, endpoints=tuple(detectors)))
    return _DetectorRegistry(detectors, missing_families)


def _sentence_ranges(text: str, locale: str) -> list[tuple[int, int]]:
    """Return nonempty sentence ranges with boundary whitespace excluded."""
    spans = break_sentence_spans(text, locale)
    nonempty = [span for span in spans if span["text"].strip()]
    # Sentence breaking must not split dotted acronym suffixes or a consumed uppercase
    # abbreviation period from its following word. Advance one monotone pointer so many
    # protected spans and boundaries stay linear.
    protected = tuple(
        sorted(
            (
                *_dotted_acronym_suffix_spans(text),
                *_uppercase_abbreviation_period_spans(text, locale),
            )
        )
    )
    protected_index = 0
    abbreviation_ends = _period_ending_abbreviation_ends(text, locale)

    # Locale breakers can treat periods in numeric dates as sentence terminators
    # (notably Korean ``2024. 6. 30.``). Recognition is authoritative for those
    # spans, so keep each boundary strictly inside a date edge in one unit.
    crossing_dates = sorted(
        (int(edge["start"]), int(edge["end"]))
        for edge in _date_detector_gang(locale).detect(text)
        if str(edge.get("type", "")).startswith("date:")
    )
    date_index = 0
    ranges: list[tuple[int, int]] = []
    for span in nonempty:
        surface = span["text"]
        left = len(surface) - len(surface.lstrip())
        right = len(surface.rstrip())
        start = int(span["start"]) + left
        end = int(span["start"]) + right
        while protected_index < len(protected) and protected[protected_index][1] <= start:
            protected_index += 1
        crosses_protected = protected_index < len(protected) and (
            protected[protected_index][0] < start < protected[protected_index][1]
        )
        previous_end = ranges[-1][1] if ranges else start
        while date_index < len(crossing_dates) and crossing_dates[date_index][1] <= previous_end:
            date_index += 1
        crosses_date = date_index < len(crossing_dates) and (
            crossing_dates[date_index][0] < previous_end and crossing_dates[date_index][1] > start
        )
        follows_period_ending_abbreviation = (
            ranges and previous_end == start and start in abbreviation_ends
        )
        if ranges and (crosses_protected or crosses_date or follows_period_ending_abbreviation):
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))
    return ranges


def _alternative_text(alternative: SpokenAlternative) -> str:
    if alternative.provenance == "surface:passthrough":
        return alternative.text
    return f" {alternative.text} "


def _append_alternative_part(
    parts: list[str],
    previous_text: str | None,
    alternative: SpokenAlternative,
) -> None:
    """Append one unit exactly as the public renderer does."""
    part = _alternative_text(alternative)
    if previous_text is not None and not unit_gap(previous_text, alternative.text):
        parts[-1] = parts[-1].rstrip(" ")
        part = part.lstrip(" ")
    parts.append(part)


def _joined_unit_texts(units: tuple[VerbalizedUnit, ...]) -> list[str]:
    """Render units while closing only ICU-declared unspaced boundaries."""
    parts: list[str] = []
    previous_text: str | None = None
    for unit in units:
        alternative = unit.best
        _append_alternative_part(parts, previous_text, alternative)
        previous_text = alternative.text
    return parts


def _is_paired_quote_measure(text: str, detection: dict) -> bool:
    """Reject an ASCII quote as a unit when the same quote opens the number."""
    start = int(detection["start"])
    end = int(detection["end"])
    quote = {"measure:foot": "'", "measure:inch": '"'}.get(detection.get("type"))
    return (
        quote is not None
        and start > 0
        and text[start - 1] == quote
        and text[end - 1 : end] == quote
    )


def _is_bare_degree_temperature_measure(text: str, detection: dict) -> bool:
    """Reject a temperature scale that is absent from the written measure."""
    type_ = detection.get("type")
    if type_ in {"measure:celsius", "measure:fahrenheit"}:
        inferred_temperature = True
    elif type_ == "measure:range":
        value = detection.get("value")
        inferred_temperature = {
            getattr(getattr(value, "start", None), "unit", None),
            getattr(getattr(value, "end", None), "unit", None),
        } & {"celsius", "fahrenheit"}
    else:
        return False
    start = int(detection["start"])
    end = int(detection["end"])
    return bool(inferred_temperature) and text[start:end].rstrip().endswith("°")


def _is_empty_dotted_plural_abbreviation(
    detection: dict, plural_period_spans: frozenset[tuple[int, int]]
) -> bool:
    """Reject the lexicon's empty longer match, but keep genuine abbreviations."""
    return (
        detection.get("type") == "abbreviation"
        and (int(detection["start"]), int(detection["end"])) in plural_period_spans
        and not getattr(detection.get("value"), "expansions", ())
    )


def _sentence(
    text: str,
    *,
    source_offset: int,
    symbol_code_ranges: tuple[tuple[int, int], ...],
    raw_text: str,
    fold: InputFold | None,
    locale: str,
    profile: str | None,
    case_variant_lookup: bool,
    offsets: bool,
    max_input_chars: int | None,
    max_unit_chars: int | None,
    symbol_run_threshold: int,
    groups: GroupOrders | None,
    work_budget: TiergraphWorkBudget,
) -> tuple[list[tuple[str, ReadingEdge, VerbalizedUnit]] | None, str]:
    detections = _reading_detectors(locale, symbol_run_threshold).detect(text)
    plural_period_spans = _dotted_plural_acronym_period_spans(text)
    detections = [
        detection
        for detection in detections
        if not _is_paired_quote_measure(text, detection)
        and not _is_bare_degree_temperature_measure(text, detection)
        and not _is_empty_dotted_plural_abbreviation(detection, plural_period_spans)
        and (
            detection.get("type") != "symbol:run"
            or not _symbol_overlaps(
                symbol_code_ranges,
                source_offset + int(detection["start"]),
                source_offset + int(detection["end"]),
            )
        )
    ]
    lattice = _resolve_lattice_validated(
        detections,
        source_text=text,
        raw_source_text=raw_text,
        fold=fold,
        locale=locale,
        max_input_chars=max_input_chars,
        max_unit_chars=max_unit_chars,
        work_budget=work_budget,
    )
    edges = {edge.id: edge for edge in lattice.edges} if offsets else None
    verbalized = _verbalize_lattice_validated(
        lattice,
        profile=profile,
        case_variant_lookup=case_variant_lookup,
        max_input_chars=max_input_chars,
        max_unit_chars=max_unit_chars,
        groups=groups,
    )
    rendered: list[tuple[str, ReadingEdge, VerbalizedUnit]] | None = [] if offsets else None
    edge_ids = verbalized.best_path.edge_ids
    units = verbalized.best_path.units
    parts = _joined_unit_texts(units)
    for edge_id, unit, part in zip(
        edge_ids,
        units,
        parts,
        strict=True,
    ):
        if rendered is not None:
            assert edges is not None
            rendered.append((part, edges[edge_id], unit))
    return rendered, "".join(parts)


@overload
def normalize(
    text: str,
    *,
    locale: str = "en_US",
    profile: str | None = None,
    case_variant_lookup: bool = False,
    fold: InputFold | None = "typographic",
    offsets: Literal[False] = False,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
    max_unit_chars: int | None = DEFAULT_MAX_UNIT_CHARS,
    symbol_run_threshold: int = DEFAULT_SYMBOL_RUN_THRESHOLD,
    groups: GroupOrders | None = None,
    work_budget: TiergraphWorkBudget = DEFAULT_TIERGRAPH_WORK_BUDGET,
) -> str: ...


@overload
def normalize(
    text: str,
    *,
    locale: str = "en_US",
    profile: str | None = None,
    case_variant_lookup: bool = False,
    fold: InputFold | None = "typographic",
    offsets: Literal[True],
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
    max_unit_chars: int | None = DEFAULT_MAX_UNIT_CHARS,
    symbol_run_threshold: int = DEFAULT_SYMBOL_RUN_THRESHOLD,
    groups: GroupOrders | None = None,
    work_budget: TiergraphWorkBudget = DEFAULT_TIERGRAPH_WORK_BUDGET,
) -> NormalizedText: ...


@overload
def normalize(
    text: str,
    *,
    locale: str = "en_US",
    profile: str | None = None,
    case_variant_lookup: bool = False,
    fold: InputFold | None = "typographic",
    offsets: bool,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
    max_unit_chars: int | None = DEFAULT_MAX_UNIT_CHARS,
    symbol_run_threshold: int = DEFAULT_SYMBOL_RUN_THRESHOLD,
    groups: GroupOrders | None = None,
    work_budget: TiergraphWorkBudget = DEFAULT_TIERGRAPH_WORK_BUDGET,
) -> str | NormalizedText: ...


def normalize(
    text: str,
    *,
    locale: str = "en_US",
    profile: str | None = None,
    case_variant_lookup: bool = False,
    fold: InputFold | None = "typographic",
    offsets: bool = False,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
    max_unit_chars: int | None = DEFAULT_MAX_UNIT_CHARS,
    symbol_run_threshold: int = DEFAULT_SYMBOL_RUN_THRESHOLD,
    groups: GroupOrders | None = None,
    work_budget: TiergraphWorkBudget = DEFAULT_TIERGRAPH_WORK_BUDGET,
) -> str | NormalizedText:
    """Return the first-choice spoken form of a plain-text document.

    Sentence resolution stays bounded by ``max_unit_chars``. With ``offsets=True``,
    each output unit also records its code-point span and originating source span, and
    :class:`NormalizedText` records the applied ``fold`` and any optional detector
    families that ICU could not build for the locale. The plain-string form carries no
    metadata; callers that need fold or detector-capability provenance must request
    offsets.
    """
    if not isinstance(offsets, bool):
        raise TypeError(f"offsets must be a bool, got {type(offsets).__name__}")
    if not isinstance(case_variant_lookup, bool):
        raise TypeError(
            f"case_variant_lookup must be a bool, got {type(case_variant_lookup).__name__}"
        )
    validate_input(text, max_input_chars=max_input_chars)
    validate_unit_length(0, max_unit_chars=max_unit_chars)
    SymbolDetector("root", run_threshold=symbol_run_threshold)
    raw_text = text
    text = apply_input_fold(raw_text, fold)
    symbol_code_ranges = _symbol_code_ranges(text)
    locale = canonical_locale(locale)
    profile = validate_profile(profile)
    groups = None if groups is None else dict(validate_groups(groups) or ())
    meter = work_meter(work_budget)
    ranges = _sentence_ranges(text, locale)
    registry = _reading_detectors(locale, symbol_run_threshold) if ranges or offsets else None
    if not ranges:
        if not offsets:
            return ""
        units = (
            []
            if not text
            else [
                NormalizedUnit(
                    (0, 0),
                    (0, len(text)),
                    None,
                    "surface:boundary-whitespace",
                )
            ]
        )
        assert registry is not None
        return NormalizedText("", units, fold, registry.missing_families)

    # TODO(icukit): use icukit's reflow(text, mode) -> str once it is released.
    for start, end in ranges:
        length = end - start
        if max_unit_chars is not None and length > max_unit_chars:
            raise InputValidationError(
                f"sentence has {length} code points, exceeding max_unit_chars={max_unit_chars}; "
                "icukit's forced sentence break is not yet released, so frend refuses "
                "the unit instead of inventing a breaker"
            )

    parts: list[str] = []
    aligned: list[NormalizedUnit] | None = [] if offsets else None
    output_at = 0
    previous_end = 0
    for index, (start, end) in enumerate(ranges):
        if start > previous_end:
            separator = "" if index == 0 else " "
            parts.append(separator)
            if aligned is not None:
                aligned.append(
                    NormalizedUnit(
                        (output_at, output_at + len(separator)),
                        (previous_end, start),
                        None,
                        (
                            "surface:boundary-whitespace"
                            if index == 0
                            else "surface:inter-sentence-whitespace"
                        ),
                    )
                )
            output_at += len(separator)
        sentence = text[start:end]
        rendered, spoken = _sentence(
            sentence,
            source_offset=start,
            symbol_code_ranges=symbol_code_ranges,
            raw_text=raw_text[start:end],
            fold=fold,
            locale=locale,
            profile=profile,
            case_variant_lookup=case_variant_lookup,
            offsets=offsets,
            max_input_chars=max_input_chars,
            max_unit_chars=max_unit_chars,
            symbol_run_threshold=symbol_run_threshold,
            groups=groups,
            work_budget=meter,
        )
        parts.append(spoken)
        if aligned is not None:
            assert rendered is not None
            for unit_text, edge, unit in rendered:
                unit_end = output_at + len(unit_text)
                reader = None if edge.detection is None else str(edge.detection.get("type"))
                aligned.append(
                    NormalizedUnit(
                        (output_at, unit_end),
                        (start + edge.start, start + edge.end),
                        reader,
                        unit.best.provenance,
                    )
                )
                output_at = unit_end
        else:
            output_at += len(spoken)
        previous_end = end
    if previous_end < len(text) and aligned is not None:
        aligned.append(
            NormalizedUnit(
                (output_at, output_at),
                (previous_end, len(text)),
                None,
                "surface:boundary-whitespace",
            )
        )
    normalized = "".join(parts)
    if aligned is None:
        return normalized
    assert registry is not None
    return NormalizedText(normalized, aligned, fold, registry.missing_families)
