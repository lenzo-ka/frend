"""Plain text-in, spoken text-out normalization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, overload

from icukit import break_sentence_spans
from icukit.detectors import date_detectors, detect
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
from frend.letters import LettersDetector
from frend.locale_data import LOCALE_CACHE, canonical_locale
from frend.profiles import GroupOrders, validate_groups, validate_profile
from frend.ranges import RangeDetector, date_interval_readers, icu_range_readers
from frend.symbols import (
    DEFAULT_SYMBOL_RUN_THRESHOLD,
    SymbolDetector,
)
from frend.symbols import (
    _code_ranges as _symbol_code_ranges,
)
from frend.symbols import (
    _overlaps as _symbol_overlaps,
)
from frend.verbalize import VerbalizedUnit, _verbalize_lattice_validated
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


@dataclass(frozen=True)
class NormalizedUnit:
    """One output unit and the original source span that produced it."""

    output_span: tuple[int, int]
    source_span: tuple[int, int]
    reader: str | None
    provenance: str


@dataclass(frozen=True)
class NormalizedText:
    """Normalized text with source alignment for each spoken or passthrough unit."""

    text: str
    units: list[NormalizedUnit]
    fold: InputFold | None = None


@lru_cache(maxsize=LOCALE_CACHE)
def _reading_detectors(
    locale: str, symbol_run_threshold: int = DEFAULT_SYMBOL_RUN_THRESHOLD
) -> tuple[object, ...]:
    """The evaluator's recognition profile, packaged for the public API."""
    from icukit.abbreviation_recognize import AbbreviationDetector

    runs = AlphanumericRunsDetector(locale)
    written = WrittenFormsDetector(locale)
    dates = date_detectors(locale, _DATE_SKELETONS).with_(
        FlexibleDateDetector(locale),
        FlexibleTextDateDetector(locale),
        PluralNumeralDetector(locale),
        runs,
    )
    numbers = (
        FlexibleNumberDetector(locale),
        FlexibleCompactDetector(locale, "long"),
        FlexibleCompactDetector(locale, "short"),
    )
    cardinals = numbers + (LetterNameDetector(locale), SingleLetterWordDetector(locale))
    measures = (
        FlexiblePercentDetector(locale),
        *(FlexibleMeasureDetector(locale, unit) for unit in _MEASURE_UNITS),
        *(FlexibleMixedMeasureDetector(locale, mixed) for mixed in _MIXED_MEASURES),
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
        (FlexibleNumberDetector(locale), written),
        (SymbolDetector(locale, run_threshold=symbol_run_threshold),),
        numbers,
        (*dates.detectors, *date_interval_readers(locale), written),
        (FlexibleFractionDetector(locale),),
        (FlexibleOrdinalDetector(locale), FlexibleNumberDetector(locale), runs),
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
    return tuple(detectors)


def _sentence_ranges(text: str, locale: str) -> list[tuple[int, int]]:
    """Return nonempty sentence ranges with boundary whitespace excluded."""
    spans = break_sentence_spans(text, locale)
    nonempty = [span for span in spans if span["text"].strip()]
    ranges: list[tuple[int, int]] = []
    for span in nonempty:
        surface = span["text"]
        left = len(surface) - len(surface.lstrip())
        right = len(surface.rstrip())
        start = int(span["start"]) + left
        end = int(span["start"]) + right
        ranges.append((start, end))
    return ranges


def _unit_text(unit: VerbalizedUnit) -> str:
    if unit.best.provenance == "surface:passthrough":
        return unit.best.text
    return f" {unit.best.text} "


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
) -> tuple[list[tuple[str, ReadingEdge, VerbalizedUnit]] | None, str]:
    detections = detect(text, _reading_detectors(locale, symbol_run_threshold))
    detections = [
        detection
        for detection in detections
        if detection.get("type") != "symbol:run"
        or not _symbol_overlaps(
            symbol_code_ranges,
            source_offset + int(detection["start"]),
            source_offset + int(detection["end"]),
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
    parts = []
    for edge_id, unit in zip(
        verbalized.best_path.edge_ids,
        verbalized.best_path.units,
        strict=True,
    ):
        part = _unit_text(unit)
        parts.append(part)
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
) -> str | NormalizedText:
    """Return the first-choice spoken form of a plain-text document.

    Sentence resolution stays bounded by ``max_unit_chars``. With ``offsets=True``,
    each output unit also records its code-point span and originating source span, and
    :class:`NormalizedText` records the applied ``fold``. The plain-string form carries
    no metadata; callers that need fold provenance must request offsets.
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
    ranges = _sentence_ranges(text, locale)
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
        return NormalizedText("", units, fold)

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
    return NormalizedText(normalized, aligned, fold) if aligned is not None else normalized
