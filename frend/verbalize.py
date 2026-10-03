"""Reflective spoken alternatives for a distilled reading lattice."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from functools import cache, lru_cache
from itertools import product
from pathlib import Path
from types import MappingProxyType

import icu
from icukit import AbbreviationValue, DateTimeFormatter
from icukit.detectors import DateTimeValue, MeasureValue, NumberValue
from icukit.measure import WIDTH_WIDE, format_measure

from frend.abbreviation_variants import (
    AS_WRITTEN_SOURCE,
    SPELLED_SOURCE,
    VARIANT_TYPE,
    abbreviation_weights,
    chain_expansions,
    is_chain,
    measured_case,
    measures_spelled,
    upper_variant_expansions,
)
from frend.britishisms import load_britishisms, respell_from_table
from frend.context import (
    CONTEXT_THRESHOLD,
    ContextChoice,
    TextContext,
    between_numbers,
    cldr_range_separator,
    connector_probability,
    connector_words,
    rerank,
    rerank_range,
)
from frend.electronic import (
    ElectronicValue,
    digit_forms,
    digit_probabilities,
    letter_key,
    letter_probabilities,
    separator_names,
    tld_positions,
)
from frend.input_limits import (
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_UNIT_CHARS,
    validate_input,
    validate_unit_length,
)
from frend.lattice import ReadingEdge, ReadingLattice
from frend.letters import (
    LettersValue,
    cv_pattern,
    is_letter_run,
    letter_names,
    spelled,
    spelled_token_prior,
    spelled_token_rule,
    split_acronym_surface,
)
from frend.locale_data import LOCALE_CACHE, canonical_locale, lexical_forms
from frend.number_priors import load_number_priors
from frend.profiles import GOOGLE_TN, google_tn_profile_path, validate_profile
from frend.ranges import (
    RangeValue,
    digit_groups,
    from_icukit,
    load_range_priors,
    range_class_key,
    range_separator_classes,
    range_sub_key,
    written_sub_key,
)
from frend.spoken_priors import measurement_sub_key, normalize_spoken, source_prior
from frend.symbols import ScriptRunValue, SymbolValue
from frend.written_forms import DigitsValue

__all__ = [
    "RangeConnector",
    "RangeSlot",
    "SpokenAlternative",
    "VerbalizedLattice",
    "VerbalizedPath",
    "VerbalizedUnit",
    "register_curated_alternative",
    "range_connector",
    "range_separators",
    "verbalize_edge",
    "verbalize_lattice",
]

LEXICAL_SOURCE = "lexical:en_US"


def lexical_source(locale: str) -> str:
    return _lexical_source(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _lexical_source(locale: str) -> str:
    return f"lexical:{locale}"


# Every form ICU and CLDR do not give, and the corpus says, is a hand-written lexical
# form: it lives with its reason in ``data/<locale>/lexical.json`` and is read here by
# key. A locale with no table has no lexical forms, and each feature that needs one is
# off for it. The provenance string above stays as it is: the measured tables key on it.


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@lru_cache(maxsize=LOCALE_CACHE)
def _lexical_for(locale: str) -> Mapping[str, object]:
    return MappingProxyType(
        {key: _freeze(entry["value"]) for key, entry in lexical_forms(locale).items()}
    )


def _lexical(key: str, locale: str):
    """The locale's lexical form ``key`` (``data/<locale>/lexical.json``), or ``None``."""
    return _lexical_for(canonical_locale(locale)).get(key)


def _lexical_pattern(key: str, locale: str, *texts: str) -> str | None:
    pattern = _lexical(key, locale)
    return None if pattern is None else pattern.format(*texts)


@dataclass(frozen=True)
class SpokenAlternative:
    """One spoken form, where it came from, and an optional genuine weight."""

    text: str
    provenance: str
    weight: Decimal | None = None


@dataclass(frozen=True)
class VerbalizedUnit:
    """One edge's spoken alternatives and trace back to its reading prior.

    ``unspoken`` holds every capture of the reading that no alternative speaks, so
    written material the verbalizer does not say is known here rather than lost. A
    surface fallback speaks the whole span and so leaves nothing unspoken.
    ``context`` is what the context tree said about the alternatives' order
    (``frend.context``), or ``None`` where no tree applies.
    """

    edge_id: str
    alternatives: tuple[SpokenAlternative, ...]
    tier: str | None
    provenance: str | None
    verbalized: bool
    unspoken: tuple[object, ...] = ()
    context: ContextChoice | None = None

    @property
    def best(self) -> SpokenAlternative:
        return self.alternatives[0]


@dataclass(frozen=True)
class VerbalizedPath:
    """One ranked path; alternatives stay factored by constituent edge."""

    rank: int
    edge_ids: tuple[str, ...]
    units: tuple[VerbalizedUnit, ...]
    spoken: str

    @property
    def alternatives(self) -> tuple[tuple[SpokenAlternative, ...], ...]:
        return tuple(unit.alternatives for unit in self.units)


@dataclass(frozen=True)
class VerbalizedLattice:
    paths: tuple[VerbalizedPath, ...]
    best_path: VerbalizedPath
    ambiguous: bool
    truncated: bool


CuratedKey = tuple[str, str]
CuratedSupplements = Mapping[CuratedKey, Sequence[SpokenAlternative]]
_CURATED: dict[CuratedKey, list[SpokenAlternative]] = {}


def register_curated_alternative(
    reading_class: str,
    value_or_shape: object,
    text: str,
    source: str,
    *,
    weight: Decimal | None = None,
) -> None:
    """Register an additive curated form; callers own the source and any weight."""
    alternative = SpokenAlternative(text, f"curated:{source}", weight)
    _CURATED.setdefault((reading_class, str(value_or_shape)), []).append(alternative)


def _ranked(alternatives: Sequence[SpokenAlternative]) -> tuple[SpokenAlternative, ...]:
    indexed = list(enumerate(alternatives))
    indexed.sort(
        key=lambda item: (
            item[1].weight is None,
            -(item[1].weight or Decimal(0)) if item[1].weight is not None else Decimal(0),
            item[0],
        )
    )
    seen: set[str] = set()
    return tuple(
        alternative
        for _, alternative in indexed
        if not (alternative.text in seen or seen.add(alternative.text))
    )


def _measured_kind(type_: str, value: object) -> str | None:
    """Map date, time, ordinal, fraction, money, decimal, and cardinal reading families.

    ``date:*`` and ``number:plural*`` (decades, filed as DATE by the corpus) map to
    date; ``time:*`` to time; ``measure:*`` and ``number:percent`` (the corpus files
    percent under MEASURE) to measure; ``ordinal:*`` to ordinal; ``fraction:*`` and
    ``number:fraction*`` to fraction;
    ``money:*``, ``number:currency*``, or a number carrying currency to money;
    ``number:decimal*`` to decimal; and ``number:cardinal*`` or ``number:int*``
    to cardinal. Other families have no spoken-source measurement.
    """
    if type_.startswith(("date:", "number:plural")):
        return "date"
    if type_.startswith("time:"):
        return "time"
    if type_.startswith("measure:duration"):
        return "time"
    if type_.startswith("symbol:"):
        return "symbol"
    if type_.startswith("measure:") or type_ == "number:percent":
        return "measure"
    if type_.startswith("ordinal:"):
        return "ordinal"
    if type_.startswith(("fraction:", "number:fraction")):
        return "fraction"
    if type_.startswith(("money:", "number:currency")) or (
        isinstance(value, NumberValue) and value.currency is not None
    ):
        return "money"
    if type_.startswith("number:decimal"):
        return "decimal"
    if type_.startswith(("number:cardinal", "number:int")):
        return "cardinal"
    return None


def _source_tier(provenance: str) -> int:
    """Return the declared unmeasured-source tier.

    A composed provenance is lexical/curated when any component has either
    prefix. Hand-written forms outrank generated ones when neither is attested;
    this is a tier, not a measurement.
    """
    components = provenance.split("+")
    return 2 if any(part.startswith(("lexical:", "curated:")) for part in components) else 3


def _rank_final(
    alternatives: Sequence[SpokenAlternative],
    kind: str | None,
    sub_key: str | None,
    locale: str,
) -> tuple[SpokenAlternative, ...]:
    """Apply carried weights, one measurement scope, then source tiers stably.

    If the table contains the reading's sub-key, every alternative's share blends
    that row toward the kind row by the sub-key's evidence (``source_prior``), so all
    alternatives on an edge are scored by the same rule; otherwise every alternative
    uses the kind row. A source measured at neither level falls back to its tier.
    """
    ranked: list[tuple[int, Decimal, int, SpokenAlternative]] = []
    for index, alternative in enumerate(alternatives):
        if alternative.weight is not None:
            ranked.append((0, -alternative.weight, index, alternative))
            continue
        measurement = (
            source_prior(kind, alternative.provenance, sub_key, locale=locale)
            if kind is not None
            else None
        )
        if measurement is not None:
            weighted = SpokenAlternative(
                alternative.text, alternative.provenance, measurement.share
            )
            ranked.append((1, -measurement.share, index, weighted))
            continue
        ranked.append((_source_tier(alternative.provenance), Decimal(0), index, alternative))
    ranked = _zero_shares(ranked, kind, locale)
    ranked.sort(key=lambda item: item[:3])
    return tuple(item[3] for item in ranked)


def _zero_priors(*, locale: str = "en_US") -> dict[str, dict[str, int]]:
    """The locale's measured zero words by kind (``data/<locale>/zero_priors.json``);
    empty when the locale has no table. Cached on the canonical locale."""
    from frend.locale_data import canonical_locale

    return _zero_priors_for(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _zero_priors_for(locale: str) -> dict[str, dict[str, int]]:
    from frend.locale_data import measured_table

    table = measured_table("zero_priors", locale)
    return {} if table is None else table["kinds"]


def _zero_words(locale: str = "en_US") -> frozenset[str]:
    """The words a zero digit is said as (``lexical.json`` ``zero.words``); none without."""
    return frozenset(_lexical("zero.words", locale) or ())


def _zero_key(text: str, locale: str = "en_US") -> tuple[str, ...]:
    zeros = _zero_words(locale)
    return tuple("0" if word in zeros else word for word in text.replace("-", " ").split())


def _zero_shares(
    ranked: list[tuple[int, Decimal, int, SpokenAlternative]], kind: str | None, locale: str
) -> list[tuple[int, Decimal, int, SpokenAlternative]]:
    """Readings that differ only in how a zero is said ("point zero five", "point o
    five"; ICU's "oh-five", "o five") share their weight by how the corpus says a zero
    in this kind of reading (``data/en/zero_priors.json``, ``tools/build_zero_priors.py``),
    each zero counted once, add-one: a date's zero is "o", a decimal's mostly "o"."""
    counts = _zero_priors(locale=locale).get(kind or "")
    zeros = _zero_words(locale)
    if not counts or not zeros:
        return ranked
    total = sum(counts.get(word, 0) + 1 for word in zeros)
    groups: dict[tuple[str, ...], list[int]] = {}
    for position, (_, _, _, alternative) in enumerate(ranked):
        if zeros & set(alternative.text.replace("-", " ").split()):
            groups.setdefault(_zero_key(alternative.text, locale), []).append(position)
    out = list(ranked)
    for members in groups.values():
        weights = [ranked[at][3].weight for at in members]
        if len(members) < 2 or not any(weights):
            continue
        pooled = sum((weight or Decimal(0) for weight in weights), Decimal(0))
        scores = []
        for at in members:
            score = Decimal(1)
            for word in ranked[at][3].text.replace("-", " ").split():
                if word in zeros:
                    score *= Decimal(counts.get(word, 0) + 1) / total
            scores.append(score)
        tier = min(ranked[at][0] for at in members)
        for at, score in zip(members, scores, strict=True):
            _, _, index, alternative = ranked[at]
            weight = pooled * score / sum(scores)
            shared = SpokenAlternative(alternative.text, alternative.provenance, weight)
            out[at] = (tier, -weight, index, shared)
    return out


def _with_curated(
    alternatives: Sequence[SpokenAlternative],
    reading_class: str,
    value_or_shape: object,
    supplements: CuratedSupplements | None,
) -> tuple[SpokenAlternative, ...]:
    key = (reading_class, str(value_or_shape))
    extra = [*_CURATED.get(key, ()), *((supplements or {}).get(key, ()))]
    for alternative in extra:
        if not alternative.provenance.startswith("curated:"):
            raise ValueError("curated supplement provenance must start with 'curated:'")
    return _ranked([*alternatives, *extra])


def _surface(edge: ReadingEdge, source_text: str | None) -> str:
    if source_text is not None:
        return source_text[edge.start : edge.end]
    if edge.detection is not None and "text" in edge.detection:
        return str(edge.detection["text"])
    raise ValueError(f"edge {edge.id!r} needs surface text, but the lattice has no source_text")


@cache
def _spellout_formatter(locale: str) -> icu.RuleBasedNumberFormat:
    return icu.RuleBasedNumberFormat(icu.URBNFRuleSetTag.SPELLOUT, icu.Locale(locale))


@cache
def _spellout_rule_sets(locale: str) -> tuple[str, ...]:
    formatter = _spellout_formatter(locale)
    return tuple(
        formatter.getRuleSetName(index) for index in range(formatter.getNumberOfRuleSetNames())
    )


def _applicable_rule_sets(kind: str, locale: str) -> tuple[str, ...]:
    names = _spellout_rule_sets(locale)
    if kind == "ordinal":
        return tuple(name for name in names if "ordinal" in name)
    if kind == "year":
        return tuple(
            name
            for name in names
            if "ordinal" not in name
            and ("numbering-year" in name or "cardinal" in name or "numbering" in name)
        )
    return tuple(
        name
        for name in names
        if "ordinal" not in name
        and "numbering-year" not in name
        and ("cardinal" in name or "numbering" in name)
    )


def _format_exact(formatter: icu.RuleBasedNumberFormat, value: Decimal, ruleset: str) -> str:
    """Cross into PyICU only through Decimal or an exact arbitrary-size integer."""
    try:
        return formatter.format(value, ruleset)
    except Exception as exc:
        if value == value.to_integral_value():
            return formatter.format(int(value), ruleset)
        raise NotImplementedError(
            "the installed PyICU binding cannot accept a non-integral Decimal exactly"
        ) from exc


def _number_leaf(value: Decimal, kind: str, locale: str) -> tuple[SpokenAlternative, ...]:
    formatter = _spellout_formatter(locale)
    return _ranked(
        [
            SpokenAlternative(_format_exact(formatter, value, ruleset), f"icu-rbnf:{ruleset}")
            for ruleset in _applicable_rule_sets(kind, locale)
        ]
    )


def _decimal_separator_word(formatter: icu.RuleBasedNumberFormat, locale: str) -> SpokenAlternative:
    rules = formatter.getRules()
    for rule_set in _applicable_rule_sets("cardinal", locale):
        start = rules.find(f"{rule_set}:")
        if start < 0:
            continue
        end = rules.find("\n%", start + 1)
        section = rules[start : len(rules) if end < 0 else end]
        match = re.search(r"^x\.x:.*?<%[^<]+<\s+([A-Za-z]+)\s+>%[^>]+>;$", section, re.MULTILINE)
        if match:
            # DecimalFormatSymbols provides only punctuation; the RBNF rule is
            # the locale data that states how that punctuation is spoken.
            return SpokenAlternative(match.group(1), f"icu-rbnf:{rule_set}")
    raise NotImplementedError("the locale's RBNF rules do not name a decimal separator")


def _spoken_decimal(
    value: Decimal, locale: str, *, omit_zero_integer: bool = False
) -> tuple[SpokenAlternative, ...]:
    sign, digits, exponent = value.as_tuple()
    if exponent >= 0:
        return _number_leaf(value, "cardinal", locale)
    digit_text = "".join(str(digit) for digit in digits).rjust(-exponent + 1, "0")
    integer_digits = digit_text[:exponent]
    fractional_digits = digit_text[exponent:]
    integer = Decimal(integer_digits or "0")
    integer_alternatives = _signed_integer_leaf(integer, bool(sign), locale)
    formatter = _spellout_formatter(locale)
    parts = [] if omit_zero_integer and not sign and not integer else [integer_alternatives]
    parts.append((_decimal_separator_word(formatter, locale),))
    zero = _lexical("zero.digit", locale)
    for digit in fractional_digits:
        words = list(_number_leaf(Decimal(digit), "cardinal", locale))
        if digit == "0" and zero is not None:
            # ICU has no digit-reading rule that calls zero "o"; the corpus uses
            # that spelling for zero digits in decimals.
            words.append(SpokenAlternative(zero, lexical_source(locale)))
        parts.append(_ranked(words))
    alternatives = list(_compose(parts, " ".join("{}" for _ in parts)))
    if set(fractional_digits) <= {"0"} and value == value.to_integral_value():
        alternatives.extend(_number_leaf(value, "cardinal", locale))
    return _ranked(alternatives)


def _signed_integer_leaf(
    absolute: Decimal, negative: bool, locale: str
) -> tuple[SpokenAlternative, ...]:
    alternatives = _number_leaf(-absolute if negative else absolute, "cardinal", locale)
    if not negative or absolute:
        return alternatives
    positive = _number_leaf(Decimal(1), "cardinal", locale)[0].text
    negative_one = _number_leaf(Decimal(-1), "cardinal", locale)[0]
    prefix = negative_one.text.removesuffix(positive).strip()
    return tuple(
        SpokenAlternative(f"{prefix} {item.text}", item.provenance) for item in alternatives
    )


@cache
def _measure_formatter(locale: str) -> icu.MeasureFormat:
    return icu.MeasureFormat(icu.Locale(locale), icu.UMeasureFormatWidth.WIDE)


@cache
def _percent_name(locale: str) -> str:
    return _measure_formatter(locale).getUnitDisplayName(icu.MeasureUnit.createPercent())


@cache
def _currency_name(currency: str, plural: bool, locale: str) -> str:
    rendered = _measure_formatter(locale).formatMeasure(
        icu.Measure(2 if plural else 1, icu.CurrencyUnit(currency))
    )
    return rendered.split(" ", 1)[1]


@cache
def _plural_rules(locale: str) -> icu.PluralRules:
    return icu.PluralRules.forLocale(icu.Locale(locale))


def _capture_integer(detection: object, name: str) -> Decimal | None:
    for capture in detection.get("captures", ()):  # type: ignore[union-attr]
        if getattr(capture, "name", None) == name:
            try:
                return Decimal(str(capture.value))
            except (InvalidOperation, ValueError):
                return None
    return None


def _capture(detection: object, name: str) -> object | None:
    return next(
        (
            capture
            for capture in detection.get("captures", ())  # type: ignore[union-attr]
            if getattr(capture, "name", None) == name
        ),
        None,
    )


def _compact_scale(magnitude: int, locale: str) -> SpokenAlternative:
    formatter = icu.CompactDecimalFormat.createInstance(
        icu.Locale(locale), icu.UNumberCompactStyle.LONG
    )
    rendered = formatter.format(10**magnitude)
    words = re.findall(r"[^\W\d_]+", rendered, re.UNICODE)
    if not words:
        raise NotImplementedError("the locale's long compact pattern has no scale word")
    return SpokenAlternative(" ".join(words), "icu-compact:long")


def _spoken_compact(detection: object, locale: str) -> tuple[SpokenAlternative, ...]:
    integer = _capture(detection, "integer")
    compact = _capture(detection, "compact")
    if integer is None or compact is None:
        raise NotImplementedError("compact recognition did not retain mantissa and scale")
    fraction = _capture(detection, "fraction")
    mantissa_text = str(integer.value)
    if fraction is not None:
        mantissa_text = f"{mantissa_text}.{fraction.value}"
    sign = _capture(detection, "sign")
    if sign is not None and str(sign.text).strip() == "-":
        mantissa_text = f"-{mantissa_text}"
    mantissa = Decimal(mantissa_text)
    mantissa_words = (
        _spoken_decimal(mantissa, locale)
        if fraction is not None
        else _number_leaf(mantissa, "cardinal", locale)
    )
    scale = _compact_scale(int(compact.value), locale)
    return _compose([mantissa_words, (scale,)], "{} {}")


def _compose(
    parts: Sequence[Sequence[SpokenAlternative]], template: str
) -> tuple[SpokenAlternative, ...]:
    composed = []
    for combination in product(*parts):
        composed.append(
            SpokenAlternative(
                template.format(*(item.text for item in combination)),
                "+".join(item.provenance for item in combination),
                sum((item.weight for item in combination), Decimal(0))
                if all(item.weight is not None for item in combination)
                else None,
            )
        )
    return _ranked(composed)


def _spoken_fraction(detection: object, locale: str) -> tuple[SpokenAlternative, ...]:
    whole = _capture_integer(detection, "whole")
    numerator = _capture_integer(detection, "numerator")
    denominator = _capture_integer(detection, "denominator")
    if numerator is None or denominator is None:
        raise NotImplementedError("fraction recognition did not retain numerator/denominator")
    sign = _capture(detection, "sign")
    negative = sign is not None and str(sign.text).strip() == "-"
    numerator_words = tuple(
        SpokenAlternative(item.text.replace("-", " "), item.provenance)
        for item in _signed_integer_leaf(numerator, negative, locale)
    )
    denominator_words = []
    singular = _plural_rules(locale).select(int(numerator)) == "one"
    for ordinal in _number_leaf(denominator, "ordinal", locale):
        denominator_words.append(
            SpokenAlternative(
                (ordinal.text if singular else f"{ordinal.text}s").replace("-", " "),
                ordinal.provenance,
            )
        )
    irregular = (_lexical("fraction.denominators", locale) or {}).get(str(int(denominator)))
    if irregular is not None:
        denominator_words.insert(
            0, SpokenAlternative(irregular[0 if singular else 1], lexical_source(locale))
        )
    fraction = _compose([numerator_words, _ranked(denominator_words)], "{} {}")
    over = _compose([numerator_words, _number_leaf(denominator, "cardinal", locale)], "{} over {}")
    if whole is not None:
        whole_words = _number_leaf(whole, "cardinal", locale)
        mixed_fraction = list(fraction)
        single = (
            None if irregular is None else _lexical_pattern("fraction.one", locale, irregular[0])
        )
        if numerator == 1 and single is not None:
            mixed_fraction.insert(0, SpokenAlternative(single, lexical_source(locale)))
        mixed = _compose([whole_words, _ranked(mixed_fraction)], "{} and {}")
        mixed_over = _compose([whole_words, over], "{} and {}")
        return _ranked([*mixed, *mixed_over])
    return _ranked([*fraction, *over])


def _currency_units(currency: str, locale: str):
    """The corpus's bare major and minor names for ``currency`` (``currency.units``)."""
    return (_lexical("currency.units", locale) or {}).get(currency)


def _currency_unit_name(
    currency: str, amount: Decimal, minor: bool, locale: str
) -> SpokenAlternative:
    units = _currency_units(currency, locale)
    if units is None:
        raise NotImplementedError(f"no bare currency terms for {currency} in {locale}")
    names = units[minor]
    plural = (
        amount != amount.to_integral_value() or _plural_rules(locale).select(int(amount)) != "one"
    )
    return SpokenAlternative(names[plural], lexical_source(locale))


def _currency_wide_names(
    currency: str, amount: Decimal, locale: str
) -> tuple[SpokenAlternative, ...]:
    plural = amount != 1
    names = [SpokenAlternative(_currency_name(currency, plural, locale), "icu-measure:wide")]
    region = (_lexical("currency.region_names", locale) or {}).get(currency)
    if region is not None:
        names.append(SpokenAlternative(region[plural], lexical_source(locale)))
    return _ranked(names)


@cache
def _currency_fraction_digits(currency: str, locale: str) -> int:
    formatter = icu.NumberFormat.createCurrencyInstance(icu.Locale(locale))
    formatter.setCurrency(currency)
    return formatter.getMaximumFractionDigits()


def _spoken_money(value: Decimal, currency: str, locale: str) -> tuple[SpokenAlternative, ...]:
    negative = value.is_signed()
    absolute = abs(value)
    major = int(absolute)
    fraction_digits = _currency_fraction_digits(currency, locale)
    scale = Decimal(10) ** fraction_digits
    scaled = absolute * scale
    lexical_units = _currency_units(currency, locale)
    wide_names = _currency_wide_names(currency, absolute, locale)
    if scaled != scaled.to_integral_value() or (absolute != major and lexical_units is None):
        names = list(wide_names)
        if lexical_units is not None:
            names.insert(0, _currency_unit_name(currency, absolute, False, locale))
        return _compose([_spoken_decimal(value, locale), names], "{} {}")
    minor = int(scaled) % int(scale) if fraction_digits else 0
    major_words = tuple(
        SpokenAlternative(item.text.replace("-", " "), item.provenance)
        for item in _signed_integer_leaf(Decimal(major), negative, locale)
    )
    alternatives = list(_compose([major_words, wide_names], "{} {}"))
    if lexical_units is None:
        return _ranked(alternatives)
    bare_name = _currency_unit_name(currency, Decimal(major), False, locale)
    if minor:
        minor_words = tuple(
            SpokenAlternative(item.text.replace("-", " "), item.provenance)
            for item in _number_leaf(Decimal(minor), "cardinal", locale)
        )
        minor_name = _currency_unit_name(currency, Decimal(minor), True, locale)
        return _ranked(
            [
                *_compose(
                    [major_words, (bare_name,), minor_words, (minor_name,)], "{} {} and {} {}"
                ),
                *_compose([major_words, wide_names, minor_words, (minor_name,)], "{} {} and {} {}"),
            ]
        )
    alternatives[0:0] = _compose([major_words, (bare_name,)], "{} {}")
    return _ranked(alternatives)


def _month_name(month: int, calendar: str, locale: str) -> SpokenAlternative:
    if not 1 <= month <= 12:
        raise ValueError(f"invalid month {month!r}")
    text = DateTimeFormatter(locale, calendar=calendar).format(date(2000, month, 1), pattern="LLLL")
    return SpokenAlternative(text, "icu-datetime:LLLL")


# Captures each verbalizer speaks, from the value struct or by reading the capture.
# Any other capture on a verbalized reading is recorded as unspoken, never dropped.
# A minus sign is spoken through the signed value, a plus sign as a leading "plus";
# any other sign character is recorded as unspoken.
_SPOKEN_CAPTURES = {
    "number": frozenset(
        {
            "integer",
            "fraction",
            "decimal-separator",
            "sign",
            "percent",
            "currency",
            "compact",
            "whole",
            "reporter",
            "numerator",
            "denominator",
        }
    ),
    "date": frozenset(
        {
            "y",
            "M",
            "d",
            "month",
            "weekday",
            "era",
            "quarter",
            "H",
            "m",
            "day-period",
            "time-zone",
            "datetime-glue",
        }
    ),
    "ordinal": frozenset({"integer", "ordinal-affix"}),
    "abbreviation": frozenset({"surface"}),
    "time": frozenset({"H", "m", "day-period", "time-zone"}),
    "plural": frozenset({"number", "suffix", "apostrophe", "elision"}),
    "runs": frozenset({"digits", "letters", "separator"}),
    "relative": frozenset({"relative", "integer", "relative-marker"}),
    "digits": frozenset({"digits"}),
    "symbol": frozenset({"symbol"}),
    "letters": frozenset({"letters", "suffix", "period"}),
    "electronic": frozenset({"digits", "letters", "separator"}),
    "measure": frozenset({"integer", "decimal-separator", "fraction", "unit"}),
    "mixed-measure": frozenset({"integer", "unit"}),
    "duration": frozenset({"h", "m", "s", "decimal-separator", "fraction"}),
    "unit": frozenset({"unit"}),
    "roman": frozenset({"integer", "apostrophe", "suffix"}),
    "range": frozenset({"start", "separator", "end"}),
}
_MINUS_SIGNS = frozenset({"-", "−"})
_PLUS_SIGNS = frozenset({"+"})


@cache
def _zone_display_names(zone: str, locale: str) -> frozenset[str]:
    """ICU's long names for an IANA zone: standard, daylight and generic."""
    timezone = icu.TimeZone.createTimeZone(zone)
    icu_locale = icu.Locale(locale)
    return frozenset(
        timezone.getDisplayName(daylight, style, icu_locale)
        for daylight in (False, True)
        for style in (icu.TimeZone.LONG, icu.TimeZone.LONG_GENERIC)
    )


@cache
def _zone_expansions(locale: str) -> dict[str, tuple[SpokenAlternative, ...]]:
    """ICU's long names for each zone abbreviation, as icukit lists them."""
    from icukit import icu_abbreviations

    names: dict[str, list[SpokenAlternative]] = {}
    for row in icu_abbreviations(locale, kinds=["time-zone"]):
        for expansion in row.expansions:
            alternative = SpokenAlternative(expansion, "icukit-abbreviation:time-zone")
            if alternative not in names.setdefault(row.surface, []):
                names[row.surface].append(alternative)
    return {surface: tuple(items) for surface, items in names.items()}


def _spoken_time(
    value: DateTimeValue, detection: object, locale: str
) -> tuple[SpokenAlternative, ...]:
    """Speak an hour, optional minutes and a written day period ("5pm" -> "five p m").

    With a day period the written 12-hour hour is spoken, not the 24-hour value
    ("12am" is "twelve a m"). A zero-led minute reads "oh five" or "o five"; a bare
    or on-the-hour time also reads with "o'clock". A written time zone follows as
    its letters, as the corpus reads it ("10 PM ET" "ten p m e t", "18:00 UTC"
    "eighteen hundred u t c"), and also by each long name icukit lists for it from ICU
    ("EST" also "Eastern Standard Time"; "IST" both India and Irish). A day period or
    zone written in words ("in the
    afternoon", "Eastern Standard Time", icukit #116) is said as those words, never
    spelled. Seconds are not verbalized.
    """
    fields = dict(value.fields)
    if "H" not in fields or not set(fields) <= {"H", "m"}:
        raise NotImplementedError("time verbalization covers hours and minutes only")
    period = _capture(detection, "day-period")
    written_hour = _capture_integer(detection, "H")
    hour = written_hour if period is not None and written_hour is not None else Decimal(fields["H"])
    hours = _number_leaf(hour, "cardinal", locale)
    tail = []
    for capture in (period, _capture(detection, "time-zone")):
        words = str(getattr(capture, "text", "")).split()
        if len(words) > 1:
            tail.append((SpokenAlternative(" ".join(words), "surface:words"),))
            continue
        letters = "".join(ch for ch in "".join(words) if ch.isalpha()).lower()
        if letters:
            spelled = SpokenAlternative(" ".join(letters), "surface:letters")
            expansions = (
                _zone_expansions(locale).get("".join(words), ()) if capture is not period else ()
            )
            zone = str(getattr(capture, "value", "") or "")
            if expansions and "/" in zone:
                # icukit 0.7 captures the zone's IANA ID ("IST" read as Asia/Kolkata and
                # as Europe/Dublin): keep the names ICU gives that zone. Where none of the
                # listed names matches ICU's display names literally (CLDR's metazone
                # "Saint-Pierre-et-Miquelon Daylight Time" against the zone's "St. Pierre
                # & Miquelon Daylight Time"), every listed name is kept rather than none.
                named = _zone_display_names(zone, locale)
                expansions = tuple(item for item in expansions if item.text in named) or expansions
            tail.append((spelled, *expansions))
    minute = fields.get("m")
    forms: list[SpokenAlternative] = []
    zeros = _lexical("zero.minute", locale)
    if minute:
        minutes = _number_leaf(Decimal(minute), "cardinal", locale)
        if minute < 10 and zeros:
            # A zero-led minute is said with its zero ("oh five", "o five"), which no
            # ICU time pattern says; without the locale's words it is ICU's cardinal.
            minutes = tuple(
                SpokenAlternative(
                    f"{zero} {item.text}", f"{lexical_source(locale)}+{item.provenance}"
                )
                for zero in zeros
                for item in minutes
            )
        parts = [hours, minutes, *tail]
        forms.extend(_compose(parts, " ".join("{}" for _ in parts)))
    else:
        oclock = _lexical("clock.oclock", locale)
        spoken_hours = [[hours, *tail]]
        if oclock is not None:
            spoken_hours.append(
                [hours, (SpokenAlternative(oclock, lexical_source(locale)),), *tail]
            )
        for parts in spoken_hours:
            forms.extend(_compose(parts, " ".join("{}" for _ in parts)))
        hundred = _lexical("clock.hundred", locale)
        if minute == 0 and period is None and hundred is not None:
            # A written on-the-hour 24-hour time: the corpus reads "20:00" "twenty hundred".
            parts = [hours, (SpokenAlternative(hundred, lexical_source(locale)),), *tail]
            forms.extend(_compose(parts, " ".join("{}" for _ in parts)))
    return _ranked(forms)


def _plural_word(word: str, rules: Sequence[Mapping]) -> str:
    """The plural of one number word by the locale's ``numeral.plural`` rules: the first
    rule whose ending fits (and whose ending does not follow a ``not_after`` letter)
    strips ``strip`` letters and adds ``add``. CLDR has no plural of a numeral."""
    for rule in rules:
        for ending in rule["ends"]:
            if not word.endswith(ending):
                continue
            before = word[: len(word) - len(ending)][-1:]
            if before in rule.get("not_after", ()):
                continue
            return word[: len(word) - rule.get("strip", 0)] + rule["add"]
    raise NotImplementedError(f"no plural rule fits {word!r}")


def _plural_phrase(text: str, locale: str) -> str:
    rules = _lexical("numeral.plural", locale)
    if not rules:
        raise NotImplementedError(f"no plural of a number word for {locale}")
    head, space, last = text.rpartition(" ")
    stem, hyphen, final = last.rpartition("-")
    return f"{head}{space}{stem}{hyphen}{_plural_word(final, rules)}"


def _spoken_plural(detection: object, locale: str) -> tuple[SpokenAlternative, ...]:
    """Speak a plural numeral ("1990s") as year-style and cardinal-style plurals.

    "1990s" gives "nineteen nineties", "1900s" "nineteen hundreds", "'90s"
    "nineties", "100s" "one hundreds"; no decade or century is guessed.
    """
    number = _capture_integer(detection, "number")
    if number is None:
        raise NotImplementedError("plural numeral did not retain its number")
    return _ranked(
        [
            SpokenAlternative(
                _plural_phrase(item.text, locale),
                f"{item.provenance}+{lexical_source(locale)}",
            )
            for kind in ("year", "cardinal")
            for item in _number_leaf(number, kind, locale)
        ]
    )


def _spoken_runs(value: object, locale: str) -> tuple[SpokenAlternative, ...]:
    """Speak a letter-digit token as its runs ("3D" -> "three d", "1080p" -> "ten eighty p").

    Each digit run reads as a cardinal and as a year; each letter run is spelled by
    its letters; a separator is silent. The corpus speaks such tokens this way.
    """
    parts: list[tuple[SpokenAlternative, ...]] = []
    for kind, text in getattr(value, "runs", ()):
        if kind == "digits" and len(text) > 1 and text.startswith("0"):
            # A zero-led run ("007", the "000" of "1,000th") is read digit by digit.
            digits = [_number_leaf(Decimal(digit), "cardinal", locale)[0] for digit in text]
            parts.append(
                (SpokenAlternative(" ".join(d.text for d in digits), digits[0].provenance),)
            )
        elif kind == "digits":
            number = Decimal(text)
            parts.append(
                _ranked(
                    [
                        *_number_leaf(number, "cardinal", locale),
                        *_number_leaf(number, "year", locale),
                    ]
                )
            )
        elif kind == "letters":
            letter_form = spelled(text, locale)
            if letter_form is None:
                raise NotImplementedError(f"no authoritative letter names for {locale}")
            parts.append(
                (
                    SpokenAlternative(letter_form.text, "surface:letters")
                    if locale == "en_US"
                    else letter_form,
                )
            )
    if not parts:
        raise NotImplementedError("alphanumeric token has no speakable run")
    return _compose(parts, " ".join("{}" for _ in parts))


def _roman_readings(
    alternatives: Sequence[SpokenAlternative], value: NumberValue, detection: object, locale: str
) -> tuple[SpokenAlternative, ...]:
    """A Roman numeral reads as a cardinal or an ordinal ("II": "two", "second", "the second").

    The corpus reads "Henry II" as "the second"; which reading fits is left to the
    ranking. A written possessive ("II's") adds "'s" to every form.
    """
    ordinals = _number_leaf(Decimal(value.decimal), "ordinal", locale)
    forms = [*alternatives, *ordinals, *_with_article(ordinals, locale)]
    if _capture(detection, "suffix") is not None:
        suffix = _lexical("possessive.suffix", locale)
        if suffix is None:
            raise NotImplementedError(f"no spoken possessive for {locale}")
        forms = [
            SpokenAlternative(
                f"{item.text}{suffix}",
                f"{item.provenance}+{lexical_source(locale)}",
                item.weight,
            )
            for item in forms
        ]
    return _ranked(forms)


def _with_article(
    ordinals: Sequence[SpokenAlternative], locale: str
) -> tuple[SpokenAlternative, ...]:
    """Each ordinal said with the locale's article (``ordinal.article``: "the second")."""
    if _lexical("ordinal.article", locale) is None:
        return ()
    return tuple(
        SpokenAlternative(
            _lexical_pattern("ordinal.article", locale, item.text),
            f"{lexical_source(locale)}+{item.provenance}",
        )
        for item in ordinals
    )


def _spoken_ordinal(
    value: NumberValue, detection: object, locale: str
) -> tuple[SpokenAlternative, ...]:
    """Speak a written ordinal ("29th") through ICU's ordinal spellout.

    A Roman ordinal ("V.", "XIVth") also reads with "the", as the corpus reads
    "V." ("the fifth") and as a Roman cardinal already does.
    """
    try:
        number = Decimal(value.decimal)
    except InvalidOperation as exc:
        raise ValueError(f"invalid captured ordinal {value.decimal!r}") from exc
    if number != number.to_integral_value() or number < 0:
        raise NotImplementedError("ordinal spellout covers non-negative integers only")
    ordinals = _number_leaf(number, "ordinal", locale)
    if getattr(_capture(detection, "integer"), "form", None) != "roman":
        return ordinals
    return _ranked([*ordinals, *_with_article(ordinals, locale)])


# A run is read in NFC ("E\u0301CO" as "ÉCO"), a combining mark kept on its letter.
_NFC = icu.Normalizer2.getNFCInstance()
_MARKS = icu.UnicodeSet("[:M:]")
_MARKS.freeze()


# A spell-out ("MD" read "M D") names each letter; an expansion reads the text as words.
def _spoken_letters(
    value: LettersValue, locale: str, profile: str | None = None
) -> tuple[SpokenAlternative, ...]:
    """A letter token spelled or read as a word, weighted by its measured population.

    Capital runs use the acronym prior; relevant bounded non-uppercase tokens use their
    exact spell-or-say prior. A plural or possessive rides on an acronym's last letter.
    An initial is its letter.
    """
    letters = _NFC.normalize(value.letters)
    if value.suffix == ".":
        letter_form = spelled(letters, locale)
        if letter_form is None:
            raise NotImplementedError(f"no authoritative letter names for {locale}")
        return (
            SpokenAlternative(letter_form.text, "surface:letter")
            if locale == "en_US"
            else letter_form,
        )
    if not is_letter_run(letters):
        spelling = spelled(letters, locale)
        if spelling is None:
            raise NotImplementedError(f"no authoritative letter names for {locale}")
        word = _NFC.normalize(value.surface).lower()
        entry = spelled_token_prior(value.letters, locale)
        if entry is None:
            decision = spelled_token_rule(value.letters)
            other = "say" if decision == "spell" else "spell"
            shares = {decision: 1, other: 0}
            source = "rule:spelled-token"
        else:
            shares = entry["shares"]
            source = "measured:spelled-token"
        return _ranked(
            (
                SpokenAlternative(
                    spelling.text,
                    f"{source}-spell",
                    Decimal(str(shares["spell"])),
                ),
                SpokenAlternative(
                    word,
                    f"{source}-say",
                    Decimal(str(shares["say"])),
                ),
            )
        )
    suffix = "'s" if value.suffix else ""
    word = _NFC.normalize(value.surface).lower()
    parsed_surface = split_acronym_surface(value.surface)
    readings = tuple(
        SpokenAlternative(
            f"{form.text}{suffix}" if form.provenance.endswith("spelled") else word,
            form.provenance,
            form.weight,
        )
        for form in _with_acronym_readings(
            letters,
            (),
            locale,
            profile=profile,
            profile_surface=parsed_surface[0],
            surface_subkey=parsed_surface[1],
        )
    )
    if not readings:
        raise NotImplementedError(f"no authoritative letter names for {locale}")
    return readings


# The source says which, so a consumer can tell spelled letters from a written long form.
_ABBREVIATION_SOURCES = {"expansion": "icukit-abbreviation", "spell-out": "icukit-spell-out"}


def _acronym_priors(*, locale: str = "en_US") -> dict[str, dict[str, int]]:
    """The locale's measured acronym readings by key (``data/<locale>/acronym_priors.json``);
    empty when the locale has no table. Cached on the canonical locale."""
    from frend.locale_data import canonical_locale

    return _acronym_priors_for(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _acronym_priors_for(locale: str) -> dict[str, dict[str, int]]:
    from frend.locale_data import measured_table

    table = measured_table("acronym_priors", locale)
    return {} if table is None else table["keys"]


def _acronym_surface_priors(
    *, locale: str = "en_US"
) -> tuple[dict[str, dict[str, dict[str, int]]], int, Decimal]:
    """Load Google-TN exact surfaces from its configured, external profile file."""
    _table_locale, surfaces, minimum_support, parent_strength, _source_shards = (
        _acronym_surface_profile(locale=locale)
    )
    return surfaces, minimum_support, parent_strength


def _acronym_surface_profile(*, locale: str = "en_US"):
    """Load the acronym table, including its verified source identities."""
    key = _profile_file_key(google_tn_profile_path())
    table_locale, surfaces, minimum_support, parent_strength, source_shards = (
        _acronym_surface_priors_for(*key)
    )
    locale = canonical_locale(locale)
    if locale.split("_", 1)[0] != table_locale.split("_", 1)[0]:
        raise ValueError(
            f"{GOOGLE_TN!r} profile data at {key[0]} is for {table_locale}, not {locale}"
        )
    return table_locale, surfaces, minimum_support, parent_strength, source_shards


def _google_tn_britishisms(*, locale: str = "en_US"):
    """Load both profile tables and require their verified corpus identities to agree."""
    *_settings, acronym_shards = _acronym_surface_profile(locale=locale)
    britishisms = load_britishisms(locale=locale)
    if acronym_shards != britishisms.source_shards:
        raise ValueError(
            f"{GOOGLE_TN!r} profile table source-shard digests do not match: "
            "acronym and spelling tables must be rebuilt from the same verified inventory"
        )
    return britishisms


_PROFILE_FILE_KEYS: dict[str, tuple[tuple[int, int, int, int, int], tuple[str, int, int, str]]] = {}


def _missing_profile(path: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"{GOOGLE_TN!r} profile data is missing at {path}; build it from your licensed "
        "Google TN corpus with tools/build_acronym_priors.py --profile-out PATH"
    )


def _profile_file_key(path: Path) -> tuple[str, int, int, str]:
    """Resolved path, mtime, size and digest, refreshed on filesystem replacement."""
    resolved = path.expanduser().resolve()
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        raise _missing_profile(resolved) from None
    freshness = (stat.st_dev, stat.st_ino, stat.st_ctime_ns, stat.st_mtime_ns, stat.st_size)
    cached = _PROFILE_FILE_KEYS.get(str(resolved))
    if cached is not None and cached[0] == freshness:
        return cached[1]
    try:
        content = resolved.read_bytes()
        after = resolved.stat()
    except FileNotFoundError:
        raise _missing_profile(resolved) from None
    after_freshness = (
        after.st_dev,
        after.st_ino,
        after.st_ctime_ns,
        after.st_mtime_ns,
        after.st_size,
    )
    if freshness != after_freshness:
        raise ValueError(f"{GOOGLE_TN!r} profile data changed while being read at {resolved}")
    key = (str(resolved), stat.st_mtime_ns, stat.st_size, hashlib.sha256(content).hexdigest())
    _PROFILE_FILE_KEYS[str(resolved)] = (freshness, key)
    return key


@lru_cache(maxsize=LOCALE_CACHE)
def _acronym_surface_priors_for(
    path_text: str, mtime_ns: int, size: int, sha256: str
) -> tuple[
    str,
    dict[str, dict[str, dict[str, int]]],
    int,
    Decimal,
    tuple[tuple[str, str], ...],
]:
    path = Path(path_text)
    try:
        content = path.read_bytes()
        stat = path.stat()
    except FileNotFoundError:
        raise _missing_profile(path) from None
    if (
        stat.st_mtime_ns != mtime_ns
        or stat.st_size != size
        or hashlib.sha256(content).hexdigest() != sha256
    ):
        raise ValueError(f"{GOOGLE_TN!r} profile data changed while being loaded at {path}")
    table = json.loads(content)
    if not isinstance(table, dict) or table.get("schema_version") != 1:
        raise ValueError(f"invalid {GOOGLE_TN!r} profile data at {path}: wrong schema version")
    if table.get("profile") != GOOGLE_TN:
        raise ValueError(f"invalid {GOOGLE_TN!r} profile data at {path}: wrong profile name")
    if not isinstance(table.get("locale"), str) or not table["locale"]:
        raise ValueError(f"invalid {GOOGLE_TN!r} profile data at {path}: no locale")
    table_locale = canonical_locale(table["locale"])
    selection = table.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"invalid {GOOGLE_TN!r} profile data at {path}: no selection")
    provenance = table.get("provenance")
    source_shards = provenance.get("source_shards") if isinstance(provenance, dict) else None
    if not isinstance(source_shards, list) or not source_shards:
        raise ValueError(f"invalid {GOOGLE_TN!r} profile data at {path}: no verified source shards")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("relative_path"), str)
        or not isinstance(item.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        for item in source_shards
    ):
        raise ValueError(
            f"invalid {GOOGLE_TN!r} profile data at {path}: invalid source-shard digest"
        )
    surfaces = table.get("surfaces")
    if not isinstance(surfaces, dict):
        raise ValueError(f"invalid {GOOGLE_TN!r} profile data at {path}: no surfaces table")
    for surface, subkeys in surfaces.items():
        if not isinstance(surface, str) or not isinstance(subkeys, dict):
            raise ValueError(
                f"invalid {GOOGLE_TN!r} profile data at {path}: invalid surfaces table"
            )
        for subkey, counts in subkeys.items():
            if (
                subkey not in {"bare", "plural", "possessive"}
                or not isinstance(counts, dict)
                or not counts
                or not set(counts) <= {"spelled", "word"}
                or any(
                    not isinstance(count, int) or isinstance(count, bool) or count < 0
                    for count in counts.values()
                )
            ):
                raise ValueError(
                    f"invalid {GOOGLE_TN!r} profile data at {path}: invalid surface row"
                )
    try:
        minimum_support = int(selection["minimum_support"])
        parent_strength = Decimal(str(selection["parent_strength"]))
    except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid {GOOGLE_TN!r} profile data at {path}: invalid selection"
        ) from exc
    if minimum_support <= 0 or parent_strength < 0:
        raise ValueError(f"invalid {GOOGLE_TN!r} profile data at {path}: invalid selection")
    return (
        table_locale,
        surfaces,
        minimum_support,
        parent_strength,
        tuple(sorted((item["relative_path"], item["sha256"]) for item in source_shards)),
    )


def _with_acronym_readings(
    surface: str,
    alternatives: tuple[SpokenAlternative, ...],
    locale: str,
    *,
    profile: str | None = None,
    profile_surface: str | None = None,
    surface_subkey: str = "bare",
) -> tuple[SpokenAlternative, ...]:
    """An acronym ("FBI", "NASA") also reads spelled and as a word, weighted as measured.

    kal ruled that frend says both: the share the corpus spells an all-capitals token
    weights "f b i", the rest weights "fbi" (``data/en/acronym_priors.json``,
    ``tools/build_acronym_priors.py``), by the acronym's own counts where icukit's lexicon
    lists it or icukit reads it as a Roman numeral, blended toward its consonant-vowel
    pattern and then its shape (``letter_key``: length, vowel). An adequately supported
    exact surface and suffix row then replaces that share, smoothed toward it; a sparse
    or absent row leaves it unchanged. icukit's long forms follow. A spelled form
    icukit already gives ("M D") takes the weight, not a copy.
    """
    letters = "".join(ch for ch in surface if ch.isalpha() or _MARKS.contains(ch))
    if len(letters) < 2 or not letters.isupper():
        return alternatives
    spelling = spelled(letters, locale)
    if spelling is None:
        return alternatives
    table = _acronym_priors(locale=locale)
    if not table:
        return _ranked((spelling, *alternatives))

    def blend(counts: dict[str, int], parent: Decimal) -> Decimal:
        return (Decimal(counts.get("spelled", 0)) + 5 * parent) / (sum(counts.values()) + 5)

    overall = table["*"]
    share = blend(
        table.get(letter_key(letters, locale), {}),
        Decimal(overall["spelled"]) / sum(overall.values()),
    )
    if (pattern := cv_pattern(letters, locale)) is not None and f"cv:{pattern}" in table:
        # Its consonant-vowel pattern ("GUS" is mostly said, "GWR" spelled) over its shape.
        share = blend(table[f"cv:{pattern}"], share)
    for key in (f"surface:{letters}", f"roman:{letters}"):
        if key in table:
            # The run's own evidence ("NASA" is a word 2118 times to 4, "XI" 955 to 7)
            # over its shape's; a Roman numeral's number readings are not counted here.
            own = {label: table[key].get(label, 0) for label in ("spelled", "word")}
            share = blend(own, share)
    if profile == GOOGLE_TN:
        surfaces, minimum_support, parent_strength = _acronym_surface_priors(locale=locale)
        exact_surface = surface if profile_surface is None else profile_surface
        surface_counts = surfaces.get(exact_surface, {}).get(surface_subkey, {})
        support = sum(surface_counts.get(label, 0) for label in ("spelled", "word"))
        if support >= minimum_support > 0:
            # Sparse rows abstain completely, leaving the established shape/CV prior
            # unchanged. Retaining the suffix prevents a plural from training a bare run.
            share = (Decimal(surface_counts.get("spelled", 0)) + parent_strength * share) / (
                support + parent_strength
            )
    # Each capital by its own lower case ("İB" is "i b", "ΟΣ" "ο σ"), not the run's.
    forms = [SpokenAlternative(spelling.text, "measured:acronym-spelled", share)]
    if letters == surface:
        # A dotted surface ("U.S.") is spelled or expanded, never read as a word ("us").
        forms.append(SpokenAlternative(letters.lower(), "measured:acronym-word", 1 - share))
    measured = {normalize_spoken(form.text): form for form in forms}
    kept = []
    for item in alternatives:
        form = measured.pop(normalize_spoken(item.text), None)
        # A form icukit already gives ("M D") keeps its text and source, with the weight.
        kept.append(
            item if form is None else SpokenAlternative(item.text, item.provenance, form.weight)
        )
    return (*measured.values(), *kept)


def _spoken_abbreviation(value: AbbreviationValue) -> tuple[SpokenAlternative, ...]:
    """Every lexicon expansion is an alternative; ambiguity is kept, never resolved here.

    Sense and cue ride in the provenance ("Dr." gives Doctor as title/precedes-name
    and Drive as thoroughfare/address), so a later context rule can choose among
    them; a spell-out ("MD" read "M D") carries the ``icukit-spell-out`` source
    rather than ``icukit-abbreviation``. A surface the lexicon gives no expansion
    is not verbalized here.
    """
    if not value.expansions:
        raise NotImplementedError(f"no lexicon expansion for {value.surface!r}")
    return _expansion_alternatives(value.expansions)


def _expansion_alternatives(expansions: Sequence[object]) -> tuple[SpokenAlternative, ...]:
    """Each lexicon expansion as a spoken alternative, its sense and cue in its source."""
    return _ranked(
        [
            SpokenAlternative(
                expansion.text,
                _ABBREVIATION_SOURCES.get(
                    getattr(expansion, "type", "expansion"), "icukit-abbreviation"
                )
                + f":{expansion.sense}"
                + (f"/{expansion.cue}" if expansion.cue else ""),
            )
            for expansion in expansions
        ]
    )


def _abbreviation_ranked(
    written: str, alternatives: tuple[SpokenAlternative, ...], locale: str
) -> tuple[SpokenAlternative, ...]:
    """Rank an abbreviation's readings by how the corpus says its written form.

    Each reading the table measures takes its category's share from
    ``abbreviation_priors.json`` (``abbreviation_variants.abbreviation_weights``: by the
    token's fold, and its written case where the corpus writes that case), in place of
    any weight it carried (the key's own evidence is narrower than an acronym's shape).
    Measured readings lead by share; an unmeasured one follows, the token as written
    first, since with no evidence a variant reads as it did before it was recognized.
    """
    weights = abbreviation_weights(written, [item.text for item in alternatives], locale=locale)
    if weights is None:
        weights = (None,) * len(alternatives)
    weighted = [
        item if weight is None else SpokenAlternative(item.text, item.provenance, weight)
        for item, weight in zip(alternatives, weights, strict=True)
    ]
    order = sorted(
        range(len(weighted)),
        key=lambda index: (
            weighted[index].weight is None,
            -(weighted[index].weight or Decimal(0)),
            weighted[index].provenance != AS_WRITTEN_SOURCE,
            index,
        ),
    )
    return tuple(weighted[index] for index in order)


def _unspoken(detection: object, path: str) -> tuple[object, ...]:
    spoken = _SPOKEN_CAPTURES[path]
    return tuple(
        capture
        for capture in detection.get("captures", ())  # type: ignore[union-attr]
        if getattr(capture, "name", None) not in spoken
        or (
            capture.name == "sign"
            and str(getattr(capture, "text", "")).strip() not in _MINUS_SIGNS | _PLUS_SIGNS
        )
    )


# ICU numbers weekdays from Sunday = 1; this date is a Sunday.
_ICU_FIRST_WEEKDAY = date(2000, 1, 2)


def _weekday_name(capture: object, calendar: str, locale: str) -> SpokenAlternative:
    """Speak a weekday capture, which icukit encodes as ICU's number or as a name."""
    formatter = DateTimeFormatter(locale, calendar=calendar)
    names = [
        formatter.format(_ICU_FIRST_WEEKDAY + timedelta(days=offset), pattern="EEEE")
        for offset in range(7)
    ]
    value = getattr(capture, "value", None)
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 7:
        return SpokenAlternative(names[value - 1], "icu-datetime:EEEE")
    if isinstance(value, str):
        matches = [name for name in names if name.casefold() == value.casefold()]
        if len(matches) == 1:
            return SpokenAlternative(matches[0], "icu-datetime:EEEE")
    raise ValueError(f"invalid weekday capture value {value!r}")


@cache
def _era_variants(locale: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """CLDR's variant era names from ICU's data: abbreviated ("BCE", "CE") and wide
    ("Before Common Era", "Common Era"), empty where the locale has none."""
    try:
        table = (
            icu.ResourceBundle("", icu.Locale(locale))
            .getWithFallback("calendar")
            .getWithFallback("gregorian")
            .getWithFallback("eras")
        )
    except icu.ICUError:
        return (), ()
    found = []
    for key in ("abbreviated%variant", "wide%variant"):
        try:
            forms = table.getWithFallback(key)
        except icu.ICUError:
            found.append(())
            continue
        found.append(tuple(forms.get(index).getString() for index in range(forms.getSize())))
    return found[0], found[1]


def _era_names(era: int, detection: object, locale: str) -> tuple[SpokenAlternative, ...]:
    """An era reads as it is written: an abbreviation letter by letter, a name as words.

    The corpus reads "500 BC" as "five hundred b c" and "300 BCE" as "three hundred b c e":
    the written abbreviation, letter by letter. A written wide name ("Before Christ",
    "Common Era"; icukit's capture form ``wide``) is said as its words. An abbreviation
    also reads as the wide name of its own family from ICU's CLDR data: "BC" as "Before
    Christ", the variant "BCE" as "Before Common Era".
    """
    symbols = icu.DateFormatSymbols(icu.Locale(locale))
    names = symbols.getEraNames()
    if not 0 <= era < len(names):
        raise ValueError(f"invalid era {era!r}")
    written = _capture(detection, "era")
    text = str(getattr(written, "text", "")) or symbols.getEras()[era]
    if getattr(written, "form", None) == "wide":
        return (SpokenAlternative(" ".join(text.split()), "surface:words"),)
    letters = "".join(ch for ch in text if ch.isalpha())
    names_for_letters = letter_names(letters, locale)
    variant_short, variant_wide = _era_variants(locale)
    wide = SpokenAlternative(names[era], "icu-datetime:GGGG")
    if era < len(variant_short) and era < len(variant_wide):
        if text.casefold() == variant_short[era].casefold():
            wide = SpokenAlternative(variant_wide[era], "icu-datetime:GGGG%variant")
    spelled_era = (
        ()
        if names_for_letters is None
        else (SpokenAlternative(" ".join(names_for_letters.spoken), "surface:letters"),)
    )
    return (*spelled_era, wide)


def _spoken_era_year(
    value: DateTimeValue, detection: object, locale: str
) -> tuple[SpokenAlternative, ...]:
    """Speak a year with its era ("500 BC" -> "five hundred b c"), year first as written.

    The corpus reads the year of an era date as a cardinal ("two thousand a d"); the
    year-style reading is kept as an alternative for the ranking to weigh.
    """
    fields = dict(value.fields)
    year = Decimal(fields["y"])
    years = _ranked([*_number_leaf(year, "cardinal", locale), *_number_leaf(year, "year", locale)])
    eras = _era_names(fields["G"], detection, locale)
    era, written_year = _capture(detection, "era"), _capture(detection, "y")
    if era is not None and written_year is not None and era.start < written_year.start:
        # Written era first ("A.D. 1066"): said in the written order.
        return _compose([eras, years], "{} {}")
    return _compose([years, eras], "{} {}")


def _year_leaf(value: Decimal, locale: str) -> tuple[SpokenAlternative, ...]:
    """ICU's year readings, plus "o" where ICU says "oh" ("1908": "nineteen o eight").

    ICU's year rule set says a zero-led second half "oh" ("nineteen oh-eight"); the
    corpus says "o" (a date's zero is "o" 47,752 times, "oh" never), and
    ``_zero_shares`` ranks it so. No locale data spells "o", so that form is lexical
    over ICU's, as a zero-led minute already is.
    """
    forms = list(_number_leaf(value, "year", locale))
    said = _lexical("zero.year", locale) or {}
    for item in tuple(forms):
        words = item.text.replace("-", " ").split(" ")
        if said.keys() & set(words):
            text = " ".join(said.get(word, word) for word in words)
            forms.append(SpokenAlternative(text, f"{item.provenance}+{lexical_source(locale)}"))
    return _ranked(forms)


def _quarter_names(quarter: int, calendar: str, locale: str) -> tuple[SpokenAlternative, ...]:
    """A quarter as ICU names it: the wide name with its ordinal said ("1st quarter" ->
    "first quarter"), and the short name spelled ("Q1" -> "q one")."""
    if not 1 <= quarter <= 4:
        raise ValueError(f"invalid quarter {quarter!r}")
    formatter = DateTimeFormatter(locale, calendar=calendar)
    day = date(2000, 3 * quarter - 2, 1)
    wide = formatter.format(day, pattern="QQQQ")
    short = formatter.format(day, pattern="QQQ")
    number = _number_leaf(Decimal(quarter), "ordinal", locale)[0].text
    wide_spoken = re.sub(r"^\d+\S*", number, wide)
    letters = " ".join(ch.lower() for ch in short if ch.isalpha())
    digit = _number_leaf(Decimal(quarter), "cardinal", locale)[0].text
    return (
        SpokenAlternative(wide_spoken, "icu-datetime:QQQQ+icu-rbnf:%spellout-ordinal"),
        SpokenAlternative(f"{letters} {digit}", "icu-datetime:QQQ+surface:letters"),
    )


def _spoken_date_parts(
    value: DateTimeValue, detection: object, locale: str
) -> tuple[SpokenAlternative, ...]:
    """Speak a date that carries an era, a quarter, a weekday or a time (icukit #115,
    #118, #121, #123) part by part in written order, each part by frend's own speech.

    The date's y/M/d read as any date; an era after them as its letters or name; a
    quarter as ICU names it; a time as any time, joined by "at" where "at" is written
    (ICU's long date-time pattern) and by nothing where a comma is. A weekday field is
    not spoken here: its capture leads every form, as for any date.
    """
    fields = dict(value.fields)
    ymd = tuple((key, item) for key, item in value.fields if key in ("y", "M", "d"))
    pieces: list[tuple[int, tuple[SpokenAlternative, ...]]] = []

    def start(name: str, default: int) -> int:
        capture = _capture(detection, name)
        return capture.start if capture is not None else default

    if "Q" in fields:
        pieces.append((start("quarter", 0), _quarter_names(fields["Q"], value.calendar, locale)))
        if "y" in fields:
            pieces.append((start("y", 1), _year_leaf(Decimal(fields["y"]), locale)))
    elif ymd:
        if set(dict(ymd)) == {"y"} and "G" in fields:
            pieces.append(
                (
                    start("y", 0),
                    _spoken_era_year(
                        DateTimeValue((("G", fields["G"]), ("y", fields["y"])), value.calendar),
                        detection,
                        locale,
                    ),
                )
            )
        else:
            # The date part ranks by its own measured shares, as a plain date does; the
            # other parts carry no weight, so the combination keeps that order.
            dates = _rank_final(
                _spoken_date(DateTimeValue(ymd, value.calendar), detection, locale),
                "date",
                measurement_sub_key("date", detection, locale=locale),
                locale,
            )
            first = min(
                (
                    capture.start
                    for name in ("y", "month", "M", "d")
                    if (capture := _capture(detection, name))
                ),
                default=0,
            )
            pieces.append((first, dates))
            if "G" in fields:
                pieces.append((start("era", first + 1), _era_names(fields["G"], detection, locale)))
    if "H" in fields:
        time_value = DateTimeValue(
            tuple((key, item) for key, item in value.fields if key in ("H", "m")), value.calendar
        )
        glue = str(getattr(_capture(detection, "datetime-glue"), "text", "")).strip(" ,")
        times = _spoken_time(time_value, detection, locale)
        if glue:
            times = tuple(
                SpokenAlternative(f"{glue} {item.text}", f"surface:words+{item.provenance}")
                for item in times
            )
        pieces.append((start("H", 10**6), times))
    if not pieces:
        raise NotImplementedError(f"no speakable date part in {value.fields!r}")
    ordered = [
        tuple(
            item
            if item.weight is not None
            else SpokenAlternative(item.text, item.provenance, Decimal(0))
            for item in alternatives
        )
        if any(item.weight is not None for part in pieces for item in part[1])
        else alternatives
        for _, alternatives in sorted(pieces, key=lambda piece: piece[0])
    ]
    return _compose(ordered, " ".join("{}" for _ in ordered))


def _bare_four_digit(detection: Mapping, locale: str) -> bool:
    """Whether a bare four-digit token has measured locale-specific choices."""
    written = str(detection.get("text", ""))
    return (
        load_number_priors(locale) is not None
        and len(written) == 4
        and written.isascii()
        and written.isdigit()
    )


def _spoken_date(
    value: DateTimeValue, detection: object, locale: str
) -> tuple[SpokenAlternative, ...]:
    """Emit month-first and day-first alternatives because value alone hides written order."""
    fields = dict(value.fields)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in fields.values()):
        raise ValueError(f"date fields must be integers: {value.fields!r}")
    if set(fields) == {"G", "y"} and len(value.fields) == 2:
        return _spoken_era_year(value, detection, locale)
    if set(fields) & {"G", "Q", "H", "E"} and len(fields) == len(value.fields):
        return _spoken_date_parts(value, detection, locale)
    if len(fields) != len(value.fields) or not fields or not set(fields) <= {"y", "M", "d"}:
        raise NotImplementedError("v1 date assembly supports unique y/M/d fields only")
    parts: list[tuple[SpokenAlternative, ...]] = []
    if "M" in fields:
        parts.append((_month_name(fields["M"], value.calendar, locale),))
    if "d" in fields:
        parts.append(_number_leaf(Decimal(fields["d"]), "ordinal", locale))
    if "y" in fields:
        parts.append(_year_leaf(Decimal(fields["y"]), locale))
    bare = str(detection.get("text", ""))
    if set(fields) == {"y"} and _bare_four_digit(detection, locale):
        # A bare four-digit token may be a digit string.  Dates with any other
        # structure remain icukit's reading and never gain this choice.
        return _ranked(
            [*_year_leaf(Decimal(fields["y"]), locale), *_spoken_digits(DigitsValue(bare), locale)]
        )
    if set(fields) == {"M", "d", "y"}:
        template = "{} {}, {}"
    elif set(fields) == {"M", "d"}:
        template = "{} {}"
    else:
        template = " ".join("{}" for _ in parts)
    month_first = _compose(parts, template)
    if not {"M", "d"} <= set(fields):
        return month_first
    # The corpus reads a day-first date as "the eighteenth of September", with or
    # without a year; no locale pattern says it, so the frame is lexical.
    day_words = tuple(
        SpokenAlternative(item.text.replace("-", " "), item.provenance)
        for item in _number_leaf(Decimal(fields["d"]), "ordinal", locale)
    )
    day_parts = [day_words, (_month_name(fields["M"], value.calendar, locale),)]
    if "y" in fields:
        day_parts.append(_year_leaf(Decimal(fields["y"]), locale))
    day_first = _compose(day_parts, "the {} of " + " ".join("{}" for _ in day_parts[1:]))
    return _ranked([*month_first, *day_first])


def _bare_number_ranked(
    alternatives: Sequence[SpokenAlternative], detection: Mapping, locale: str
) -> tuple[SpokenAlternative, ...]:
    """Rank a bare four-digit date/cardinal/digit choice by its measured century row."""
    if not _bare_four_digit(detection, locale):
        return tuple(alternatives)
    written = str(detection.get("text", ""))
    table = load_number_priors(locale)
    assert table is not None

    def choice(item: SpokenAlternative) -> str:
        if "numbering-year" in item.provenance:
            return "date"
        if item.provenance.startswith("icu-rbnf:%spellout-cardinal"):
            return "digit"
        return "cardinal"

    weighted = []
    for item in alternatives:
        prior = table.lookup(written, choice(item))
        weighted.append(
            item if prior is None else SpokenAlternative(item.text, item.provenance, prior.share)
        )
    return _ranked(weighted)


def _spoken_number(
    type_: str, value: NumberValue, detection: object, locale: str
) -> tuple[SpokenAlternative, ...]:
    try:
        decimal = Decimal(value.decimal)
    except InvalidOperation as exc:
        raise ValueError(f"invalid captured decimal {value.decimal!r}") from exc
    if type_.startswith("fraction:") or type_.startswith("number:fraction"):
        return _spoken_fraction(detection, locale)
    compact = _capture(detection, "compact")
    if compact is not None:
        alternatives = _spoken_compact(detection, locale)
        if value.currency is None:
            return alternatives
        names = list(_currency_wide_names(value.currency, decimal, locale))
        if _currency_units(value.currency, locale) is not None:
            names.insert(0, _currency_unit_name(value.currency, decimal, False, locale))
        return _compose([alternatives, _ranked(names)], "{} {}")
    if value.currency is not None or type_.startswith(("money:", "number:currency")):
        if value.currency is None:
            raise NotImplementedError("currency reading has no captured ISO currency code")
        return _spoken_money(decimal, value.currency, locale)
    if type_.startswith("number:decimal") and decimal.as_tuple().exponent < 0:
        integer_capture = _capture(detection, "integer")
        written = None if integer_capture is None else str(integer_capture.text)
        if written == "":
            return _spoken_decimal(decimal, locale, omit_zero_integer=True)
        alternatives = _spoken_decimal(decimal, locale)
        if written is not None and set(written) == {"0"}:
            # A written leading zero ("0.75") is often not said: "point seven five".
            alternatives = _ranked(
                [*alternatives, *_spoken_decimal(decimal, locale, omit_zero_integer=True)]
            )
        return alternatives
    if type_ == "number:percent":
        # The amount is read as any written number is. The value is a fraction of one
        # and has dropped trailing zeros, so a written fraction is read from its
        # captures ("79.20%" is "seventy nine point two o percent").
        fraction = _capture(detection, "fraction")
        integer = _capture(detection, "integer")
        amount = decimal.scaleb(2)
        if fraction is not None and integer is not None:
            digits = "".join(ch for ch in str(integer.text) if ch.isdigit())
            amount = Decimal(f"{'-' if decimal < 0 else ''}{digits or '0'}.{fraction.text}")
        suffix = _percent_name(locale)
        return tuple(
            SpokenAlternative(f"{item.text} {suffix}", item.provenance, item.weight)
            for item in _spoken_decimal(amount, locale)
        )
    if type_ == "number:cardinal:citation":
        # A case citation's volume ("339 U.S."): the corpus says the number alone; the
        # reporter's letters are also offered, for the ranking to weigh.
        base = _number_leaf(decimal, "cardinal", locale)
        reporter = _capture(detection, "reporter")
        letters = "".join(ch for ch in str(getattr(reporter, "text", "")) if ch.isalpha()).lower()
        spelled = (
            tuple(
                SpokenAlternative(
                    f"{item.text} {' '.join(letters)}", f"{item.provenance}+surface:letters"
                )
                for item in base
            )
            if letters
            else ()
        )
        return _ranked([*base, *spelled])
    if type_.startswith(("number:cardinal", "number:int", "number:decimal")):
        cardinals = _number_leaf(decimal, "cardinal", locale)
        written = str(getattr(_capture(detection, "integer"), "text", ""))
        if (
            len(written) >= 2
            and written.isdigit()
            and _capture(detection, "sign") is None
            and _capture(detection, "fraction") is None
        ):
            # A digit string also reads digit by digit, as spelled-out letters do ("2013"
            # "two zero one three", "068" "o six eight"); the ranking is the corpus's.
            return _ranked([*cardinals, *_spoken_digits(DigitsValue(written), locale)])
        return cardinals
    raise NotImplementedError(f"unsupported NumberValue reading class {type_!r}")


def _measure_template(amount: Decimal, unit: str, locale: str) -> str:
    """ICU's wide measure form for an amount, with ICU's own formatted number cut out.

    Formatting 60 kilometers in the locale and removing ICU's "60" leaves "{} kilometers":
    the unit's wide name in the plural the amount selects, in the locale's order, as CLDR
    states it. Nothing about the unit is written here.
    """
    number = float(amount)
    formatted = format_measure(number, unit, locale, WIDTH_WIDE)
    written = icu.NumberFormat.createInstance(icu.Locale(locale)).format(number)
    if formatted.count(written) != 1:
        raise NotImplementedError(f"cannot locate the amount in {formatted!r}")
    return formatted.replace(written, "{}", 1)


def _spoken_measure(value: MeasureValue, locale: str) -> tuple[SpokenAlternative, ...]:
    """Speak a measure: the amount as frend reads any number, the unit as ICU names it.

    A rate ("578.3/km²", unit ``per-square-kilometer``) is ICU's "per square kilometer";
    it also reads with the unit's plural after "per", as the corpus does ("per square
    kilometers"). Both unit forms are ICU's; putting the plural there is the corpus's
    choice, so that form is lexical over them.
    """
    amount = Decimal(value.decimal)
    head = _measure_template(amount, value.unit, locale)
    templates = [(head, "icu-measure:wide")]
    if value.unit.startswith("per-") and _lexical("measure.per_plural", locale):
        base = value.unit.removeprefix("per-")
        singular = _measure_template(Decimal(1), base, locale).replace("{}", "").strip()
        plural = _measure_template(amount, base, locale).replace("{}", "").strip()
        if singular != plural and head.count(singular) == 1:
            templates.append(
                (head.replace(singular, plural), f"icu-measure:wide+{lexical_source(locale)}")
            )
    return _ranked(
        [
            SpokenAlternative(template.format(item.text), f"{item.provenance}+{source}")
            for template, source in templates
            for item in _spoken_decimal(amount, locale)
        ]
    )


_DURATION_UNITS = {"h": "hour", "m": "minute", "s": "second"}


def _unit_joiners(locale: str) -> tuple[icu.ListFormatter, ...]:
    """ICU's list patterns that join unit phrases: narrow units ("one minute nineteen
    seconds") and wide "and" ("one minute and nineteen seconds")."""
    return tuple(
        icu.ListFormatter.createInstance(icu.Locale(locale), kind, width)
        for kind, width in (
            (icu.UListFormatterType.UNITS, icu.UListFormatterWidth.NARROW),
            (icu.UListFormatterType.AND, icu.UListFormatterWidth.WIDE),
        )
    )


def _spoken_duration(detection: object, locale: str) -> tuple[SpokenAlternative, ...]:
    """Speak a numeric duration ("1:47.22") component by component in ICU's wide units.

    icukit captures each field ("m" 1, "s" 47); each reads as a cardinal in ICU's wide
    unit form, and ICU's list patterns for units join them. A written fraction of the
    last field reads two ways: as ICU's decimal ("forty seven point two two seconds"),
    and as the corpus reads a race time, the fraction's digits as a count of
    milliseconds after "and" ("... seconds and twenty two milliseconds"); that reading
    is the corpus's, so it is lexical over ICU's unit forms.
    """
    fields = [
        (Decimal(str(capture.value)), _DURATION_UNITS[capture.name])
        for capture in detection.get("captures", ())  # type: ignore[union-attr]
        if capture.name in _DURATION_UNITS
    ]
    if not fields:
        raise NotImplementedError("duration without a captured field")
    fraction = _capture(detection, "fraction")

    def phrase(amount: Decimal, unit: str, reader) -> list[tuple[str, str]]:
        template = _measure_template(amount, unit, locale)
        return [
            (template.format(item.text), f"{item.provenance}+icu-measure:wide")
            for item in reader(amount)
        ]

    def cardinal(amount: Decimal) -> tuple[SpokenAlternative, ...]:
        return _number_leaf(amount, "cardinal", locale)

    def decimal(amount: Decimal) -> tuple[SpokenAlternative, ...]:
        return _spoken_decimal(amount, locale)

    head = [phrase(amount, unit, cardinal) for amount, unit in fields[:-1]]
    last_amount, last_unit = fields[-1]
    readings: list[tuple[list[list[tuple[str, str]]], str]] = []
    if fraction is None:
        readings.append(([*head, phrase(last_amount, last_unit, cardinal)], ""))
    else:
        exact = Decimal(f"{last_amount}.{fraction.text}")
        readings.append(([*head, phrase(exact, last_unit, decimal)], ""))
        if last_unit == "second" and _lexical("duration.milliseconds", locale) is not None:
            count = Decimal(str(fraction.text))
            milliseconds = phrase(count, "millisecond", cardinal)
            readings.append(
                ([*head, phrase(last_amount, last_unit, cardinal), milliseconds], "and")
            )
    forms = []
    for groups, tail in readings:
        for combination in product(*groups):
            texts = [text for text, _ in combination]
            source = "+".join(provenance for _, provenance in combination)
            if tail:
                # The corpus's race-time reading: "and" before the milliseconds only.
                body = _unit_joiners(locale)[0].format(texts[:-1])
                forms.append(
                    SpokenAlternative(
                        _lexical_pattern("duration.milliseconds", locale, body, texts[-1]),
                        f"{source}+icu-list:units+{lexical_source(locale)}",
                    )
                )
                continue
            for joiner in _unit_joiners(locale):
                forms.append(SpokenAlternative(joiner.format(texts), f"{source}+icu-list:units"))
    return _ranked(forms)


def _spoken_unit(value: object, locale: str) -> tuple[SpokenAlternative, ...]:
    """Speak a unit written with no amount ("/km²", "per second"): ICU's wide form of the
    unit for one, with ICU's number cut out ("per square kilometer").

    icukit reads such a rate as a ``UnitValue`` (icukit #102); it is matched by name so
    frend still runs on an icukit release without it.
    """
    unit = str(getattr(value, "unit", ""))
    phrase = _measure_template(Decimal(1), unit, locale).replace("{}", "").strip()
    if not phrase:
        raise NotImplementedError(f"no ICU form for unit {unit!r}")
    return (SpokenAlternative(phrase, "icu-measure:wide"),)


def _spoken_mixed_measure(detection: object, locale: str) -> tuple[SpokenAlternative, ...]:
    """Speak each component of a mixed measure in its own unit, joined as ICU joins units.

    icukit captures each component's integer and unit ("5'10\\"": 5 foot, 10 inch); each
    reads as a cardinal in ICU's wide unit form, and ICU's list pattern for units joins
    them ("five feet, ten inches").
    """
    pairs: list[tuple[Decimal, str]] = []
    amount = None
    for capture in detection.get("captures", ()):  # type: ignore[union-attr]
        if capture.name == "integer":
            amount = Decimal(str(capture.value))
        elif capture.name == "unit" and amount is not None and capture.value:
            pairs.append((amount, str(capture.value)))
            amount = None
    if len(pairs) < 2:
        raise NotImplementedError("mixed measure without two components")
    joiner = icu.ListFormatter.createInstance(
        icu.Locale(locale), icu.UListFormatterType.UNITS, icu.UListFormatterWidth.WIDE
    )
    components = [
        [
            SpokenAlternative(
                _measure_template(number, unit, locale).format(item.text),
                f"{item.provenance}+icu-measure:wide",
            )
            for item in _number_leaf(number, "cardinal", locale)
        ]
        for number, unit in pairs
    ]
    return _ranked(
        [
            SpokenAlternative(
                joiner.format([item.text for item in combination]),
                "+".join(item.provenance for item in combination) + "+icu-list:units",
            )
            for combination in product(*components)
        ]
    )


# How many ranked readings of a URL or email address are kept: its runs multiply.
ELECTRONIC_BEAM = 8
ELECTRONIC_SOURCE = "measured:electronic"
# The corpus holds no email address, so "@" has no measured name, and locale data has
# no spoken name for it (ICU's character name is COMMERCIAL AT): "at" is lexical, in
# ``lexical.json`` ``separator.words``.


def _spoken_electronic(value: ElectronicValue, locale: str) -> tuple[SpokenAlternative, ...]:
    """Speak a URL, email address or domain run by run, as the corpus is measured to.

    Each letter run reads as a word or spelled, each digit run as one of its readings
    (ICU's cardinal or year, or digit by digit), each separator by its spoken name, all
    with probabilities from ``data/en/electronic_priors.json``. The runs multiply, so the
    best readings by their product are kept (``ELECTRONIC_BEAM``), each weighted by it.
    A separator the corpus never names is not verbalized, except one the locale names
    in ``lexical.json`` (``separator.words``: "@" "at").
    """
    from frend.electronic import load_electronic_priors

    if load_electronic_priors(locale=locale) is None:
        raise NotImplementedError(f"no measured electronic prior for {locale}")
    tlds = tld_positions(value.parts)
    unmeasured_names = _lexical("separator.words", locale) or {}
    unmeasured = False
    choices: list[dict[str, Decimal]] = []
    for index, (kind, text) in enumerate(value.parts):
        options: dict[str, Decimal] = {}
        if kind == "letters":
            lower = text.lower()
            probabilities = letter_probabilities(text, tld=index in tlds, locale=locale)
            if len(lower) == 1:
                options[lower] = Decimal(1)
            else:
                options[lower] = probabilities.get("word", Decimal(0))
                spelled = " ".join(lower)
                options[spelled] = options.get(spelled, Decimal(0)) + probabilities.get(
                    "spelled", Decimal(0)
                )
        elif kind == "digits":
            probabilities = digit_probabilities(text, locale=locale)
            for form, readings in digit_forms(text, locale).items():
                spoken = readings[0][0]
                options[spoken] = options.get(spoken, Decimal(0)) + probabilities.get(
                    form, Decimal(0)
                )
        else:
            names = {
                name: share
                for name, share in separator_names(text, locale=locale).items()
                if name != "sil"
            }
            if not names and text in unmeasured_names:
                names = {unmeasured_names[text]: Decimal(1)}
                unmeasured = True
            if not names:
                raise NotImplementedError(f"no measured spoken name for {text!r}")
            options.update(names)
        choices.append(options)
    beam: list[tuple[Decimal, tuple[str, ...]]] = [(Decimal(1), ())]
    for options in choices:
        extended = [
            (probability * share, words + (spoken,))
            for probability, words in beam
            for spoken, share in options.items()
        ]
        extended.sort(key=lambda item: -item[0])
        beam = extended[:ELECTRONIC_BEAM]
    source = f"{ELECTRONIC_SOURCE}+{lexical_source(locale)}" if unmeasured else ELECTRONIC_SOURCE
    return tuple(
        SpokenAlternative(" ".join(words), source, probability) for probability, words in beam
    )


def _spoken_digits(value: DigitsValue, locale: str) -> tuple[SpokenAlternative, ...]:
    """Say spaced digits one by one by ICU's cardinal ("6 3" -> "six three"); a zero is
    also "o", which ICU has no rule for, so that form is lexical."""
    words = [_number_leaf(Decimal(digit), "cardinal", locale)[0].text for digit in value.digits]
    forms = [SpokenAlternative(" ".join(words), "icu-rbnf:%spellout-cardinal")]
    zero = _lexical("zero.digit", locale)
    if "0" in value.digits and zero is not None:
        spoken = " ".join(zero if d == "0" else w for d, w in zip(value.digits, words, strict=True))
        forms.append(
            SpokenAlternative(spoken, f"icu-rbnf:%spellout-cardinal+{lexical_source(locale)}")
        )
    return _ranked(forms)


_RELATIVE_DIRECTIONS = {-2: "LAST_2", -1: "LAST", 0: "THIS", 1: "NEXT", 2: "NEXT_2"}


@cache
def _relative_formatter(locale: str) -> icu.RelativeDateTimeFormatter:
    """ICU's relative-date formatter at its wide (LONG) style, built as icukit builds it."""
    icu_locale = icu.Locale(locale)
    return icu.RelativeDateTimeFormatter(
        icu_locale,
        icu.NumberFormat.createInstance(icu_locale),
        icu.UDateRelativeDateTimeFormatterStyle.LONG,
        icu.UDisplayContext.CAPITALIZATION_NONE,
    )


def _spoken_relative(
    value: object, detection: object, locale: str
) -> tuple[SpokenAlternative, ...]:
    """Speak a relative date (icukit's ``date:relative``) as ICU's wide style says it.

    A named phrase ("yesterday", "next Tue.", "last mo.") reads as ICU's wide phrase for
    its direction and unit ("yesterday", "next Tuesday", "last month"). A numeric one ("1
    hr. ago", "in 2h") reads as ICU's wide numeric form with ICU's number cut out and
    frend's number spoken in its place ("one hour ago", "in two hours").
    """
    offset = int(value.offset)
    unit = str(value.unit).upper()
    formatter = _relative_formatter(locale)
    if _capture(detection, "integer") is None:
        direction = getattr(icu.UDateDirection, _RELATIVE_DIRECTIONS.get(offset, ""), None)
        absolute = getattr(icu.UDateAbsoluteUnit, unit, None)
        if unit == "NOW":
            direction = icu.UDateDirection.PLAIN
        phrase = formatter.format(direction, absolute) if direction and absolute else ""
        if not phrase:
            raise NotImplementedError(f"ICU names no relative phrase for {offset} {unit}")
        return (SpokenAlternative(phrase, "icu-relative:long"),)
    numeric_unit = getattr(icu.URelativeDateTimeUnit, unit, None)
    if numeric_unit is None:
        raise NotImplementedError(f"ICU has no relative unit {unit}")
    formatted = formatter.formatNumeric(offset, numeric_unit)
    written = icu.NumberFormat.createInstance(icu.Locale(locale)).format(abs(offset))
    if formatted.count(written) != 1:
        raise NotImplementedError(f"cannot locate the amount in {formatted!r}")
    template = formatted.replace(written, "{}", 1)
    return tuple(
        SpokenAlternative(template.format(item.text), f"{item.provenance}+icu-relative:long")
        for item in _number_leaf(Decimal(abs(offset)), "cardinal", locale)
    )


@dataclass(frozen=True)
class RangeSlot:
    index: str
    attributes: Mapping[str, str]


@dataclass(frozen=True)
class RangeConnector:
    id: str
    pattern: str
    words: str
    slots: tuple[RangeSlot, ...]
    provenance: str


def range_connector(locale: str) -> RangeConnector | None:
    """The words a range's two ends are joined by ("to"), from the lexical table's
    ``range.connector`` "to" pattern; ``None`` when the locale has none (the feature is
    off) or its pattern cannot be said by a separator alone."""
    return _range_connector(canonical_locale(locale), _lexical, _lexical_for)


@lru_cache(maxsize=LOCALE_CACHE)
def _range_connector(locale: str, lexical, lexical_for) -> RangeConnector | None:
    del lexical_for  # Its identity invalidates the cache when tests or callers replace the loader.
    for form in (lexical("range.connector", locale) or {}).get("range", ()):
        identifier = str(form.get("id", "")).strip()
        pattern = str(form.get("pattern", ""))
        words = connector_words(pattern)
        if identifier and words:
            slots = tuple(
                RangeSlot(str(index), MappingProxyType(dict(attributes)))
                for index, attributes in sorted((form.get("slots") or {}).items())
            )
            return RangeConnector(
                identifier,
                pattern,
                words,
                slots,
                _lexical_source(locale),
            )
    return None


def range_separators(locale: str) -> frozenset[str]:
    """The characters a written range is joined by: the lexical table's
    ``range.separator`` ranges (the hyphen-minus, which CLDR writes nowhere in en) and
    CLDR's own number-range separator (the en dash). Empty when the locale has no spoken
    connector."""
    return _range_separators(canonical_locale(locale), _lexical, _lexical_for)


@lru_cache(maxsize=LOCALE_CACHE)
def _range_separators(locale: str, lexical, lexical_for) -> frozenset[str]:
    if _range_connector(locale, lexical, lexical_for) is None:
        return frozenset()
    written = set((lexical("range.separator", locale) or {}).get("range", ()))
    cldr = cldr_range_separator(locale)
    return frozenset(written | ({cldr} if cldr else set()))


# The digit groups a separator joins (``frend.ranges.digit_groups``, R3's chain).
_digit_groups = digit_groups


def _is_identifier(text: str, start: int, end: int, separators: frozenset[str]) -> bool:
    """Whether the separator at ``text[start:end]`` joins the digit groups of an
    identifier, not the ends of a range: a chain of three or more groups (an ISBN,
    "1-800-555-1212") or a phone number's local shape, three digits then four
    ("555-1212")."""
    groups = _digit_groups(text, start, end, separators)
    return len(groups) >= 3 or groups == [3, 4]


def _range_to(
    context: TextContext | None, start: int, end: int, locale: str
) -> RangeConnector | None:
    """The locale's range connector ("to") when ``[start, end)`` of the lattice's text
    is a range separator with a number on either side in ``context``, spaced alike on
    both sides, and not inside an identifier's digit groups; else ``None``.

    The range class ("-", "–") is #43's; a lone ratio separator (":", spaced) between
    numbers is offered "to" too (R12), where the range table emits its neighbors' sub-key (E:
    "10 : 30" is a clock's, and says no "to")."""
    if context is None:
        return None
    connector = range_connector(locale)
    if connector is None:
        return None
    a, b = context.offset + start, context.offset + end
    written = context.text[a:b]
    ratio = range_separator_classes(locale).get("ratio", frozenset())
    if written in range_separators(locale):
        separators = range_separators(locale)
    elif written in ratio:
        separators = ratio
    else:
        return None
    if not between_numbers(context.text, a, b):
        return None
    if _is_identifier(context.text, a, b, separators):
        return None
    if written in ratio and not (
        _spaced(context.text, a, b) and _lone_ratio_emits(context.text, a, b, written, locale)
    ):
        # R12 is for a lone ":" (spaced, or a corpus token read alone); a joined one is
        # a range span's, or no range ("20:2008's").
        return None
    return connector


def _spaced(text: str, start: int, end: int) -> bool:
    return text[start - 1 : start].isspace() and text[end : end + 1].isspace()


def _lone_ratio_emits(text: str, start: int, end: int, separator: str, locale: str) -> bool:
    """E for a lone ratio separator: the sub-key of the numbers either side of it."""
    table = load_range_priors(locale)
    if table is None:
        return False
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    left = re.search(r"[0-9]+$", before)
    right = re.match(r"[0-9]+", after)
    if left is None or right is None:
        return False
    return table.emits(written_sub_key("ratio", left.group(0), separator, right.group(0), locale))


def _is_ratio_separator(context: TextContext | None, start: int, end: int, locale: str) -> bool:
    if context is None:
        return False
    written = context.text[context.offset + start : context.offset + end]
    return written in range_separator_classes(locale).get("ratio", frozenset())


def _lone_ratio_first(
    alternatives: tuple[SpokenAlternative, ...], locale: str
) -> tuple[SpokenAlternative, ...]:
    """R12: a lone ":" between numbers says "to" first where the range table's ratio
    class shares say "to" more often than nothing (``connector_share``)."""
    table = load_range_priors(locale)
    if table is None:
        return alternatives
    connector = range_connector(locale)
    if connector is None:
        return alternatives
    said = table.connector_share("ratio", connector.id)
    silent = table.connector_share("ratio", "silent")
    if said is None or silent is None or said <= silent:
        return alternatives
    at = next(
        (
            i
            for i, item in enumerate(alternatives)
            if item.provenance.startswith(connector.provenance)
        ),
        None,
    )
    if not at:
        return alternatives
    return (alternatives[at], *alternatives[:at], *alternatives[at + 1 :])


def _connector_first(
    alternatives: tuple[SpokenAlternative, ...],
    context: TextContext,
    start: int,
    end: int,
    locale: str,
    threshold: float,
) -> tuple[tuple[SpokenAlternative, ...], ContextChoice | None]:
    """A separator written inside a range ("5-10", "10:30-11:45"): the standalone
    separator's tree, read at the same place, puts the first connector reading first
    when it gives "to" at least ``threshold``."""
    a, b = context.offset + start, context.offset + end
    said = connector_probability(
        context.text, a, b, locale=locale, bos=context.bos, eos=context.eos
    )
    if said is None:
        return alternatives, None
    label, probability = said
    connector = range_connector(locale)
    if connector is None:
        return alternatives, None
    at = next(
        (
            i
            for i, item in enumerate(alternatives)
            if item.provenance.startswith(connector.provenance)
        ),
        None,
    )
    applied = at is not None and at > 0 and probability >= threshold
    choice = ContextChoice(f"connector:{context.text[a:b]}", label, probability, applied)
    if not applied:
        return alternatives, choice
    return (alternatives[at], *alternatives[:at], *alternatives[at + 1 :]), choice


def _may_end_a_range(type_: str) -> bool:
    """Whether a signed reading of this type may be a written range's right end, its
    sign the separator ("5-10", "5-10%")."""
    return type_.startswith(("number:cardinal", "number:int", "number:decimal", "number:percent"))


def _connector_reading(connector: RangeConnector) -> SpokenAlternative:
    """The range connector alone, said for a separator between two numbers ("to");
    ``connector`` is read from the lexical table (``range_connector``)."""
    return SpokenAlternative(connector.words, connector.provenance)


def _connected(
    connector: RangeConnector, right_end: Sequence[SpokenAlternative]
) -> tuple[SpokenAlternative, ...]:
    """The right end of a written range after the table's connector ("to ten")."""
    return tuple(
        SpokenAlternative(
            f"{connector.words} {item.text}", f"{connector.provenance}+{item.provenance}"
        )
        for item in right_end
    )


def _left_end_digits(context: TextContext, start: int) -> int:
    """How many decimal digits the number just before ``start`` (the lattice's
    coordinates) is written with, white space skipped."""
    before = context.text[: context.offset + start].rstrip()
    count = 0
    while count < len(before) and before[-1 - count].isdecimal():
        count += 1
    return count


def _unsigned(
    type_: str, value: NumberValue, detection: Mapping, locale: str, *, year_first: bool
) -> tuple[SpokenAlternative, ...]:
    """A signed number's readings without its sign: the right end of a written range.

    With ``year_first`` (the left end is written as a year is, four digits: "1990-1995",
    "1992-93"), an integer end is read as a year first, as the corpus reads such ranges
    ("nineteen ninety to nineteen ninety five"); until the range readers measure it
    (the ranges plan's P6), this follows the left end's form, not a measurement.
    """
    decimal = abs(Decimal(value.decimal))
    unsigned_value = dataclasses.replace(value, decimal=str(decimal))
    unsigned_detection = {
        **detection,
        "captures": tuple(
            capture
            for capture in detection.get("captures", ())
            if getattr(capture, "name", None) != "sign"
        ),
    }
    alternatives = _spoken_number(type_, unsigned_value, unsigned_detection, locale)
    kind = _measured_kind(type_, unsigned_value)
    ranked = _rank_final(
        alternatives,
        kind,
        measurement_sub_key(kind, unsigned_detection, locale=locale),
        locale,
    )
    if year_first and type_ != "number:percent" and decimal == decimal.to_integral_value():
        return _year_first(decimal, ranked, locale)
    return ranked


def _year_first(
    decimal: Decimal, ranked: Sequence[SpokenAlternative], locale: str
) -> tuple[SpokenAlternative, ...]:
    """A whole range end read as a year first, then its own readings: the interim rule
    for a range whose left end is written with four digits ("1990-1995", "1992-93",
    "1990–95"), shared by the hyphen (``_unsigned``) and ICU's ranges (``_range_end``).
    It follows the left end's form, not a measurement; the ranges plan's P6 measures
    and replaces it."""
    seen: set[str] = set()
    return tuple(
        item
        for item in (*_year_leaf(decimal, locale), *ranked)
        if not (item.text in seen or seen.add(item.text))
    )


# At most this many readings of one range: each end's readings, joined by each of the
# locale's connector patterns, in order of how far each is from the first of its kind.
_RANGE_CAP = 16


def _fill_slot(detection: Mapping, slot: Mapping, locale: str) -> SpokenAlternative | None:
    """A range end said in the case and gender its pattern's slot asks for ("от {0} до
    {1}": genitive "пяти"), by the locale's own rule set ``%spellout-cardinal-{g}-{c}``;
    ``None`` when the end is no whole number or the locale has no such rule set (the
    pattern is then not said: no fallback to another case)."""
    value = detection.get("value")
    if not isinstance(value, NumberValue):
        return None
    try:
        amount = Decimal(value.decimal)
    except InvalidOperation:
        return None
    if amount != amount.to_integral_value():
        return None
    ruleset = f"%spellout-cardinal-{slot.get('gender')}-{slot.get('case')}"
    if ruleset not in _spellout_rule_sets(locale):
        return None
    text = _format_exact(_spellout_formatter(locale), amount, ruleset)
    return SpokenAlternative(text, f"icu-rbnf:{ruleset}")


def _written_digits(detection: Mapping) -> int:
    text = str(detection.get("text", ""))
    return len(text) if text.isdigit() else 0


def _range_end(
    detection: Mapping, locale: str, *, bare: bool, year: bool, apply_source_priors: bool
) -> tuple[tuple[SpokenAlternative, str], ...]:
    """One end's readings, each with the kind it is read as (``cardinal``, ``decimal``,
    ``money``, ``measure``, ``date``, ``time``): the end read as frend reads that value
    alone, ranked by that kind's measured shares. ``bare`` reads an amount without the
    unit or currency the other end carries; ``year`` reads a whole number as a year
    first (``_year_first``, the interim rule the hyphen follows)."""
    type_ = str(detection["type"])
    value = detection["value"]
    if isinstance(value, DateTimeValue):
        if type_.startswith("time:"):
            alternatives = _spoken_time(value, detection, locale)
            kind = "time"
        else:
            alternatives = _spoken_date(value, detection, locale)
            kind = "date"
        amount = None
    else:
        amount = Decimal(value.decimal)
        number = detection.get("number")
        if (bare or type_ == "number:decimal") and number is not None:
            # The number the end writes, read as written ("1.00" "one point o o", "79.20"
            # in "79.20%"): a cardinal where it writes no fraction, else a decimal.
            amount = Decimal(number["value"].decimal)
            whole = _capture(number, "fraction") is None
            alternatives = _spoken_number("number:decimal", number["value"], number, locale)
            kind = "cardinal" if whole else "decimal"
        elif bare or type_ == "number:decimal":
            if type_ == "number:percent":
                amount = amount.scaleb(2)
            whole = amount == amount.to_integral_value()
            type_ = "number:cardinal" if whole else "number:decimal"
            plain = NumberValue(decimal=str(amount), currency=None)
            alternatives = _spoken_number(type_, plain, {"captures": ()}, locale)
            kind = "cardinal" if whole else "decimal"
        elif type_.startswith("measure:"):
            alternatives = _spoken_measure(value, locale)
            kind = "measure"
        else:
            alternatives = _spoken_number(type_, value, detection, locale)
            kind = _measured_kind(type_, value) or "cardinal"
    if apply_source_priors:
        alternatives = _rank_final(
            alternatives, kind, measurement_sub_key(kind, detection, locale=locale), locale
        )
    if (
        year
        and kind == "cardinal"
        and amount is not None
        and str(detection["type"]) != "number:percent"
    ):
        years = {item.text for item in _year_leaf(amount, locale)}
        return tuple(
            (item, "date" if item.text in years else kind)
            for item in _year_first(amount, alternatives, locale)
        )
    return tuple((item, kind) for item in alternatives)


def _measured_end(
    detection: Mapping, locale: str, *, bare: bool, apply_source_priors: bool
) -> tuple[tuple[SpokenAlternative, str], ...]:
    """One end's readings labeled as the range table counts them (``_range_end`` without
    the interim year rule), each with its kind:

    - R11: a cardinal said digit by digit (``%spellout-cardinal``: "05" "o five") is kind
      ``digits``;
    - R8: a plain whole end (``number:decimal``, no unit, integral) also offers its year
      readings (``%spellout-numbering-year``, and their "o" forms), kind ``date``,
      ranked as frend ranks a year alone; an end written with four digits labels a
      reading that is also its year reading ``date`` ("two thousand seven"), so a range
      of years has one source."""
    items = [
        (item, "digits" if kind == "cardinal" and "%spellout-cardinal" in item.provenance else kind)
        for item, kind in _range_end(
            detection, locale, bare=bare, year=False, apply_source_priors=apply_source_priors
        )
    ]
    value = detection.get("value")
    if (
        str(detection.get("type")) == "number:decimal"
        and not detection.get("writes_unit")
        and isinstance(value, NumberValue)
    ):
        try:
            amount = Decimal(value.decimal)
        except InvalidOperation:
            amount = None
        if amount is not None and amount == amount.to_integral_value() and amount >= 0:
            years = tuple(
                item for item in _year_leaf(amount, locale) if "numbering-year" in item.provenance
            )
            if apply_source_priors:
                years = _rank_final(years, "date", None, locale)
            written = str(detection.get("text", ""))
            if len(written) == 4 and set(written) <= set("0123456789"):
                year_texts = {item.text for item in years}
                items = [
                    (item, "date" if item.text in year_texts else kind) for item, kind in items
                ]
            own = {item.text for item, _ in items}
            items += [(item, "date") for item in years if item.text not in own]
    return tuple(items)


def _range_candidates(
    value: RangeValue,
    locale: str,
    *,
    apply_source_priors: bool,
    patterns: Sequence[Mapping] | None = None,
    measured: bool,
    placements_out: list[int] | None = None,
) -> list[SpokenAlternative]:
    """Every reading of a range, in P5's order (how far each is from the first reading
    of its end and from the first pattern), uncapped and not deduplicated.

    ``measured`` labels the ends as the range table counts them (``_measured_end``: R8,
    R11); otherwise the interim four-digit-year rule of #43 and #44 applies (a four-digit
    left end reads both plain ends as years first, ``_year_first``)."""
    if patterns is None:
        connectors = _lexical("range.connector", locale) or {}
        patterns = tuple(connectors.get(value.separator_class, ()))
    left, right = value.left[0], value.right[0]
    # The interim four-digit-year rule (``_year_first``) is the hyphen's, for two plain
    # numbers: a range with a unit or currency keeps its kind's order ("1990–95 km").
    plain = all(
        end.get("type") == "number:decimal" and not end.get("writes_unit") for end in (left, right)
    )
    year = not measured and plain and _written_digits(left) == 4
    left_unit, right_unit = bool(left.get("writes_unit")), bool(right.get("writes_unit"))
    placements = [(False, False)]
    if left_unit != right_unit:
        # In place (the bare end said bare), then the unit moved to the end.
        placements = list(dict.fromkeys([(not left_unit, not right_unit), (True, False)]))
    candidates: list[tuple[tuple[int, ...], SpokenAlternative]] = []
    for placement, (bare_left, bare_right) in enumerate(placements):
        if measured:
            ends = [
                _measured_end(end, locale, bare=bare, apply_source_priors=apply_source_priors)
                for end, bare in ((left, bare_left), (right, bare_right))
            ]
        else:
            ends = [
                _range_end(
                    end, locale, bare=bare, year=year, apply_source_priors=apply_source_priors
                )
                for end, bare in ((left, bare_left), (right, bare_right))
            ]
        for number, pattern in enumerate(patterns):
            template = str(pattern["pattern"])
            slots = pattern.get("slots") or {}
            if slots:
                filled = [
                    _fill_slot(end, slots[key], locale) if key in slots else None
                    for key, end in (("0", left), ("1", right))
                ]
                if any(item is None for item in filled):
                    continue
                source = f"range:cardinal+{pattern['id']}+cardinal"
                said = template.format(*(item.text for item in filled))
                candidates.append(
                    ((number + placement, number, placement, 0, 0), SpokenAlternative(said, source))
                )
                continue
            for i, (a, a_kind) in enumerate(ends[0]):
                for j, (b, b_kind) in enumerate(ends[1]):
                    said = template.format(a.text, b.text)
                    source = f"range:{a_kind}+{pattern['id']}+{b_kind}"
                    order = (i + j + number + placement, number, placement, i, j)
                    candidates.append((order, SpokenAlternative(said, source)))
    candidates.sort(key=lambda item: item[0])
    if placements_out is not None:
        placements_out.extend(order[2] for order, _ in candidates)
    return [item for _, item in candidates]


def _capped_by_placement(
    candidates: Sequence[SpokenAlternative], placements: Sequence[int]
) -> tuple[SpokenAlternative, ...]:
    """P5's order capped at ``_RANGE_CAP``, keeping each unit placement's first reading
    (it replaces the last readings the cap would keep)."""
    seen: set[str] = set()
    distinct = []
    for item, placement in zip(candidates, placements, strict=True):
        if item.text not in seen:
            seen.add(item.text)
            distinct.append((item, placement))
    leads = {}
    for index, (_, placement) in enumerate(distinct):
        leads.setdefault(placement, index)
    missing = [index for index in leads.values() if index >= _RANGE_CAP]
    kept = [i for i in range(min(len(distinct), _RANGE_CAP - len(missing)))] + missing
    return tuple(distinct[i][0] for i in sorted(kept))


def _money_range(value: RangeValue) -> bool:
    return any(
        str(end.get("type", "")).startswith("number:currency")
        for end in (*value.left, *value.right)
    )


def _spoken_range(
    value: RangeValue,
    locale: str,
    *,
    apply_source_priors: bool,
    patterns: Sequence[Mapping] | None = None,
    sub_key: str | None = None,
) -> tuple[SpokenAlternative, ...]:
    """Speak a range: each end's readings, joined by each connector pattern the locale's
    lexical table gives the range's class (``range.connector``: "{0} to {1}", "{0} {1}").

    - With the locale's range table (``frend.ranges.load_range_priors``): each end is
      labeled as the table counts it (R8, R11; ``_measured_end``), and with
      ``apply_source_priors`` the readings are stably sorted by J's share of their joint
      source at ``sub_key`` (``RangePriorTable.lookup``: blended toward the class row),
      each measured reading carrying its share as its weight; unmeasured sources follow
      in P5's order. A range with a currency end keeps P5's order (R10: the corpus says
      no money range to measure once its punctuation dashes are set aside). Builder mode
      (``apply_source_priors=False``) keeps P5's order and carries no weight.
    - With no table, #43's and #44's reading, unchanged: P5's order, and a four-digit
      left end reads both plain ends as years first (``_year_first``).
    - A unit or currency written on one end only ("$5–10") is said in place and also
      moved to the end ("five dollars to ten", "five to ten dollars").
    - A pattern slot with a case and a gender is said by ``_fill_slot``, or the pattern
      is not said.
    - Each reading's source is ``range:<left kind>+<connector id>+<right kind>``; at most
      ``_RANGE_CAP`` readings, after the sort.

    ``patterns`` replaces the locale's connector patterns (a test's fixture).
    """
    table = load_range_priors(locale)
    placements: list[int] = []
    candidates = _range_candidates(
        value,
        locale,
        apply_source_priors=apply_source_priors,
        patterns=patterns,
        measured=table is not None,
        placements_out=placements,
    )
    if table is not None and _money_range(value):
        # R10: both unit placements stay offered under the cap ("$5-10": "five dollars
        # to ten" and "five to ten dollars"), in P5's order.
        return _capped_by_placement(candidates, placements)
    if table is not None and apply_source_priors:
        cls = range_class_key(value.separator_class)
        weighted = []
        for item in candidates:
            measurement = table.lookup(cls, item.provenance, sub_key)
            weighted.append(
                item
                if measurement is None
                else SpokenAlternative(item.text, item.provenance, measurement.share)
            )
        candidates = weighted
    return _ranked(candidates)[:_RANGE_CAP]


def verbalize_edge(
    edge: ReadingEdge,
    *,
    source_text: str | None = None,
    locale: str = "en_US",
    supplements: CuratedSupplements | None = None,
    apply_source_priors: bool = True,
    context: TextContext | None = None,
    rerank_by_context: bool = True,
    context_threshold: float = CONTEXT_THRESHOLD,
    profile: str | None = None,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
    max_unit_chars: int | None = DEFAULT_MAX_UNIT_CHARS,
) -> VerbalizedUnit:
    """Validate the source text, then verbalize one edge."""
    validate_input(source_text, max_input_chars=max_input_chars)
    validate_input(context.text if context is not None else None, max_input_chars=max_input_chars)
    unit_length = len(source_text) if source_text is not None else edge.end
    validate_unit_length(
        unit_length,
        max_unit_chars=max_unit_chars,
        max_input_chars=max_input_chars,
    )
    if context is not None:
        validate_unit_length(len(context.text), max_unit_chars=max_unit_chars)
    return _verbalize_edge(
        edge,
        source_text=source_text,
        locale=locale,
        supplements=supplements,
        apply_source_priors=apply_source_priors,
        context=context,
        rerank_by_context=rerank_by_context,
        context_threshold=context_threshold,
        profile=profile,
    )


def _verbalize_edge(
    edge: ReadingEdge,
    *,
    source_text: str | None = None,
    locale: str = "en_US",
    supplements: CuratedSupplements | None = None,
    apply_source_priors: bool = True,
    context: TextContext | None = None,
    rerank_by_context: bool = True,
    context_threshold: float = CONTEXT_THRESHOLD,
    profile: str | None = None,
) -> VerbalizedUnit:
    """Verbalize one edge and optionally apply shipped source measurements.

    ``apply_source_priors=False`` is the measurement-builder mode. A builder
    cannot measure a source table through the ranking that table itself drives
    without making the measurement circular; it reads no context either.

    ``context`` is the running text around the edge (``frend.context``; by default
    ``source_text`` itself). A range separator between two numbers ("5 - 10", "5-10")
    is also offered the locale's range connector ("to"), and a context tree may put
    another reading first (``frend.context.rerank``, at ``context_threshold`` or more).
    ``rerank_by_context=False`` keeps frend's own order with the offers (the context
    trees' builder mode).
    """
    locale = canonical_locale(locale)
    profile = validate_profile(profile)
    if profile == GOOGLE_TN:
        _acronym_surface_priors(locale=locale)
    if context is None and source_text is not None:
        context = TextContext(source_text)
    elif context is not None and source_text is not None:
        placed = context.text[context.offset : context.offset + len(source_text)]
        if placed != source_text:
            raise ValueError("the context text does not hold the source text at its offset")
    if not apply_source_priors:
        context = None
    prior = edge.prior
    tier = prior.tier if prior is not None else None
    provenance = prior.provenance if prior is not None else None
    range_span: tuple[int, int] | None = None  # a separator between numbers, said "to"?
    if edge.kind == "passthrough":
        alternatives: tuple[SpokenAlternative, ...] = (
            SpokenAlternative(_surface(edge, source_text), "surface:passthrough"),
        )
        choice = None
        connector = _range_to(context, edge.start, edge.end, locale)
        if connector is not None:
            # A separator written inside a range ("10:30-11:45") may be said "to".
            alternatives = (*alternatives, _connector_reading(connector))
            if _is_ratio_separator(context, edge.start, edge.end, locale):
                alternatives = _lone_ratio_first(alternatives, locale)
            if rerank_by_context:
                alternatives, choice = _connector_first(
                    alternatives, context, edge.start, edge.end, locale, context_threshold
                )
        return VerbalizedUnit(edge.id, alternatives, tier, provenance, True, context=choice)
    detection = edge.detection
    if detection is None:
        raise ValueError(f"reading edge {edge.id!r} has no detection")
    type_ = str(detection["type"])
    value = detection["value"]
    try:
        # A range written in running text ("5-10", "16:79", "3x4"; ``RangeDetector``), or
        # one ICU writes ("1990–1995", "5–10 km", "May 3 – 5, 2020"): each end read as
        # that value alone, joined by the locale's connector patterns.
        ranged = detection if isinstance(value, RangeValue) else from_icukit(detection, locale)
        if ranged is not None:
            alternatives = _spoken_range(
                ranged["value"],
                locale,
                apply_source_priors=apply_source_priors,
                sub_key=range_sub_key(ranged),
            )
            if not alternatives:
                raise NotImplementedError(f"no spoken connector for {type_!r} in {locale!r}")
            key_value = str(detection.get("text", ""))
            path = "range"
        elif isinstance(value, NumberValue) and type_.startswith("ordinal:"):
            alternatives = _spoken_ordinal(value, detection, locale)
            key_value: object = value.decimal
            path = "ordinal"
        elif isinstance(value, NumberValue) and type_.startswith("number:plural"):
            alternatives = _spoken_plural(detection, locale)
            key_value = value.decimal
            path = "plural"
        elif isinstance(value, NumberValue) and type_.startswith("number:cardinal:roman"):
            alternatives = _roman_readings(
                _spoken_number(type_, value, detection, locale), value, detection, locale
            )
            key_value = value.decimal
            path = "roman"
        elif isinstance(value, NumberValue):
            alternatives = _spoken_number(type_, value, detection, locale)
            key_value = value.decimal
            path = "number"
        elif isinstance(value, AbbreviationValue):
            # A lexicon entry with no expansion ("J.R.R.", a sentence-break exception)
            # is still spelled when it is capitals; only a surface with neither fails.
            written = str(detection.get("text", value.surface))
            expansions = value.expansions or chain_expansions(written, locale)
            # A dotted chain in another case than the lexicon's ("E.G.") borrows its
            # expansions ("for example"), after the letters, as the corpus reads it.
            expanded = _expansion_alternatives(expansions) if expansions else ()
            if type_ == VARIANT_TYPE:
                # A written variant ("Mr", "st", "no") may be the word it spells, so its
                # expansions stand beside the token as written; the corpus ranks them.
                alternatives = (*expanded, SpokenAlternative(written, AS_WRITTEN_SOURCE))
                if measures_spelled(written, locale=locale):
                    # A key the corpus spells ("Lt", "ch", "Rt") also reads letter by
                    # letter; the table ranks the letters with the other readings.
                    if (letter_form := spelled(written.replace(".", ""), locale)) is not None:
                        alternatives = (
                            *alternatives,
                            SpokenAlternative(letter_form.text, SPELLED_SOURCE)
                            if locale == "en_US"
                            else letter_form,
                        )
            else:
                alternatives = _with_acronym_readings(written, expanded, locale, profile=profile)
                if is_chain(written) and not written.isupper():
                    # A dotted chain of any case is spelled ("e.g." "e g", "j.r.r." "j r
                    # r"): the corpus spells every one it writes.
                    if (letter_form := spelled(written.replace(".", ""), locale)) is not None:
                        alternatives = (
                            *alternatives,
                            SpokenAlternative(letter_form.text, SPELLED_SOURCE)
                            if locale == "en_US"
                            else letter_form,
                        )
            if not alternatives:
                raise NotImplementedError(f"no reading for {value.surface!r}")
            if apply_source_priors:
                alternatives = _abbreviation_ranked(written, alternatives, locale)
            key_value = value.surface
            path = "abbreviation"
        elif isinstance(value, DateTimeValue) and type_.startswith("date:"):
            alternatives = _spoken_date(value, detection, locale)
            key_value = tuple(value.fields)
            path = "date"
        elif isinstance(value, DateTimeValue) and type_.startswith("time:"):
            alternatives = _spoken_time(value, detection, locale)
            key_value = tuple(value.fields)
            path = "time"
        elif isinstance(value, MeasureValue) and type_.startswith("measure:duration"):
            alternatives = _spoken_duration(detection, locale)
            key_value = (value.decimal, value.unit)
            path = "duration"
        elif isinstance(value, MeasureValue) and "-and-" in type_:
            alternatives = _spoken_mixed_measure(detection, locale)
            key_value = (value.decimal, value.unit)
            path = "mixed-measure"
        elif isinstance(value, SymbolValue):
            # A standalone character reads by its names or as nothing; the corpus says
            # most punctuation, a lone dash and "風" as nothing, "&" as "and".
            alternatives = (
                *(SpokenAlternative(name, source) for name, source in value.names),
                SpokenAlternative("", "surface:silence"),
            )
            connector = _range_to(context, edge.start, edge.end, locale)
            if connector is not None:
                # A lone separator between two numbers ("5 - 10") may be said "to".
                alternatives = (*alternatives, _connector_reading(connector))
                range_span = (edge.start, edge.end)
            key_value = value.char
            path = "symbol"
        elif isinstance(value, ScriptRunValue):
            # A run of another script's letters is one reading: its ICU transliteration
            # as this locale reads it, its letters' names, as written, or nothing; the
            # script's measured shares rank them as they rank its lone letters.
            alternatives = (
                *(SpokenAlternative(name, source) for name, source in value.names),
                SpokenAlternative("", "surface:silence"),
            )
            key_value = value.text
            path = "symbol"
        elif isinstance(value, LettersValue):
            alternatives = _spoken_letters(value, locale, profile)
            if not value.suffix:
                # A run that is a lexicon abbreviation in capitals ("MR", "DR") is also
                # offered its expansions.
                expansions = upper_variant_expansions(value.letters, locale)
                alternatives = (*alternatives, *_expansion_alternatives(expansions))
                if (
                    expansions
                    and apply_source_priors
                    and measured_case(value.letters, locale=locale)
                ):
                    # Where the corpus measures the run in capitals ("MR": mister 562
                    # of 611), that row ranks the expansions and the letters; a reading
                    # it does not measure follows, its acronym weight dropped.
                    alternatives = _abbreviation_ranked(
                        value.letters,
                        tuple(SpokenAlternative(a.text, a.provenance) for a in alternatives),
                        locale,
                    )
            key_value = value.surface
            path = "letters"
        elif isinstance(value, DigitsValue):
            alternatives = _spoken_digits(value, locale)
            key_value = value.digits
            path = "digits"
        elif type(value).__name__ == "RelativeDateValue":
            alternatives = _spoken_relative(value, detection, locale)
            key_value = (value.offset, value.unit)
            path = "relative"
        elif type(value).__name__ == "UnitValue":
            alternatives = _spoken_unit(value, locale)
            key_value = value.unit
            path = "unit"
        elif isinstance(value, MeasureValue):
            alternatives = _spoken_measure(value, locale)
            key_value = (value.decimal, value.unit)
            path = "measure"
        elif isinstance(value, ElectronicValue):
            alternatives = _spoken_electronic(value, locale)
            key_value = value.parts
            path = "electronic"
        elif type(value).__name__ == "AlphanumericRunsValue":
            alternatives = _spoken_runs(value, locale)
            key_value = value.runs
            path = "runs"
        else:
            raise NotImplementedError(f"unsupported value struct {type(value).__name__}")
    except NotImplementedError:
        fallback = SpokenAlternative(_surface(edge, source_text), "surface:unsupported")
        return VerbalizedUnit(edge.id, (fallback,), tier, provenance, False)
    alternatives = _with_curated(alternatives, type_, key_value, supplements)
    if apply_source_priors and path != "range":
        # A range's readings are ranked within it (``_spoken_range``): each end by its
        # own kind's measured shares.
        kind = _measured_kind(type_, value)
        alternatives = _rank_final(
            alternatives, kind, measurement_sub_key(kind, detection, locale=locale), locale
        )
        if path == "date" and type_ == "date:y":
            alternatives = _bare_number_ranked(alternatives, detection, locale)
    weekday = _capture(detection, "weekday") if path == "date" else None
    if weekday is not None:
        # The weekday is written but not in the date's value; it leads every form,
        # after ranking, so curated keys and measured source shares are unchanged.
        day = _weekday_name(weekday, value.calendar, locale)
        alternatives = tuple(
            SpokenAlternative(
                f"{day.text}, {item.text}", f"{day.provenance}+{item.provenance}", item.weight
            )
            for item in alternatives
        )
    sign = _capture(detection, "sign") if path == "number" else None
    if (
        sign is not None
        and str(getattr(sign, "text", "")).strip() in _PLUS_SIGNS
        and _lexical("sign.plus", locale) is not None
    ):
        # A written plus is usually said and sometimes dropped: "plus five" leads, and
        # the unsigned form stays as the fallback. ICU spells no plus, so it is lexical.
        alternatives = (
            *(
                SpokenAlternative(
                    _lexical_pattern("sign.plus", locale, item.text),
                    f"{lexical_source(locale)}+{item.provenance}",
                    item.weight,
                )
                for item in alternatives
            ),
            *alternatives,
        )
    choice = None
    sign = _capture(detection, "sign") if path == "number" else None
    connector = (
        None
        if sign is None or not _may_end_a_range(type_)
        else _range_to(context, sign.start, sign.end, locale)
    )
    if range_span is not None:
        # One question for every separator between numbers, however written: the
        # separator's tree says whether "to" is said (a lone ":" has no tree; the range
        # table's ratio class orders it, R12).
        if _is_ratio_separator(context, *range_span, locale):
            alternatives = _lone_ratio_first(alternatives, locale)
        if rerank_by_context:
            alternatives, choice = _connector_first(
                alternatives, context, *range_span, locale, context_threshold
            )
    elif connector is not None:
        # A range written as one word ("5-10", "1990-1995"): the sign is the range's
        # separator, and the number its right end, said after the connector.
        right_end = _unsigned(
            type_, value, detection, locale, year_first=_left_end_digits(context, sign.start) == 4
        )
        alternatives = (*alternatives, *_connected(connector, right_end))
        if rerank_by_context:
            alternatives, choice = _connector_first(
                alternatives, context, sign.start, sign.end, locale, context_threshold
            )
    elif path == "range" and load_range_priors(locale) is not None:
        # A range's readings are one problem per separator class (``range:<class>``):
        # its tree may put another joint source first.
        if context is not None and rerank_by_context:
            alternatives, choice = rerank_range(
                alternatives,
                context.text,
                context.offset + edge.start,
                context.offset + edge.end,
                ranged["value"],
                locale=locale,
                bos=context.bos,
                eos=context.eos,
            )
    elif (
        context is not None
        and rerank_by_context
        and not (path == "date" and type_ == "date:y" and _bare_four_digit(detection, locale))
    ):
        alternatives, choice = rerank(
            alternatives,
            context.text,
            context.offset + edge.start,
            context.offset + edge.end,
            locale=locale,
            bos=context.bos,
            eos=context.eos,
            threshold=context_threshold,
        )
    return VerbalizedUnit(
        edge.id, alternatives, tier, provenance, True, _unspoken(detection, path), choice
    )


def _respelling_chunks(source: str, target: str) -> tuple[str, ...]:
    """Align ``target`` onto ``source`` code points while preserving unit geometry."""
    chunks = [""] * len(source)
    for tag, left_start, _left_end, right_start, right_end in SequenceMatcher(
        None, source, target
    ).get_opcodes():
        if tag == "equal":
            for offset, char in enumerate(target[right_start:right_end]):
                chunks[left_start + offset] = char
        elif tag == "replace":
            at = left_start if left_start < len(chunks) else len(chunks) - 1
            if at >= 0:
                chunks[at] += target[right_start:right_end]
        elif tag == "insert":
            at = max(0, left_start - 1)
            if chunks:
                chunks[at] += target[right_start:right_end]
    return tuple(chunks)


def _respell_passthrough_words(units, edge_ids, edges, source_text, table):
    """Respell complete alphabetic passthrough runs, leaving spoken readings untouched."""
    changed = list(units)
    at = 0
    while at < len(edge_ids):
        edge = edges[edge_ids[at]]
        if edge.kind != "passthrough" or not source_text[edge.start : edge.end].isalpha():
            at += 1
            continue
        end = at + 1
        last = edge.end
        while end < len(edge_ids):
            following = edges[edge_ids[end]]
            if (
                following.kind != "passthrough"
                or following.start != last
                or not source_text[following.start : following.end].isalpha()
            ):
                break
            last = following.end
            end += 1
        word = source_text[edge.start : last]
        if (edge.start and source_text[edge.start - 1].isalpha()) or (
            last < len(source_text) and source_text[last].isalpha()
        ):
            at = end
            continue
        converted = respell_from_table(word, table)
        if converted != word:
            for index, chunk in zip(
                range(at, end), _respelling_chunks(word, converted), strict=True
            ):
                changed[index] = dataclasses.replace(
                    changed[index],
                    alternatives=(SpokenAlternative(chunk, "surface:passthrough"),),
                )
        at = end
    return tuple(changed)


def verbalize_lattice(
    lattice: ReadingLattice,
    *,
    locale: str | None = None,
    supplements: CuratedSupplements | None = None,
    context: TextContext | None = None,
    profile: str | None = None,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
    max_unit_chars: int | None = DEFAULT_MAX_UNIT_CHARS,
    _input_validated: bool = False,
) -> VerbalizedLattice:
    """Verbalize every projected path without expanding alternatives across units.

    ``context`` is the running text the lattice's source text sits in (by default the
    source text itself): it is what the context trees read (``verbalize_edge``).
    """
    if not _input_validated:
        validate_input(lattice.source_text, max_input_chars=max_input_chars)
        validate_input(
            context.text if context is not None else None,
            max_input_chars=max_input_chars,
        )
    validate_unit_length(
        lattice.text_length,
        max_unit_chars=max_unit_chars,
        max_input_chars=max_input_chars,
    )
    if context is not None:
        validate_unit_length(len(context.text), max_unit_chars=max_unit_chars)
    effective = lattice.locale
    profile = validate_profile(profile)
    britishisms = None
    if profile == GOOGLE_TN:
        britishisms = _google_tn_britishisms(locale=effective)
    if locale is not None and canonical_locale(locale) != effective:
        raise ValueError(f"locale {locale!r} does not match lattice locale {effective!r}")
    edges = {edge.id: edge for edge in lattice.edges}
    if britishisms is None:
        paths = tuple(
            VerbalizedPath(
                path.rank,
                path.edge_ids,
                units := tuple(
                    _verbalize_edge(
                        edges[edge_id],
                        source_text=lattice.source_text,
                        locale=effective,
                        supplements=supplements,
                        context=context,
                        profile=profile,
                    )
                    for edge_id in path.edge_ids
                ),
                " ".join(unit.best.text for unit in units),
            )
            for path in lattice.paths
        )
        by_rank = {path.rank: path for path in paths}
        return VerbalizedLattice(
            paths, by_rank[lattice.best_path.rank], lattice.ambiguous, lattice.truncated
        )
    paths = []
    for path in lattice.paths:
        units = tuple(
            _verbalize_edge(
                edges[edge_id],
                source_text=lattice.source_text,
                locale=effective,
                supplements=supplements,
                context=context,
                profile=profile,
            )
            for edge_id in path.edge_ids
        )
        if britishisms is not None and lattice.source_text is not None:
            units = _respell_passthrough_words(
                units, path.edge_ids, edges, lattice.source_text, britishisms
            )
        paths.append(
            VerbalizedPath(
                path.rank,
                path.edge_ids,
                units,
                " ".join(unit.best.text for unit in units),
            )
        )
    paths = tuple(paths)
    by_rank = {path.rank: path for path in paths}
    return VerbalizedLattice(
        paths, by_rank[lattice.best_path.rank], lattice.ambiguous, lattice.truncated
    )
