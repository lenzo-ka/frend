"""Reflective spoken alternatives for a distilled reading lattice."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from functools import cache
from itertools import product

import icu
from icukit import AbbreviationValue, DateTimeFormatter
from icukit.detectors import DateTimeValue, MeasureValue, NumberValue
from icukit.measure import WIDTH_WIDE, format_measure

from frend.electronic import (
    ElectronicValue,
    digit_forms,
    digit_probabilities,
    letter_key,
    letter_probabilities,
    separator_names,
    tld_positions,
)
from frend.lattice import ReadingEdge, ReadingLattice
from frend.spoken_priors import measurement_sub_key, normalize_spoken, source_prior
from frend.written_forms import DigitsValue

__all__ = [
    "SpokenAlternative",
    "VerbalizedLattice",
    "VerbalizedPath",
    "VerbalizedUnit",
    "register_curated_alternative",
    "verbalize_edge",
    "verbalize_lattice",
]

LEXICAL_SOURCE = "lexical:en_US"

# CLDR exposes wide currency names and currency precision, but not these
# corpus-attested bare major names or minor-unit names. Keeping the missing English
# lexical facts together makes their locale boundary explicit.
_EN_CURRENCY_UNITS = {
    "USD": (("dollar", "dollars"), ("cent", "cents")),
    "EUR": (("euro", "euros"), ("cent", "cents")),
    "GBP": (("pound", "pounds"), ("pence", "pence")),
    "JPY": (("yen", "yen"), ("sen", "sen")),
    "CNY": (("yuan", "yuan"), ("fen", "fen")),
    "INR": (("rupee", "rupees"), ("paisa", "paise")),
    "CAD": (("dollar", "dollars"), ("cent", "cents")),
    "AUD": (("dollar", "dollars"), ("cent", "cents")),
    "KRW": (("won", "won"), ("jeon", "jeon")),
    "RUB": (("ruble", "rubles"), ("kopek", "kopeks")),
}

# CLDR's en_US display name is "US Dollar" rather than the corpus's
# region-expanded name.
_EN_USD_REGION_NAMES = ("united states dollar", "united states dollars")

# English RBNF has regular ordinal names such as "second" and "fourth", but
# carries neither of the corpus's conventional fraction words.
_EN_FRACTION_DENOMINATORS = {
    2: ("half", "halves"),
    4: ("quarter", "quarters"),
}


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
    """

    edge_id: str
    alternatives: tuple[SpokenAlternative, ...]
    tier: str | None
    provenance: str | None
    verbalized: bool
    unspoken: tuple[object, ...] = ()

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
    alternatives: Sequence[SpokenAlternative], kind: str | None, sub_key: str | None
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
            source_prior(kind, alternative.provenance, sub_key) if kind is not None else None
        )
        if measurement is not None:
            weighted = SpokenAlternative(
                alternative.text, alternative.provenance, measurement.share
            )
            ranked.append((1, -measurement.share, index, weighted))
            continue
        ranked.append((_source_tier(alternative.provenance), Decimal(0), index, alternative))
    ranked.sort(key=lambda item: item[:3])
    return tuple(item[3] for item in ranked)


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
    for digit in fractional_digits:
        words = list(_number_leaf(Decimal(digit), "cardinal", locale))
        if digit == "0":
            # ICU has no digit-reading rule that calls zero "o"; the corpus uses
            # that spelling for zero digits in decimals.
            words.append(SpokenAlternative("o", LEXICAL_SOURCE))
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
    irregular = _EN_FRACTION_DENOMINATORS.get(int(denominator))
    if irregular is not None:
        denominator_words.insert(
            0, SpokenAlternative(irregular[0 if singular else 1], LEXICAL_SOURCE)
        )
    fraction = _compose([numerator_words, _ranked(denominator_words)], "{} {}")
    over = _compose([numerator_words, _number_leaf(denominator, "cardinal", locale)], "{} over {}")
    if whole is not None:
        whole_words = _number_leaf(whole, "cardinal", locale)
        mixed_fraction = list(fraction)
        if numerator == 1 and irregular is not None:
            mixed_fraction.insert(0, SpokenAlternative(f"a {irregular[0]}", LEXICAL_SOURCE))
        mixed = _compose([whole_words, _ranked(mixed_fraction)], "{} and {}")
        mixed_over = _compose([whole_words, over], "{} and {}")
        return _ranked([*mixed, *mixed_over])
    return _ranked([*fraction, *over])


def _currency_unit_name(currency: str, amount: Decimal, minor: bool) -> SpokenAlternative:
    try:
        names = _EN_CURRENCY_UNITS[currency][minor]
    except KeyError as exc:
        raise NotImplementedError(f"no bare English currency terms for {currency}") from exc
    plural = (
        amount != amount.to_integral_value() or _plural_rules("en_US").select(int(amount)) != "one"
    )
    return SpokenAlternative(names[plural], LEXICAL_SOURCE)


def _currency_wide_names(
    currency: str, amount: Decimal, locale: str
) -> tuple[SpokenAlternative, ...]:
    plural = amount != 1
    names = [SpokenAlternative(_currency_name(currency, plural, locale), "icu-measure:wide")]
    if currency == "USD":
        names.append(SpokenAlternative(_EN_USD_REGION_NAMES[plural], LEXICAL_SOURCE))
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
    lexical_units = _EN_CURRENCY_UNITS.get(currency)
    wide_names = _currency_wide_names(currency, absolute, locale)
    if scaled != scaled.to_integral_value() or (absolute != major and lexical_units is None):
        names = list(wide_names)
        if lexical_units is not None:
            names.insert(0, _currency_unit_name(currency, absolute, False))
        return _compose([_spoken_decimal(value, locale), names], "{} {}")
    minor = int(scaled) % int(scale) if fraction_digits else 0
    major_words = tuple(
        SpokenAlternative(item.text.replace("-", " "), item.provenance)
        for item in _signed_integer_leaf(Decimal(major), negative, locale)
    )
    alternatives = list(_compose([major_words, wide_names], "{} {}"))
    if lexical_units is None:
        return _ranked(alternatives)
    bare_name = _currency_unit_name(currency, Decimal(major), False)
    if minor:
        minor_words = tuple(
            SpokenAlternative(item.text.replace("-", " "), item.provenance)
            for item in _number_leaf(Decimal(minor), "cardinal", locale)
        )
        minor_name = _currency_unit_name(currency, Decimal(minor), True)
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
    "electronic": frozenset({"digits", "letters", "separator"}),
    "measure": frozenset({"integer", "decimal-separator", "fraction", "unit"}),
    "mixed-measure": frozenset({"integer", "unit"}),
    "duration": frozenset({"h", "m", "s", "decimal-separator", "fraction"}),
    "unit": frozenset({"unit"}),
    "roman": frozenset({"integer", "apostrophe", "suffix"}),
}
_MINUS_SIGNS = frozenset({"-", "−"})
_PLUS_SIGNS = frozenset({"+"})


_OCLOCK = SpokenAlternative("o'clock", LEXICAL_SOURCE)
_HUNDRED = SpokenAlternative("hundred", LEXICAL_SOURCE)


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
            tail.append((spelled, *expansions))
    minute = fields.get("m")
    forms: list[SpokenAlternative] = []
    if minute:
        if minute < 10:
            digits = _number_leaf(Decimal(minute), "cardinal", locale)
            minutes = tuple(
                SpokenAlternative(f"{zero} {item.text}", f"{LEXICAL_SOURCE}+{item.provenance}")
                for zero in ("oh", "o")
                for item in digits
            )
        else:
            minutes = _number_leaf(Decimal(minute), "cardinal", locale)
        parts = [hours, minutes, *tail]
        forms.extend(_compose(parts, " ".join("{}" for _ in parts)))
    else:
        for parts in ([hours, *tail], [hours, (_OCLOCK,), *tail]):
            forms.extend(_compose(parts, " ".join("{}" for _ in parts)))
        if minute == 0 and period is None:
            # A written on-the-hour 24-hour time: the corpus reads "20:00" "twenty hundred".
            parts = [hours, (_HUNDRED,), *tail]
            forms.extend(_compose(parts, " ".join("{}" for _ in parts)))
    return _ranked(forms)


def _plural_word(word: str) -> str:
    """English plural of one number word; CLDR has no plural of a numeral, so lexical."""
    if word.endswith("y") and word[-2:-1] not in ("a", "e", "i", "o", "u"):
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


def _plural_phrase(text: str) -> str:
    head, space, last = text.rpartition(" ")
    stem, hyphen, final = last.rpartition("-")
    return f"{head}{space}{stem}{hyphen}{_plural_word(final)}"


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
            SpokenAlternative(_plural_phrase(item.text), f"{item.provenance}+{LEXICAL_SOURCE}")
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
            parts.append((SpokenAlternative(" ".join(text.lower()), "surface:letters"),))
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
    forms = [
        *alternatives,
        *ordinals,
        *(
            SpokenAlternative(f"the {item.text}", f"{LEXICAL_SOURCE}+{item.provenance}")
            for item in ordinals
        ),
    ]
    if _capture(detection, "suffix") is not None:
        forms = [
            SpokenAlternative(f"{item.text}'s", f"{item.provenance}+{LEXICAL_SOURCE}", item.weight)
            for item in forms
        ]
    return _ranked(forms)


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
    return _ranked(
        [
            *ordinals,
            *(
                SpokenAlternative(f"the {item.text}", f"{LEXICAL_SOURCE}+{item.provenance}")
                for item in ordinals
            ),
        ]
    )


# A spell-out ("MD" read "M D") names each letter; an expansion reads the text as words.
# The source says which, so a consumer can tell spelled letters from a written long form.
_ABBREVIATION_SOURCES = {"expansion": "icukit-abbreviation", "spell-out": "icukit-spell-out"}


@cache
def _acronym_priors() -> dict[str, dict[str, int]]:
    from importlib.resources import files

    data = files("frend").joinpath("data/acronym_priors.json").read_text(encoding="utf-8")
    return json.loads(data)["keys"]


def _with_acronym_readings(
    surface: str, alternatives: tuple[SpokenAlternative, ...]
) -> tuple[SpokenAlternative, ...]:
    """An acronym ("FBI", "NASA") also reads spelled and as a word, weighted as measured.

    kal ruled that frend says both: the share the corpus spells an all-capitals token of
    this shape (``letter_key``: length, vowel) weights "f b i", the rest weights "fbi"
    (``data/acronym_priors.json``, ``tools/build_acronym_priors.py``). icukit's long forms
    follow. A spelled form icukit already gives ("M D") takes the weight, not a copy.
    """
    letters = "".join(ch for ch in surface if ch.isalpha())
    if len(letters) < 2 or not letters.isupper():
        return alternatives
    table = _acronym_priors()
    overall = table["*"]
    shaped = table.get(letter_key(letters), {})
    total = sum(shaped.values()) + 5
    share = (
        Decimal(shaped.get("spelled", 0)) + 5 * Decimal(overall["spelled"]) / sum(overall.values())
    ) / total
    spelled = SpokenAlternative(" ".join(letters.lower()), "measured:acronym-spelled", share)
    word = SpokenAlternative(letters.lower(), "measured:acronym-word", 1 - share)
    measured = {normalize_spoken(form.text): form for form in (spelled, word)}
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
            for expansion in value.expansions
        ]
    )


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
    letters = "".join(ch for ch in text if ch.isalpha()).lower()
    variant_short, variant_wide = _era_variants(locale)
    wide = SpokenAlternative(names[era], "icu-datetime:GGGG")
    if era < len(variant_short) and era < len(variant_wide):
        if text.casefold() == variant_short[era].casefold():
            wide = SpokenAlternative(variant_wide[era], "icu-datetime:GGGG%variant")
    return (SpokenAlternative(" ".join(letters), "surface:letters"), wide)


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
    corpus reads it "o" as often. No locale data spells "o", so that form is lexical
    over ICU's, as a zero-led minute already is.
    """
    forms = list(_number_leaf(value, "year", locale))
    for item in tuple(forms):
        words = item.text.replace("-", " ").split(" ")
        if "oh" in words:
            text = " ".join("o" if word == "oh" else word for word in words)
            forms.append(SpokenAlternative(text, f"{item.provenance}+{LEXICAL_SOURCE}"))
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
            dates = _spoken_date(DateTimeValue(ymd, value.calendar), detection, locale)
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
                SpokenAlternative(f"{glue} {item.text}", f"{LEXICAL_SOURCE}+{item.provenance}")
                for item in times
            )
        pieces.append((start("H", 10**6), times))
    if not pieces:
        raise NotImplementedError(f"no speakable date part in {value.fields!r}")
    ordered = [alternatives for _, alternatives in sorted(pieces, key=lambda piece: piece[0])]
    return _compose(ordered, " ".join("{}" for _ in ordered))


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
        if value.currency in _EN_CURRENCY_UNITS:
            names.insert(0, _currency_unit_name(value.currency, decimal, False))
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
        return _number_leaf(decimal, "cardinal", locale)
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
    if value.unit.startswith("per-"):
        base = value.unit.removeprefix("per-")
        singular = _measure_template(Decimal(1), base, locale).replace("{}", "").strip()
        plural = _measure_template(amount, base, locale).replace("{}", "").strip()
        if singular != plural and head.count(singular) == 1:
            templates.append((head.replace(singular, plural), f"icu-measure:wide+{LEXICAL_SOURCE}"))
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
        if last_unit == "second":
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
                        f"{body} and {texts[-1]}", f"{source}+icu-list:units+{LEXICAL_SOURCE}"
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
# no spoken name for it (ICU's character name is COMMERCIAL AT). "at" is lexical.
_UNMEASURED_SEPARATORS = {"@": "at"}


def _spoken_electronic(value: ElectronicValue, locale: str) -> tuple[SpokenAlternative, ...]:
    """Speak a URL, email address or domain run by run, as the corpus is measured to.

    Each letter run reads as a word or spelled, each digit run as one of its readings
    (ICU's cardinal or year, or digit by digit), each separator by its spoken name, all
    with probabilities from ``data/electronic_priors.json``. The runs multiply, so the
    best readings by their product are kept (``ELECTRONIC_BEAM``), each weighted by it.
    A separator the corpus never names is not verbalized, except "@" (see
    ``_UNMEASURED_SEPARATORS``).
    """
    tlds = tld_positions(value.parts)
    unmeasured = False
    choices: list[dict[str, Decimal]] = []
    for index, (kind, text) in enumerate(value.parts):
        options: dict[str, Decimal] = {}
        if kind == "letters":
            lower = text.lower()
            probabilities = letter_probabilities(text, tld=index in tlds)
            if len(lower) == 1:
                options[lower] = Decimal(1)
            else:
                options[lower] = probabilities.get("word", Decimal(0))
                spelled = " ".join(lower)
                options[spelled] = options.get(spelled, Decimal(0)) + probabilities.get(
                    "spelled", Decimal(0)
                )
        elif kind == "digits":
            probabilities = digit_probabilities(text)
            for form, readings in digit_forms(text, locale).items():
                spoken = readings[0][0]
                options[spoken] = options.get(spoken, Decimal(0)) + probabilities.get(
                    form, Decimal(0)
                )
        else:
            names = {name: share for name, share in separator_names(text).items() if name != "sil"}
            if not names and text in _UNMEASURED_SEPARATORS:
                names = {_UNMEASURED_SEPARATORS[text]: Decimal(1)}
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
    source = f"{ELECTRONIC_SOURCE}+{LEXICAL_SOURCE}" if unmeasured else ELECTRONIC_SOURCE
    return tuple(
        SpokenAlternative(" ".join(words), source, probability) for probability, words in beam
    )


def _spoken_digits(value: DigitsValue, locale: str) -> tuple[SpokenAlternative, ...]:
    """Say spaced digits one by one by ICU's cardinal ("6 3" -> "six three"); a zero is
    also "o", which ICU has no rule for, so that form is lexical."""
    words = [_number_leaf(Decimal(digit), "cardinal", locale)[0].text for digit in value.digits]
    forms = [SpokenAlternative(" ".join(words), "icu-rbnf:%spellout-cardinal")]
    if "0" in value.digits:
        spoken = " ".join("o" if d == "0" else w for d, w in zip(value.digits, words, strict=True))
        forms.append(SpokenAlternative(spoken, f"icu-rbnf:%spellout-cardinal+{LEXICAL_SOURCE}"))
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


def verbalize_edge(
    edge: ReadingEdge,
    *,
    source_text: str | None = None,
    locale: str = "en_US",
    supplements: CuratedSupplements | None = None,
    apply_source_priors: bool = True,
) -> VerbalizedUnit:
    """Verbalize one edge and optionally apply shipped source measurements.

    ``apply_source_priors=False`` is the measurement-builder mode. A builder
    cannot measure a source table through the ranking that table itself drives
    without making the measurement circular.
    """
    if locale != "en_US":
        raise NotImplementedError(f"verbalization v1 supports only en_US, got {locale!r}")
    prior = edge.prior
    tier = prior.tier if prior is not None else None
    provenance = prior.provenance if prior is not None else None
    if edge.kind == "passthrough":
        alternative = SpokenAlternative(_surface(edge, source_text), "surface:passthrough")
        return VerbalizedUnit(edge.id, (alternative,), tier, provenance, True)
    detection = edge.detection
    if detection is None:
        raise ValueError(f"reading edge {edge.id!r} has no detection")
    type_ = str(detection["type"])
    value = detection["value"]
    try:
        if isinstance(value, NumberValue) and type_.startswith("ordinal:"):
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
            alternatives = _with_acronym_readings(value.surface, _spoken_abbreviation(value))
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
    if apply_source_priors:
        kind = _measured_kind(type_, value)
        alternatives = _rank_final(alternatives, kind, measurement_sub_key(kind, detection))
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
    if sign is not None and str(getattr(sign, "text", "")).strip() in _PLUS_SIGNS:
        # A written plus is usually said and sometimes dropped: "plus five" leads, and
        # the unsigned form stays as the fallback. ICU spells no plus, so it is lexical.
        alternatives = (
            *(
                SpokenAlternative(
                    f"plus {item.text}", f"{LEXICAL_SOURCE}+{item.provenance}", item.weight
                )
                for item in alternatives
            ),
            *alternatives,
        )
    return VerbalizedUnit(edge.id, alternatives, tier, provenance, True, _unspoken(detection, path))


def verbalize_lattice(
    lattice: ReadingLattice,
    *,
    locale: str = "en_US",
    supplements: CuratedSupplements | None = None,
) -> VerbalizedLattice:
    """Verbalize every projected path without expanding alternatives across units."""
    edges = {edge.id: edge for edge in lattice.edges}
    paths = tuple(
        VerbalizedPath(
            path.rank,
            path.edge_ids,
            units := tuple(
                verbalize_edge(
                    edges[edge_id],
                    source_text=lattice.source_text,
                    locale=locale,
                    supplements=supplements,
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
