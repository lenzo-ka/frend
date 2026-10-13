"""Sourced clock composition through precompiled tiergraph rule projections."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import cache
from importlib.resources import files
from itertools import product
from types import MappingProxyType

import icu
from icukit.detectors import DateTimeValue
from tiergraph import Realization

from frend.generated_rule_locales import locales_for
from frend.locale_data import canonical_locale
from frend.normalization_components import load_runtime_records

_LOCALES = locales_for("time")
_SHAPES = ("H", "HM", "HMS", "PH", "PHM", "PHMS")
_EVENTS = {"load": 0, "lower": 0, "generate": 0}
_WRITTEN_UHR_MARKER = "_frend_written_uhr"


def compose_german_written_uhr(detections, source_text: str, locale: str) -> list:
    """Extend a German hour-minute reading across its following written ``Uhr``."""
    detections = list(detections)
    if canonical_locale(locale) != "de_DE":
        return detections
    composed = []
    suffix = " Uhr"
    for detection in detections:
        value = detection.get("value")
        if (
            detection.get("type") != "time:flexible"
            or not isinstance(value, DateTimeValue)
            or tuple(name for name, _ in value.fields) != ("H", "m")
        ):
            continue
        end = int(detection["end"])
        extended_end = end + len(suffix)
        if source_text[end:extended_end] != suffix or (
            extended_end < len(source_text) and source_text[extended_end].isalnum()
        ):
            continue
        extended = dict(detection)
        extended["end"] = extended_end
        extended["text"] = source_text[int(detection["start"]) : extended_end]
        extended[_WRITTEN_UHR_MARKER] = True
        composed.append(extended)
    return [*detections, *composed]


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def time_rule_events() -> Mapping[str, int]:
    return MappingProxyType(dict(_EVENTS))


@dataclass(frozen=True)
class TimeRuleRecipe:
    tokens: tuple[str, ...]
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class TimeRuleBundle:
    locale: str
    recipes: Mapping[str, tuple[TimeRuleRecipe, ...]]
    periods: Mapping[int, str]
    records: Mapping[str, Mapping[str, object]]


@cache
def load_time_rule_bundle(locale: str) -> TimeRuleBundle:
    _EVENTS["load"] += 1
    locale = canonical_locale(locale)
    if locale not in _LOCALES:
        raise ValueError(f"no generated time-rule bundle for {locale!r}")
    resource = files("frend").joinpath("data", "time_rules", f"{locale}.json")
    document = json.loads(resource.read_text(encoding="utf-8"))
    required = {
        "schema",
        "provenance",
        "locale",
        "icu_version",
        "normalization_sha256",
        "projection_sha256",
        "recipes",
        "periods",
    }
    if set(document) != required or document.get("schema") != 3 or document.get("locale") != locale:
        raise ValueError(f"time-rule bundle identity mismatch for {locale!r}")
    if document.get("icu_version") != icu.ICU_VERSION:
        raise ValueError(
            f"time-rule bundle {locale!r} was built with ICU {document.get('icu_version')}, "
            f"not runtime ICU {icu.ICU_VERSION}"
        )
    projection = {key: document.get(key) for key in ("recipes", "periods")}
    if (
        document.get("projection_sha256")
        != hashlib.sha256(_canonical_bytes(projection)).hexdigest()
    ):
        raise ValueError(f"time-rule bundle {locale!r} projection hash mismatch")
    raw_recipes = document.get("recipes")
    if not isinstance(raw_recipes, Mapping) or set(raw_recipes) != set(_SHAPES):
        raise ValueError(f"time-rule bundle {locale!r} has incomplete compiled recipes")
    observed = {
        shape: tuple(
            TimeRuleRecipe(tuple(item["tokens"]), tuple(item["provenance"]))
            for item in raw_recipes[shape]
        )
        for shape in _SHAPES
    }
    if any(len(items) not in {0, 1} for items in observed.values()):
        raise ValueError(f"time-rule bundle {locale!r} has duplicate recipes")
    records = load_runtime_records(
        f"time_{locale}.json", locale, document.get("normalization_sha256")
    )
    periods = document.get("periods")
    if not isinstance(periods, Mapping) or set(periods) != {str(hour) for hour in range(24)}:
        raise ValueError(f"time-rule bundle {locale!r} has a non-total period selector")
    return TimeRuleBundle(
        locale,
        MappingProxyType(observed),
        MappingProxyType({int(hour): str(value) for hour, value in periods.items()}),
        MappingProxyType(records),
    )


def _selected_number(value: int, locale: str, *, gender: str | None, korean_hour: bool):
    from frend.verbalize import _number_leaf

    forms = _number_leaf(Decimal(value), "cardinal", locale)
    marker = None
    if korean_hour:
        marker = "%spellout-cardinal-native-attributive"
    elif value == 1 and gender in {"feminine", "masculine", "neuter"}:
        marker = f"%spellout-cardinal-{gender}"
    if marker is not None:
        selected = tuple(
            item
            for item in forms
            if any(part.endswith(f":{marker}") for part in item.provenance.split("+"))
        )
        if selected:
            forms = selected
    normal = tuple(
        item
        for item in forms
        if any(part.endswith(":%spellout-numbering") for part in item.provenance.split("+"))
    )
    if marker is None and normal:
        forms = normal
    return tuple(Realization((item.text,), (item.provenance,), item.weight) for item in forms[:1])


def _record_value(bundle: TimeRuleBundle, identifier: str) -> str:
    record = bundle.records.get(identifier)
    forms = None if record is None else record.get("forms")
    if not isinstance(forms, Mapping) or list(forms) != ["default"]:
        raise ValueError(f"time record {identifier!r} has no single default form")
    return str(forms["default"])


def _unit_realization(bundle: TimeRuleBundle, symbol: str, value: int) -> Realization:
    category = str(icu.PluralRules.forLocale(icu.Locale(bundle.locale)).select(value))
    identifier = f"time-unit-{symbol.lower()}-{category}"
    text = _record_value(bundle, identifier)
    return Realization((text,), (f"normalization-record:{identifier}",), None)


def _capture_integer(detection: Mapping[str, object], name: str) -> int | None:
    from frend.verbalize import _capture_integer as capture_integer

    value = capture_integer(detection, name)
    return None if value is None else int(value)


def _period_realization(bundle: TimeRuleBundle, hour: int) -> Realization:
    text = bundle.periods[hour]
    record = next(
        item
        for item in bundle.records.values()
        if item.get("kind") == "time-period"
        and item.get("forms") == {"default": text}
        and hour in item.get("hours", ())
    )
    return Realization((text,), (f"normalization-record:{record['id']}",), None)


def _field_realizations(
    bundle: TimeRuleBundle,
    symbol: str,
    values: Mapping[str, int],
    *,
    period: bool,
    semantic_hour: int,
) -> tuple[Realization, ...]:
    if symbol == "P":
        return (_period_realization(bundle, semantic_hour),)
    if symbol in {"UH", "UM", "US"}:
        return (_unit_realization(bundle, symbol, values[symbol[-1]]),)
    gender = {"H": "feminine", "M": "masculine", "S": "masculine"}[symbol]
    if bundle.locale == "fr_FR" and symbol in {"M", "S"}:
        gender = "feminine"
    if bundle.locale == "de_DE" and symbol == "H":
        gender = "neuter"
    return _selected_number(
        values[symbol],
        bundle.locale,
        gender=gender,
        korean_hour=bundle.locale == "ko_KR" and symbol == "H" and period,
    )


def _preserve_unchanged_ranking(generated, previous):
    """Keep every legacy reading in exact order/metadata, then add genuinely new text."""
    remaining = list(generated)
    preserved = []
    for old in previous:
        match = next((item for item in remaining if item.text == old.text), None)
        if match is None:
            preserved.append(old)
            continue
        remaining.remove(match)
        preserved.append(
            replace(
                match,
                weight=old.weight,
                prior_provenance=old.prior_provenance or old.provenance,
            )
        )
    return tuple((*preserved, *remaining))


def generate_time_alternatives(
    value: DateTimeValue,
    detection: Mapping[str, object],
    locale: str,
    *,
    max_derivations: int = 128,
):
    """Generate a complete sourced clock reading, refusing incomplete captures."""
    from frend.verbalize import SpokenAlternative, _legacy_spoken_time, _time_tail

    bundle = load_time_rule_bundle(locale)
    values = dict(value.fields)
    if len(values) != len(value.fields) or "H" not in values or not set(values) <= {"H", "m", "s"}:
        raise NotImplementedError("time rules require unique H with optional m and s")
    normalized = {"H": int(values["H"]), "M": int(values.get("m", 0)), "S": int(values.get("s", 0))}
    if (
        not 0 <= normalized["H"] <= 23
        or not 0 <= normalized["M"] <= 59
        or not 0 <= normalized["S"] <= 59
    ):
        raise NotImplementedError("time fields are outside the clock domain")
    has_period = any(
        getattr(capture, "name", None) == "day-period" for capture in detection.get("captures", ())
    )
    if has_period:
        written_hour = _capture_integer(detection, "H")
        semantic = normalized["H"]
        if (
            written_hour is None
            or not 1 <= written_hour <= 12
            or written_hour % 12 != semantic % 12
        ):
            raise NotImplementedError("period-bearing time has no congruent written 1..12 hour")
        realized_hour = written_hour
    else:
        realized_hour = normalized["H"]
    realized = dict(normalized)
    realized["H"] = realized_hour
    shape = ("P" if has_period else "") + "H"
    written_uhr = detection.get(_WRITTEN_UHR_MARKER) is True
    if written_uhr and bundle.locale != "de_DE":
        raise NotImplementedError("written Uhr composition is German-only")
    if "m" in values and not (written_uhr and normalized["M"] == 0):
        shape += "M"
    if "s" in values:
        shape += "S"
    recipes = bundle.recipes.get(shape, ())
    if not recipes:
        raise NotImplementedError(f"time rules have no sourced {shape} recipe for {bundle.locale}")
    recipe = recipes[0]
    symbols = [token[1:-1] for token in recipe.tokens if token.startswith("\ufff0")]
    realizations = [
        _field_realizations(
            bundle,
            symbol,
            realized,
            period=has_period,
            semantic_hour=normalized["H"],
        )
        for symbol in symbols
    ]
    tails = _time_tail(detection, bundle.locale)
    exact_count = math.prod(len(items) for items in realizations) * max(1, len(tails))
    if exact_count < 1 or exact_count > max_derivations:
        raise ValueError(f"time generation for {bundle.locale} has {exact_count} derivations")
    alternatives = []
    for selected in product(*realizations):
        by_symbol = dict(zip(symbols, selected, strict=True))
        rendered = []
        provenance = [f"time-rule:{bundle.locale}", *recipe.provenance]
        weights = []
        prior = []
        for token in recipe.tokens:
            symbol = token[1:-1] if token.startswith("\ufff0") else None
            realization = by_symbol.get(symbol)
            if realization is None:
                rendered.append(token)
                continue
            rendered.extend(realization.tokens)
            provenance.extend(realization.provenance)
            prior.extend(realization.provenance[:1])
            weights.append(realization.weight)
        core = "".join(rendered)
        tail_options = tails or (((), (), None),)
        for tail_tokens, tail_provenance, tail_weight in tail_options:
            text = " ".join(item for item in (core, *tail_tokens) if item)
            sources = tuple(dict.fromkeys((*provenance, *tail_provenance)))
            all_weights = (*weights, tail_weight)
            alternatives.append(
                SpokenAlternative(
                    text,
                    "+".join(sources),
                    sum((item for item in all_weights if item is not None), Decimal(0))
                    if all_weights and all(item is not None for item in all_weights)
                    else None,
                    prior_provenance="+".join(dict.fromkeys(prior)) or None,
                )
            )
    if written_uhr:
        return tuple(alternatives)
    try:
        previous = _legacy_spoken_time(value, detection, bundle.locale)
    except NotImplementedError:
        previous = ()
    return _preserve_unchanged_ranking(alternatives, previous)
