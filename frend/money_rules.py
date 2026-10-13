"""Sourced MONEY composition through precompiled tiergraph rule projections."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from importlib.resources import files
from types import MappingProxyType

import icu
from icukit.detectors import NumberValue
from tiergraph import (
    GrammarInput,
    GrammarInputToken,
    Realization,
)

from frend.generated_rule_locales import locales_for
from frend.locale_data import canonical_locale
from frend.normalization_components import load_runtime_records
from frend.rule_composition import (
    FieldProvider,
    RuleRecipe,
    compose_recipes,
    exact_composition_count,
    preserve_legacy_ranking,
)

_LOCALES = locales_for("money")
_EVENTS = {"load": 0, "lower": 0, "generate": 0}


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def money_rule_events() -> Mapping[str, int]:
    return MappingProxyType(dict(_EVENTS))


@dataclass(frozen=True)
class MoneyRuleBundle:
    locale: str
    recipes: Mapping[str, tuple[RuleRecipe, ...]]
    records: Mapping[str, Mapping[str, object]]


@cache
def load_money_rule_bundle(locale: str) -> MoneyRuleBundle:
    _EVENTS["load"] += 1
    locale = canonical_locale(locale)
    if locale not in _LOCALES:
        raise ValueError(f"no generated money-rule bundle for {locale!r}")
    resource = files("frend").joinpath("data", "money_rules", f"{locale}.json")
    document = json.loads(resource.read_text(encoding="utf-8"))
    required = {
        "schema",
        "provenance",
        "locale",
        "icu_version",
        "normalization_sha256",
        "projection_sha256",
        "recipes",
    }
    if set(document) != required or document.get("schema") != 3 or document.get("locale") != locale:
        raise ValueError(f"money-rule bundle identity mismatch for {locale!r}")
    if document.get("icu_version") != icu.ICU_VERSION:
        raise ValueError(
            f"money-rule bundle {locale!r} was built with ICU "
            f"{document.get('icu_version')}, not runtime ICU {icu.ICU_VERSION}"
        )
    projection = {"recipes": document.get("recipes")}
    projection_hash = hashlib.sha256(_canonical_bytes(projection)).hexdigest()
    if document.get("projection_sha256") != projection_hash:
        raise ValueError(f"money-rule bundle {locale!r} projection hash mismatch")
    raw = document.get("recipes")
    if not isinstance(raw, Mapping):
        raise ValueError(f"money-rule bundle {locale!r} has invalid recipes")
    observed = {
        key: tuple(RuleRecipe(tuple(item["tokens"]), tuple(item["provenance"])) for item in values)
        for key, values in raw.items()
    }
    if any(len(values) != 1 for values in observed.values()):
        raise ValueError(f"money-rule bundle {locale!r} has duplicate recipes")
    records = load_runtime_records(
        f"money_{locale}.json", locale, document.get("normalization_sha256")
    )
    return MoneyRuleBundle(locale, MappingProxyType(observed), records)


def _capture(detection: Mapping[str, object], name: str) -> object | None:
    return next(
        (item for item in detection.get("captures", ()) if getattr(item, "name", None) == name),
        None,
    )


def _fraction_digits(currency: str, locale: str) -> int:
    formatter = icu.NumberFormat.createCurrencyInstance(icu.Locale(locale))
    formatter.setCurrency(currency)
    return formatter.getMaximumFractionDigits()


def _shape(value: NumberValue, detection: Mapping[str, object], bundle: MoneyRuleBundle) -> str:
    currency = value.currency
    if currency is None:
        raise NotImplementedError("money rules require a captured ISO currency code")
    if _capture(detection, "compact") is not None:
        return "compact_major"
    amount = abs(Decimal(value.decimal))
    digits = _fraction_digits(currency, bundle.locale)
    scaled = amount * (Decimal(10) ** digits)
    if scaled != scaled.to_integral_value():
        return "major"
    minor = int(scaled) % (10**digits) if digits else 0
    if not minor:
        return "major"
    if not int(amount) and f"minor_only:{currency}" in bundle.recipes:
        return "minor_only"
    return "major_minor"


def money_grammar_input(
    value: NumberValue, detection: Mapping[str, object], locale: str
) -> GrammarInput:
    """Return the exact typed tiergraph input routed for a complete money edge."""
    bundle = load_money_rule_bundle(locale)
    shape = _shape(value, detection, bundle)
    currency = value.currency
    key = f"{shape}:{currency}"
    recipes = bundle.recipes.get(key, ())
    if not recipes:
        raise NotImplementedError(f"money rules have no sourced {key} recipe for {bundle.locale}")
    symbols = tuple(token[1:-1] for token in recipes[0].tokens if token.startswith("\ufff0"))
    return GrammarInput(
        tuple(
            GrammarInputToken(
                f"FIELD:{symbol}",
                (Realization((symbol,), (f"money-input:{symbol}",), None),),
                (f"money-field:{symbol}",),
            )
            for symbol in symbols
        )
    )


def _number_realizations(
    amount: Decimal, locale: str, record: Mapping[str, object] | None = None
) -> tuple[Realization, ...]:
    from frend.verbalize import _number_leaf, _spoken_decimal

    if amount != amount.to_integral_value():
        forms = _spoken_decimal(amount, locale)
    else:
        forms = _number_leaf(amount, "cardinal", locale)
    marker = None
    if record is not None and record.get("rbnf_variant") == "sinokorean":
        marker = "%spellout-cardinal-sinokorean"
    elif record is not None and record.get("gender") in {"masculine", "feminine", "neuter"}:
        marker = f"%spellout-cardinal-{record['gender']}"
    if marker is not None:
        selected = tuple(
            item
            for item in forms
            if any(part.endswith(f":{marker}") for part in item.provenance.split("+"))
        )
        if selected:
            forms = selected
    return tuple(
        Realization((item.text.replace("-", " "),), (item.provenance,), item.weight)
        for item in forms[:1]
    )


def _currency_record(
    bundle: MoneyRuleBundle, currency: str, *, minor: bool
) -> Mapping[str, object]:
    kind = "minor" if minor else "currency"
    return next(
        record
        for record in bundle.records.values()
        if record.get("kind") == kind and record.get("code") == currency
    )


def _currency_realization(
    bundle: MoneyRuleBundle, currency: str, amount: Decimal, *, minor: bool
) -> Realization:
    record = _currency_record(bundle, currency, minor=minor)
    forms = record["forms"]
    operand = int(abs(amount)) if amount == amount.to_integral_value() else float(abs(amount))
    category = str(icu.PluralRules.forLocale(icu.Locale(bundle.locale)).select(operand))
    text = forms.get(category, forms.get("other"))
    if not isinstance(text, str):
        raise ValueError(f"money record {record['id']!r} has no form for {category!r}")
    return Realization((text,), (f"normalization-record:{record['id']}",), None)


def generate_money_alternatives(
    value: NumberValue,
    detection: Mapping[str, object],
    locale: str,
    *,
    max_derivations: int = 128,
):
    """Add sourced non-English readings for one complete recognized currency edge."""
    from frend.verbalize import SpokenAlternative, _spoken_money

    bundle = load_money_rule_bundle(locale)
    currency = value.currency
    if currency is None:
        raise NotImplementedError("money rules require a captured ISO currency code")
    amount = Decimal(value.decimal)
    shape = _shape(value, detection, bundle)
    recipes = bundle.recipes.get(f"{shape}:{currency}", ())
    if not recipes:
        raise NotImplementedError(
            f"money rules have no sourced {shape}:{currency} recipe for {bundle.locale}"
        )
    digits = _fraction_digits(currency, bundle.locale)
    scale = 10**digits
    absolute = abs(amount)
    major = Decimal(int(absolute))
    minor = Decimal(int(absolute * scale) % scale)
    signed_major = major.copy_sign(amount) if major else major
    signed_minor = minor.copy_sign(amount) if amount.is_signed() and not major else minor
    major_record = _currency_record(bundle, currency, minor=False)
    values = {
        "AMOUNT": _number_realizations(amount, bundle.locale, major_record),
        "MAJOR": _number_realizations(signed_major, bundle.locale, major_record),
        "MINOR": _number_realizations(signed_minor, bundle.locale),
        "CURRENCY": (
            _currency_realization(
                bundle,
                currency,
                minor if shape == "minor_only" else (amount if shape == "major" else major),
                minor=shape == "minor_only",
            ),
        ),
    }
    if shape == "major_minor":
        values["MINOR_CURRENCY"] = (_currency_realization(bundle, currency, minor, minor=True),)
    fields = {symbol: FieldProvider(realizations) for symbol, realizations in values.items()}
    exact_count = exact_composition_count(recipes, fields)
    if exact_count < 1 or exact_count > max_derivations:
        raise ValueError(f"money generation for {bundle.locale} has {exact_count} derivations")
    generated = compose_recipes(
        recipes,
        fields,
        base_provenance=(f"money-rule:{bundle.locale}",),
        make=lambda text, provenance, weight, prior: SpokenAlternative(
            text, provenance, weight, prior_provenance=prior
        ),
        render=lambda core, suffix, selected: "".join(core),
        max_derivations=max_derivations,
        exact_count=exact_count,
        provenance_filter=lambda item: bool(item) and not item.startswith("money-key:"),
    )
    previous = _spoken_money(amount, currency, bundle.locale)
    return preserve_legacy_ranking(generated, previous, unmatched="retain")
