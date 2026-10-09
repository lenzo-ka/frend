"""CLDR-derived date composition executed through tiergraph generation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from importlib.resources import files
from itertools import product
from types import MappingProxyType
from typing import Literal

import icu
from icukit.detectors import DateTimeValue
from tiergraph import (
    GeneratedDerivation,
    GrammarDeclaration,
    GrammarHole,
    GrammarInput,
    GrammarInputToken,
    GrammarTerminal,
    LoweredGrammar,
    ParseForest,
    Realization,
    generate,
    grammar_loads,
    lower_grammar,
    recognize,
)

from frend.locale_data import canonical_locale

GENERATED_DATE_LOCALES = (
    "es_MX",
    "es_ES",
    "fr_FR",
    "de_DE",
    "pt_BR",
    "pt_PT",
    "it_IT",
    "zh_CN",
    "ko_KR",
    "ja_JP",
)
_EVENTS = {"load": 0, "lower": 0, "generate": 0}


def date_rule_events() -> Mapping[str, int]:
    """Return process-local instrumentation for cold-path gate receipts."""
    return MappingProxyType(dict(_EVENTS))


@dataclass(frozen=True)
class DateRuleBundle:
    schema: int
    locale: str
    icu_version: str
    patterns: Mapping[str, Mapping[str, str]]
    declaration: GrammarDeclaration
    lowered: LoweredGrammar
    day_policy: str


@dataclass(frozen=True)
class DateRuleRecipe:
    tokens: tuple[str, ...]
    provenance: tuple[str, ...]


def _validate_field_holes(declaration: GrammarDeclaration) -> None:
    expected = ({"M", "D"}, {"Y", "M", "D"})
    observed: list[set[str]] = []
    for rule in declaration.rules:
        if rule.left != declaration.start:
            continue
        source = {
            hole.nonterminal.local_name for hole in rule.source if isinstance(hole, GrammarHole)
        }
        target = {
            hole.nonterminal.local_name for hole in rule.target if isinstance(hole, GrammarHole)
        }
        if source != target or len(source) != sum(
            isinstance(item, GrammarHole) for item in rule.source
        ):
            raise ValueError("date grammar must consume and emit every field hole exactly once")
        observed.append(source)
    if tuple(observed) != expected:
        raise ValueError(f"date grammar field rules are {observed!r}, expected {expected!r}")
    for rule in declaration.rules:
        if rule.left == declaration.start:
            continue
        terminals = [item.text.lexical for item in rule.source if isinstance(item, GrammarTerminal)]
        if terminals != [f"FIELD:{rule.left.local_name}"] or rule.source != rule.target:
            raise ValueError(f"date field rule {rule.left.local_name!r} is not an exact echo")


@cache
def load_date_rule_bundle(locale: str) -> DateRuleBundle:
    _EVENTS["load"] += 1
    locale = canonical_locale(locale)
    if locale not in GENERATED_DATE_LOCALES:
        raise ValueError(f"no generated date-rule bundle for {locale!r}")
    resource = files("frend").joinpath("data", "date_rules", f"{locale}.json")
    document = json.loads(resource.read_text(encoding="utf-8"))
    required = {
        "schema",
        "locale",
        "icu_version",
        "sources",
        "field_policy",
        "grammar",
        "provenance",
    }
    if set(document) != required:
        raise ValueError(f"date-rule bundle {locale!r} has keys {sorted(document)!r}")
    if document["schema"] != 1 or document["locale"] != locale:
        raise ValueError(f"date-rule bundle identity mismatch for {locale!r}")
    if document["icu_version"] != icu.ICU_VERSION:
        raise ValueError(
            f"date-rule bundle {locale!r} was built with ICU {document['icu_version']}, "
            f"not runtime ICU {icu.ICU_VERSION}"
        )
    declaration = grammar_loads(json.dumps(document["grammar"], ensure_ascii=False))
    _validate_field_holes(declaration)
    lowered = lower_grammar(declaration)
    _EVENTS["lower"] += 1
    sources = MappingProxyType(
        {key: MappingProxyType(dict(value)) for key, value in document["sources"].items()}
    )
    return DateRuleBundle(
        1,
        locale,
        icu.ICU_VERSION,
        sources,
        declaration,
        lowered,
        str(document["field_policy"]["day"]),
    )


def _realization(item) -> Realization:
    return Realization((item.text,), (item.provenance,), item.weight)


def date_field_realizations(
    field: Literal["Y", "M", "D"],
    value: int,
    *,
    calendar: str,
    locale: str,
) -> tuple[Realization, ...]:
    """Return unit-free ICU leaves plus the narrowly sourced date exceptions."""
    from frend.verbalize import SpokenAlternative, _lexical, _month_name, _number_leaf, _year_leaf

    locale = canonical_locale(locale)
    if field == "Y":
        forms = _year_leaf(Decimal(value), locale)
    elif field == "M" and locale not in {"zh_CN", "ko_KR", "ja_JP"}:
        forms = (_month_name(value, calendar, locale),)
    elif field == "D" and locale == "de_DE":
        forms = tuple(
            item
            for item in _number_leaf(Decimal(value), "ordinal", locale)
            if item.provenance.endswith(":%spellout-ordinal-r")
        )
    else:
        forms = _number_leaf(Decimal(value), "cardinal", locale)[:1]

    sourced: list[SpokenAlternative] = []
    if field == "D" and value == 1 and locale in {"es_MX", "es_ES", "fr_FR", "pt_BR", "pt_PT"}:
        if word := _lexical("date.day_one", locale):
            sourced.append(SpokenAlternative(str(word), f"lexical:{locale}"))
    if field == "M" and locale == "ko_KR" and value in {6, 10}:
        if word := _lexical(f"date.month.{value}", locale):
            sourced.append(SpokenAlternative(str(word), f"lexical:{locale}"))
    selected = (*sourced, *forms)
    if not selected:
        raise ValueError(f"no date realization for {locale} {field}={value}")
    return tuple(_realization(item) for item in selected)


def date_grammar_input(
    value: DateTimeValue,
    detection: Mapping[str, object],
    locale: str,
) -> GrammarInput:
    del detection
    fields = dict(value.fields)
    if len(fields) != len(value.fields) or set(fields) not in ({"M", "d"}, {"y", "M", "d"}):
        raise NotImplementedError("generated date rules require unique M,d or y,M,d fields")
    order = ("Y", "M", "D") if "y" in fields else ("M", "D")
    values = {"Y": fields.get("y"), "M": fields["M"], "D": fields["d"]}
    return GrammarInput(
        tuple(
            GrammarInputToken(
                f"FIELD:{field}",
                date_field_realizations(
                    field, int(values[field]), calendar=value.calendar, locale=locale
                ),
                (f"date-field:{field}",),
            )
            for field in order
        )
    )


def render_date_derivation(derivation: GeneratedDerivation) -> str:
    return "".join(derivation.tokens)


def exact_date_generation_count(forest: ParseForest, grammar_input: GrammarInput) -> int:
    return forest.count() * math.prod(len(token.realization) for token in grammar_input.tokens)


def _date_source_path_count(bundle: DateRuleBundle, symbols: tuple[str, ...]) -> int:
    wanted = tuple(symbol.removeprefix("FIELD:") for symbol in symbols)
    return sum(
        tuple(item.nonterminal.local_name for item in rule.source if isinstance(item, GrammarHole))
        == wanted
        for rule in bundle.declaration.rules
        if rule.left == bundle.declaration.start
    )


@cache
def _date_rule_recipes(locale: str, kind: str) -> tuple[DateRuleRecipe, ...]:
    """Compile a locale/field-shape rule once, never a date value or input text."""
    bundle = load_date_rule_bundle(locale)
    symbols = ("FIELD:M", "FIELD:D") if kind == "Md" else ("FIELD:Y", "FIELD:M", "FIELD:D")
    placeholders = {symbol: f"\ufff0{symbol}\ufff1" for symbol in symbols}
    grammar_input = GrammarInput(
        tuple(
            GrammarInputToken(
                symbol,
                (Realization((placeholders[symbol],), (f"recipe:{symbol}",), None),),
                (f"date-field:{symbol[-1]}",),
            )
            for symbol in symbols
        )
    )
    forest = recognize(bundle.lowered, grammar_input, collapse_units=False)
    count = forest.count()
    _EVENTS["generate"] += 1
    result = generate(forest, count=count)
    if result.truncated or len(result.derivations) != count:
        raise ValueError(
            f"date recipe for {locale} {kind} returned {len(result.derivations)} of "
            f"{count} derivations (truncated={result.truncated})"
        )
    return tuple(
        DateRuleRecipe(
            tuple(derivation.tokens),
            tuple(
                value.lexical
                for application in derivation.applications
                for value in application.provenance
            ),
        )
        for derivation in result.derivations
    )


def generate_date_alternatives(
    value: DateTimeValue,
    detection: Mapping[str, object],
    locale: str,
    *,
    max_derivations: int = 128,
):
    """Generate every declared date derivation, refusing rather than truncating."""
    from frend.verbalize import SpokenAlternative

    locale = canonical_locale(locale)
    bundle = load_date_rule_bundle(locale)
    grammar_input = date_grammar_input(value, detection, locale)
    kind = "yMd" if len(grammar_input.tokens) == 3 else "Md"
    source_paths = _date_source_path_count(
        bundle, tuple(token.symbol for token in grammar_input.tokens)
    )
    exact_count = source_paths * math.prod(len(token.realization) for token in grammar_input.tokens)
    realization_counts = tuple(len(token.realization) for token in grammar_input.tokens)
    if exact_count > max_derivations:
        raise ValueError(
            f"date generation for {locale} has field realization counts "
            f"{realization_counts}, {source_paths} source paths, {exact_count} targets, "
            f"above cap {max_derivations}"
        )
    if exact_count < 1:
        raise ValueError(f"date grammar for {locale} recognized no complete derivation")
    recipes = _date_rule_recipes(locale, kind)
    if len(recipes) != source_paths:
        raise ValueError(
            f"date grammar for {locale} {kind} compiled {len(recipes)} of "
            f"{source_paths} source paths"
        )
    by_placeholder = {f"\ufff0{token.symbol}\ufff1": token for token in grammar_input.tokens}
    alternatives = []
    for recipe in recipes:
        fields = [token for token in recipe.tokens if token in by_placeholder]
        for realizations in product(*(by_placeholder[field].realization for field in fields)):
            selected = dict(zip(fields, realizations, strict=True))
            rendered: list[str] = []
            provenance = [f"date-rule:{locale}", *recipe.provenance]
            for token in recipe.tokens:
                realization = selected.get(token)
                if realization is None:
                    rendered.append(token)
                    continue
                field = by_placeholder[token]
                rendered.extend(realization.tokens)
                provenance.extend(field.provenance)
                provenance.extend(realization.provenance)
            unique = tuple(dict.fromkeys(provenance))
            alternatives.append(SpokenAlternative("".join(rendered), "+".join(unique)))
    if len(alternatives) != exact_count:
        raise ValueError(
            f"date generation for {locale} returned {len(alternatives)} of "
            f"{exact_count} derivations"
        )
    return tuple(alternatives)
