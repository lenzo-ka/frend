"""Sourced fraction and percent composition executed through tiergraph generation."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import cache
from importlib.resources import files
from itertools import product
from types import MappingProxyType

import icu
from tiergraph import (
    GrammarDeclaration,
    GrammarHole,
    GrammarInput,
    GrammarInputToken,
    GrammarTerminal,
    Realization,
    grammar_loads,
)

from frend.locale_data import canonical_locale
from frend.normalization_schema import validate_bundle

GENERATED_FRACTION_LOCALES = (
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


def fraction_rule_events() -> Mapping[str, int]:
    return MappingProxyType(dict(_EVENTS))


@dataclass(frozen=True)
class FractionRuleBundle:
    locale: str
    declaration: GrammarDeclaration
    recipes: Mapping[str, tuple[FractionRuleRecipe, ...]]
    records: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class FractionRuleRecipe:
    tokens: tuple[str, ...]
    provenance: tuple[str, ...]


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _receipt_index(document: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for source in document.get("sources", []):
        if not isinstance(source, Mapping) or not isinstance(source.get("receipt"), str):
            continue
        reference = source["receipt"]
        receipt = json.loads(
            files("frend").joinpath("data", "normalization", reference).read_text("utf-8")
        )
        result[reference] = receipt
        result[hashlib.sha256(_canonical_bytes(receipt)).hexdigest()] = receipt
    return result


def _validate_holes(declaration: GrammarDeclaration) -> None:
    observed: set[tuple[str, ...]] = set()
    for rule in declaration.rules:
        if rule.left != declaration.start:
            continue
        source = tuple(
            item.nonterminal.local_name for item in rule.source if isinstance(item, GrammarHole)
        )
        target = tuple(
            item.nonterminal.local_name for item in rule.target if isinstance(item, GrammarHole)
        )
        if len(source) != len(set(source)) or sorted(source) != sorted(target):
            raise ValueError("fraction grammar must consume and emit every field hole exactly once")
        observed.add(source)
    expected = {
        ("SIGN", "NUM", "DEN"),
        ("SIGN", "WHOLE", "NUM", "DEN"),
        ("SIGN", "AMOUNT"),
    }
    if observed != expected:
        raise ValueError(f"fraction grammar field rules are {observed!r}, expected {expected!r}")
    for rule in declaration.rules:
        if rule.left == declaration.start:
            continue
        terminals = [item.text.lexical for item in rule.source if isinstance(item, GrammarTerminal)]
        if terminals != [f"FIELD:{rule.left.local_name}"] or rule.source != rule.target:
            raise ValueError(f"fraction field rule {rule.left.local_name!r} is not an exact echo")


def _validate_compiled_recipes(
    declaration: GrammarDeclaration,
    recipes: Mapping[str, Sequence[FractionRuleRecipe]],
    locale: str,
) -> None:
    """Require the fast-path recipes to be an exact projection of the parsed grammar."""
    kinds = {
        ("SIGN", "NUM", "DEN"): "fraction",
        ("SIGN", "WHOLE", "NUM", "DEN"): "mixed",
        ("SIGN", "AMOUNT"): "percent",
    }
    expected: dict[str, list[FractionRuleRecipe]] = {kind: [] for kind in kinds.values()}
    for rule in declaration.rules:
        if rule.left != declaration.start:
            continue
        symbols = tuple(
            item.nonterminal.local_name for item in rule.source if isinstance(item, GrammarHole)
        )
        kind = kinds.get(symbols)
        if kind is None:
            continue
        tokens = tuple(
            f"\ufff0{item.nonterminal.local_name}\ufff1"
            if isinstance(item, GrammarHole)
            else item.text.lexical
            for item in rule.target
        )
        provenance = tuple(value.lexical for value in rule.provenance)
        expected[kind].append(FractionRuleRecipe(tokens, provenance))
    observed = {kind: tuple(items) for kind, items in recipes.items()}
    wanted = {kind: tuple(items) for kind, items in expected.items()}
    if observed != wanted:
        raise ValueError(f"fraction-rule bundle {locale!r} compiled recipes do not match grammar")


@cache
def load_fraction_rule_bundle(locale: str) -> FractionRuleBundle:
    _EVENTS["load"] += 1
    locale = canonical_locale(locale)
    if locale not in GENERATED_FRACTION_LOCALES:
        raise ValueError(f"no generated fraction-rule bundle for {locale!r}")
    normalization = json.loads(
        files("frend").joinpath("data", "normalization", f"{locale}.json").read_text("utf-8")
    )
    validate_bundle(normalization, receipt_index=_receipt_index(normalization))
    if normalization["locale"] != locale or normalization["versions"].get("icu") != icu.ICU_VERSION:
        raise ValueError(f"fraction normalization record identity/version mismatch for {locale!r}")
    resource = files("frend").joinpath("data", "fraction_rules", f"{locale}.json")
    document = json.loads(resource.read_text(encoding="utf-8"))
    if document.get("schema") != 1 or document.get("locale") != locale:
        raise ValueError(f"fraction-rule bundle identity mismatch for {locale!r}")
    if document.get("icu_version") != icu.ICU_VERSION:
        raise ValueError(
            f"fraction-rule bundle {locale!r} was built with ICU {document.get('icu_version')}, "
            f"not runtime ICU {icu.ICU_VERSION}"
        )
    if (
        document.get("normalization_sha256")
        != hashlib.sha256(_canonical_bytes(normalization)).hexdigest()
    ):
        raise ValueError(f"fraction-rule bundle source-record drift for {locale!r}")
    declaration = grammar_loads(json.dumps(document["grammar"], ensure_ascii=False))
    _validate_holes(declaration)
    raw_recipes = document.get("recipes")
    if not isinstance(raw_recipes, Mapping) or set(raw_recipes) != {
        "fraction",
        "mixed",
        "percent",
    }:
        raise ValueError(f"fraction-rule bundle {locale!r} has no complete compiled recipes")
    recipes: dict[str, tuple[FractionRuleRecipe, ...]] = {}
    for kind, items in raw_recipes.items():
        if not isinstance(kind, str) or not isinstance(items, list) or not items:
            raise ValueError(f"fraction-rule bundle {locale!r} has invalid {kind!r} recipes")
        compiled = []
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError(f"fraction-rule bundle {locale!r} has a non-mapping recipe")
            tokens = item.get("tokens")
            provenance = item.get("provenance")
            if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
                raise ValueError(f"fraction-rule bundle {locale!r} has invalid recipe tokens")
            if not isinstance(provenance, list) or not all(
                isinstance(source, str) for source in provenance
            ):
                raise ValueError(f"fraction-rule bundle {locale!r} has invalid recipe provenance")
            compiled.append(FractionRuleRecipe(tuple(tokens), tuple(provenance)))
        recipes[kind] = tuple(compiled)
    _validate_compiled_recipes(declaration, recipes, locale)
    records = {
        str(item["id"]): MappingProxyType(dict(item))
        for section in ("resources", "rules")
        for item in normalization.get(section, [])
    }
    return FractionRuleBundle(
        locale,
        declaration,
        MappingProxyType(recipes),
        MappingProxyType(records),
    )


def _realization(item, *, text: str | None = None, source: str | None = None) -> Realization:
    provenance = (item.provenance,) if source is None else (item.provenance, source)
    return Realization(((item.text if text is None else text),), provenance, item.weight)


def _record_value(bundle: FractionRuleBundle, identifier: str) -> str:
    record = bundle.records.get(identifier)
    forms = None if record is None else record.get("forms")
    if not isinstance(forms, Mapping) or list(forms) != ["default"]:
        raise ValueError(f"fraction record {identifier!r} is missing one default value")
    value = forms["default"]
    if not isinstance(value, str):
        raise ValueError(f"fraction record {identifier!r} is not textual")
    return value


def _record_realizations(record: Mapping[str, object]) -> tuple[Realization, ...]:
    forms = record.get("forms")
    if (
        not isinstance(forms, Mapping)
        or not forms
        or not all(
            isinstance(key, str) and isinstance(value, str) and value
            for key, value in forms.items()
        )
    ):
        raise ValueError(f"fraction record {record.get('id')!r} has invalid textual forms")
    identifier = str(record["id"])
    return tuple(
        Realization((value,), (f"normalization-record:{identifier}",), None)
        for value in forms.values()
    )


def _normal_cardinals(value: Decimal, locale: str):
    from frend.verbalize import _number_leaf

    forms = _number_leaf(value, "cardinal", locale)
    normal = tuple(
        item
        for item in forms
        if any(part.endswith(":%spellout-numbering") for part in item.provenance.split("+"))
    )
    return normal or forms[:1]


def _masculine_one(locale: str):
    from frend.verbalize import _number_leaf

    forms = _number_leaf(Decimal(1), "cardinal", locale)
    wanted = {
        "es_MX": "%spellout-cardinal-masculine",
        "es_ES": "%spellout-cardinal-masculine",
        "de_DE": "%spellout-cardinal-neuter",
        "it_IT": "%spellout-cardinal-masculine",
    }.get(locale)
    if wanted is not None:
        selected = tuple(
            item
            for item in forms
            if any(part.endswith(f":{wanted}") for part in item.provenance.split("+"))
        )
        if selected:
            return selected
    return _normal_cardinals(Decimal(1), locale)


def _cardinal_realizations(value: Decimal, locale: str, *, masculine_one: bool = False):
    forms = (
        _masculine_one(locale) if masculine_one and value == 1 else _normal_cardinals(value, locale)
    )
    return tuple(_realization(item) for item in forms)


def _ordinal_realizations(value: int, locale: str, *, plural: bool):
    from frend.verbalize import _number_leaf

    marker = "%spellout-ordinal-masculine-plural" if plural else "%spellout-ordinal-masculine"
    forms = tuple(
        item
        for item in _number_leaf(Decimal(value), "ordinal", locale)
        if any(part.endswith(f":{marker}") for part in item.provenance.split("+"))
    )
    if not forms and not plural:
        adjective = f"{marker}-adjective"
        forms = tuple(
            item
            for item in _number_leaf(Decimal(value), "ordinal", locale)
            if any(part.endswith(f":{adjective}") for part in item.provenance.split("+"))
        )
    if not forms and not plural:
        forms = tuple(
            item
            for item in _number_leaf(Decimal(value), "ordinal", locale)
            if any(part.endswith(":%spellout-ordinal") for part in item.provenance.split("+"))
        )[:1]
    return forms


def _sourced_plural_ordinals(
    forms: Sequence[object], bundle: FractionRuleBundle
) -> tuple[Realization, ...]:
    record = bundle.records.get("denominator-plural-rule")
    if record is None:
        return ()
    rule = _record_value(bundle, "denominator-plural-rule")
    if rule != "terminal-o-to-i":
        raise ValueError(f"unknown denominator plural rule {rule!r} for {bundle.locale}")
    return tuple(
        _realization(
            item,
            text=f"{item.text[:-1]}i",
            source="normalization-record:denominator-plural-rule",
        )
        for item in forms
        if item.text.endswith("o")
    )


def _cardinal_denominator_alternatives(
    denominator: int, bundle: FractionRuleBundle
) -> tuple[Realization, ...]:
    record = bundle.records.get("denominator-cardinal-alternatives")
    if record is None:
        return ()
    forms = record.get("forms")
    if not isinstance(forms, Mapping):
        raise ValueError("denominator-cardinal-alternatives has no textual forms")
    identifier = str(record["id"])
    return tuple(
        _realization(
            cardinal,
            text=str(pattern).format(cardinal=cardinal.text),
            source=f"normalization-record:{identifier}",
        )
        for cardinal in _normal_cardinals(Decimal(denominator), bundle.locale)
        for pattern in forms.values()
    )


def _written_denominator(denominator: int) -> tuple[Realization, ...]:
    return (
        Realization(
            (str(denominator),),
            ("written-surface:fraction-denominator",),
            None,
        ),
    )


def _strip_accents(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value)
        if not unicodedata.combining(character)
    )


def _denominator_realizations(
    denominator: int, numerator: int, bundle: FractionRuleBundle
) -> tuple[Realization, ...]:
    locale = bundle.locale
    plural = icu.PluralRules.forLocale(icu.Locale(locale)).select(numerator) != "one"
    irregular = bundle.records.get(
        f"denominator-{denominator}-{'plural' if plural else 'singular'}"
    )
    cardinal_alternatives = _cardinal_denominator_alternatives(denominator, bundle)
    if irregular is not None:
        return (*_record_realizations(irregular), *cardinal_alternatives)
    strategy = _record_value(bundle, "denominator-strategy")
    if strategy == "icu-ordinal":
        forms = _ordinal_realizations(denominator, locale, plural=plural)
        ordinals = tuple(_realization(item) for item in forms)
        if plural and not ordinals:
            ordinals = _sourced_plural_ordinals(
                _ordinal_realizations(denominator, locale, plural=False), bundle
            )
        return (*ordinals, *cardinal_alternatives) or _written_denominator(denominator)
    if strategy == "german-cardinal-tel":
        cardinal = _normal_cardinals(Decimal(denominator), locale)[0]
        stem = cardinal.text
        if denominator >= 100 and stem.startswith("ein"):
            stem = stem[3:]
        if stem.endswith("eins"):
            text = f"{stem[:-4]}eintel"
        elif denominator == 3:
            text = "drittel"
        elif denominator == 7:
            text = "siebtel"
        elif denominator >= 20 and not 1 <= denominator % 100 <= 19:
            text = f"{stem}stel"
        else:
            text = f"{stem}tel"
        return (
            _realization(
                cardinal,
                text=text[0].upper() + text[1:],
                source="normalization-record:denominator-strategy",
            ),
        )
    if strategy == "portuguese-precedence":
        if denominator <= 10 or denominator in {100, 1000}:
            forms = _ordinal_realizations(denominator, locale, plural=False)
            suffix = "s" if plural else ""
            result = tuple(
                _realization(
                    item,
                    text=f"{item.text}{suffix}",
                    source="normalization-record:denominator-strategy",
                )
                for item in forms
            )
            return result or _written_denominator(denominator)
        suffix = " avos"
        return tuple(
            _realization(
                item,
                text=f"{item.text}{suffix}",
                source="normalization-record:denominator-strategy",
            )
            for item in _normal_cardinals(Decimal(denominator), locale)
        )
    if strategy == "spanish-avo":
        if (
            denominator <= 10
            or denominator
            in {
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
            }
            or denominator % 1000 == 0
        ):
            result = []
            for item in _ordinal_realizations(denominator, locale, plural=plural):
                text = item.text
                if denominator == 1_000_000 and text.startswith("un "):
                    text = text[3:]
                result.append(_realization(item, text=text.replace(" ", "")))
            return tuple(result) or _written_denominator(denominator)
        cardinal = _normal_cardinals(Decimal(denominator), locale)[0]
        text = _strip_accents(cardinal.text)
        stem_record = bundle.records.get(f"denominator-stem-{denominator}")
        if stem_record is not None:
            text = _record_value(bundle, str(stem_record["id"]))
        elif text.endswith("uno"):
            text = f"{text[:-3]}un"
        text = text.replace(" y ", "i").replace(" ", "")
        avos = (
            _realization(
                cardinal,
                text=f"{text}avo{'s' if plural else ''}",
                source="normalization-record:denominator-strategy",
            ),
        )
        if 20 <= denominator <= 90 and denominator % 10 == 0:
            ordinals = tuple(
                _realization(item, source="normalization-record:denominator-strategy")
                for item in _ordinal_realizations(denominator, locale, plural=plural)
            )
            return (*avos, *ordinals)
        return avos
    if strategy == "cardinal":
        return _cardinal_realizations(Decimal(denominator), locale)
    raise ValueError(f"unknown denominator strategy {strategy!r} for {locale}")


def _sign_realizations(negative: bool, locale: str) -> tuple[Realization, ...]:
    if not negative:
        return (Realization(("",), (), Decimal(0)),)
    from frend.verbalize import _signed_integer_leaf

    positive = _normal_cardinals(Decimal(1), locale)[0].text
    result = []
    for item in _signed_integer_leaf(Decimal(1), True, locale):
        if not item.text.endswith(positive):
            continue
        prefix = item.text.removesuffix(positive).rstrip()
        if locale not in {"zh_CN", "ja_JP"}:
            prefix += " "
        result.append(_realization(item, text=prefix))
    if not result:
        raise ValueError(f"cannot isolate ICU sign realization for {locale}")
    return tuple(result)


def _amount_realizations(amount: Decimal, locale: str) -> tuple[Realization, ...]:
    from frend.verbalize import _spoken_decimal

    return tuple(_realization(item) for item in _spoken_decimal(abs(amount), locale))


def _grammar_input(symbols: Sequence[tuple[str, tuple[Realization, ...]]]) -> GrammarInput:
    return GrammarInput(
        tuple(
            GrammarInputToken(f"FIELD:{symbol}", realizations, (f"fraction-field:{symbol}",))
            for symbol, realizations in symbols
        )
    )


def _source_path_count(bundle: FractionRuleBundle, symbols: tuple[str, ...]) -> int:
    return sum(
        tuple(item.nonterminal.local_name for item in rule.source if isinstance(item, GrammarHole))
        == symbols
        for rule in bundle.declaration.rules
        if rule.left == bundle.declaration.start
    )


@cache
def _fraction_rule_recipes(locale: str, kind: str) -> tuple[FractionRuleRecipe, ...]:
    bundle = load_fraction_rule_bundle(locale)
    try:
        return bundle.recipes[kind]
    except KeyError as exc:
        raise ValueError(f"unknown fraction recipe kind {kind!r}") from exc


def render_fraction_tokens(tokens: Sequence[str]) -> str:
    return "".join(tokens)


def _korean_mixed_particle(whole: str) -> str:
    final = whole.rstrip()[-1]
    if not "\uac00" <= final <= "\ud7a3":
        raise ValueError(f"Korean mixed-number whole {whole!r} has no final Hangul syllable")
    return "\uacfc" if (ord(final) - 0xAC00) % 28 else "\uc640"


def _generate(
    bundle: FractionRuleBundle,
    kind: str,
    grammar_input: GrammarInput,
    *,
    max_derivations: int,
    omit_numerator: bool = False,
):
    from frend.verbalize import SpokenAlternative

    symbols = tuple(token.symbol.removeprefix("FIELD:") for token in grammar_input.tokens)
    paths = _source_path_count(bundle, symbols)
    counts = tuple(len(token.realization) for token in grammar_input.tokens)
    exact_count = paths * math.prod(counts)
    if exact_count > max_derivations:
        raise ValueError(
            f"fraction generation for {bundle.locale} has field realization counts {counts}, "
            f"{paths} source paths, {exact_count} targets, above cap {max_derivations}"
        )
    if exact_count < 1:
        raise ValueError(f"fraction grammar for {bundle.locale} recognized no complete derivation")
    recipes = _fraction_rule_recipes(bundle.locale, kind)
    if len(recipes) != paths:
        raise ValueError(
            f"fraction grammar for {bundle.locale} {kind} compiled "
            f"{len(recipes)} of {paths} source paths"
        )
    by_placeholder = {
        f"\ufff0{token.symbol.removeprefix('FIELD:')}\ufff1": token
        for token in grammar_input.tokens
    }
    alternatives = []
    for recipe in recipes:
        fields = [token for token in recipe.tokens if token in by_placeholder]
        for realizations in product(*(by_placeholder[field].realization for field in fields)):
            selected = dict(zip(fields, realizations, strict=True))
            rendered: list[str] = []
            provenance = [
                f"fraction-rule:{bundle.locale}",
                f"lexical:{bundle.locale}",
                *recipe.provenance,
            ]
            prior: list[str] = []
            weights: list[Decimal | None] = []
            skip_space_after_omitted_numerator = False
            for token in recipe.tokens:
                realization = selected.get(token)
                if realization is None:
                    if skip_space_after_omitted_numerator and token.isspace():
                        skip_space_after_omitted_numerator = False
                        continue
                    rendered.append(token)
                    continue
                if omit_numerator and token == "\ufff0NUM\ufff1":
                    skip_space_after_omitted_numerator = True
                    continue
                rendered.extend(realization.tokens)
                provenance.extend(realization.provenance)
                prior.extend(realization.provenance[:1])
                weights.append(realization.weight)
            provenance = list(
                dict.fromkeys(
                    item for item in provenance if item and not item.startswith("recipe:")
                )
            )
            prior = list(
                dict.fromkeys(item for item in prior if item and not item.startswith("recipe:"))
            )
            text = render_fraction_tokens(rendered)
            if kind == "mixed" and bundle.locale == "de_DE":
                text = text.lower()
            if "{{WA_GWA}}" in text:
                whole = selected["\ufff0WHOLE\ufff1"].tokens[0]
                text = text.replace("{{WA_GWA}}", _korean_mixed_particle(whole))
            alternatives.append(
                SpokenAlternative(
                    text,
                    "+".join(provenance),
                    sum(weights, Decimal(0))
                    if weights and all(item is not None for item in weights)
                    else None,
                    prior_provenance="+".join(prior) or None,
                )
            )
    if len(alternatives) != exact_count:
        raise ValueError(
            f"fraction generation for {bundle.locale} returned "
            f"{len(alternatives)} of {exact_count} derivations"
        )
    return tuple(alternatives)


def _preserve_unchanged_ranking(generated, previous):
    """Keep the old measurement key, weight, and order for byte-identical readings."""
    remaining = list(generated)
    preserved = []
    for old in previous:
        match = next((item for item in remaining if item.text == old.text), None)
        if match is None:
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


def generate_fraction_alternatives(
    detection: Mapping[str, object], locale: str, *, max_derivations: int = 128
):
    from frend.verbalize import _capture_integer

    bundle = load_fraction_rule_bundle(locale)
    whole = _capture_integer(detection, "whole")
    numerator = _capture_integer(detection, "numerator")
    denominator = _capture_integer(detection, "denominator")
    if numerator is None or denominator is None:
        raise NotImplementedError("fraction recognition did not retain numerator/denominator")
    negative = any(
        getattr(item, "name", None) == "sign"
        and str(getattr(item, "text", "")).strip() in {"-", "−", "负", "負", "마이너스", "マイナス"}
        for item in detection.get("captures", ())
    )
    fields = [("SIGN", _sign_realizations(negative, bundle.locale))]
    if whole is not None:
        fields.append(("WHOLE", _cardinal_realizations(whole, bundle.locale, masculine_one=True)))
    fields.extend(
        (
            ("NUM", _cardinal_realizations(numerator, bundle.locale, masculine_one=True)),
            ("DEN", _denominator_realizations(int(denominator), int(numerator), bundle)),
        )
    )
    generated = _generate(
        bundle,
        "mixed" if whole is not None else "fraction",
        _grammar_input(fields),
        max_derivations=max_derivations,
        omit_numerator=(
            whole is not None
            and numerator == 1
            and denominator == 2
            and "mixed-half-omit-one" in bundle.records
        ),
    )
    from frend.verbalize import _spoken_fraction

    try:
        previous = _spoken_fraction(detection, bundle.locale)
    except NotImplementedError:
        previous = ()
    return _preserve_unchanged_ranking(generated, previous)


def generate_percent_alternatives(
    amount: Decimal, detection: Mapping[str, object], locale: str, *, max_derivations: int = 128
):
    del detection
    bundle = load_fraction_rule_bundle(locale)
    grammar_input = _grammar_input(
        (
            ("SIGN", _sign_realizations(amount.is_signed(), bundle.locale)),
            ("AMOUNT", _amount_realizations(amount, bundle.locale)),
        )
    )
    generated = _generate(bundle, "percent", grammar_input, max_derivations=max_derivations)
    from frend.verbalize import SpokenAlternative, _percent_name, _spoken_decimal

    previous = tuple(
        SpokenAlternative(
            f"{item.text} {_percent_name(bundle.locale)}", item.provenance, item.weight
        )
        for item in _spoken_decimal(amount, bundle.locale)
    )
    return _preserve_unchanged_ranking(generated, previous)
