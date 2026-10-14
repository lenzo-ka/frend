"""Sourced electronic composition through precompiled tiergraph rule projections."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from types import MappingProxyType

import icu
from tiergraph import GrammarInput, GrammarInputToken, Realization

from frend.artifact_policy import ELECTRONIC_RULE_POLICY, require_policy
from frend.electronic import ElectronicValue
from frend.generated_rule_locales import locales_for
from frend.locale_data import canonical_locale
from frend.rule_composition import (
    FieldProvider,
    RuleRecipe,
    compose_recipes,
    exact_composition_count,
    placeholder,
    preserve_legacy_ranking,
)

_LOCALES = locales_for("electronic")
_EVENTS = {"load": 0, "input": 0, "generate": 0}


class ElectronicInputRefusal(NotImplementedError):
    """A recognized electronic value lacks a complete sourced private reading."""


@dataclass(frozen=True)
class ElectronicRuleBundle:
    locale: str
    recipes: Mapping[str, tuple[RuleRecipe, ...]]
    records: Mapping[str, Mapping[str, object]]


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def electronic_rule_events() -> Mapping[str, int]:
    return MappingProxyType(dict(_EVENTS))


@cache
def load_electronic_rule_bundle(locale: str) -> ElectronicRuleBundle:
    _EVENTS["load"] += 1
    locale = canonical_locale(locale)
    if locale not in _LOCALES:
        raise ValueError(f"no generated electronic-rule bundle for {locale!r}")
    resource = files("frend").joinpath("data", "electronic_rules", f"{locale}.json")
    document = json.loads(resource.read_text(encoding="utf-8"))
    required = {
        "schema",
        "provenance",
        "locale",
        "icu_version",
        "normalization_sha256",
        "projection_sha256",
        "policy",
        "recipes",
        "records",
    }
    if set(document) != required or document.get("schema") != 1 or document.get("locale") != locale:
        raise ValueError(f"electronic-rule bundle identity mismatch for {locale!r}")
    if document.get("icu_version") != icu.ICU_VERSION:
        raise ValueError(
            f"electronic-rule bundle {locale!r} was built with ICU "
            f"{document.get('icu_version')}, not runtime ICU {icu.ICU_VERSION}"
        )
    require_policy(document.get("policy"), ELECTRONIC_RULE_POLICY, "electronic-rule")
    projection = {"recipes": document.get("recipes"), "records": document.get("records")}
    if (
        document.get("projection_sha256")
        != hashlib.sha256(_canonical_bytes(projection)).hexdigest()
    ):
        raise ValueError(f"electronic-rule bundle {locale!r} projection hash mismatch")
    raw = document.get("recipes")
    if not isinstance(raw, Mapping):
        raise ValueError(f"electronic-rule bundle {locale!r} has invalid recipes")
    recipes = {
        kind: tuple(RuleRecipe(tuple(item["tokens"]), tuple(item["provenance"])) for item in values)
        for kind, values in raw.items()
    }
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise ValueError(f"electronic-rule bundle {locale!r} has invalid records")
    records = {
        str(item["id"]): MappingProxyType(dict(item))
        for item in raw_records
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if len(records) != len(raw_records):
        raise ValueError(f"electronic-rule bundle {locale!r} has duplicate records")
    return ElectronicRuleBundle(locale, MappingProxyType(recipes), records)


def electronic_letter_names(text: str, locale: str) -> tuple[Realization, ...]:
    bundle = load_electronic_rule_bundle(locale)
    names = []
    for character in text:
        if not character.isascii() or not character.isalpha():
            raise ElectronicInputRefusal(f"no sourced electronic letter name for {character!r}")
        identifier = f"letter-{character.upper()}"
        record = bundle.records.get(identifier)
        if record is None:
            raise ElectronicInputRefusal(f"no sourced electronic letter record {identifier!r}")
        names.append(str(record["forms"]["default"]))
    if not names:
        raise ElectronicInputRefusal("empty electronic letter run")
    return (
        Realization(
            (" ".join(names),),
            tuple(f"normalization-record:letter-{character.upper()}" for character in text),
            None,
        ),
    )


def _digit_realizations(text: str, locale: str) -> tuple[Realization, ...]:
    from frend.verbalize import _spoken_digits
    from frend.written_forms import DigitsValue

    forms = _spoken_digits(DigitsValue(text), locale)
    if not forms:
        raise ElectronicInputRefusal(f"no ICU digit-by-digit realization for {text!r}")
    return (Realization((forms[0].text,), (forms[0].provenance,), None),)


def _separator_realizations(
    text: str, kind: str, bundle: ElectronicRuleBundle
) -> tuple[Realization, ...]:
    records = sorted(
        (
            record
            for record in bundle.records.values()
            if record.get("kind") == "electronic-separator"
            and record.get("value") == text
            and record.get("electronic_kind") == kind
        ),
        key=lambda record: int(record.get("priority", 0)),
    )
    if not records:
        raise ElectronicInputRefusal(f"no sourced {kind} separator reading for {text!r}")
    return tuple(
        Realization(
            (str(record["forms"]["default"]),),
            (f"normalization-record:{record['id']}",),
            None,
        )
        for record in records
    )


def _run_realizations(
    value: ElectronicValue, bundle: ElectronicRuleBundle, *, max_derivations: int
) -> tuple[Realization, ...]:
    providers = {}
    tokens = []
    for index, (part_kind, text) in enumerate(value.parts):
        symbol = f"PART{index}"
        if tokens:
            tokens.append(" ")
        tokens.append(placeholder(symbol))
        if part_kind == "letters":
            choices = electronic_letter_names(text, bundle.locale)
        elif part_kind == "digits":
            choices = _digit_realizations(text, bundle.locale)
        elif part_kind == "separator":
            choices = _separator_realizations(text, value.kind, bundle)
        else:
            raise ElectronicInputRefusal(f"unknown electronic run kind {part_kind!r}")
        providers[symbol] = FieldProvider(choices, prior=False)
    if not providers:
        raise ElectronicInputRefusal("electronic value has no parts")
    recipe = RuleRecipe(tuple(tokens), ())
    count = exact_composition_count((recipe,), providers)
    if count > max_derivations:
        raise ElectronicInputRefusal(f"electronic run composition has {count} derivations")
    return compose_recipes(
        (recipe,),
        providers,
        base_provenance=(),
        make=lambda text, provenance, weight, _prior: Realization(
            (text,), tuple(filter(None, provenance.split("+"))), weight
        ),
        render=lambda core, suffix, selected: "".join(core),
        max_derivations=max_derivations,
        exact_count=count,
    )


def electronic_grammar_input(value: ElectronicValue, locale: str) -> GrammarInput:
    _EVENTS["input"] += 1
    bundle = load_electronic_rule_bundle(locale)
    if value.kind not in bundle.recipes:
        raise ElectronicInputRefusal(
            f"electronic rules have no sourced {value.kind!r} recipe for {bundle.locale}"
        )
    limit = int(ELECTRONIC_RULE_POLICY["max_total_alternatives"]) - 1
    runs = _run_realizations(value, bundle, max_derivations=limit)
    return GrammarInput((GrammarInputToken("FIELD:RUNS", runs, ("electronic-field:RUNS",)),))


def generate_electronic_alternatives(value: ElectronicValue, locale: str):
    from frend.verbalize import SpokenAlternative

    _EVENTS["generate"] += 1
    bundle = load_electronic_rule_bundle(locale)
    recipes = bundle.recipes.get(value.kind, ())
    if not recipes:
        raise ElectronicInputRefusal(
            f"electronic rules have no sourced {value.kind!r} recipe for {bundle.locale}"
        )
    previous = (
        SpokenAlternative(
            "".join(text for _kind, text in value.parts),
            "surface:unsupported",
            None,
        ),
    )
    total_cap = int(ELECTRONIC_RULE_POLICY["max_total_alternatives"])
    available = total_cap - len(previous)
    grammar_input = electronic_grammar_input(value, locale)
    fields = {"RUNS": FieldProvider(grammar_input.tokens[0].realization)}
    count = exact_composition_count(recipes, fields)
    if count < 1 or count > available:
        raise ElectronicInputRefusal(
            f"electronic generation for {bundle.locale} has {count} derivations"
        )
    generated = compose_recipes(
        recipes,
        fields,
        base_provenance=(f"electronic-rule:{bundle.locale}:{value.kind}",),
        make=lambda text, provenance, weight, prior: SpokenAlternative(
            text, provenance, weight, prior_provenance=prior
        ),
        render=lambda core, suffix, selected: "".join(core),
        max_derivations=available,
        exact_count=count,
        provenance_filter=lambda item: (
            bool(item) and not item.startswith(("electronic-kind:", "recipe:"))
        ),
    )
    result = preserve_legacy_ranking(generated, previous, unmatched="retain")
    if len(result) > total_cap:
        raise ValueError(f"electronic generation returned {len(result)} total alternatives")
    if any(item.weight is not None or "measured:" in item.provenance for item in generated):
        raise ValueError("sourced electronic generation carried measured ranking")
    return result
