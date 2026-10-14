"""Sourced regional telephone composition through generated tiergraph recipes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from types import MappingProxyType

import phonenumbers
from tiergraph import Realization

from frend.artifact_policy import TELEPHONE_REGIONAL_POLICY, require_policy
from frend.generated_rule_locales import locales_for
from frend.locale_data import canonical_locale
from frend.normalization_components import load_normalization_document
from frend.rule_composition import (
    FieldProvider,
    RuleRecipe,
    compose_recipes,
    exact_composition_count,
    preserve_legacy_ranking,
)

_LOCALES = locales_for("telephone")
_EVENTS = {"load": 0, "lower": 0, "generate": 0}


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True, slots=True)
class RegionalTelephoneRule:
    id: str
    region: str
    scope: str
    descriptor_sha256: str
    number_types: frozenset[int]
    trunk_prefix: str
    modes: tuple[str, ...]
    recipe: RuleRecipe


@cache
def regional_telephone_formats(locale: str) -> tuple[RegionalTelephoneRule, ...]:
    """Load a locale projection on first use and verify its policy and source records."""
    _EVENTS["load"] += 1
    locale = canonical_locale(locale)
    if locale not in _LOCALES:
        raise ValueError(f"no generated telephone-rule bundle for {locale!r}")
    resource = files("frend").joinpath("data", "telephone_rules", f"{locale}.json")
    document = json.loads(resource.read_text(encoding="utf-8"))
    required = {
        "schema",
        "locale",
        "metadata_version",
        "normalization_sha256",
        "policy",
        "provenance",
        "projection_sha256",
        "rules",
    }
    if (
        set(document) != required
        or document.get("schema") != 1
        or document.get("locale") != locale
        or document.get("provenance") != {"source": "frend/curated"}
    ):
        raise ValueError(f"telephone-rule bundle identity mismatch for {locale!r}")
    require_policy(document.get("policy"), TELEPHONE_REGIONAL_POLICY, "telephone regional")
    # The dependency floor is open. Newer releases may load the projection, while
    # telephone._regional_match enables only descriptors whose complete digest still
    # matches the v9.0.41 build authority.
    projection = {"rules": document.get("rules")}
    if (
        document.get("projection_sha256")
        != hashlib.sha256(_canonical_bytes(projection)).hexdigest()
    ):
        raise ValueError(f"telephone-rule bundle {locale!r} projection hash mismatch")
    authority = load_normalization_document(f"telephone_{locale}.json")
    if (
        document.get("normalization_sha256")
        != hashlib.sha256(_canonical_bytes(authority)).hexdigest()
    ):
        raise ValueError(f"telephone-rule bundle {locale!r} normalization hash mismatch")
    records = {
        item["id"]: item
        for item in authority.get("rules", ())
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    type_values = {
        "fixed": phonenumbers.PhoneNumberType.FIXED_LINE,
        "mobile": phonenumbers.PhoneNumberType.MOBILE,
    }
    parsed = []
    for raw in document["rules"]:
        identifier = raw.get("id")
        record = records.get(identifier)
        if record is None or record.get("provenance_status") != "provisional":
            raise ValueError(f"telephone rule {identifier!r} lacks provisional source authority")
        if record.get("source_category") != "vendor documentation":
            raise ValueError(f"telephone rule {identifier!r} is not vendor documentation")
        number_types = raw.get("number_types")
        try:
            parsed_types = frozenset(type_values[item] for item in number_types)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"telephone rule {identifier!r} has invalid number types") from exc
        recipe_raw = raw.get("recipe")
        parsed.append(
            RegionalTelephoneRule(
                str(identifier),
                str(raw["region"]),
                str(raw["scope"]),
                str(raw["descriptor_sha256"]),
                parsed_types,
                str(raw.get("trunk_prefix", "")),
                tuple(raw["modes"]),
                RuleRecipe(tuple(recipe_raw["tokens"]), tuple(recipe_raw["provenance"])),
            )
        )
    return tuple(parsed)


def telephone_rule_events() -> Mapping[str, int]:
    return MappingProxyType(dict(_EVENTS))


def generate_telephone_alternatives(value, locale: str, *, max_derivations: int = 128):
    """Generate one sourced regional telephone reading without changing English ranking."""
    from frend.verbalize import SpokenAlternative, _telephone_group

    matched = value.matched_format
    if matched is None:
        raise NotImplementedError("regional telephone generation requires a matched format")
    rule = next(item for item in regional_telephone_formats(locale) if item.id == matched.rule)
    if len(rule.modes) != len(matched.groups):
        raise ValueError(f"telephone rule {rule.id!r} does not consume every group")
    fields = {}
    for index, (digits, mode) in enumerate(zip(matched.groups, rule.modes, strict=True), start=1):
        realizations = tuple(
            Realization((item.text,), (item.provenance,), item.weight)
            for item in _telephone_group(digits, mode, locale)
        )
        fields[f"GROUP{index}"] = FieldProvider(realizations)
    recipes = (rule.recipe,)
    count = exact_composition_count(recipes, fields)
    limit = min(max_derivations, int(TELEPHONE_REGIONAL_POLICY["max_derivations"]))
    generated = compose_recipes(
        recipes,
        fields,
        base_provenance=(
            f"telephone-rule:{locale}",
            "source:vendor-documentation:provisional",
        ),
        make=lambda text, provenance, weight, prior: SpokenAlternative(
            text, provenance, weight, prior_provenance=prior
        ),
        render=lambda core, suffix, selected: "".join(core),
        max_derivations=limit,
        exact_count=count,
    )
    _EVENTS["generate"] += 1
    return preserve_legacy_ranking(generated, (), unmatched="drop")
