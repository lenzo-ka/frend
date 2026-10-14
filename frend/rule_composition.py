"""Private deterministic execution for generated-rule recipe families."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import lru_cache
from itertools import product
from typing import Literal

from tiergraph import Realization

_OPEN = "\ufff0"
_CLOSE = "\ufff1"


@dataclass(frozen=True, slots=True)
class RuleRecipe:
    tokens: tuple[str, ...]
    provenance: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FieldProvider:
    realization: tuple[Realization, ...]
    provenance: tuple[str, ...] = ()
    prior: bool = True


def placeholder(symbol: str) -> str:
    if not symbol or _OPEN in symbol or _CLOSE in symbol:
        raise ValueError(f"invalid recipe placeholder symbol {symbol!r}")
    return f"{_OPEN}{symbol}{_CLOSE}"


def placeholder_symbol(token: str) -> str | None:
    if token.startswith(_OPEN) and token.endswith(_CLOSE) and len(token) > 2:
        return token[1:-1]
    return None


@lru_cache(maxsize=256)
def ordered_placeholders(recipe: RuleRecipe) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            symbol for token in recipe.tokens if (symbol := placeholder_symbol(token)) is not None
        )
    )


def exact_composition_count(
    recipes: Sequence[RuleRecipe],
    fields: Mapping[str, FieldProvider],
    *,
    suffixes: Sequence[Realization] | None = None,
) -> int:
    suffix_count = 1 if suffixes is None else len(suffixes)
    return sum(
        math.prod(len(fields[symbol].realization) for symbol in ordered_placeholders(recipe))
        * suffix_count
        for recipe in recipes
    )


def compose_recipes[T](
    recipes: Sequence[RuleRecipe],
    fields: Mapping[str, FieldProvider],
    *,
    base_provenance: Sequence[str],
    make: Callable[[str, str, Decimal | None, str | None], T],
    render: Callable[[tuple[str, ...], tuple[str, ...], Mapping[str, Realization]], str],
    max_derivations: int,
    exact_count: int | None = None,
    suffixes: Sequence[Realization] | None = None,
    omit_placeholders: frozenset[str] = frozenset(),
    provenance_filter: Callable[[str], bool] = bool,
    combine_weights: bool = True,
) -> tuple[T, ...]:
    """Realize recipes without changing declared path, field, or suffix order."""
    if exact_count is None:
        exact_count = exact_composition_count(recipes, fields, suffixes=suffixes)
    if exact_count < 1 or exact_count > max_derivations:
        raise ValueError(f"rule composition has {exact_count} derivations")
    suffix_options: Sequence[Realization | None] = (None,) if suffixes is None else suffixes
    output = []
    for recipe in recipes:
        symbols = ordered_placeholders(recipe)
        try:
            choices = tuple(fields[symbol].realization for symbol in symbols)
        except KeyError as exc:
            raise ValueError(f"recipe has no field provider for {exc.args[0]!r}") from exc
        for selected_values in product(*choices):
            selected = dict(zip(symbols, selected_values, strict=True))
            core: list[str] = []
            provenance = [*base_provenance, *recipe.provenance]
            prior: list[str] = []
            weights: list[Decimal | None] = []
            folded: set[str] = set()
            skip_space = False
            for token in recipe.tokens:
                symbol = placeholder_symbol(token)
                realization = selected.get(symbol) if symbol is not None else None
                if realization is None:
                    if skip_space and token.isspace():
                        skip_space = False
                        continue
                    core.append(token)
                    continue
                if symbol in omit_placeholders:
                    skip_space = True
                    continue
                if symbol not in folded:
                    provider = fields[symbol]
                    provenance.extend(provider.provenance)
                    provenance.extend(realization.provenance)
                    if provider.prior:
                        prior.extend(realization.provenance[:1])
                    if combine_weights:
                        weights.append(realization.weight)
                    folded.add(symbol)
                core.extend(realization.tokens)
            for suffix in suffix_options:
                combined_provenance = list(provenance)
                combined_weights = list(weights)
                suffix_tokens: tuple[str, ...] = ()
                if suffix is not None:
                    suffix_tokens = suffix.tokens
                    combined_provenance.extend(suffix.provenance)
                    if combine_weights:
                        combined_weights.append(suffix.weight)
                sources = tuple(
                    dict.fromkeys(item for item in combined_provenance if provenance_filter(item))
                )
                prior_sources = tuple(
                    dict.fromkeys(item for item in prior if provenance_filter(item))
                )
                weight = (
                    sum(combined_weights, Decimal(0))
                    if combined_weights and all(item is not None for item in combined_weights)
                    else None
                )
                output.append(
                    make(
                        render(tuple(core), suffix_tokens, selected),
                        "+".join(sources),
                        weight,
                        "+".join(prior_sources) or None,
                    )
                )
    if len(output) != exact_count:
        raise ValueError(f"rule composition returned {len(output)} of {exact_count} derivations")
    return tuple(output)


def preserve_legacy_ranking[T](
    generated: Sequence[T],
    previous: Sequence[T],
    *,
    unmatched: Literal["retain", "drop"],
) -> tuple[T, ...]:
    """Match duplicate texts first-to-first while preserving real-reading order.

    A raw passthrough is a completeness fallback, not a spoken reading.  Keep its
    relative order with other raw fallbacks, but place it after every generated
    spoken reading.  The order of all non-fallback legacy and generated readings is
    otherwise unchanged.
    """
    remaining = list(generated)
    preserved = []
    raw_fallbacks = []
    for old in previous:
        match = next((item for item in remaining if item.text == old.text), None)
        if match is None:
            if unmatched == "retain":
                target = (
                    raw_fallbacks
                    if old.provenance in {"surface:passthrough", "surface:unsupported"}
                    else preserved
                )
                target.append(old)
            continue
        remaining.remove(match)
        target = (
            raw_fallbacks
            if old.provenance in {"surface:passthrough", "surface:unsupported"}
            else preserved
        )
        target.append(
            replace(
                match,
                weight=old.weight,
                prior_provenance=old.prior_provenance or old.provenance,
            )
        )
    return tuple((*preserved, *remaining, *raw_fallbacks))
