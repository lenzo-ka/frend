from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest
from tiergraph import Realization

from frend.rule_composition import (
    FieldProvider,
    RuleRecipe,
    compose_recipes,
    exact_composition_count,
    ordered_placeholders,
    preserve_legacy_ranking,
)


@dataclass(frozen=True)
class Alternative:
    text: str
    provenance: str
    weight: Decimal | None = None
    prior_provenance: str | None = None


def _make(text, provenance, weight, prior):
    return Alternative(text, provenance, weight, prior)


def test_recipe_order_literals_repeated_placeholders_and_provenance_are_stable():
    recipe = RuleRecipe(("<", "\ufff0A\ufff1", ":", "\ufff0A\ufff1", ">"), ("rule", "dupe"))
    fields = {
        "A": FieldProvider(
            (
                Realization(("one",), ("dupe", "leaf-one"), Decimal("1")),
                Realization(("two",), ("leaf-two",), Decimal("2")),
            ),
            ("field",),
        )
    }

    assert ordered_placeholders(recipe) == ("A",)
    assert exact_composition_count((recipe,), fields) == 2
    assert compose_recipes(
        (recipe,),
        fields,
        base_provenance=("base", "dupe"),
        make=_make,
        render=lambda core, suffix, selected: "".join((*core, *suffix)),
        max_derivations=2,
    ) == (
        Alternative("<one:one>", "base+dupe+rule+field+leaf-one", Decimal("1"), "dupe"),
        Alternative("<two:two>", "base+dupe+rule+field+leaf-two", Decimal("2"), "leaf-two"),
    )


def test_partial_weights_cap_omission_suffixes_and_render_hook():
    recipes = (RuleRecipe(("\ufff0A\ufff1", " ", "\ufff0B\ufff1"), ()),)
    fields = {
        "A": FieldProvider((Realization(("a",), ("a",), Decimal("1")),)),
        "B": FieldProvider((Realization(("b",), ("b",), None),)),
    }
    suffixes = (Realization(("tail",), ("tail",), Decimal("2")),)
    with pytest.raises(ValueError, match="1 derivations"):
        compose_recipes(
            recipes,
            fields,
            base_provenance=(),
            make=_make,
            render=lambda core, suffix, selected: "".join(core),
            max_derivations=0,
            suffixes=suffixes,
        )
    assert compose_recipes(
        recipes,
        fields,
        base_provenance=(),
        make=_make,
        render=lambda core, suffix, selected: f"{'|'.join(core)}:{','.join(suffix)}",
        max_derivations=1,
        suffixes=suffixes,
        omit_placeholders=frozenset({"A"}),
    ) == (Alternative("b:tail", "b+tail", None, "b"),)


@pytest.mark.parametrize(
    ("unmatched", "expected"),
    [
        ("retain", ("old-only", "same", "same", "new-only")),
        ("drop", ("same", "same", "new-only")),
    ],
)
def test_legacy_ranking_uses_first_duplicate_match_and_explicit_unmatched_policy(
    unmatched, expected
):
    generated = (
        Alternative("same", "generated-first"),
        Alternative("same", "generated-second"),
        Alternative("new-only", "new"),
    )
    previous = (
        Alternative("old-only", "old"),
        Alternative("same", "legacy-first", Decimal("1")),
        Alternative("same", "legacy-second", Decimal("2")),
    )

    result = preserve_legacy_ranking(generated, previous, unmatched=unmatched)

    assert tuple(item.text for item in result) == expected
    same = tuple(item for item in result if item.text == "same")
    assert tuple(item.provenance for item in same) == ("generated-first", "generated-second")
    assert tuple(item.weight for item in same) == (Decimal("1"), Decimal("2"))


@pytest.mark.parametrize("provenance", ("surface:passthrough", "surface:unsupported"))
def test_sourced_spoken_readings_rank_before_raw_surface_fallbacks(provenance):
    generated = (
        Alternative("spoken first", "normalization-record:first"),
        Alternative("spoken second", "normalization-record:second"),
    )
    previous = (
        Alternative("raw", provenance),
        Alternative("legacy spoken", "lexical:locale"),
    )

    result = preserve_legacy_ranking(generated, previous, unmatched="retain")

    assert tuple(item.text for item in result) == (
        "legacy spoken",
        "spoken first",
        "spoken second",
        "raw",
    )


def test_as_written_reading_is_not_treated_as_a_raw_passthrough():
    result = preserve_legacy_ranking(
        (Alternative("generated", "normalization-record:new"),),
        (Alternative("as written", "surface:as-written"),),
        unmatched="retain",
    )

    assert tuple(item.text for item in result) == ("as written", "generated")
