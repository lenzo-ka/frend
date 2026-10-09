"""Validation for declarative normalization bundles."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from frend import data_sources
from frend.data_sources import register_receipt
from frend.normalization_schema import bundle_refusals, validate_bundle

_FIXTURES = Path(__file__).parent / "data" / "normalization"


def _load(name: str) -> object:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _receipt_index() -> dict[str, dict[str, object]]:
    receipts = {}
    for path in sorted((_FIXTURES / "receipts").glob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        canonical = json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        receipts[hashlib.sha256(canonical).hexdigest()] = receipt
        receipts[f"receipts/{path.name}"] = receipt
    return receipts


def _valid_bundle() -> dict[str, object]:
    bundle = _load("ru_RU-RUB.json")
    assert isinstance(bundle, dict)
    return bundle


def _assert_refused(bundle: object, reason: str) -> None:
    refusals = bundle_refusals(bundle, receipt_index=_receipt_index())
    assert any(item.startswith(f"{reason}:") for item in refusals), refusals
    with pytest.raises(ValueError, match=reason):
        validate_bundle(bundle, receipt_index=_receipt_index())


def test_recipe_ru_ru_rub_example_validates():
    bundle = _valid_bundle()

    assert bundle_refusals(bundle, receipt_index=_receipt_index()) == ()
    validate_bundle(bundle, receipt_index=_receipt_index())


def test_recipe_example_accepts_receipts_from_the_process_registry(monkeypatch):
    monkeypatch.setattr(data_sources, "_RECEIPT_INDEX", {})
    bundle = _valid_bundle()
    for source in bundle["sources"]:
        receipt = _receipt_index()[source["receipt"]]
        source["receipt"] = register_receipt(receipt)

    validate_bundle(bundle)


def test_missing_derived_from_is_refused_with_named_reason():
    bundle = _valid_bundle()
    del bundle["resources"][0]["derived_from"]

    _assert_refused(bundle, "missing-derived-from")


@pytest.mark.parametrize("section", ["weights", "curated_rows"])
def test_every_optional_fact_entry_needs_derived_from(section):
    bundle = _valid_bundle()
    bundle[section] = [{"id": "uncited"}]

    _assert_refused(bundle, "missing-derived-from")


def test_ancestry_cycle_is_refused_with_named_reason():
    bundle = _valid_bundle()
    bundle["ancestry"]["cldr/48#grammar"] = ["cldr/48#RUB"]

    _assert_refused(bundle, "ancestry-cycle")


def test_unregistered_leaf_is_refused_with_named_reason():
    bundle = _valid_bundle()
    bundle["ancestry"]["cldr/48#grammar"] = ["f" * 64]

    _assert_refused(bundle, "unregistered-leaf")


def test_ldc_leaf_is_refused_with_named_reason():
    bundle = _valid_bundle()
    bundle["ancestry"]["cldr/48#grammar"] = ["LDC93S6A"]

    refusals = bundle_refusals(bundle, receipt_index=_receipt_index())

    assert any(
        refusal.startswith("ldc-leaf:") and "ancestry reaches forbidden leaf" in refusal
        for refusal in refusals
    )


def test_ldc_marker_away_from_a_leaf_is_also_refused():
    bundle = _valid_bundle()
    bundle["resources"][0]["note"] = "consulted LDC93S6A"

    _assert_refused(bundle, "ldc-leaf")


def test_non_total_selector_is_refused_with_named_reason():
    bundle = _valid_bundle()
    del bundle["selectors"][0]["mapping"]["many.genitive"]

    _assert_refused(bundle, "non-total-selector")


def test_unknown_selector_value_is_refused_with_named_reason():
    bundle = _valid_bundle()
    selector = bundle["selectors"][0]
    selector["mapping"]["one.instrumental"] = selector["mapping"].pop("one.genitive")

    _assert_refused(bundle, "unknown-selector-value")


def test_cardinal_reachable_values_must_match_icu_for_the_locale():
    bundle = _valid_bundle()
    selector = bundle["selectors"][0]
    selector["reachable"]["number"] = ["one", "other"]
    selector["mapping"] = {
        key: value
        for key, value in selector["mapping"].items()
        if key.split(".")[0] in {"one", "other"}
    }

    _assert_refused(bundle, "invalid-selector")


def test_case_reachable_values_must_match_the_declared_cases():
    bundle = _valid_bundle()
    selector = bundle["selectors"][0]
    selector["reachable"]["case"].append("instrumental")
    selector["mapping"].update(
        {
            f"{number}.instrumental": f"nominative.{number}"
            for number in ("one", "few", "many", "other")
        }
    )

    _assert_refused(bundle, "invalid-selector")


def test_selector_outputs_must_exist_on_resources_a_rule_can_use():
    bundle = _valid_bundle()
    bundle["selectors"][0]["mapping"]["one.nominative"] = "nominative.oen"

    _assert_refused(bundle, "missing-resource-form")


def test_rule_fields_must_supply_the_selected_selector_inputs():
    bundle = _valid_bundle()
    bundle["rules"][0]["fields"].remove("case")

    _assert_refused(bundle, "invalid-rule")


@pytest.mark.parametrize(
    ("entry", "detail"),
    [
        ({"source": "nemo", "use": "ranking"}, "nemo may be used"),
        ({"marker": "internal-only"}, "internal-only ancestry"),
        (
            {"digest": "5d1b84c18b57c62e2c369ce93b3fdc607a2487de17acb74ec72503e242300034"},
            "cataloged internal digest",
        ),
    ],
)
def test_package_boundary_refusals_are_preserved(entry, detail):
    bundle = _valid_bundle()
    bundle["weights"] = [{"id": "w", "derived_from": ["cldr/48#grammar"], **entry}]

    refusals = bundle_refusals(bundle, receipt_index=_receipt_index())

    assert any(
        refusal.startswith("package-boundary:") and detail in refusal for refusal in refusals
    )


def test_source_id_must_match_the_resolved_receipt():
    bundle = _valid_bundle()
    bundle["sources"][0]["receipt"] = "receipts/public-ru-grammar.json"

    _assert_refused(bundle, "source-receipt-mismatch")


def test_source_labeled_ancestry_must_close_at_its_declared_receipt():
    bundle = _valid_bundle()
    bundle["ancestry"]["cldr/48#grammar"] = [
        "06de57dd0eb5136b40470749fdac9ef5e2361a37b59e7d717566437cf803b2d3"
    ]

    _assert_refused(bundle, "source-ancestry-mismatch")


def test_every_ancestry_leaf_must_be_declared_as_a_source_receipt():
    bundle = _valid_bundle()
    bundle["sources"] = [bundle["sources"][0]]

    _assert_refused(bundle, "undeclared-leaf")


def test_validation_does_not_mutate_the_bundle():
    bundle = _valid_bundle()
    before = copy.deepcopy(bundle)

    validate_bundle(bundle, receipt_index=_receipt_index())

    assert bundle == before
