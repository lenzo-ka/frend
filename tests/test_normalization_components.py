import hashlib
import json
from copy import deepcopy

import pytest

from frend.normalization_components import (
    expand_normalization_bundle,
    load_normalization_bundle,
)
from frend.normalization_schema import validate_bundle

_EXPANDED_HASHES = {
    "es_ES": "71e397cec2ddb35499e2cd5de40d81086ec3e1e016daad7d613f87a0d6e2baf7",
    "es_MX": "061fdd5dc59db02fbcd0c148175f3f8ec325c9d77f2affb2fc9a73b3b7c949f9",
    "pt_BR": "1da1b9dff5e01e7830eaf07b37790e460a070986c295a7b44485a9e140189393",
    "pt_PT": "bccaeeed7774cd2d98887289379ba8d2ce4af680a4ad0b53985da02a7a37a519",
}


def _digest(document: object) -> str:
    canonical = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize(("locale", "expected"), _EXPANDED_HASHES.items())
def test_regional_defaults_expand_to_the_measured_preconsolidation_document(locale, expected):
    assert _digest(load_normalization_bundle(locale)) == expected


def test_percent_ancestry_remains_region_specific():
    for first, second in (("es_ES", "es_MX"), ("pt_BR", "pt_PT")):
        first_keys = {
            key for key in load_normalization_bundle(first)["ancestry"] if "percent" in key
        }
        second_keys = {
            key for key in load_normalization_bundle(second)["ancestry"] if "percent" in key
        }
        assert len(first_keys) == len(second_keys) == 1
        assert first_keys != second_keys


def _component() -> dict[str, object]:
    return {
        "schema": "frend-normalization-component/1",
        "language": "es",
        "sources": [],
        "selectors": [],
        "resources": [
            {
                "id": "word",
                "kind": "test",
                "forms": {"default": "base"},
                "derived_from": ["source#row"],
            }
        ],
        "rules": [],
        "weights": [],
        "curated_rows": [],
        "ancestry": {"source#row": ["a" * 64]},
    }


def _regional(locale: str) -> dict[str, object]:
    return {
        "schema": "frend-normalization/2",
        "locale": locale,
        "provenance": {"source": "test"},
        "versions": {"icu": "78.3"},
        "components": ["es-default.json"],
        "resources": [],
        "ancestry": {},
    }


def test_explicit_override_changes_only_its_region():
    first = _regional("es_ES")
    first["resources"] = [
        {
            "id": "word",
            "override": True,
            "kind": "test",
            "forms": {"default": "regional"},
            "derived_from": ["source#row"],
        }
    ]
    second = _regional("es_MX")

    def loader(_name):
        return _component()

    assert expand_normalization_bundle(first, component_loader=loader)["resources"][0]["forms"] == {
        "default": "regional"
    }
    assert expand_normalization_bundle(second, component_loader=loader)["resources"][0][
        "forms"
    ] == {"default": "base"}


def test_implicit_duplicate_and_unknown_override_are_rejected():
    duplicate = _regional("es_ES")
    duplicate["resources"] = [deepcopy(_component()["resources"][0])]
    with pytest.raises(ValueError, match="implicitly duplicates"):
        expand_normalization_bundle(duplicate, component_loader=lambda _name: _component())
    unknown = _regional("es_ES")
    unknown["resources"] = [
        {
            "id": "unknown",
            "override": True,
            "kind": "test",
            "forms": {"default": "regional"},
            "derived_from": ["source#row"],
        }
    ]
    with pytest.raises(ValueError, match="overrides unknown"):
        expand_normalization_bundle(unknown, component_loader=lambda _name: _component())


def test_component_language_drift_and_missing_ancestry_are_rejected():
    wrong = _component()
    wrong["language"] = "pt"
    with pytest.raises(ValueError, match="does not match"):
        expand_normalization_bundle(_regional("es_ES"), component_loader=lambda _name: wrong)
    missing = _component()
    del missing["resources"][0]["derived_from"]
    document = expand_normalization_bundle(
        _regional("es_ES"), component_loader=lambda _name: missing
    )
    with pytest.raises(ValueError, match="missing-derived-from"):
        validate_bundle(document)
