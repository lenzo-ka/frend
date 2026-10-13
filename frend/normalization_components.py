"""Expand language normalization components into locale-complete documents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType

_COMPONENT_SCHEMA = "frend-normalization-component/1"
_ENTRY_SECTIONS = ("selectors", "resources", "rules", "weights", "curated_rows")
_NORMALIZATION_DATA = files("frend").joinpath("data", "normalization")


@lru_cache(maxsize=64)
def _packaged_component(name: str) -> Mapping[str, object]:
    resource = _NORMALIZATION_DATA.joinpath(name)
    document = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"normalization component {name!r} is not an object")
    return document


def _merge_entries(base: list[object], overlay: object, *, section: str) -> list[object]:
    if not isinstance(overlay, list):
        raise ValueError(f"normalization overlay {section!r} is not a list")
    result = list(base)
    positions = {
        item.get("id"): index
        for index, item in enumerate(result)
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if len(positions) != len(result):
        raise ValueError(f"normalization component has invalid or duplicate {section} IDs")
    for raw in overlay:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
            raise ValueError(f"normalization overlay has invalid {section} entry")
        item = dict(raw)
        identifier = str(item["id"])
        overriding = item.pop("override", False)
        if overriding is not False and overriding is not True:
            raise ValueError(f"normalization override marker for {identifier!r} is not boolean")
        if identifier in positions:
            if not overriding:
                raise ValueError(f"normalization overlay implicitly duplicates {identifier!r}")
            result[positions[identifier]] = item
        elif overriding:
            raise ValueError(f"normalization overlay overrides unknown {identifier!r}")
        else:
            positions[identifier] = len(result)
            result.append(item)
    return result


def expand_normalization_bundle(
    document: Mapping[str, object],
    *,
    component_loader: Callable[[str], Mapping[str, object]] = _packaged_component,
) -> dict[str, object]:
    """Return a component-free locale document, applying explicit stable-ID overrides."""
    components = document.get("components")
    if components is None:
        return dict(document)
    if (
        not isinstance(components, list)
        or not components
        or not all(isinstance(item, str) and item for item in components)
    ):
        raise ValueError("normalization components must be a nonempty string list")
    locale = document.get("locale")
    if not isinstance(locale, str) or "_" not in locale:
        raise ValueError("component normalization bundle has no regional locale")
    language = locale.partition("_")[0]
    merged_sections: dict[str, list[object]] = {section: [] for section in _ENTRY_SECTIONS}
    merged_sources: list[object] = []
    merged_ancestry: dict[str, object] = {}
    for name in components:
        component = component_loader(name)
        if component.get("schema") != _COMPONENT_SCHEMA or component.get("language") != language:
            raise ValueError(f"normalization component {name!r} does not match {locale!r}")
        sources = component.get("sources")
        ancestry = component.get("ancestry")
        if not isinstance(sources, list) or not isinstance(ancestry, Mapping):
            raise ValueError(f"normalization component {name!r} is incomplete")
        merged_sources.extend(sources)
        for key, value in ancestry.items():
            if key in merged_ancestry:
                raise ValueError(f"normalization components repeat ancestry {key!r}")
            merged_ancestry[str(key)] = value
        for section in _ENTRY_SECTIONS:
            merged_sections[section] = _merge_entries(
                merged_sections[section], component.get(section, []), section=section
            )

    expanded = {key: value for key, value in document.items() if key != "components"}
    overlay_sources = expanded.get("sources", [])
    overlay_ancestry = expanded.get("ancestry", {})
    if not isinstance(overlay_sources, list) or not isinstance(overlay_ancestry, Mapping):
        raise ValueError("normalization overlay sources or ancestry is invalid")
    expanded["sources"] = [*merged_sources, *overlay_sources]
    for key, value in overlay_ancestry.items():
        if key in merged_ancestry:
            raise ValueError(f"normalization overlay repeats ancestry {key!r}")
        merged_ancestry[str(key)] = value
    expanded["ancestry"] = merged_ancestry
    for section in _ENTRY_SECTIONS:
        expanded[section] = _merge_entries(
            merged_sections[section], expanded.get(section, []), section=section
        )
    if expanded.get("locale") != locale:
        raise ValueError("normalization expansion changed locale identity")
    return expanded


@lru_cache(maxsize=64)
def load_normalization_document(name: str) -> dict[str, object]:
    """Load and expand one packaged normalization document by basename."""
    if not name or name != name.rsplit("/", 1)[-1] or not name.endswith(".json"):
        raise ValueError(f"invalid normalization document name {name!r}")
    resource = _NORMALIZATION_DATA.joinpath(name)
    document = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"normalization document {name!r} is not an object")
    return expand_normalization_bundle(document)


@lru_cache(maxsize=16)
def load_normalization_bundle(locale: str) -> dict[str, object]:
    """Load and expand a packaged locale normalization document once."""
    expanded = load_normalization_document(f"{locale}.json")
    if expanded.get("locale") != locale:
        raise ValueError(f"normalization bundle {locale!r} has locale drift")
    return expanded


def load_runtime_records(
    name: str, locale: str, normalization_sha256: object
) -> Mapping[str, Mapping[str, object]]:
    """Verify a normalization authority and index its runtime resource records."""
    document = load_normalization_document(name)
    if document.get("locale") != locale:
        raise ValueError(f"normalization document {name!r} has locale drift")
    canonical = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    if normalization_sha256 != hashlib.sha256(canonical).hexdigest():
        raise ValueError(f"normalization document {name!r} hash mismatch")
    raw = document.get("resources")
    if not isinstance(raw, list):
        raise ValueError(f"normalization document {name!r} has invalid resources")
    records = {
        str(item["id"]): MappingProxyType(dict(item))
        for item in raw
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if len(records) != len(raw):
        raise ValueError(f"normalization document {name!r} has invalid or duplicate resources")
    return MappingProxyType(records)
