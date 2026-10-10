"""Schema and ancestry validation for declarative normalization bundles.

This module validates bundle data only.  Runtime normalization deliberately does not
load these bundles yet.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from itertools import product

import icu

from frend.data_sources import resolve_receipt, shipping_refusals, source_class

__all__ = ["bundle_refusals", "validate_bundle"]

_SCHEMA = "frend-normalization/2"
_ENTRY_SECTIONS = ("selectors", "resources", "rules", "weights", "curated_rows")


def _reason(name: str, detail: str) -> str:
    return f"{name}: {detail}"


def _is_string_list(value: object, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _entry_refusals(bundle: Mapping[object, object]) -> tuple[list[str], list[str]]:
    refusals: list[str] = []
    roots: list[str] = []
    for section in _ENTRY_SECTIONS:
        entries = bundle.get(section, [])
        if not isinstance(entries, list):
            refusals.append(_reason("invalid-section", f"{section!r} is not a list"))
            continue
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            location = f"{section}[{index}]"
            if not isinstance(entry, Mapping):
                refusals.append(_reason("invalid-entry", f"{location} is not a mapping"))
                continue
            identifier = entry.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                refusals.append(_reason("invalid-entry", f"{location} has no nonempty id"))
            elif identifier in seen:
                refusals.append(_reason("duplicate-entry-id", f"{section} repeats {identifier!r}"))
            else:
                seen.add(identifier)
            derived_from = entry.get("derived_from")
            if not _is_string_list(derived_from, nonempty=True):
                refusals.append(
                    _reason("missing-derived-from", f"{location} has no nonempty ancestry")
                )
            else:
                roots.extend(derived_from)
    return refusals, roots


def _selector_refusals(bundle: Mapping[object, object]) -> list[str]:
    refusals: list[str] = []
    selectors = bundle.get("selectors", [])
    if not isinstance(selectors, list):
        return refusals
    selector_ids: set[str] = set()
    for index, selector in enumerate(selectors):
        if not isinstance(selector, Mapping):
            continue
        identifier = selector.get("id")
        if isinstance(identifier, str):
            selector_ids.add(identifier)
        location = f"selectors[{index}]"
        inputs = selector.get("inputs")
        reachable = selector.get("reachable")
        mapping = selector.get("mapping")
        if (
            not _is_string_list(inputs, nonempty=True)
            or len(set(inputs)) != len(inputs)
            or not isinstance(reachable, Mapping)
            or set(reachable) != set(inputs)
            or not isinstance(mapping, Mapping)
        ):
            refusals.append(
                _reason(
                    "invalid-selector",
                    f"{location} requires unique inputs, matching reachable axes, and a mapping",
                )
            )
            continue

        axes: list[list[str]] = []
        valid_axes = True
        for input_name in inputs:
            values = reachable[input_name]
            if (
                not _is_string_list(values, nonempty=True)
                or len(set(values)) != len(values)
                or any("." in value for value in values)
            ):
                refusals.append(
                    _reason(
                        "invalid-selector",
                        f"{location}.reachable[{input_name!r}] is not a nonempty unique value list",
                    )
                )
                valid_axes = False
                continue
            axes.append(values)
        if not valid_axes:
            continue

        if "number" in inputs:
            number_values = reachable["number"]
            plural_source = selector.get("plural_source")
            if plural_source != "cldr-cardinal":
                refusals.append(
                    _reason(
                        "invalid-selector",
                        f"{location} number axis requires plural_source 'cldr-cardinal'",
                    )
                )
            else:
                locale = bundle.get("locale")
                if isinstance(locale, str) and locale.strip():
                    plural_values = set(icu.PluralRules.forLocale(icu.Locale(locale)).getKeywords())
                    if set(number_values) != plural_values:
                        refusals.append(
                            _reason(
                                "invalid-selector",
                                f"{location}.reachable['number'] does not match "
                                f"CLDR cardinal values {sorted(plural_values)!r}",
                            )
                        )
        if "case" in inputs:
            case_values = selector.get("case_values")
            if (
                not _is_string_list(case_values, nonempty=True)
                or len(set(case_values)) != len(case_values)
                or set(reachable["case"]) != set(case_values)
            ):
                refusals.append(
                    _reason(
                        "invalid-selector",
                        f"{location}.reachable['case'] does not match case_values",
                    )
                )

        expected = {".".join(values) for values in product(*axes)}
        observed: set[str] = set()
        for key, result in mapping.items():
            if not isinstance(key, str):
                refusals.append(
                    _reason("unknown-selector-value", f"{location} has a non-string tuple")
                )
                continue
            values = key.split(".")
            if len(values) != len(inputs) or any(
                value not in axes[position] for position, value in enumerate(values)
            ):
                refusals.append(
                    _reason("unknown-selector-value", f"{location} maps unknown tuple {key!r}")
                )
            else:
                observed.add(key)
            if not isinstance(result, str) or not result.strip():
                refusals.append(
                    _reason("invalid-selector", f"{location}.mapping[{key!r}] is empty")
                )
        missing = sorted(expected - observed)
        if missing:
            refusals.append(_reason("non-total-selector", f"{location} does not map {missing!r}"))

    rules = bundle.get("rules", [])
    if isinstance(rules, list):
        for index, rule in enumerate(rules):
            if not isinstance(rule, Mapping) or "select" not in rule:
                continue
            selected = rule["select"]
            if selected not in selector_ids:
                refusals.append(_reason("unknown-selector", f"rules[{index}] selects {selected!r}"))
    return refusals


def _content_refusals(bundle: Mapping[object, object]) -> list[str]:
    refusals: list[str] = []
    resources = bundle.get("resources", [])
    resource_forms: list[tuple[str, Mapping[object, object]]] = []
    if isinstance(resources, list):
        for index, resource in enumerate(resources):
            if not isinstance(resource, Mapping):
                continue
            if not isinstance(resource.get("kind"), str) or not resource["kind"].strip():
                refusals.append(
                    _reason("invalid-resource", f"resources[{index}] has no nonempty kind")
                )
            forms = resource.get("forms")
            if (
                not isinstance(forms, Mapping)
                or not forms
                or not all(
                    isinstance(key, str)
                    and bool(key.strip())
                    and isinstance(value, str)
                    and bool(value.strip())
                    for key, value in forms.items()
                )
            ):
                refusals.append(
                    _reason(
                        "invalid-resource",
                        f"resources[{index}].forms is not a nonempty string mapping",
                    )
                )
            else:
                resource_forms.append((resource["kind"], forms))

    selector_inputs: dict[str, set[str]] = {}
    selector_outputs: dict[str, set[str]] = {}
    selectors = bundle.get("selectors", [])
    if isinstance(selectors, list):
        for selector in selectors:
            if not isinstance(selector, Mapping) or not isinstance(selector.get("id"), str):
                continue
            inputs = selector.get("inputs")
            mapping = selector.get("mapping")
            if _is_string_list(inputs, nonempty=True):
                selector_inputs[selector["id"]] = set(inputs)
            if isinstance(mapping, Mapping):
                selector_outputs[selector["id"]] = {
                    result
                    for result in mapping.values()
                    if isinstance(result, str) and bool(result.strip())
                }

    rules = bundle.get("rules", [])
    if isinstance(rules, list):
        for index, rule in enumerate(rules):
            if not isinstance(rule, Mapping):
                continue
            if not _is_string_list(rule.get("fields"), nonempty=True):
                refusals.append(_reason("invalid-rule", f"rules[{index}] has no nonempty fields"))
                fields: set[str] = set()
            else:
                fields = set(rule["fields"])
            selected = rule.get("select")
            required_inputs = selector_inputs.get(selected, set())
            missing_inputs = sorted(required_inputs - fields)
            if missing_inputs:
                refusals.append(
                    _reason(
                        "invalid-rule",
                        f"rules[{index}] omits selector inputs {missing_inputs!r}",
                    )
                )
            outputs = selector_outputs.get(selected, set())
            for kind, forms in resource_forms:
                if kind not in fields:
                    continue
                missing_forms = sorted(outputs - set(forms))
                if missing_forms:
                    refusals.append(
                        _reason(
                            "missing-resource-form",
                            f"rules[{index}] can select absent {kind!r} forms {missing_forms!r}",
                        )
                    )
            directions = rule.get("directions")
            if not _is_string_list(directions, nonempty=True) or any(
                direction not in {"tn", "itn"} for direction in directions
            ):
                refusals.append(
                    _reason(
                        "invalid-rule",
                        f"rules[{index}] directions are not declared TN/ITN directions",
                    )
                )
    return refusals


def _leaf_refusals(
    leaf: str,
    receipt_index: Mapping[str, Mapping[str, object]] | None,
    declared_receipts: set[str],
) -> list[str]:
    boundary = shipping_refusals({"ancestors": [leaf]}, receipt_index=receipt_index)
    if any("LDC" in refusal for refusal in boundary):
        return [_reason("ldc-leaf", f"ancestry reaches forbidden leaf {leaf!r}")]
    if boundary:
        return [_reason("unregistered-leaf", f"ancestry reaches {leaf!r}")]
    if leaf not in declared_receipts:
        return [_reason("undeclared-leaf", f"ancestry reaches undeclared receipt {leaf!r}")]
    return []


def _ancestry_refusals(
    bundle: Mapping[object, object],
    roots: Sequence[str],
    receipt_index: Mapping[str, Mapping[str, object]] | None,
    source_receipts: Mapping[str, set[str]],
) -> list[str]:
    ancestry = bundle.get("ancestry")
    if not isinstance(ancestry, Mapping):
        return [_reason("invalid-ancestry", "ancestry is not a mapping")]

    refusals: list[str] = []
    graph: dict[str, list[str]] = {}
    for node, parents in ancestry.items():
        if (
            not isinstance(node, str)
            or not node.strip()
            or not _is_string_list(parents, nonempty=True)
        ):
            refusals.append(
                _reason("invalid-ancestry", f"node {node!r} has no nonempty parent list")
            )
            continue
        graph[node] = parents

    finished: set[str] = set()
    visiting: list[str] = []

    def walk(node: str) -> None:
        if node in finished:
            return
        if node in visiting:
            start = visiting.index(node)
            cycle = " -> ".join((*visiting[start:], node))
            refusals.append(_reason("ancestry-cycle", cycle))
            return
        parents = graph.get(node)
        if parents is None:
            refusals.extend(
                _leaf_refusals(
                    node,
                    receipt_index,
                    set().union(*source_receipts.values()) if source_receipts else set(),
                )
            )
            finished.add(node)
            return
        visiting.append(node)
        for parent in parents:
            walk(parent)
        visiting.pop()
        finished.add(node)

    for root in (*roots, *graph):
        walk(root)

    def leaves(node: str, seen: set[str]) -> set[str]:
        if node in seen:
            return set()
        parents = graph.get(node)
        if parents is None:
            return {node}
        return set().union(*(leaves(parent, seen | {node}) for parent in parents))

    for node in graph:
        source_id = node.partition("#")[0]
        expected = source_receipts.get(source_id)
        if expected is None:
            continue
        wrong = sorted(leaves(node, set()) - expected)
        if wrong:
            refusals.append(
                _reason(
                    "source-ancestry-mismatch",
                    f"{node!r} closes at receipts not declared for {source_id!r}: {wrong!r}",
                )
            )
    return refusals


def _receipt_fingerprint(receipt: Mapping[str, object]) -> str:
    canonical = json.dumps(
        dict(receipt), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _source_refusals(
    bundle: Mapping[object, object],
    receipt_index: Mapping[str, Mapping[str, object]] | None,
) -> tuple[list[str], list[str], dict[str, set[str]]]:
    sources = bundle.get("sources")
    if not isinstance(sources, list) or not sources:
        return [_reason("invalid-sources", "sources is not a nonempty list")], [], {}
    refusals: list[str] = []
    receipts: list[str] = []
    source_receipts: dict[str, set[str]] = {}
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            refusals.append(_reason("invalid-source", f"sources[{index}] is not a mapping"))
            continue
        identifier = source.get("id")
        if not isinstance(identifier, str) or source_class(identifier) is None:
            refusals.append(
                _reason("unregistered-source", f"sources[{index}] names {identifier!r}")
            )
        receipt = source.get("receipt")
        if not isinstance(receipt, str) or not receipt.strip():
            refusals.append(_reason("unregistered-leaf", f"sources[{index}] has no receipt"))
        else:
            resolved = resolve_receipt(receipt, receipt_index)
            if resolved is None:
                refusals.append(
                    _reason(
                        "unregistered-leaf",
                        f"sources[{index}] receipt {receipt!r} is not registered",
                    )
                )
                continue
            fingerprint = _receipt_fingerprint(resolved)
            receipts.append(fingerprint)
            if resolved.get("source") != identifier:
                refusals.append(
                    _reason(
                        "source-receipt-mismatch",
                        f"sources[{index}] id {identifier!r} does not match receipt source "
                        f"{resolved.get('source')!r}",
                    )
                )
            if isinstance(identifier, str):
                source_receipts.setdefault(identifier, set()).add(fingerprint)
    return refusals, receipts, source_receipts


def bundle_refusals(
    bundle: object,
    *,
    receipt_index: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[str, ...]:
    """Return named reasons why a ``frend-normalization/2`` bundle is invalid."""
    if not isinstance(bundle, Mapping):
        return (_reason("invalid-bundle", "bundle is not a mapping"),)

    refusals: list[str] = []
    if bundle.get("schema") != _SCHEMA:
        refusals.append(_reason("wrong-schema", f"expected {_SCHEMA!r}"))
    locale = bundle.get("locale")
    if not isinstance(locale, str) or not locale.strip():
        refusals.append(_reason("invalid-locale", "locale is missing or empty"))
    versions = bundle.get("versions")
    if (
        not isinstance(versions, Mapping)
        or not versions
        or not all(
            isinstance(key, str)
            and bool(key.strip())
            and isinstance(value, str)
            and bool(value.strip())
            for key, value in versions.items()
        )
    ):
        refusals.append(_reason("invalid-versions", "versions is not a nonempty mapping"))

    entry_refusals, roots = _entry_refusals(bundle)
    source_refusals, receipts, source_receipts = _source_refusals(bundle, receipt_index)
    refusals.extend(entry_refusals)
    refusals.extend(source_refusals)
    refusals.extend(_selector_refusals(bundle))
    refusals.extend(_content_refusals(bundle))
    for boundary_refusal in shipping_refusals(bundle, receipt_index=receipt_index):
        if "LDC" in boundary_refusal:
            refusals.append(_reason("ldc-leaf", "bundle contains a forbidden LDC marker"))
        else:
            refusals.append(_reason("package-boundary", boundary_refusal))
    refusals.extend(
        _ancestry_refusals(
            bundle,
            (*roots, *receipts),
            receipt_index,
            source_receipts,
        )
    )
    return tuple(dict.fromkeys(refusals))


def validate_bundle(
    bundle: object,
    *,
    receipt_index: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Raise ``ValueError`` with named reasons unless ``bundle`` is valid."""
    refusals = bundle_refusals(bundle, receipt_index=receipt_index)
    if refusals:
        raise ValueError("; ".join(refusals))
