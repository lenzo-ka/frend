#!/usr/bin/env python3
"""Compile sourced normalization-v2 telephone records into tiergraph projections."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import phonenumbers
from tiergraph import (
    AttributeValue,
    GrammarDeclaration,
    GrammarHole,
    GrammarInput,
    GrammarInputToken,
    GrammarRule,
    GrammarTerminal,
    QualifiedName,
    Realization,
    XsdType,
    generate,
    lower_grammar,
    recognize,
)

from frend.artifact_policy import TELEPHONE_REGIONAL_POLICY, policy_document
from frend.generated_rule_locales import locales_for
from frend.normalization_schema import validate_bundle

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "frend" / "data" / "normalization"
DATA = REPO / "frend" / "data" / "telephone_rules"
LOCALES = locales_for("telephone")
_RULE_KEYS = {
    "id",
    "kind",
    "fields",
    "directions",
    "target",
    "value",
    "source_category",
    "provenance_status",
    "gender",
    "number",
    "derived_from",
}
_VALUE_KEYS = {
    "region",
    "scope",
    "metadata_version",
    "descriptor_sha256",
    "number_types",
    "trunk_prefix",
    "group_modes",
    "max_derivations",
}


def _bytes(document: object) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _receipt_index(document: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    result = {}
    for source in document["sources"]:
        reference = source["receipt"]
        receipt = json.loads((RECORDS / reference).read_text("utf-8"))
        result[reference] = receipt
        result[hashlib.sha256(_canonical_bytes(receipt)).hexdigest()] = receipt
    return result


def _string(name: QualifiedName, value: str) -> AttributeValue:
    return AttributeValue(name, XsdType.STRING, value)


def _validate_family(document: Mapping[str, object]) -> None:
    for record in document["rules"]:
        if set(record) != _RULE_KEYS or record.get("kind") != "telephone-speech-rule":
            raise ValueError(f"invalid telephone record {record.get('id')!r}")
        value = record.get("value")
        if not isinstance(value, Mapping) or set(value) != _VALUE_KEYS:
            raise ValueError(f"invalid telephone value {record.get('id')!r}")
        fields = record["fields"]
        modes = value["group_modes"]
        target_fields = [item.casefold() for item in record["target"] if item.startswith("GROUP")]
        if (
            value["scope"] != "national"
            or value["metadata_version"] != phonenumbers.__version__
            or value["max_derivations"] != TELEPHONE_REGIONAL_POLICY["max_derivations"]
            or value["number_types"] not in (["fixed"], ["mobile"])
            or len(fields) != len(modes)
            or len(fields) > TELEPHONE_REGIONAL_POLICY["max_groups"]
            or target_fields != fields
            or len(set(fields)) != len(fields)
            or any(mode not in {"digits", "cardinal"} for mode in modes)
            or record["source_category"] != "vendor documentation"
            or record["provenance_status"] != "provisional"
            or record["gender"] is not None
            or record["number"] is not None
        ):
            raise ValueError(f"telephone record {record['id']!r} violates the family contract")


def _compile_recipes(document: Mapping[str, object]) -> list[dict[str, object]]:
    namespace = f"urn:frend:telephone:v1:{document['locale']}"
    start = QualifiedName(namespace, "TELEPHONE")
    variable = QualifiedName(namespace, "variable")
    rules = []
    nonterminals = [start]
    emitted_fields: set[str] = set()
    for record in document["rules"]:
        names = {
            field.upper(): QualifiedName(namespace, field.upper()) for field in record["fields"]
        }
        holes = {
            field: GrammarHole(_string(variable, field.casefold()), name)
            for field, name in names.items()
        }
        nonterminals.extend(names.values())
        target = tuple(
            holes[item]
            if item in holes
            else GrammarTerminal(_string(QualifiedName(namespace, "text"), item))
            for item in record["target"]
        )
        rules.append(
            GrammarRule(
                start,
                tuple(holes[field] for field in names),
                target,
                provenance=(
                    _string(QualifiedName(namespace, "rule"), f"telephone-id:{record['id']}"),
                    *(
                        _string(
                            QualifiedName(namespace, f"source-{index}"),
                            f"normalization-record:{record['id']}:{source}",
                        )
                        for index, source in enumerate(record["derived_from"], start=1)
                    ),
                ),
            )
        )
        for field, name in names.items():
            if field in emitted_fields:
                continue
            emitted_fields.add(field)
            terminal = GrammarTerminal(_string(QualifiedName(namespace, "field"), f"FIELD:{field}"))
            rules.append(GrammarRule(name, (terminal,), (terminal,)))
    declaration = GrammarDeclaration(tuple(dict.fromkeys(nonterminals)), start, tuple(rules))
    lowered = lower_grammar(declaration)
    output = []
    for record in document["rules"]:
        fields = tuple(field.upper() for field in record["fields"])
        grammar_input = GrammarInput(
            tuple(
                GrammarInputToken(
                    f"FIELD:{field}",
                    (Realization((f"\ufff0{field}\ufff1",), (f"recipe:{field}",), None),),
                    (f"telephone-field:{field}",),
                )
                for field in fields
            )
        )
        forest = recognize(lowered, grammar_input, collapse_units=False)
        result = generate(forest, count=forest.count())
        marker = f"telephone-id:{record['id']}"
        matching = [
            derivation
            for derivation in result.derivations
            if any(
                value.lexical == marker
                for app in derivation.applications
                for value in app.provenance
            )
        ]
        if result.truncated or len(matching) != 1:
            raise ValueError(f"telephone record {record['id']!r} did not lower exactly once")
        derivation = matching[0]
        value = record["value"]
        output.append(
            {
                "id": record["id"],
                "region": value["region"],
                "scope": value["scope"],
                "descriptor_sha256": value["descriptor_sha256"],
                "number_types": value["number_types"],
                "trunk_prefix": value["trunk_prefix"],
                "modes": value["group_modes"],
                "recipe": {
                    "tokens": list(derivation.tokens),
                    "provenance": list(
                        dict.fromkeys(
                            value.lexical
                            for app in derivation.applications
                            for value in app.provenance
                            if not value.lexical.startswith("telephone-id:")
                        )
                    ),
                },
            }
        )
    return output


def build_bundle(locale: str) -> dict[str, object]:
    document = json.loads((RECORDS / f"telephone_{locale}.json").read_text("utf-8"))
    validate_bundle(document, receipt_index=_receipt_index(document))
    _validate_family(document)
    projection = {"rules": _compile_recipes(document)}
    return {
        "schema": 1,
        "locale": locale,
        "provenance": {"source": "frend/curated"},
        "metadata_version": phonenumbers.__version__,
        "normalization_sha256": hashlib.sha256(_canonical_bytes(document)).hexdigest(),
        "policy": policy_document(TELEPHONE_REGIONAL_POLICY),
        "projection_sha256": hashlib.sha256(_canonical_bytes(projection)).hexdigest(),
        **projection,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--locale", choices=LOCALES, action="append")
    args = parser.parse_args(argv)
    DATA.mkdir(parents=True, exist_ok=True)
    for locale in args.locale or LOCALES:
        path = DATA / f"{locale}.json"
        expected = _bytes(build_bundle(locale))
        if args.check:
            if path.read_text("utf-8") != expected:
                raise SystemExit(f"telephone-rule bundle drift: {path}")
        else:
            path.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
