#!/usr/bin/env python3
"""Compile sourced normalization-v2 electronic records into tiergraph projections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import icu
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

from frend.artifact_policy import ELECTRONIC_RULE_POLICY, policy_document
from frend.generated_rule_locales import locales_for
from frend.normalization_schema import validate_bundle

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "frend" / "data" / "normalization"
DATA = REPO / "frend" / "data" / "electronic_rules"
LOCALES = locales_for("electronic")
ELECTRONIC_KINDS = frozenset({"email", "url", "domain", "hashtag", "document"})


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


def _terminal(namespace: str, value: str) -> GrammarTerminal:
    return GrammarTerminal(_string(QualifiedName(namespace, "text"), value))


def validate_source_claims(locale: str, document: Mapping[str, object], claims_path: Path) -> None:
    with claims_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "claim_id",
        "record_id",
        "electronic_kind",
        "claim",
        "source_id",
        "stable_locator",
        "quoted_passage",
        "verdict",
        "reviewer",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError("electronic source-claim table has an invalid schema")
    indexed: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        indexed.setdefault(row["record_id"], []).append(row)
    for resource in document["resources"]:
        key = f"{locale}:{resource['id']}"
        matches = indexed.get(key, ())
        if len(matches) != 1 or not matches[0]["verdict"].startswith("SUPPORTED"):
            raise ValueError(f"{key} must join exactly one supported source claim")
    for rule in document["rules"]:
        if rule.get("kind") != "electronic-rule":
            continue
        kind = rule.get("electronic_kind")
        if kind not in ELECTRONIC_KINDS:
            raise ValueError(f"{locale} rule {rule.get('id')!r} has invalid electronic_kind")
        key = f"{locale}:{rule['id']}"
        matches = indexed.get(key, ())
        if len(matches) != 1:
            raise ValueError(f"{key} must join exactly one construction claim")
        claim = matches[0]
        if claim["electronic_kind"] != kind:
            raise ValueError(f"{key} source claim has cross-kind electronic_kind")
        if not claim["verdict"].startswith("SUPPORTED") or "kal approved" not in claim["reviewer"]:
            raise ValueError(f"{key} source claim is not approved")


def compile_grammar(locale: str, document: Mapping[str, object]) -> GrammarDeclaration:
    namespace = f"urn:frend:electronic:v1:{locale}"
    start = QualifiedName(namespace, "ELECTRONIC")
    runs = QualifiedName(namespace, "RUNS")
    variable = QualifiedName(namespace, "variable")
    rules = []
    seen: set[str] = set()
    for record in document["rules"]:
        if record.get("kind") != "electronic-rule":
            continue
        kind = record.get("electronic_kind")
        if kind not in ELECTRONIC_KINDS or kind in seen:
            raise ValueError(f"{locale} repeats or mislabels electronic kind {kind!r}")
        seen.add(kind)
        fields = tuple(str(item) for item in record["fields"])
        holes = {field: GrammarHole(_string(variable, field.lower()), runs) for field in fields}
        target = tuple(
            holes[item] if item in holes else _terminal(namespace, item)
            for item in record["target"]
        )
        target_fields = tuple(
            item.nonterminal.local_name for item in target if isinstance(item, GrammarHole)
        )
        if len(target_fields) != len(set(target_fields)) or set(target_fields) != set(fields):
            raise ValueError(
                f"{locale} {kind} must consume and emit each field once: "
                f"{target_fields!r} != {fields!r}"
            )
        provenance = (f"electronic-kind:{kind}", f"normalization-record:{record['id']}")
        rules.append(
            GrammarRule(
                start,
                tuple(holes[field] for field in fields),
                target,
                provenance=tuple(
                    _string(QualifiedName(namespace, f"source-{index}"), source)
                    for index, source in enumerate(provenance, start=1)
                ),
            )
        )
    terminal = _terminal(namespace, "FIELD:RUNS")
    rules.append(GrammarRule(runs, (terminal,), (terminal,)))
    return GrammarDeclaration((start, runs), start, tuple(rules))


def compile_recipes(declaration: GrammarDeclaration, document: Mapping[str, object]):
    lowered = lower_grammar(declaration)
    result = {}
    for record in document["rules"]:
        if record.get("kind") != "electronic-rule":
            continue
        kind = str(record["electronic_kind"])
        grammar_input = GrammarInput(
            (
                GrammarInputToken(
                    "FIELD:RUNS",
                    (Realization(("\ufff0RUNS\ufff1",), ("recipe:RUNS",), None),),
                    ("electronic-field:RUNS",),
                ),
            )
        )
        forest = recognize(lowered, grammar_input, collapse_units=False)
        count = forest.count()
        generated = generate(forest, count=count)
        matching = [
            derivation
            for derivation in generated.derivations
            if any(
                value.lexical == f"electronic-kind:{kind}"
                for application in derivation.applications
                for value in application.provenance
            )
        ]
        if generated.truncated or len(generated.derivations) != count or len(matching) != 1:
            raise ValueError(
                f"electronic recipe {locale if (locale := document['locale']) else ''} "
                f"{kind} returned {len(matching)} matching derivations from forest count "
                f"{count} (truncated={generated.truncated})"
            )
        derivation = matching[0]
        result[kind] = [
            {
                "tokens": list(derivation.tokens),
                "provenance": list(
                    dict.fromkeys(
                        value.lexical
                        for application in derivation.applications
                        for value in application.provenance
                    )
                ),
            }
        ]
    return result


def build_bundle(locale: str, claims_path: Path) -> dict[str, object]:
    document = json.loads((RECORDS / f"electronic_{locale}.json").read_text("utf-8"))
    validate_bundle(document, receipt_index=_receipt_index(document))
    validate_source_claims(locale, document, claims_path)
    declaration = compile_grammar(locale, document)
    projection = {
        "recipes": compile_recipes(declaration, document),
        "records": [
            {
                key: record[key]
                for key in (
                    "id",
                    "kind",
                    "value",
                    "forms",
                    "electronic_kind",
                    "priority",
                    "derived_from",
                )
                if key in record
            }
            for record in document["resources"]
        ],
    }
    return {
        "schema": 1,
        "provenance": {"source": "frend/curated"},
        "locale": locale,
        "icu_version": icu.ICU_VERSION,
        "normalization_sha256": hashlib.sha256(_canonical_bytes(document)).hexdigest(),
        "projection_sha256": hashlib.sha256(_canonical_bytes(projection)).hexdigest(),
        "policy": policy_document(ELECTRONIC_RULE_POLICY),
        **projection,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--locale", choices=LOCALES, action="append")
    args = parser.parse_args(argv)
    DATA.mkdir(parents=True, exist_ok=True)
    for locale in args.locale or LOCALES:
        path = DATA / f"{locale}.json"
        expected = _bytes(build_bundle(locale, args.claims))
        if args.check:
            if path.read_text("utf-8") != expected:
                raise SystemExit(f"electronic-rule bundle drift: {path}")
        else:
            path.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
