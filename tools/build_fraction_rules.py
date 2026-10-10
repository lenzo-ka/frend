"""Compile validated normalization-v2 fraction records into tiergraph declarations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
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

from frend.fraction_rules import GENERATED_FRACTION_LOCALES
from frend.normalization_schema import validate_bundle

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "frend" / "data" / "normalization"
DATA = REPO / "frend" / "data" / "fraction_rules"


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


def compile_grammar(locale: str, document: Mapping[str, object]) -> GrammarDeclaration:
    namespace = f"urn:frend:fraction:v1:{locale}"
    start = QualifiedName(namespace, "COMPOSE")
    names = {
        key: QualifiedName(namespace, key) for key in ("SIGN", "WHOLE", "NUM", "DEN", "AMOUNT")
    }
    variable = QualifiedName(namespace, "variable")
    provenance = QualifiedName(namespace, "source")
    rules = []
    expected = {
        "fraction": ("SIGN", "NUM", "DEN"),
        "mixed": ("SIGN", "WHOLE", "NUM", "DEN"),
        "percent": ("SIGN", "AMOUNT"),
    }
    for record in document["rules"]:
        identifier = str(record["id"])
        if identifier not in expected:
            continue
        fields = tuple(str(item).upper() for item in record["fields"])
        if fields != expected[identifier]:
            raise ValueError(
                f"{locale} {identifier} fields are {fields!r}, expected {expected[identifier]!r}"
            )
        target = record.get("target")
        if not isinstance(target, list) or not target:
            raise ValueError(f"{locale} {identifier} has no target")
        holes = {
            field: GrammarHole(_string(variable, field.lower()), names[field]) for field in fields
        }
        target_items = tuple(
            holes[item] if item in holes else _terminal(namespace, item) for item in target
        )
        rules.append(
            GrammarRule(
                start,
                tuple(holes[field] for field in fields),
                target_items,
                provenance=tuple(
                    _string(provenance, f"normalization-record:{source}")
                    for source in record["derived_from"]
                ),
            )
        )
    for field, name in names.items():
        terminal = _terminal(namespace, f"FIELD:{field}")
        rules.append(GrammarRule(name, (terminal,), (terminal,)))
    return GrammarDeclaration((start, *names.values()), start, tuple(rules))


def compile_recipes(declaration: GrammarDeclaration) -> dict[str, list[dict[str, object]]]:
    result = {}
    for kind, symbols in {
        "fraction": ("SIGN", "NUM", "DEN"),
        "mixed": ("SIGN", "WHOLE", "NUM", "DEN"),
        "percent": ("SIGN", "AMOUNT"),
    }.items():
        grammar_input = GrammarInput(
            tuple(
                GrammarInputToken(
                    f"FIELD:{symbol}",
                    (
                        Realization(
                            (f"\ufff0{symbol}\ufff1",),
                            (f"recipe:{symbol}",),
                            None,
                        ),
                    ),
                    (f"fraction-field:{symbol}",),
                )
                for symbol in symbols
            )
        )
        forest = recognize(lower_grammar(declaration), grammar_input, collapse_units=False)
        count = forest.count()
        generated = generate(forest, count=count)
        if generated.truncated or len(generated.derivations) != count or count < 1:
            raise ValueError(
                f"fraction recipe {kind!r} returned {len(generated.derivations)} of "
                f"{count} derivations (truncated={generated.truncated})"
            )
        result[kind] = [
            {
                "tokens": list(derivation.tokens),
                "provenance": [
                    value.lexical
                    for application in derivation.applications
                    for value in application.provenance
                ],
            }
            for derivation in generated.derivations
        ]
    return result


def build_bundle(locale: str) -> dict[str, object]:
    path = RECORDS / f"{locale}.json"
    document = json.loads(path.read_text("utf-8"))
    validate_bundle(document, receipt_index=_receipt_index(document))
    declaration = compile_grammar(locale, document)
    return {
        "schema": 1,
        "locale": locale,
        "icu_version": icu.ICU_VERSION,
        "normalization_sha256": hashlib.sha256(_canonical_bytes(document)).hexdigest(),
        "grammar": declaration.to_data(),
        "recipes": compile_recipes(declaration),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    for locale in GENERATED_FRACTION_LOCALES:
        path = DATA / f"{locale}.json"
        expected = _bytes(build_bundle(locale))
        if args.check:
            if path.read_text("utf-8") != expected:
                raise SystemExit(f"fraction-rule bundle drift: {path}")
        else:
            path.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
