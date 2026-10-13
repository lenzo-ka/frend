#!/usr/bin/env python3
"""Compile validated normalization-v2 clock records into tiergraph projections."""

from __future__ import annotations

import argparse
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

from frend.generated_rule_locales import locales_for
from frend.normalization_schema import validate_bundle

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "frend" / "data" / "normalization"
DATA = REPO / "frend" / "data" / "time_rules"
LOCALES = locales_for("time")
SHAPES = ("H", "HM", "HMS", "PH", "PHM", "PHMS")


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
    namespace = f"urn:frend:time:v1:{locale}"
    start = QualifiedName(namespace, "CLOCK")
    symbols = ("P", "H", "M", "S", "UH", "UM", "US")
    names = {symbol: QualifiedName(namespace, symbol) for symbol in symbols}
    variable = QualifiedName(namespace, "variable")
    rules = []
    seen: set[str] = set()
    for record in document["rules"]:
        shape = record.get("shape")
        if shape not in SHAPES:
            continue
        if shape in seen:
            raise ValueError(f"{locale} repeats time shape {shape!r}")
        seen.add(shape)
        fields = tuple(str(item) for item in record["fields"])
        if len(fields) != len(set(fields)) or any(field not in names for field in fields):
            raise ValueError(f"{locale} {shape} has invalid or repeated fields {fields!r}")
        target = record.get("target")
        if not isinstance(target, list) or not target:
            raise ValueError(f"{locale} {shape} has no target")
        holes = {
            field: GrammarHole(_string(variable, field.lower()), names[field]) for field in fields
        }
        target_items = tuple(
            holes[item] if item in holes else _terminal(namespace, item) for item in target
        )
        target_fields = tuple(
            item.nonterminal.local_name for item in target_items if isinstance(item, GrammarHole)
        )
        if len(target_fields) != len(set(target_fields)) or set(target_fields) != set(fields):
            raise ValueError(
                f"{locale} {shape} must consume and emit every declared hole exactly once"
            )
        rules.append(
            GrammarRule(
                start,
                tuple(holes[field] for field in fields),
                target_items,
                provenance=tuple(
                    _string(
                        QualifiedName(namespace, f"source-{index}"),
                        f"normalization-record:{source}",
                    )
                    for index, source in enumerate(record["derived_from"], start=1)
                ),
            )
        )
    if not {"H", "HM"} <= seen:
        raise ValueError(f"{locale} time shapes omit required hour/minute routes: {sorted(seen)!r}")
    if any(shape.startswith("P") and shape[1:] not in seen for shape in seen):
        raise ValueError(f"{locale} period shape has no period-free counterpart: {sorted(seen)!r}")
    for symbol, name in names.items():
        terminal = _terminal(namespace, f"FIELD:{symbol}")
        rules.append(GrammarRule(name, (terminal,), (terminal,)))
    return GrammarDeclaration((start, *names.values()), start, tuple(rules))


def compile_recipes(
    declaration: GrammarDeclaration, document: Mapping[str, object]
) -> dict[str, list[dict[str, object]]]:
    result = {shape: [] for shape in SHAPES}
    fields_by_shape = {
        str(record["shape"]): tuple(str(item) for item in record["fields"])
        for record in document["rules"]
        if record.get("shape") in SHAPES
    }
    lowered = lower_grammar(declaration)
    for shape in SHAPES:
        if shape not in fields_by_shape:
            continue
        symbols = fields_by_shape[shape]
        grammar_input = GrammarInput(
            tuple(
                GrammarInputToken(
                    f"FIELD:{symbol}",
                    (Realization((f"\ufff0{symbol}\ufff1",), (f"recipe:{symbol}",), None),),
                    (f"time-field:{symbol}",),
                )
                for symbol in symbols
            )
        )
        forest = recognize(lowered, grammar_input, collapse_units=False)
        count = forest.count()
        generated = generate(forest, count=count)
        if generated.truncated or len(generated.derivations) != count or count != 1:
            raise ValueError(
                f"time recipe {locale if (locale := document.get('locale')) else ''} {shape} "
                f"returned {len(generated.derivations)} of {count} derivations"
            )
        result[shape] = [
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


def _periods(document: Mapping[str, object]) -> dict[str, str]:
    selector = next(
        item for item in document["selectors"] if item.get("id") == "time-period-semantic-hour"
    )
    records = {item["id"]: item for item in document["resources"]}
    result = {}
    for hour, record_id in selector["mapping"].items():
        record = records[record_id]
        result[str(hour)] = record["forms"]["default"]
    if set(result) != {str(hour) for hour in range(24)}:
        raise ValueError(f"{document['locale']} period selector is not total over 0..23")
    return result


def build_bundle(locale: str) -> dict[str, object]:
    path = RECORDS / f"time_{locale}.json"
    document = json.loads(path.read_text("utf-8"))
    validate_bundle(document, receipt_index=_receipt_index(document))
    declaration = compile_grammar(locale, document)
    projection = {
        "grammar": declaration.to_data(),
        "recipes": compile_recipes(declaration, document),
        "periods": _periods(document),
        "records": document["resources"],
    }
    return {
        "schema": 2,
        "provenance": {"source": "frend/curated"},
        "locale": locale,
        "icu_version": icu.ICU_VERSION,
        "normalization_sha256": hashlib.sha256(_canonical_bytes(document)).hexdigest(),
        "projection_sha256": hashlib.sha256(_canonical_bytes(projection)).hexdigest(),
        **projection,
    }


def check_bundle(path: Path, expected: str) -> None:
    if path.read_text("utf-8") != expected:
        raise SystemExit(f"time-rule bundle drift: {path}")


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
            check_bundle(path, expected)
        else:
            path.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
