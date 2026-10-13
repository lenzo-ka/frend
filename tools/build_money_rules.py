#!/usr/bin/env python3
"""Compile sourced normalization-v2 MONEY records into tiergraph projections."""

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
DATA = REPO / "frend" / "data" / "money_rules"
LOCALES = locales_for("money")
SHAPE_SYMBOLS = {
    "major": ("AMOUNT", "CURRENCY"),
    "major_minor": ("MAJOR", "CURRENCY", "MINOR", "MINOR_CURRENCY"),
    "minor_only": ("MINOR", "CURRENCY"),
    "compact_major": ("AMOUNT", "SCALE", "CURRENCY"),
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


def _terminal(namespace: str, value: str) -> GrammarTerminal:
    return GrammarTerminal(_string(QualifiedName(namespace, "text"), value))


def compile_grammar(locale: str, document: Mapping[str, object]) -> GrammarDeclaration:
    namespace = f"urn:frend:money:v1:{locale}"
    start = QualifiedName(namespace, "MONEY")
    all_symbols = tuple(
        dict.fromkeys(symbol for values in SHAPE_SYMBOLS.values() for symbol in values)
    )
    names = {symbol: QualifiedName(namespace, symbol) for symbol in all_symbols}
    variable = QualifiedName(namespace, "variable")
    rules = []
    seen: set[tuple[str, str]] = set()
    for record in document["rules"]:
        shape = record.get("shape")
        currency = record.get("currency")
        if shape not in SHAPE_SYMBOLS or not isinstance(currency, str):
            continue
        key = (shape, currency)
        if key in seen:
            raise ValueError(f"{locale} repeats money rule {key!r}")
        seen.add(key)
        symbols = SHAPE_SYMBOLS[shape]
        holes = {
            symbol: GrammarHole(_string(variable, symbol.lower()), names[symbol])
            for symbol in symbols
        }
        raw_target = tuple(str(item) for item in record["target"])
        if shape == "minor_only":
            raw_target = tuple(
                "CURRENCY" if item == "MINOR_CURRENCY" else item for item in raw_target
            )
        target = tuple(
            holes[item] if item in holes else _terminal(namespace, item) for item in raw_target
        )
        target_fields = tuple(
            item.nonterminal.local_name for item in target if isinstance(item, GrammarHole)
        )
        if len(target_fields) != len(set(target_fields)) or set(target_fields) != set(symbols):
            raise ValueError(
                f"{locale} {shape}/{currency} must consume and emit each field once: "
                f"{target_fields!r} != {symbols!r}"
            )
        provenance = [f"money-key:{shape}:{currency}", *record["derived_from"]]
        if shape == "major_minor":
            minor = next(
                item
                for item in document["resources"]
                if item.get("kind") == "minor" and item.get("code") == currency
            )
            provenance.append(minor["id"])
        rules.append(
            GrammarRule(
                start,
                tuple(holes[symbol] for symbol in symbols),
                target,
                provenance=tuple(
                    _string(
                        QualifiedName(namespace, f"source-{index}"),
                        source
                        if source.startswith("money-key:")
                        else f"normalization-record:{source}",
                    )
                    for index, source in enumerate(provenance, start=1)
                ),
            )
        )
    for symbol, name in names.items():
        terminal = _terminal(namespace, f"FIELD:{symbol}")
        rules.append(GrammarRule(name, (terminal,), (terminal,)))
    return GrammarDeclaration((start, *names.values()), start, tuple(rules))


def compile_recipes(
    declaration: GrammarDeclaration, document: Mapping[str, object]
) -> dict[str, list[dict[str, object]]]:
    lowered = lower_grammar(declaration)
    result: dict[str, list[dict[str, object]]] = {}
    for record in document["rules"]:
        shape = record.get("shape")
        currency = record.get("currency")
        if shape not in SHAPE_SYMBOLS or not isinstance(currency, str):
            continue
        symbols = SHAPE_SYMBOLS[shape]
        grammar_input = GrammarInput(
            tuple(
                GrammarInputToken(
                    f"FIELD:{symbol}",
                    (Realization((f"\ufff0{symbol}\ufff1",), (f"recipe:{symbol}",), None),),
                    (f"money-field:{symbol}",),
                )
                for symbol in symbols
            )
        )
        forest = recognize(lowered, grammar_input, collapse_units=False)
        count = forest.count()
        generated = generate(forest, count=count)
        matching = [
            derivation
            for derivation in generated.derivations
            if any(
                value.lexical == f"money-key:{shape}:{currency}"
                for application in derivation.applications
                for value in application.provenance
            )
        ]
        if generated.truncated or len(matching) != 1:
            raise ValueError(
                f"money recipe {document['locale']} {shape}/{currency} returned "
                f"{len(matching)} matching derivations from forest count {count}"
            )
        derivation = matching[0]
        result[f"{shape}:{currency}"] = [
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


def build_bundle(locale: str) -> dict[str, object]:
    path = RECORDS / f"money_{locale}.json"
    document = json.loads(path.read_text("utf-8"))
    validate_bundle(document, receipt_index=_receipt_index(document))
    declaration = compile_grammar(locale, document)
    projection = {
        "grammar": declaration.to_data(),
        "recipes": compile_recipes(declaration, document),
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
                raise SystemExit(f"money-rule bundle drift: {path}")
        else:
            path.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
