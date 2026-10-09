"""Compile ICU/CLDR date patterns into deterministic tiergraph declarations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import icu
from tiergraph import (
    AttributeValue,
    GrammarDeclaration,
    GrammarHole,
    GrammarRule,
    GrammarTerminal,
    QualifiedName,
    XsdType,
)

LOCALES = (
    "es_MX",
    "es_ES",
    "fr_FR",
    "de_DE",
    "pt_BR",
    "pt_PT",
    "it_IT",
    "zh_CN",
    "ko_KR",
    "ja_JP",
)
DATA = Path(__file__).resolve().parents[1] / "frend" / "data" / "date_rules"


@dataclass(frozen=True)
class PatternPart:
    field: Literal["y", "M", "d"] | None
    literal: str


def cldr_pattern_parts(pattern: str) -> tuple[PatternPart, ...]:
    """Tokenize the supported CLDR date-pattern subset, including apostrophe quoting."""
    parts: list[PatternPart] = []
    literal: list[str] = []
    quoted = False
    index = 0

    def flush() -> None:
        if literal:
            text = "".join(literal)
            if parts and parts[-1].field is None:
                parts[-1] = PatternPart(None, parts[-1].literal + text)
            else:
                parts.append(PatternPart(None, text))
            literal.clear()

    while index < len(pattern):
        char = pattern[index]
        if char == "'":
            if index + 1 < len(pattern) and pattern[index + 1] == "'":
                literal.append("'")
                index += 2
                continue
            quoted = not quoted
            index += 1
            continue
        if not quoted and char.isascii() and char.isalpha():
            flush()
            end = index + 1
            while end < len(pattern) and pattern[end] == char:
                end += 1
            if char not in {"y", "M", "d"}:
                raise ValueError(f"unsupported CLDR date field {char!r} in pattern {pattern!r}")
            parts.append(PatternPart(char, pattern[index:end]))
            index = end
            continue
        literal.append(char)
        index += 1
    if quoted:
        raise ValueError(f"unterminated CLDR quote in pattern {pattern!r}")
    flush()
    return tuple(parts)


def _string(name: QualifiedName, value: str) -> AttributeValue:
    return AttributeValue(name, XsdType.STRING, value)


def _target_symbols(locale: str, parts: tuple[PatternPart, ...]) -> tuple[str, ...]:
    """Lower written CLDR literal evidence to exact spoken tokens."""
    cjk = locale in {"zh_CN", "ja_JP"}
    korean = locale == "ko_KR"
    raw: list[str] = []
    for part in parts:
        if part.field is not None:
            raw.append({"y": "Y", "M": "M", "d": "D"}[part.field])
            if korean and part.field == "M":
                raw.append("월")
            continue
        word = "".join(char for char in part.literal if char.isalpha())
        units = "".join(char for char in part.literal if char in "年月日년월일")
        if units:
            raw.extend(units)
        elif word:
            raw.extend(word.split())
        elif part.literal:
            raw.append(" ")

    if cjk:
        return tuple(item for item in raw if item != " ")

    output: list[str] = []
    for item in raw:
        if item == " ":
            if output and output[-1] != " ":
                output.append(item)
            continue
        if output and output[-1] != " " and not (korean and item in "년월일"):
            output.append(" ")
        output.append(item)
    while output and output[-1] == " ":
        output.pop()
    return tuple(output)


def _grammar_terminal(namespace: str, value: str) -> GrammarTerminal:
    return GrammarTerminal(_string(QualifiedName(namespace, "text"), value))


def compile_date_grammar(locale: str) -> GrammarDeclaration:
    generator = icu.DateTimePatternGenerator.createInstance(icu.Locale(locale))
    patterns = {
        "Md": generator.getBestPattern("MMMMd"),
        "yMd": generator.getBestPattern("yMMMMd"),
    }
    namespace = f"urn:frend:date:v1:{locale}"
    start = QualifiedName(namespace, "DATE")
    fields = {key: QualifiedName(namespace, key) for key in ("Y", "M", "D")}
    variable_name = QualifiedName(namespace, "variable")
    provenance_name = QualifiedName(namespace, "source")
    rules: list[GrammarRule] = []
    for skeleton, field_order in (("Md", ("M", "D")), ("yMd", ("Y", "M", "D"))):
        cldr_skeleton = "MMMMd" if skeleton == "Md" else "yMMMMd"
        variables = {field: _string(variable_name, field.lower()) for field in field_order}
        target = []
        for symbol in _target_symbols(locale, cldr_pattern_parts(patterns[skeleton])):
            if symbol in variables:
                target.append(GrammarHole(variables[symbol], fields[symbol]))
            else:
                target.append(_grammar_terminal(namespace, symbol))
        rules.append(
            GrammarRule(
                start,
                tuple(GrammarHole(variables[field], fields[field]) for field in field_order),
                tuple(target),
                provenance=(
                    _string(
                        provenance_name,
                        f"cldr:{icu.ICU_VERSION}:{locale}:{cldr_skeleton}",
                    ),
                ),
            )
        )
    for field in ("Y", "M", "D"):
        terminal = _grammar_terminal(namespace, f"FIELD:{field}")
        rules.append(GrammarRule(fields[field], (terminal,), (terminal,)))
    return GrammarDeclaration((start, *fields.values()), start, tuple(rules))


def build_bundle(locale: str) -> dict[str, object]:
    if locale not in LOCALES:
        raise ValueError(f"unsupported generated-rule locale {locale!r}")
    generator = icu.DateTimePatternGenerator.createInstance(icu.Locale(locale))
    patterns = {
        "Md": {"skeleton": "MMMMd", "pattern": generator.getBestPattern("MMMMd")},
        "yMd": {"skeleton": "yMMMMd", "pattern": generator.getBestPattern("yMMMMd")},
    }
    if locale == "ko_KR":
        month_formatter = icu.SimpleDateFormat("LLLL", icu.Locale(locale))
        for month in range(1, 13):
            instant = icu.GregorianCalendar(2000, month - 1, 1).getTime()
            if month_formatter.format(instant) != f"{month}월":
                raise ValueError(f"ko_KR month {month} is not a numeric month plus 월")
    return {
        "schema": 1,
        "locale": locale,
        "icu_version": icu.ICU_VERSION,
        "sources": patterns,
        "field_policy": {
            "day": "ordinal-r" if locale == "de_DE" else "cardinal-with-sourced-day-one",
            "month": "numeric-unit-free"
            if locale in {"zh_CN", "ko_KR", "ja_JP"}
            else "icu-format-context-MMMM",
            "year": "icu-rbnf-year",
        },
        "grammar": compile_date_grammar(locale).to_data(),
        "provenance": {"source": f"icu/{icu.ICU_VERSION}/cldr-date-patterns"},
    }


def _bytes(document: dict[str, object]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def check_bundle(path: Path, expected: dict[str, object]) -> None:
    actual = path.read_text(encoding="utf-8")
    if actual != _bytes(expected):
        raise SystemExit(f"date-rule bundle drift: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    for locale in LOCALES:
        path = DATA / f"{locale}.json"
        document = build_bundle(locale)
        if args.check:
            check_bundle(path, document)
        else:
            path.write_text(_bytes(document), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
