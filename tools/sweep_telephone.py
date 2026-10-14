#!/usr/bin/env python3
"""Crash and near-miss sweep for the existing libphonenumber-gated telephone path."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import phonenumbers
from icukit.detectors import detect

from frend import resolve_lattice
from frend.telephone import TelephoneDetector, telephone_shape
from frend.verbalize import verbalize_edge
from frend.written_forms import WrittenFormsDetector
from tools.locale_gate import GATE_LOCALES

NEGATIVES = (
    "2020-2024",
    "192.168.0.1",
    "2001:db8::1",
    "978-0-7524-4250-1",
    "0-521-56136-1",
    "123-45-6789",
    "3.12.1",
    "10001-1234",
    "123456789012345",
    "3-2",
    "2026-10-13",
)


def _speak(surface: str, locale: str) -> tuple[int, tuple[str, ...]]:
    detections = [
        item
        for item in detect(surface, (WrittenFormsDetector(locale),))
        if item["type"] == "telephone:number"
    ]
    texts = []
    for detection in detections:
        lattice = resolve_lattice([detection], source_text=surface, locale=locale)
        edge = next(item for item in lattice.edges if item.kind == "reading")
        alternatives = verbalize_edge(edge, source_text=surface, locale=locale).alternatives
        if len(alternatives) > 128:
            raise ValueError(f"{locale} {telephone_shape(surface)!r} exceeds 128 alternatives")
        texts.extend(item.text for item in alternatives)
    return len(detections), tuple(texts)


def sweep() -> dict[str, object]:
    counts: Counter[str] = Counter()
    relay: Counter[str] = Counter()
    for locale in GATE_LOCALES:
        region = locale.partition("_")[2]
        for number_kind in range(phonenumbers.PhoneNumberType.UNKNOWN + 1):
            number = phonenumbers.example_number_for_type(region, number_kind)
            if number is None:
                continue
            for scope, style in (
                ("national", phonenumbers.PhoneNumberFormat.NATIONAL),
                ("international", phonenumbers.PhoneNumberFormat.INTERNATIONAL),
            ):
                surface = phonenumbers.format_number(number, style)
                detected, alternatives = _speak(surface, locale)
                counts[f"{locale}:{scope}:examples"] += 1
                counts[f"{locale}:{scope}:detections"] += detected
                if detected:
                    if not alternatives or len(alternatives) != len(set(alternatives)):
                        raise ValueError(f"{locale} {telephone_shape(surface)!r} is unstable")
                else:
                    relay[f"{locale}:{scope}:{number_kind}"] += 1
        for negative in NEGATIVES:
            if TelephoneDetector(locale).detect(negative):
                raise ValueError(f"negative population matched for {locale}: {negative!r}")
            counts[f"{locale}:negatives"] += 1
    return {
        "schema": 1,
        "phonenumbers": phonenumbers.__version__,
        "locales": list(GATE_LOCALES),
        "exceptions": 0,
        "negative_matches": 0,
        "counts": dict(sorted(counts.items())),
        "icukit_relay": {
            "sanitized": True,
            "total": sum(relay.values()),
            "by_locale_scope_type": dict(sorted(relay.items())),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(sweep(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
