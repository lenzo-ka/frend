#!/usr/bin/env python3
"""Crash- and invariant-sweep electronic inputs across every supported locale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frend import resolve_lattice, verbalize_edge
from frend.normalize import _reading_detectors
from tools.locale_gate import GATE_LOCALES

SEPARATORS = ("@", ".", "/", "-", "_")
TEMPLATES = (
    "a@b.de",
    "a-b@c.de",
    "a_b@c.de",
    "a..b@c.de",
    "7a@b2.de",
    "https://example.com/a-b_c",
    "www.example.com",
    "#alpha-7",
    "archive/a-b_7.txt",
    "δοκιμή@example.de",
    "例子@example.de",
    "a@xn--bcher-kva.de",
    "a@b." + "x" * 512,
)


def _signature(text: str, locale: str):
    detections = _reading_detectors(locale).detect(text)
    electronic = [
        item for item in detections if str(item.get("type", "")).startswith("electronic:")
    ]
    lattice = resolve_lattice(detections, source_text=text, locale=locale)
    rows = []
    for edge in lattice.edges:
        if (
            edge.kind != "reading"
            or edge.detection is None
            or not str(edge.detection.get("type", "")).startswith("electronic:")
        ):
            continue
        unit = verbalize_edge(edge, source_text=text, locale=locale)
        if len(unit.alternatives) > 8:
            raise ValueError(f"{locale} {text!r} emitted {len(unit.alternatives)} alternatives")
        if any(
            item.weight is not None and not item.weight.is_finite() for item in unit.alternatives
        ):
            raise ValueError(f"{locale} {text!r} emitted a nonfinite weight")
        rows.append(
            (
                edge.start,
                edge.end,
                edge.detection.get("type"),
                tuple((item.text, item.provenance, str(item.weight)) for item in unit.alternatives),
            )
        )
    return (
        tuple((item["start"], item["end"], item["type"]) for item in electronic),
        tuple(rows),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = list(TEMPLATES)
    cases.extend(f"a{separator}{separator}b@c.de" for separator in SEPARATORS)
    cases.extend(("a@b.de", "a@b.de", "a-b@c.de"))
    exceptions = []
    checked = 0
    max_alternatives = 0
    for locale in GATE_LOCALES:
        for text in cases:
            try:
                first = _signature(text, locale)
                second = _signature(text, locale)
                if first != second:
                    raise ValueError("unstable electronic span or reading signature")
                max_alternatives = max(
                    max_alternatives,
                    *(len(row[3]) for row in first[1]),
                    0,
                )
            except Exception as exc:  # noqa: BLE001 - this is a crash sweep
                exceptions.append({"locale": locale, "case": checked, "error": repr(exc)})
            checked += 1
    report = {
        "schema": 1,
        "locales": list(GATE_LOCALES),
        "cases": checked,
        "exceptions": exceptions,
        "max_electronic_alternatives": max_alternatives,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("cases", "max_electronic_alternatives")}))
    return int(bool(exceptions))


if __name__ == "__main__":
    raise SystemExit(main())
