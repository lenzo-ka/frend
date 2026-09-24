"""Curate the foreign break_exceptions English seed into an icukit inventory.

Usage::

    python tools/import_break_exceptions.py
    python tools/import_break_exceptions.py --check
    python tools/import_break_exceptions.py --source /path/to/en.txt --out /tmp/en.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple, cast

from icukit.exceptions import ExceptionInventory, ExceptionRule, load_exception_inventory

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "frend" / "data" / "exceptions" / "en.json"

# These are deliberately explicit: changes to policy should produce a reviewable
# source diff, not emerge accidentally from a heuristic over the foreign file.
_KEEP = {
    "Capt.",
    "Col.",
    "Lt.",
    "Mgr.",
    "Maj.",
    "Ph.D.",
    "Prof.",
    "Pvt.",
    "Rep.",
    "Rev.",
    "Gen.",
    "Hon.",
    "St.",
    "Sta.",
    "Ste.",
    "Rd.",
    "Rt.",
    "Inc.",
    "Dept.",
    "Wm.",
    "Jr.",
    "Sr.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "U.S.",
    "U.S.A.",
}

_AMBIGUOUS = {
    "Pt.": "ambiguous point/patient notation and an ordinary sentence ending",
    "v.": "ambiguous legal versus marker and an ordinary sentence ending",
    "Vs.": "ambiguous versus marker and an ordinary sentence ending",
    "vs.": "ambiguous versus marker and an ordinary sentence ending",
    "Mon.": "day names commonly occur at real sentence endings",
    "Tue.": "day names commonly occur at real sentence endings",
    "Tues.": "day names commonly occur at real sentence endings",
    "Wed.": "day names commonly occur at real sentence endings",
    "Weds.": "day names commonly occur at real sentence endings",
    "Thu.": "day names commonly occur at real sentence endings",
    "Thurs.": "day names commonly occur at real sentence endings",
    "Fri.": "day names commonly occur at real sentence endings",
    "Sat.": "day names commonly occur at real sentence endings",
    "Sun.": "day names commonly occur at real sentence endings",
    "Feb.": "month names commonly occur at real sentence endings",
    "Mar.": "month names commonly occur at real sentence endings and Mar is a name/word",
    "Apr.": "month names commonly occur at real sentence endings",
    "Jun.": "month names commonly occur at real sentence endings and Jun is a name",
    "Jul.": "month names commonly occur at real sentence endings",
    "Aug.": "month names commonly occur at real sentence endings",
    "Sep.": "month names commonly occur at real sentence endings",
    "Sept.": "month names commonly occur at real sentence endings",
    "Oct.": "month names commonly occur at real sentence endings",
    "Nov.": "month names commonly occur at real sentence endings",
    "Dec.": "month names commonly occur at real sentence endings",
    "Md.": "ambiguous state/degree abbreviation and an ordinary sentence ending",
    "Nr.": "meaning varies by domain and locale",
    "Num.": "meaning varies by domain and may label a completed reference",
}


class SeedEntry(NamedTuple):
    value: str
    line: int
    regex: bool


class Drop(NamedTuple):
    value: str
    line: int
    reason: str


def default_source() -> Path:
    """Find the sibling checkout without baking a developer-specific path in."""
    for base in (_REPO, *_REPO.parents):
        candidate = base / "break_exceptions" / "en.txt"
        if candidate.is_file():
            return candidate
    return _REPO.parent / "break_exceptions" / "en.txt"


def parse_seed(text: str) -> list[SeedEntry]:
    """Parse literal and ``:re`` records while ignoring headers/comments."""
    entries: list[SeedEntry] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        value = stripped.split("#", 1)[0].strip()
        is_regex = value.startswith(":re ")
        entries.append(SeedEntry(value[4:].strip() if is_regex else value, line_number, is_regex))
    return entries


def _slug(surface: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", surface.lower()).strip("-")


def curate(entries: list[SeedEntry]) -> tuple[list[ExceptionRule], list[Drop]]:
    """Apply the reviewed allowlist and return rules plus auditable rejections."""
    rules: list[ExceptionRule] = []
    drops: list[Drop] = []
    for entry in entries:
        if entry.regex:
            drops.append(Drop(entry.value, entry.line, "regex seed dropped as over-broad"))
            continue
        if entry.value not in _KEEP:
            drops.append(
                Drop(
                    entry.value, entry.line, _AMBIGUOUS.get(entry.value, "not approved by curation")
                )
            )
            continue
        surface = entry.value
        rules.append(
            {
                "id": f"break-exceptions-en:{_slug(surface)}",
                "locale": "en",
                "level": ["word", "sentence"],
                "effect": "suppress",
                "surface": surface,
                "variant": "exact",
                "conditions": [
                    {
                        "id": "following-capital",
                        "kind": "unicode_set",
                        "direction": "right",
                        "set": "[[:Lu:]]",
                        "skip": {"kind": "whitespace", "max": None},
                    }
                ],
                "unconditionality": "conditional",
                "provenance": {
                    "source": "break_exceptions/en.txt",
                    "source_id": f"line-{entry.line}",
                    "note": "Foreign literal seed; curated into a conditioned sentence rule.",
                },
                "witnesses": {
                    "positive": f"We met {surface} Smith today.",
                    "near_miss": f"Prefix{surface} Smith today.",
                    "condition_negatives": [f"We met {surface} smith today."],
                },
            }
        )
    return rules, drops


def build_document(source: Path) -> tuple[ExceptionInventory, list[Drop]]:
    entries = parse_seed(source.read_text(encoding="utf-8"))
    rules, drops = curate(entries)
    inventory: ExceptionInventory = {
        "schema_version": 1,
        "corpus": "break-exceptions-en-curated",
        "named_lists": {},
        "rules": rules,
    }
    load_exception_inventory(inventory)
    return inventory, drops


def serialize(document: ExceptionInventory) -> str:
    return json.dumps(cast(dict[str, Any], document), ensure_ascii=False, indent=2) + "\n"


def _report(rules: list[ExceptionRule], drops: list[Drop]) -> None:
    print(f"kept {len(rules)} literal entries: {', '.join(rule['surface'] for rule in rules)}")
    print(f"dropped {len(drops)} entries:")
    for drop in drops:
        print(f"  line {drop.line}: {drop.value} — {drop.reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=default_source())
    parser.add_argument("--out", type=Path, default=_OUT, help="output inventory path")
    parser.add_argument("--check", action="store_true", help="exit nonzero if output drifts")
    args = parser.parse_args(argv)
    document, drops = build_document(args.source)
    rendered = serialize(document)
    _report(document["rules"], drops)
    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != rendered:
            print(
                f"drift: {args.out} is missing or out of date; rerun import_break_exceptions.py",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {args.out} is up to date")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
