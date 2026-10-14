#!/usr/bin/env python3
"""Audit the five approved PR10 rows against libphonenumber metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import phonenumbers

from frend.telephone import _descriptor_sha256, _selected_descriptor

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "pr10" / "recount.json"
ROWS = (
    ("fr-fixed", "fr_FR", "FR", "01 23 45 67 89", "pr10-001", "telephone-fr-fixed"),
    ("fr-mobile", "fr_FR", "FR", "06 12 34 56 78", "pr10-001", "telephone-fr-mobile"),
    ("br-mobile", "pt_BR", "BR", "(11) 96123-4567", "pr10-006", "telephone-br-mobile"),
    ("pt-lisbon", "pt_PT", "PT", "21 234 5678", "pr10-009", "telephone-pt-lisbon"),
    ("pt-lisbon-zero", "pt_PT", "PT", "21 203 0405", "pr10-009", "telephone-pt-lisbon"),
)


def build() -> dict[str, object]:
    rows = []
    for identifier, locale, region, surface, claim, rule in ROWS:
        number = phonenumbers.parse(surface, region)
        national = phonenumbers.national_significant_number(number)
        descriptor = _selected_descriptor(region, national)
        if descriptor is None:
            raise ValueError(f"no descriptor for {identifier}")
        match = re.fullmatch(descriptor.pattern, national)
        if (
            match is None
            or phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.NATIONAL)
            != surface
        ):
            raise ValueError(f"format audit failed for {identifier}")
        rows.append(
            {
                "id": identifier,
                "locale": locale,
                "region": region,
                "scope": "national",
                "metadata_version": phonenumbers.__version__,
                "descriptor_sha256": _descriptor_sha256(descriptor),
                "descriptor_audit": {
                    key: value for key, value in vars(descriptor).items() if key != "_mutable"
                },
                "national_significant_number": national,
                "capture_groups": list(match.groups()),
                "written_groups": re.findall(r"\d+", surface),
                "prefix_handling": "trunk-zero" if region == "FR" else "none",
                "exclusion_verdict": "eligible",
                "source_claim": claim,
                "source_record_id": rule,
                "surface_sha256": hashlib.sha256(surface.encode()).hexdigest(),
            }
        )
    return {
        "schema": 1,
        "development_only": True,
        "phonenumbers": phonenumbers.__version__,
        "counts": {"fr_FR": 2, "pt_BR": 1, "pt_PT": 2},
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        if OUTPUT.read_text("utf-8") != expected:
            raise SystemExit(f"telephone recount drift: {OUTPUT}")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
