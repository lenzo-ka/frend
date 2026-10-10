"""Build exact PR6 fraction change and recovery manifests from paired coverage reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PUBLIC_FIELDS = (
    "strict",
    "presentation",
    "insensitive",
    "first",
    "first_strict",
    "first_presentation",
    "error",
    "offer_signature",
)
RECOVERY_FIELDS = (
    "strict",
    "presentation",
    "insensitive",
    "first_strict",
    "first_presentation",
)


def _is_routed(row: Mapping[str, object]) -> bool:
    if row.get("locale") == "en_US":
        return False
    return any(
        str(edge.get("type", "")).startswith("fraction:")
        or str(edge.get("type", "")).startswith("number:fraction")
        or edge.get("type") == "number:percent"
        for edge in row.get("offer_signature", ())
        if isinstance(edge, Mapping)
    )


def _subset(row: Mapping[str, object], fields: tuple[str, ...]) -> dict[str, object]:
    return {field: row.get(field) for field in fields}


def build(
    base_path: Path,
    head_path: Path,
    recovery_witness: Mapping[str, object],
) -> tuple[dict, dict]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    head = json.loads(head_path.read_text(encoding="utf-8"))
    base_rows = {row["id"]: row for row in base["cases"]}
    head_rows = {row["id"]: row for row in head["cases"]}
    if set(base_rows) != set(head_rows):
        raise ValueError("base/head coverage row sets differ")
    routed = sorted(identifier for identifier, row in base_rows.items() if _is_routed(row))
    changes = {
        "schema": 1,
        "family": "fraction",
        "base_head": base.get("frend_head"),
        "head_head": head.get("frend_head"),
        "rows": [
            {
                "id": identifier,
                "base": _subset(base_rows[identifier], PUBLIC_FIELDS),
                "head": _subset(head_rows[identifier], PUBLIC_FIELDS),
            }
            for identifier in routed
        ],
    }
    witness_rows = recovery_witness.get("rows")
    if not isinstance(witness_rows, list) or not witness_rows:
        raise ValueError("a nonempty frozen recovery witness is required")
    by_witness_id = {str(row.get("id")): row for row in witness_rows if isinstance(row, Mapping)}
    if len(by_witness_id) != len(witness_rows):
        raise ValueError("frozen recovery witness IDs must be unique mappings")
    if not set(by_witness_id) <= set(routed):
        raise ValueError("frozen recovery witness contains a non-routed row")
    recovery_rows = []
    for identifier in sorted(by_witness_id):
        witness = by_witness_id[identifier]
        before, after = base_rows[identifier], head_rows[identifier]
        expected_base = witness.get("base")
        if expected_base != _subset(before, RECOVERY_FIELDS):
            raise ValueError(f"frozen recovery witness base drift for {identifier}")
        target = witness.get("expected_speech")
        source_record = witness.get("source_record_id")
        if not isinstance(target, str) or target not in before.get("targets", ()):
            raise ValueError(f"frozen recovery witness speech drift for {identifier}")
        if not isinstance(source_record, str) or not source_record:
            raise ValueError(f"frozen recovery witness source is missing for {identifier}")
        recovery_rows.append(
            {
                "id": identifier,
                "base": _subset(before, RECOVERY_FIELDS),
                "head": _subset(after, RECOVERY_FIELDS),
                "expected_speech": target,
                "source_record_id": source_record,
            }
        )
    witness_bytes = json.dumps(
        recovery_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    recoveries = {
        "schema": 1,
        "family": "fraction",
        "base_head": base.get("frend_head"),
        "witness_sha256": hashlib.sha256(witness_bytes).hexdigest(),
        "strict_total": sum(
            not row["base"]["strict"] and row["head"]["strict"] for row in recovery_rows
        ),
        "insensitive_total": sum(
            not row["base"]["insensitive"] and row["head"]["insensitive"] for row in recovery_rows
        ),
        "rows": recovery_rows,
    }
    return changes, recoveries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destinations = (
        REPO / "tools/fraction_expected_changes.json",
        REPO / "tools/fraction_expected_recoveries.json",
    )
    if not destinations[1].exists():
        raise SystemExit(
            "frozen recovery witness is missing; it must be created from base before head is read"
        )
    recovery_witness = json.loads(destinations[1].read_text(encoding="utf-8"))
    documents = build(args.base, args.head, recovery_witness)
    for destination, document in zip(destinations, documents, strict=True):
        expected = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.check:
            if destination.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"fraction manifest drift: {destination}")
        else:
            destination.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
