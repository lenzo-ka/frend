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


def _development_recovery(before: Mapping[str, object], after: Mapping[str, object]) -> dict | None:
    from tools.locale_gate import (
        _source_record_offers,
        insensitive_form,
        presentation_form,
        strict_form,
    )

    recovered = tuple(
        (name, form)
        for name, form in (
            ("strict", strict_form),
            ("presentation", presentation_form),
            ("insensitive", insensitive_form),
        )
        if before.get(name) is False and after.get(name) is True
    )
    if not recovered:
        return None
    targets = after.get("targets")
    target_hits = after.get("target_hits")
    if not isinstance(targets, list) or not isinstance(target_hits, Mapping):
        raise ValueError(f"coverage row {after.get('id')!r} has no target evidence")
    candidates = []
    for edge in after.get("offer_signature", ()):
        if not isinstance(edge, Mapping):
            continue
        for alternative in edge.get("alternatives", ()):
            provenance = alternative.get("provenance") if isinstance(alternative, Mapping) else None
            if not isinstance(provenance, str):
                continue
            for part in provenance.split("+"):
                if part.startswith("normalization-record:"):
                    identifier = part.removeprefix("normalization-record:")
                    if identifier not in candidates:
                        candidates.append(identifier)
    for target_index, target in enumerate(targets):
        if not isinstance(target, str):
            continue
        for name, form in recovered:
            hits = target_hits.get(name)
            if not isinstance(hits, list) or target_index >= len(hits) or not hits[target_index]:
                continue
            for source_record in candidates:
                if _source_record_offers(after, target, source_record, form=form):
                    return {
                        "id": str(after["id"]),
                        "base": _subset(before, RECOVERY_FIELDS),
                        "head": _subset(after, RECOVERY_FIELDS),
                        "expected_speech": target,
                        "source_record_id": source_record,
                    }
    raise ValueError(f"coverage row {after.get('id')!r} has no sourced recovered target route")


def build(base_path: Path, head_path: Path) -> tuple[dict, dict]:
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
        "status": "development-only",
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
    recovery_rows = []
    for identifier in routed:
        before, after = base_rows[identifier], head_rows[identifier]
        if recovery := _development_recovery(before, after):
            recovery_rows.append(recovery)
    witness_bytes = json.dumps(
        recovery_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    recoveries = {
        "schema": 1,
        "family": "fraction",
        "status": "development-only",
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
    documents = build(args.base, args.head)
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
