#!/usr/bin/env python3
"""Measure and compare the five approved PR10 telephone witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROWS = (
    ("fr-fixed", "fr_FR", "01 23 45 67 89"),
    ("fr-mobile", "fr_FR", "06 12 34 56 78"),
    ("br-mobile", "pt_BR", "(11) 96123-4567"),
    ("pt-lisbon", "pt_PT", "21 234 5678"),
    ("pt-lisbon-zero", "pt_PT", "21 203 0405"),
)

_CHILD = r"""
import dataclasses, json, sys
from icukit.detectors import detect
from frend import resolve_lattice
from frend.verbalize import verbalize_edge
from frend.written_forms import WrittenFormsDetector
identifier, locale, text = sys.argv[1:]
found = [item for item in detect(text, (WrittenFormsDetector(locale),))
         if item["type"] == "telephone:number" and item["start"] == 0 and item["end"] == len(text)]
row = {"id": identifier, "locale": locale,
       "written_sha256": __import__('hashlib').sha256(text.encode()).hexdigest(),
       "detections": []}
for item in found:
    lattice = resolve_lattice([item], source_text=text, locale=locale)
    edge = next(value for value in lattice.edges if value.kind == "reading")
    alternatives = verbalize_edge(edge, source_text=text, locale=locale).alternatives
    value = item["value"]
    matched = getattr(value, "matched_format", None)
    row["detections"].append({
        "span": [item["start"], item["end"]], "type": item["type"],
        "groups": list(value.groups), "rule": None if matched is None else matched.rule,
        "candidates": [{"text": candidate.text, "provenance": candidate.provenance,
                        "prior_provenance": candidate.prior_provenance,
                        "weight": None if candidate.weight is None else str(candidate.weight)}
                       for candidate in alternatives],
    })
print(json.dumps(row, ensure_ascii=False, sort_keys=True))
"""


def _canonical(document: object) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def measure(root: Path) -> dict[str, object]:
    root = root.resolve()
    interpreter = root / ".venv/bin/python"
    environment = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    rows = []
    for identifier, locale, text in ROWS:
        result = subprocess.run(
            [str(interpreter), "-B", "-c", _CHILD, identifier, locale, text],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        rows.append(json.loads(result.stdout))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    payload = {"schema": 1, "subject_head": head, "rows": rows}
    return {**payload, "rows_sha256": hashlib.sha256(_canonical(rows)).hexdigest()}


def compare(base: dict[str, object], head: dict[str, object], expected: dict[str, object]):
    expected_by_id = {row["id"]: row for row in expected["rows"]}
    recoveries = []
    failures = []
    for base_row, head_row in zip(base["rows"], head["rows"], strict=True):
        if base_row["id"] != head_row["id"]:
            failures.append("row identity drift")
            continue
        target = expected_by_id[head_row["id"]]
        candidates = [
            candidate
            for detection in head_row["detections"]
            for candidate in detection["candidates"]
        ]
        matching = [
            item
            for item in candidates
            if item["text"] == target["expected_text"]
            and target["source_record_id"] in item["provenance"]
        ]
        if base_row["detections"] or not matching:
            failures.append(head_row["id"])
        else:
            recoveries.append(head_row["id"])
    return {
        "schema": 1,
        "base_rows_sha256": base["rows_sha256"],
        "head_rows_sha256": head["rows_sha256"],
        "expected_sha256": hashlib.sha256(_canonical(expected)).hexdigest(),
        "recoveries": recoveries,
        "failures": failures,
        "passed": not failures and len(recoveries) == len(ROWS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", nargs=3, type=Path, metavar=("BASE", "HEAD", "EXPECTED"))
    args = parser.parse_args()
    if args.compare:
        documents = [json.loads(path.read_text("utf-8")) for path in args.compare]
        report = compare(*documents)
        if not report["passed"]:
            status = 1
        else:
            status = 0
    else:
        if args.subject_root is None:
            parser.error("--subject-root is required when not comparing")
        report = measure(args.subject_root)
        status = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
