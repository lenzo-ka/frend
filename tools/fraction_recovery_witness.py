#!/usr/bin/env python3
"""Freeze and measure PR6's held-out fraction/percent recovery witness.

``freeze`` must run with the resolved base checkout first.  It selects every row
from the explicitly named, previously unconsulted NeMo inverse-TN fixture files
whose written form and written/spoken pair do not occur in prior coverage and
whose literals do not occur in the prior PR6 reports/manifests.  ``measure``
evaluates the already-frozen rows; it never expands or reselects the witness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from locale_gate import FixtureCase, _evaluate

PUBLIC_FIELDS = (
    "strict",
    "presentation",
    "insensitive",
    "first_strict",
    "first_presentation",
    "first",
    "error",
)
SOURCE_SPECS = {
    "de": ("de_DE",),
    "es": ("es_MX", "es_ES"),
    "fr": ("fr_FR",),
    "pt": ("pt_BR", "pt_PT"),
    "it": ("it_IT",),
    "zh": ("zh_CN",),
    "ko": ("ko_KR",),
    "ja": ("ja_JP",),
}
SOURCE_FILES = (
    ("fraction", "test_cases_fraction.txt"),
    ("percent", "test_cases_measure.txt"),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def _head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _audit_label(path: Path) -> str:
    parts = path.parts
    if "nine-locales" in parts:
        return Path(*parts[parts.index("nine-locales") + 1 :]).as_posix()
    if path.parent.name == "tools":
        return f"tools/{path.name}"
    return path.name


def _source_rows(nemo_root: Path):
    source_files = []
    rows = []
    for language, locales in SOURCE_SPECS.items():
        for family, filename in SOURCE_FILES:
            path = nemo_root / language / "data_inverse_text_normalization" / filename
            if not path.exists():
                continue
            raw = path.read_bytes()
            relative = path.relative_to(nemo_root).as_posix()
            source_files.append({"path": relative, "sha256": _sha256(raw)})
            for line_number, line in enumerate(raw.decode().splitlines(), 1):
                if "~" not in line:
                    continue
                spoken, written = line.split("~", 1)
                if family == "percent" and "%" not in written:
                    continue
                for locale in locales:
                    rows.append(
                        {
                            "id": f"{locale}:{relative}:{line_number}",
                            "language": language,
                            "locale": locale,
                            "family": family,
                            "source": relative,
                            "source_line": line_number,
                            "written": written,
                            "expected_speech": spoken,
                        }
                    )
    return source_files, rows


def _evaluate_row(row: dict) -> dict:
    case = FixtureCase(
        row["language"],
        row["locale"],
        row["family"],
        row["source"],
        row["source_line"],
        row["written"],
        (row["expected_speech"],),
        False,
    )
    observed = _evaluate(case, {})
    result = {field: observed[field] for field in PUBLIC_FIELDS}
    result["offer_signature_sha256"] = _canonical_sha256(observed["offer_signature"])
    return result


def freeze(args: argparse.Namespace) -> int:
    coverage_paths = tuple(args.coverage)
    artifact_paths = tuple(args.prior_artifact)
    coverage_documents = [json.loads(path.read_text(encoding="utf-8")) for path in coverage_paths]
    coverage_rows = [row for document in coverage_documents for row in document["cases"]]
    seen_written = {row["written"] for row in coverage_rows}
    seen_pairs = {(row["written"], target) for row in coverage_rows for target in row["targets"]}
    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in artifact_paths)
    source_files, candidates = _source_rows(args.nemo_root)
    selected = []
    exclusions = Counter()
    for candidate in candidates:
        reasons = []
        if candidate["written"] in seen_written:
            reasons.append("coverage-written")
        if (candidate["written"], candidate["expected_speech"]) in seen_pairs:
            reasons.append("coverage-pair")
        if candidate["written"] in artifact_text:
            reasons.append("artifact-written")
        if candidate["expected_speech"] in artifact_text:
            reasons.append("artifact-spoken")
        if reasons:
            exclusions.update(reasons)
            continue
        candidate["base"] = _evaluate_row(candidate)
        selected.append(candidate)
    selected.sort(key=lambda row: row["id"])
    if not selected:
        raise SystemExit("no genuinely held-out fraction/percent rows exist")
    base_head = _head(args.base_repo)
    if args.expected_base and base_head != args.expected_base:
        raise SystemExit(f"base moved: expected {args.expected_base}, observed {base_head}")
    rows_sha256 = _canonical_sha256(selected)
    document = {
        "schema": 1,
        "family": "fraction-percent-heldout",
        "base_head": base_head,
        "rows_sha256": rows_sha256,
        "selection": {
            "rule": (
                "all inverse-TN fraction rows and inverse-TN measure rows containing %, "
                "excluding any written surface/pair present in prior coverage and any row "
                "whose written or spoken literal occurs in prior PR6 reports/manifests"
            ),
            "candidate_rows": len(candidates),
            "selected_rows": len(selected),
            "excluded_occurrences": dict(sorted(exclusions.items())),
            "selected_by_locale_family": {
                f"{locale}:{family}": count
                for (locale, family), count in sorted(
                    Counter((row["locale"], row["family"]) for row in selected).items()
                )
            },
        },
        "audit": {
            "coverage": [
                {
                    "path": _audit_label(path),
                    "sha256": _sha256(path.read_bytes()),
                    "rows": len(document["cases"]),
                }
                for path, document in zip(coverage_paths, coverage_documents, strict=True)
            ],
            "prior_artifacts": [
                {"path": _audit_label(path), "sha256": _sha256(path.read_bytes())}
                for path in artifact_paths
            ],
            "source": "nemo",
            "source_revision": _head(args.nemo_repo),
            "source_files": source_files,
        },
        "rows": selected,
    }
    args.out.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"base_head": base_head, "rows": len(selected), "rows_sha256": rows_sha256}))
    return 0


def measure(args: argparse.Namespace) -> int:
    witness = json.loads(args.witness.read_text(encoding="utf-8"))
    rows = witness.get("rows")
    if not isinstance(rows, list) or witness.get("rows_sha256") != _canonical_sha256(rows):
        raise SystemExit("frozen witness hash mismatch")
    source_hashes = {item["path"]: item["sha256"] for item in witness["audit"]["source_files"]}
    for path, expected in source_hashes.items():
        observed = _sha256((args.nemo_root / path).read_bytes())
        if observed != expected:
            raise SystemExit(f"held-out source drift: {path}")
    observations = []
    counts = Counter()
    for row in rows:
        observed = _evaluate_row(row)
        before = row["base"]
        if not before["strict"] and observed["strict"]:
            status = "recovered"
        elif before["strict"] and not observed["strict"]:
            status = "negative_flip"
        else:
            status = "unchanged"
        counts[status] += 1
        observations.append({"id": row["id"], "status": status, "base": before, "head": observed})
    report = {
        "schema": 1,
        "witness_rows_sha256": witness["rows_sha256"],
        "base_head": witness["base_head"],
        "measured_head": _head(args.repo),
        "counts": {name: counts[name] for name in ("recovered", "negative_flip", "unchanged")},
        "observations": observations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], sort_keys=True))
    return 1 if counts["negative_flip"] else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--base-repo", type=Path, required=True)
    freeze_parser.add_argument("--expected-base")
    freeze_parser.add_argument("--nemo-root", type=Path, required=True)
    freeze_parser.add_argument("--nemo-repo", type=Path, required=True)
    freeze_parser.add_argument("--coverage", type=Path, action="append", required=True)
    freeze_parser.add_argument("--prior-artifact", type=Path, action="append", required=True)
    freeze_parser.add_argument("--out", type=Path, required=True)
    freeze_parser.set_defaults(function=freeze)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--repo", type=Path, required=True)
    measure_parser.add_argument("--nemo-root", type=Path, required=True)
    measure_parser.add_argument("--witness", type=Path, required=True)
    measure_parser.add_argument("--out", type=Path, required=True)
    measure_parser.set_defaults(function=measure)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
