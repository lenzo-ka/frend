#!/usr/bin/env python3
"""Recount complete electronic fixture misses without importing NeMo code.

This is development-only compatibility evidence.  It reads the public NeMo fixture
text through :mod:`tools.locale_gate`, and it never reads a corpus shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from icukit.detectors import detect

from frend import apply_input_fold
from frend.electronic import _trim
from frend.normalize import _reading_detectors
from tools.locale_gate import (
    CHECKED_PT_PT,
    GATE_LOCALES,
    LOCALES,
    FixtureCase,
    _evaluate,
    load_cases,
    load_checked_cases,
    load_exclusions,
)

SCHEMA = "frend-electronic-recount/1"
REPO = Path(__file__).resolve().parents[1]
REQUIRED_FIXTURE_SENTINELS = (
    Path("en/data_text_normalization/test_cases_electronic.txt"),
    Path("es/data_text_normalization/test_cases_electronic.txt"),
)
NEMO_FIXTURE_SUBDIR = Path("tests/nemo_text_processing")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refuse_shard_99(path: Path) -> None:
    for part in path.parts:
        lowered = part.casefold()
        if "output-00099-of-00100" in lowered or lowered in {"shard-99", "shard_99"}:
            raise ValueError(f"refusing shard 99 path: {path}")


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(f"not inside the expected git checkout: {path}")
    return result.stdout.strip()


def _git_toplevel(path: Path) -> Path:
    result = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "--show-toplevel"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(f"not inside the expected git checkout: {path}")
    return Path(result.stdout.strip()).resolve()


def _validate_subjects(nemo_root: Path, *, expect_base: str, expect_nemo_commit: str) -> None:
    actual_base = _git_head(REPO)
    if actual_base != expect_base:
        raise ValueError(f"wrong frend base: expected {expect_base}, observed {actual_base}")
    actual_nemo = _git_head(nemo_root)
    if actual_nemo != expect_nemo_commit:
        raise ValueError(
            f"wrong NeMo commit: expected {expect_nemo_commit}, observed {actual_nemo}"
        )
    expected_nemo_root = _git_toplevel(nemo_root) / NEMO_FIXTURE_SUBDIR
    if nemo_root != expected_nemo_root:
        raise ValueError(
            f"wrong NeMo fixture root: expected {expected_nemo_root}, observed {nemo_root}"
        )
    missing = [str(path) for path in REQUIRED_FIXTURE_SENTINELS if not (nemo_root / path).is_file()]
    if missing:
        raise ValueError(f"wrong NeMo fixture root; missing sentinels: {missing!r}")


def _fixture_path(nemo_root: Path, language: str) -> Path:
    return nemo_root / language / "data_text_normalization" / "test_cases_electronic.txt"


def _validate_fixture_inputs(nemo_root: Path, language: str) -> None:
    """Refuse forbidden targets for every fixture that ``load_cases`` will open."""
    root = nemo_root / language / "data_text_normalization"
    for path in root.glob("test_cases_*.txt"):
        _refuse_shard_99(path.resolve())


def _checked_source_path(path: Path) -> str:
    if path == CHECKED_PT_PT.resolve():
        return "tests/data/locales/pt_PT_checked.tsv"
    return str(path)


def _complete_electronic_detection(case: FixtureCase) -> tuple[bool, bool]:
    text = apply_input_fold(case.written, "typographic")
    saw_electronic = False
    for detection in detect(text, _reading_detectors(case.locale)):
        if not str(detection.get("type", "")).startswith("electronic:"):
            continue
        saw_electronic = True
        start, end = int(detection["start"]), int(detection["end"])
        token_start, token_end = start, end
        while token_start and not text[token_start - 1].isspace():
            token_start -= 1
        while token_end < len(text) and not text[token_end].isspace():
            token_end += 1
        if start == token_start and end == _trim(text, token_start, token_end):
            return True, True
    return False, saw_electronic


def _case_key(case: FixtureCase) -> tuple[str, str, int]:
    return case.language, case.source, case.case_number


def _locale_summary(
    cases: Sequence[FixtureCase],
    evaluations: Mapping[tuple[str, str, int, str], Mapping[str, object]],
    classifications: Mapping[tuple[str, str, int, str], tuple[bool, bool]],
) -> dict[str, object]:
    physical = len(cases)
    logical = len({_case_key(case) for case in cases})
    complete = 0
    partial = 0
    no_detection = 0
    strict_pass = 0
    for case in cases:
        key = (*_case_key(case), case.locale)
        if bool(evaluations[key]["strict"]):
            strict_pass += 1
            continue
        is_complete, saw_electronic = classifications[key]
        if is_complete:
            complete += 1
        elif saw_electronic:
            partial += 1
        else:
            no_detection += 1
    return {
        "physical_rows": physical,
        "deduplicated_logical_rows": logical,
        "complete_strict_misses": complete,
        "partial_strict_misses": partial,
        "no_electronic_detection_strict_misses": no_detection,
        "strict_passes": strict_pass,
        "absent_population": physical == 0,
    }


def build_recount(nemo_root: Path, checked_path: Path = CHECKED_PT_PT) -> dict[str, object]:
    """Return the eleven-locale base recount and its exact input hashes."""
    nemo_root = nemo_root.resolve()
    checked_path = checked_path.resolve()
    _refuse_shard_99(nemo_root)
    _refuse_shard_99(checked_path)

    cases_by_locale: dict[str, list[FixtureCase]] = {locale: [] for locale in GATE_LOCALES}
    sources_by_locale: dict[str, list[dict[str, str]]] = {locale: [] for locale in GATE_LOCALES}
    for language, locales in LOCALES.items():
        _validate_fixture_inputs(nemo_root, language)
        path = _fixture_path(nemo_root, language)
        if path.exists():
            sources_by_locale_for_path = {
                "path": f"{language}/data_text_normalization/{path.name}",
                "sha256": _sha256(path),
            }
        else:
            sources_by_locale_for_path = {
                "path": f"{language}/data_text_normalization/{path.name}",
                "sha256": "absent",
            }
        electronic_cases = [
            case
            for case in load_cases(nemo_root, language, locales)
            if case.class_name == "electronic"
        ]
        for locale in locales:
            cases_by_locale[locale].extend(
                case for case in electronic_cases if case.locale == locale
            )
            sources_by_locale[locale].append(sources_by_locale_for_path)

    checked_cases = [
        case
        for case in load_checked_cases(checked_path, "pt_PT")
        if case.class_name == "electronic"
    ]
    cases_by_locale["pt_PT"].extend(checked_cases)
    sources_by_locale["pt_PT"].append(
        {
            "path": _checked_source_path(checked_path),
            "sha256": _sha256(checked_path),
        }
    )

    exclusions = load_exclusions()
    evaluations: dict[tuple[str, str, int, str], Mapping[str, object]] = {}
    classifications: dict[tuple[str, str, int, str], tuple[bool, bool]] = {}
    for cases in cases_by_locale.values():
        for case in cases:
            key = (*_case_key(case), case.locale)
            evaluations[key] = _evaluate(case, exclusions)
            classifications[key] = _complete_electronic_detection(case)

    locales = {
        locale: {
            "fixture_sources": sources_by_locale[locale],
            **_locale_summary(cases_by_locale[locale], evaluations, classifications),
        }
        for locale in GATE_LOCALES
    }
    complete_physical = sum(int(summary["complete_strict_misses"]) for summary in locales.values())
    complete_keys = {
        _case_key(case)
        for cases in cases_by_locale.values()
        for case in cases
        if not bool(evaluations[(*_case_key(case), case.locale)]["strict"])
        and classifications[(*_case_key(case), case.locale)][0]
    }
    return {
        "schema": SCHEMA,
        "locale_order": list(GATE_LOCALES),
        "locales": locales,
        "totals": {
            "complete_strict_misses_physical": complete_physical,
            "complete_strict_misses_deduplicated_logical": len(complete_keys),
        },
    }


def _canonical_bytes(document: Mapping[str, object]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--expect-base", required=True)
    parser.add_argument("--expect-nemo-commit", required=True)
    parser.add_argument("--checked-path", type=Path, default=CHECKED_PT_PT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    _validate_subjects(
        args.nemo_root.resolve(),
        expect_base=args.expect_base,
        expect_nemo_commit=args.expect_nemo_commit,
    )
    document = build_recount(args.nemo_root, args.checked_path)
    payload = _canonical_bytes(document)
    if args.output is None:
        print(payload.decode(), end="")
        return 0
    if args.check:
        if not args.output.exists() or args.output.read_bytes() != payload:
            raise SystemExit(f"electronic recount differs: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
