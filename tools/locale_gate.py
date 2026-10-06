#!/usr/bin/env python3
"""Eleven-locale fixture coverage and resource gate.

The fixture reader consumes NeMo test text only.  It never imports NeMo code and
the repository does not vendor those fixtures.  Coverage is report-only: release
decisions rest on the focused behavioral tests and on regressions reported by
``compare``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import os
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import icu
from icukit.detectors import detect

from frend import apply_input_fold, compose_choices, normalize, resolve_choices
from frend.lattice import ChoiceGraph
from frend.normalize import _reading_detectors
from frend.spacing import unit_gap

REPO = Path(__file__).resolve().parents[1]
CHECKED_PT_PT = REPO / "tests" / "data" / "locales" / "pt_PT_checked.tsv"
EXCLUSIONS_PATH = Path(__file__).with_name("nemo_exclusions.tsv")
CONVENTIONS_PATH = Path(__file__).with_name("locale_conventions.json")

LOCALES: dict[str, tuple[str, ...]] = {
    "en": ("en_US",),
    "es": ("es_MX", "es_ES"),
    "fr": ("fr_FR",),
    "de": ("de_DE",),
    "pt": ("pt_BR",),
    "it": ("it_IT",),
    "zh": ("zh_CN",),
    "ko": ("ko_KR",),
    "ja": ("ja_JP",),
}
GATE_LOCALES = tuple(locale for locales in LOCALES.values() for locale in locales) + ("pt_PT",)


@dataclass(frozen=True)
class GateBudget:
    first_hit_ratio: float = 1.05
    sequential_first_hit_total_ratio: float = 1.05
    rss_loaded_delta_mib: float = 8.0
    rss_workload_delta_mib: float = 8.0
    rss_plateau_delta_mib: float = 2.0
    rss_unseen_soak_delta_mib: float = 2.0
    warm_p50_ratio: float = 1.05
    warm_p95_ratio: float = 1.10
    fold_median_ratio: float = 1.02
    fold_row_ratio: float = 1.05
    max_alternatives_per_edge_delta: int = 0


@dataclass(frozen=True)
class FixtureCase:
    language: str
    locale: str
    class_name: str
    source: str
    case_number: int
    written: str
    targets: tuple[str, ...]
    grouped: bool

    @property
    def case_id(self) -> str:
        return f"{self.source}:{self.case_number}"


def _append_audio_group(
    cases: list[FixtureCase],
    *,
    language: str,
    locales: tuple[str, ...],
    class_name: str,
    source: str,
    case_number: int,
    written: str,
    targets: Sequence[str],
) -> None:
    for locale in locales:
        cases.append(
            FixtureCase(
                language,
                locale,
                class_name,
                source,
                case_number,
                written,
                tuple(targets),
                True,
            )
        )


def load_cases(nemo_root: Path, language: str, locales: tuple[str, ...]) -> list[FixtureCase]:
    """Read the fixture syntax used by NeMo without importing its implementation."""
    root = nemo_root / language / "data_text_normalization"
    cases: list[FixtureCase] = []
    for path in sorted(root.glob("test_cases_*.txt")):
        class_name = path.stem.removeprefix("test_cases_")
        lines = path.read_text(encoding="utf-8").splitlines()
        if class_name == "normalize_with_audio":
            written: str | None = None
            targets: list[str] = []
            group = 0
            for line in lines:
                if line.startswith("~"):
                    if written is not None:
                        group += 1
                        _append_audio_group(
                            cases,
                            language=language,
                            locales=locales,
                            class_name=class_name,
                            source=path.name,
                            case_number=group,
                            written=written,
                            targets=targets,
                        )
                    written = line.strip().replace("~", "")
                    targets = []
                else:
                    targets.append(line.strip())
            if written is not None:
                group += 1
                _append_audio_group(
                    cases,
                    language=language,
                    locales=locales,
                    class_name=class_name,
                    source=path.name,
                    case_number=group,
                    written=written,
                    targets=targets,
                )
            continue

        for line_number, line in enumerate(lines, 1):
            fields = line.split("~")
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: no ~ separator")
            if language == "de":
                written, targets = fields[1], (fields[0],)
            else:
                written, targets = fields[0], tuple(fields[1:])
            for locale in locales:
                cases.append(
                    FixtureCase(
                        language,
                        locale,
                        class_name,
                        path.name,
                        line_number,
                        written,
                        targets,
                        False,
                    )
                )
    return cases


def load_checked_cases(path: Path, locale: str) -> list[FixtureCase]:
    """Load independently sourced checked rows, rejecting every unsourced row."""
    cases: list[FixtureCase] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ["written", "spoken", "class", "source"]
        if reader.fieldnames != expected:
            raise ValueError(f"{path}: expected columns {expected!r}, got {reader.fieldnames!r}")
        for line_number, row in enumerate(reader, 2):
            if not row["source"].strip():
                raise ValueError(f"{path}:{line_number}: every checked row needs a source locator")
            if not row["written"] or not row["spoken"] or not row["class"]:
                raise ValueError(f"{path}:{line_number}: written, spoken, and class are required")
            cases.append(
                FixtureCase(
                    "pt",
                    locale,
                    row["class"],
                    "checked.tsv",
                    line_number - 1,
                    row["written"],
                    (row["spoken"],),
                    False,
                )
            )
    return cases


def _canonical_characters(text: str) -> str:
    pieces: list[str] = []
    for char in unicodedata.normalize("NFC", text).casefold():
        category = unicodedata.category(char)
        if category == "Cf":
            continue
        pieces.append(" " if category.startswith("P") or char.isspace() else char)
    return " ".join("".join(pieces).split())


def strict_form(text: str) -> str:
    """Canonicalize case and punctuation while retaining meaningful spaces."""
    return _canonical_characters(text)


def _strict_keep_cf(text: str) -> str:
    pieces: list[str] = []
    for char in unicodedata.normalize("NFC", text).casefold():
        category = unicodedata.category(char)
        pieces.append(" " if category.startswith("P") or char.isspace() else char)
    return " ".join("".join(pieces).split())


_UNSPACED_SCRIPTS: frozenset[int] = frozenset(
    code
    for code in {
        getattr(icu.UScriptCode, name)
        for name in dir(icu.UScriptCode)
        if name.isupper() and isinstance(getattr(icu.UScriptCode, name), int)
    }
    if icu.Script(code).breaksBetweenLetters()
)


def _unspaced(char: str) -> bool:
    return icu.Script.getScript(char).getScriptCode() in _UNSPACED_SCRIPTS


def presentation_form(text: str) -> str:
    """Apply the future output join only between naturally unspaced scripts."""
    canonical = strict_form(text)
    output: list[str] = []
    for index, char in enumerate(canonical):
        if char != " ":
            output.append(char)
            continue
        left = index - 1
        right = index + 1
        while left >= 0 and canonical[left] == " ":
            left -= 1
        while right < len(canonical) and canonical[right] == " ":
            right += 1
        if (
            left >= 0
            and right < len(canonical)
            and _unspaced(canonical[left])
            and _unspaced(canonical[right])
        ):
            continue
        output.append(char)
    return "".join(output)


def insensitive_form(text: str) -> str:
    return "".join(strict_form(text).split()).replace("〇", "零")


def load_exclusions(path: Path = EXCLUSIONS_PATH) -> dict[tuple[str, str, str], str]:
    exclusions: dict[tuple[str, str, str], str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ["locale", "case_id", "target", "reason", "adjudicated_by"]
        if reader.fieldnames != expected:
            raise ValueError(f"{path}: expected columns {expected!r}, got {reader.fieldnames!r}")
        for line_number, row in enumerate(reader, 2):
            key = (row["locale"], row["case_id"], row["target"])
            if not all(key) or not row["reason"] or not row["adjudicated_by"]:
                raise ValueError(f"{path}:{line_number}: incomplete exclusion")
            if key in exclusions:
                raise ValueError(f"{path}:{line_number}: duplicate exclusion {key!r}")
            exclusions[key] = row["reason"]
    return exclusions


_UPPER_TOKEN = re.compile(r"^[A-Z]{2,}$")


def _expanded_upper(text: str) -> str:
    return " ".join(
        " ".join(token) if _UPPER_TOKEN.fullmatch(token) else token for token in text.split()
    )


def admissible_targets(
    case: FixtureCase, exclusions: Mapping[tuple[str, str, str], str]
) -> tuple[str, ...]:
    values: list[str] = []
    written = strict_form(case.written)
    for target in case.targets:
        if strict_form(target) == written:
            continue
        if (case.locale, case.case_id, target) in exclusions:
            continue
        candidates = (target, _expanded_upper(target)) if case.language == "en" else (target,)
        for candidate in candidates:
            if candidate not in values:
                values.append(candidate)
    return tuple(values)


def _append_piece(current: str, piece: str, boundary: bool, form: Callable[[str], str]) -> str:
    if not piece:
        return current
    if not current or not boundary:
        return current + piece
    if not unit_gap(current, piece):
        return current + piece
    if form is insensitive_form:
        return current + piece
    return current + " " + piece


def offers(graph: ChoiceGraph, wanted: str, *, form: Callable[[str], str]) -> bool:
    """Return whether an exact complete route offers ``wanted`` under ``form``."""
    target = form(wanted)
    outgoing: dict[int, list[tuple[Any, Any]]] = defaultdict(list)
    for edge, unit in zip(graph.lattice.edges, graph.units, strict=True):
        outgoing[edge.start].append((edge, unit))
    states: dict[int, set[tuple[str, bool]]] = defaultdict(set)
    states[0].add(("", True))
    for position in range(graph.lattice.text_length + 1):
        for current, boundary in tuple(states.get(position, ())):
            for edge, unit in outgoing.get(position, ()):
                for alternative in unit.alternatives:
                    if edge.kind == "passthrough" and alternative.provenance.startswith("surface:"):
                        value = current
                        next_boundary = boundary
                        for char in unicodedata.normalize("NFC", alternative.text).casefold():
                            category = unicodedata.category(char)
                            if category == "Cf" and form is not _strict_keep_cf:
                                continue
                            if category.startswith("P") or char.isspace():
                                next_boundary = True
                                continue
                            rendered = "零" if form is insensitive_form and char == "〇" else char
                            value = _append_piece(value, rendered, next_boundary, form)
                            next_boundary = False
                    else:
                        spoken = form(alternative.text)
                        value = _append_piece(current, spoken, boundary, form)
                        next_boundary = True
                    if target.startswith(value):
                        states[edge.end].add((value, next_boundary))
    return any(value == target for value, _boundary in states[graph.lattice.text_length])


def _digit_positions(text: str) -> set[int]:
    return {index for index, char in enumerate(text) if char.isdecimal()}


@cache
def _cldr_date_time_literals(locale: str) -> frozenset[str]:
    generator = icu.DateTimePatternGenerator.createInstance(icu.Locale(locale))
    literals: set[str] = set()
    for skeleton in ("H", "Hm", "Hms", "jm", "jms", "yMd", "yMMMd", "yMMMMd"):
        pattern = generator.getBestPattern(skeleton)
        literals.update(match.group(1) for match in re.finditer(r"'([^']+)'", pattern))
    return frozenset(literal for literal in literals if strict_form(literal))


def mechanism(case: FixtureCase, detections: Sequence[Mapping]) -> str:
    """Classify the recognition/composition mechanism without using a preview."""
    folded = apply_input_fold(case.written, "typographic")
    digit_positions = _digit_positions(folded)
    family = case.class_name.removeprefix("normalize_with_audio")
    family_edges = [
        detection
        for detection in detections
        if str(detection.get("type", "")).split(":", 1)[0]
        in {family, "number" if family in {"cardinal", "decimal", "ordinal"} else family}
    ]
    complete = [
        detection
        for detection in family_edges
        if digit_positions
        and digit_positions <= set(range(int(detection["start"]), int(detection["end"])))
    ]
    if complete:
        end = max(int(detection["end"]) for detection in complete)
        suffix = folded[end:].strip()
        if case.class_name in {"date", "time"} and any(
            strict_form(suffix).startswith(strict_form(literal))
            for literal in _cldr_date_time_literals(case.locale)
        ):
            return "C-literal"
        return "C"
    if family_edges:
        return "R-partial"
    if detections:
        return "R-icukit"
    return "N"


def first_choice(case: FixtureCase) -> str:
    return normalize(case.written, locale=case.locale)


def junk_audit(output: str, locale: str) -> frozenset[str]:
    findings: set[str] = set()
    if locale == "zh_CN" and re.search(r"[初廿卅]", output):
        findings.add("days-ruleset")
    if any(char.isdecimal() for char in output):
        findings.add("latin-digits-unread")
    if "\u00ad" in output:
        findings.add("soft-hyphen")
    if not locale.startswith("en") and re.search(r"\b(and|point|over)\b", output.casefold()):
        findings.add("english-connector")
    return frozenset(findings)


def fixture_grep(nemo_root: Path, values: Sequence[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {value: [] for value in values}
    paths = sorted(nemo_root.glob("*/data_text_normalization/test_cases_*.txt"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for value in values:
            if value in text:
                result[value].append(path)
    return result


def _graph(case: FixtureCase) -> tuple[list[Mapping], ChoiceGraph]:
    recognition_text = apply_input_fold(case.written, "typographic")
    detections = list(detect(recognition_text, _reading_detectors(case.locale)))
    lattice = resolve_choices(detections, source_text=recognition_text, locale=case.locale)
    return detections, compose_choices(lattice)


def _evaluate(case: FixtureCase, exclusions: Mapping[tuple[str, str, str], str]) -> dict:
    targets = admissible_targets(case, exclusions)
    try:
        detections, graph = _graph(case)
        hits = {
            name: [offers(graph, target, form=form) for target in targets]
            for name, form in (
                ("strict", strict_form),
                ("presentation", presentation_form),
                ("insensitive", insensitive_form),
            )
        }
        first = first_choice(case)
        first_strict = any(strict_form(first) == strict_form(target) for target in targets)
        first_presentation = any(
            presentation_form(first) == presentation_form(target) for target in targets
        )
        mech = mechanism(case, detections)
        if not targets:
            mech = "E"
        elif any(hits["strict"]):
            raw_targets = tuple(
                target
                for target in case.targets
                if strict_form(target) != strict_form(case.written)
            )
            if case.language == "en" and not any(
                offers(graph, target, form=strict_form) for target in raw_targets
            ):
                mech = "S-upper"
            elif not any(offers(graph, target, form=_strict_keep_cf) for target in targets):
                mech = "S-cf"
        elif any(hits["presentation"]):
            mech = "S-join"
        elif any(hits["insensitive"]):
            mech = "S-space"
        edge_count = len(graph.lattice.edges)
        detection_count = len(detections)
        max_alternatives = max((len(unit.alternatives) for unit in graph.units), default=0)
        error = None
        junk = sorted(junk_audit(first, case.locale))
    except Exception as exc:  # A crash is a measured miss, never a hidden omission.
        hits = {name: [False] * len(targets) for name in ("strict", "presentation", "insensitive")}
        first = ""
        first_strict = first_presentation = False
        mech = "N"
        detection_count = edge_count = max_alternatives = 0
        error = f"{type(exc).__name__}: {exc}"
        junk = []
    return {
        "id": f"{case.locale}:{case.case_id}",
        "locale": case.locale,
        "language": case.language,
        "case_id": case.case_id,
        "class": case.class_name,
        "written": case.written,
        "targets": list(targets),
        "target_hits": hits,
        "strict": any(hits["strict"]),
        "presentation": any(hits["presentation"]),
        "insensitive": any(hits["insensitive"]),
        "first": first,
        "first_strict": first_strict,
        "first_presentation": first_presentation,
        "error": error,
        "mechanism": mech,
        "junk": junk,
        "detections": detection_count,
        "max_alternatives": max_alternatives,
        "edges": edge_count,
    }


def _summarize(rows: Sequence[dict]) -> dict:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)

    def counts(items: Sequence[dict]) -> dict:
        return {
            "cases": len(items),
            "strict": sum(row["strict"] for row in items),
            "presentation": sum(row["presentation"] for row in items),
            "insensitive": sum(row["insensitive"] for row in items),
            "first_strict": sum(row["first_strict"] for row in items),
            "first_presentation": sum(row["first_presentation"] for row in items),
            "errors": sum(row["error"] is not None for row in items),
            "mechanisms": dict(sorted(Counter(row["mechanism"] for row in items).items())),
            "junk": dict(sorted(Counter(item for row in items for item in row["junk"]).items())),
            "detections": sum(row["detections"] for row in items),
            "max_alternatives": max((row["max_alternatives"] for row in items), default=0),
            "edges": sum(row["edges"] for row in items),
        }

    return {
        "total": counts(rows),
        "classes": {name: counts(by_class[name]) for name in sorted(by_class)},
    }


def _merge_locale_rows(rows: Sequence[dict], locales: tuple[str, ...]) -> list[dict]:
    """Count a logical case once, succeeding when either regional locale succeeds."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["locale"] in locales:
            grouped[row["case_id"]].append(row)
    merged = []
    for case_id, variants in grouped.items():
        first = dict(variants[0])
        first["id"] = f"{'|'.join(locales)}:{case_id}"
        first["locale"] = "|".join(locales)
        for field in (
            "strict",
            "presentation",
            "insensitive",
            "first_strict",
            "first_presentation",
        ):
            first[field] = any(row[field] for row in variants)
        first["error"] = next((row["error"] for row in variants if row["error"]), None)
        first["max_alternatives"] = max(row["max_alternatives"] for row in variants)
        first["edges"] = sum(row["edges"] for row in variants)
        if not first["strict"] and "es_MX" in locales:
            conventions = json.loads(CONVENTIONS_PATH.read_text(encoding="utf-8"))["es_MX"]
            if any(re.search(item["pattern"], first["written"]) for item in conventions):
                first["mechanism"] = "L"
        merged.append(first)
    return sorted(merged, key=lambda row: row["case_id"])


def coverage(nemo_root: Path, out_dir: Path, checked_path: Path = CHECKED_PT_PT) -> dict:
    exclusions = load_exclusions()
    rows: list[dict] = []
    language_rows: dict[str, list[dict]] = defaultdict(list)
    for language, locales in LOCALES.items():
        for case in load_cases(nemo_root, language, locales):
            row = _evaluate(case, exclusions)
            rows.append(row)
            language_rows[language].append(row)
    checked_rows = [
        _evaluate(case, exclusions) for case in load_checked_cases(checked_path, "pt_PT")
    ]
    rows.extend(checked_rows)

    per_locale = {
        locale: _summarize([row for row in rows if row["locale"] == locale])
        for locale in GATE_LOCALES
    }
    logical_rows: list[dict] = []
    for language, locales in LOCALES.items():
        logical_rows.extend(
            _merge_locale_rows(language_rows[language], locales)
            if len(locales) > 1
            else language_rows[language]
        )
    nemo_summary = _summarize(logical_rows)
    report = {
        "schema": 2,
        "frend_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "icu_version": icu.ICU_VERSION,
        "locales": per_locale,
        "logical_nemo": nemo_summary,
        "pt_PT_checked": _summarize(checked_rows),
        "all_gate_workloads": _summarize(logical_rows + checked_rows),
        "cases": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _comparison(
    metric: str,
    base: Mapping[str, int],
    head: Mapping[str, int],
    limit: float,
    budget: str,
) -> dict:
    head_median = head["median"]
    outside_base_range = head_median < base["min"] or head_median > base["max"]
    breached = head_median > limit
    return {
        "metric": metric,
        "base": dict(base),
        "head": dict(head),
        "limit": limit,
        "budget": budget,
        "breached": breached,
        "outside_base_range": outside_base_range,
        "passed": not (breached and outside_base_range),
    }


def _ratio_comparison(
    metric: str,
    base: Mapping[str, int],
    head: Mapping[str, int],
    ratio: float,
    *,
    additive_ns: int = 0,
) -> dict:
    limit = max(base["median"] * ratio, base["median"] + additive_ns)
    budget = f"{ratio:.3f}x"
    if additive_ns:
        budget += f" or +{additive_ns / 1_000_000:.3f} ms"
    return _comparison(metric, base, head, limit, budget)


def _delta_comparison(
    metric: str,
    base: Mapping[str, int],
    head: Mapping[str, int],
    delta_mib: float,
) -> dict:
    delta_bytes = delta_mib * 2**20
    return _comparison(
        metric,
        base,
        head,
        base["median"] + delta_bytes,
        f"+{delta_mib:g} MiB",
    )


def _absolute_comparison(
    metric: str,
    base: Mapping[str, int],
    head: Mapping[str, int],
    limit_mib: float,
) -> dict:
    return _comparison(metric, base, head, limit_mib * 2**20, f"{limit_mib:g} MiB")


def _runtime_comparisons(base: dict, head: dict, budget: GateBudget) -> list[dict]:
    base_summary = base["summary"]
    head_summary = head["summary"]
    checks = []
    for locale in sorted(set(base_summary["first_hit_ns"]) & set(head_summary["first_hit_ns"])):
        checks.append(
            _ratio_comparison(
                f"runtime.first_hit_ns.{locale}",
                base_summary["first_hit_ns"][locale],
                head_summary["first_hit_ns"][locale],
                budget.first_hit_ratio,
                additive_ns=100_000,
            )
        )
    checks.append(
        _ratio_comparison(
            "runtime.sequential_first_hit_total_ns",
            base_summary["sequential_first_hit_total_ns"],
            head_summary["sequential_first_hit_total_ns"],
            budget.sequential_first_hit_total_ratio,
        )
    )
    checks.append(
        _delta_comparison(
            "runtime.rss_loaded_bytes",
            base_summary["rss_loaded_bytes"],
            head_summary["rss_loaded_bytes"],
            budget.rss_loaded_delta_mib,
        )
    )
    for name in ("rss_workload_pass1_bytes", "rss_workload_pass2_bytes"):
        checks.append(
            _delta_comparison(
                f"runtime.{name}",
                base_summary[name],
                head_summary[name],
                budget.rss_workload_delta_mib,
            )
        )
    plateau = {}
    for side, document in (("base", base), ("head", head)):
        values = [
            run["rss_workload_pass2_bytes"] - run["rss_workload_pass1_bytes"]
            for run in document.get("runs", ())
        ]
        if values:
            plateau[side] = _median_range(values)
    if set(plateau) == {"base", "head"}:
        checks.append(
            _absolute_comparison(
                "runtime.rss_workload_plateau_bytes",
                plateau["base"],
                plateau["head"],
                budget.rss_plateau_delta_mib,
            )
        )
    for locale in sorted(set(base_summary["locales"]) & set(head_summary["locales"])):
        for statistic, ratio in (
            ("warm_p50_ns", budget.warm_p50_ratio),
            ("warm_p95_ns", budget.warm_p95_ratio),
        ):
            checks.append(
                _ratio_comparison(
                    f"runtime.locales.{locale}.{statistic}",
                    base_summary["locales"][locale][statistic],
                    head_summary["locales"][locale][statistic],
                    ratio,
                    additive_ns=100_000,
                )
            )
    return checks


def _soak_comparisons(base: dict, head: dict, budget: GateBudget) -> list[dict]:
    return [
        _absolute_comparison(
            "soak.pass3_minus_pass1_bytes",
            base["summary"]["pass3_minus_pass1_bytes"],
            head["summary"]["pass3_minus_pass1_bytes"],
            budget.rss_unseen_soak_delta_mib,
        )
    ]


def _fold_key(row: Mapping) -> tuple[str, str, str]:
    return str(row["id"]), str(row["locale"]), str(row["mode"])


def _fold_comparisons(base: dict, head: dict, budget: GateBudget, index: int) -> list[dict]:
    base_rows = {
        _fold_key(row): row["timings_ns"]["resolve_k64"]
        for row in base["rows"]
        if "resolve_k64" in row.get("timings_ns", {})
    }
    head_rows = {
        _fold_key(row): row["timings_ns"]["resolve_k64"]
        for row in head["rows"]
        if "resolve_k64" in row.get("timings_ns", {})
    }
    common = sorted(set(base_rows) & set(head_rows))
    checks = []
    if common:
        base_median = statistics.median(base_rows[key] for key in common)
        head_median = statistics.median(head_rows[key] for key in common)
        limit = base_median * budget.fold_median_ratio
        checks.append(
            {
                "metric": f"fold[{index}].resolve_k64.median_ns",
                "base": base_median,
                "head": head_median,
                "limit": limit,
                "budget": f"{budget.fold_median_ratio:.3f}x",
                "breached": head_median > limit,
                "passed": head_median <= limit,
            }
        )
    for key in sorted(base_rows):
        head_value = head_rows.get(key)
        limit = base_rows[key] * budget.fold_row_ratio
        checks.append(
            {
                "metric": f"fold[{index}].resolve_k64.row",
                "row": {"id": key[0], "locale": key[1], "mode": key[2]},
                "base": base_rows[key],
                "head": head_value,
                "limit": limit,
                "budget": f"{budget.fold_row_ratio:.3f}x",
                "breached": head_value is None or head_value > limit,
                "passed": head_value is not None and head_value <= limit,
            }
        )
    return checks


def _max_alternative_comparisons(base: dict, head: dict, budget: GateBudget) -> list[dict]:
    checks = []
    for locale in sorted(set(base.get("locales", {})) & set(head.get("locales", {}))):
        base_value = base["locales"][locale]["total"]["max_alternatives"]
        head_value = head["locales"][locale]["total"]["max_alternatives"]
        limit = base_value + budget.max_alternatives_per_edge_delta
        checks.append(
            {
                "metric": f"coverage.locales.{locale}.max_alternatives",
                "base": base_value,
                "head": head_value,
                "limit": limit,
                "budget": f"+{budget.max_alternatives_per_edge_delta}",
                "breached": head_value > limit,
                "passed": head_value <= limit,
            }
        )
    return checks


def compare(
    base: dict,
    head: dict,
    *,
    base_runtime: dict | None = None,
    head_runtime: dict | None = None,
    base_soak: dict | None = None,
    head_soak: dict | None = None,
    base_folds: Sequence[dict] = (),
    head_folds: Sequence[dict] = (),
) -> dict:
    base_rows = {row["id"]: row for row in base.get("cases", ())}
    head_rows = {row["id"]: row for row in head.get("cases", ())}
    fields = ("strict", "presentation", "insensitive", "first_strict", "first_presentation")
    flips = []
    for identifier in sorted(set(base_rows) & set(head_rows)):
        before, after = base_rows[identifier], head_rows[identifier]
        changed = {
            field: {"base": before[field], "head": after[field]}
            for field in fields
            if before[field] != after[field]
        }
        if changed:
            flips.append(
                {
                    "id": identifier,
                    "written": after["written"],
                    "changes": changed,
                    "negative": any(
                        item == {"base": True, "head": False} for item in changed.values()
                    ),
                }
            )
    budget = GateBudget()
    resource_checks = _max_alternative_comparisons(base, head, budget)
    if base_runtime is not None and head_runtime is not None:
        resource_checks.extend(_runtime_comparisons(base_runtime, head_runtime, budget))
    if base_soak is not None and head_soak is not None:
        resource_checks.extend(_soak_comparisons(base_soak, head_soak, budget))
    for index, (base_fold, head_fold) in enumerate(zip(base_folds, head_folds, strict=True), 1):
        resource_checks.extend(_fold_comparisons(base_fold, head_fold, budget, index))
    return {
        "schema": 2,
        "base_only": sorted(set(base_rows) - set(head_rows)),
        "head_only": sorted(set(head_rows) - set(base_rows)),
        "flips": flips,
        "negative_flips": [row for row in flips if row["negative"]],
        "resources": {
            "budget": asdict(budget),
            "checks": resource_checks,
            "failures": [check for check in resource_checks if not check["passed"]],
        },
    }


def rss_bytes() -> int:
    output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(output.strip()) * 1024


def percentile(values: Sequence[int], quantile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = max(0, min(len(ordered) - 1, int(quantile * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _refuse_tracemalloc() -> None:
    if "tracemalloc" in sys.modules:
        raise RuntimeError("locale gate refuses to run while tracemalloc is imported")


def _workloads(nemo_root: Path, checked_path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {locale: [] for locale in GATE_LOCALES}
    for language, locales in LOCALES.items():
        for case in load_cases(nemo_root, language, locales):
            result[case.locale].append(case.written)
    result["pt_PT"] = [case.written for case in load_checked_cases(checked_path, "pt_PT")]
    return result


def _runtime_child(kind: str, locale: str | None, nemo_root: Path, checked_path: Path) -> dict:
    _refuse_tracemalloc()
    if kind == "first":
        assert locale is not None
        started = time.perf_counter_ns()
        normalize("123", locale=locale)
        return {"locale": locale, "first_hit_ns": time.perf_counter_ns() - started}
    workloads = _workloads(nemo_root, checked_path)
    sequential: dict[str, int] = {}
    started_total = time.perf_counter_ns()
    for item in GATE_LOCALES:
        started = time.perf_counter_ns()
        normalize("123", locale=item)
        sequential[item] = time.perf_counter_ns() - started
    sequential_total = time.perf_counter_ns() - started_total
    gc.collect()
    loaded = rss_bytes()
    samples: dict[str, list[int]] = {item: [] for item in GATE_LOCALES}
    rss_passes = []
    errors: Counter[str] = Counter()
    for pass_number in (1, 2):
        for item in GATE_LOCALES:
            for text in workloads[item]:
                started = time.perf_counter_ns()
                try:
                    normalize(text, locale=item)
                except Exception:
                    errors[item] += 1
                elapsed = time.perf_counter_ns() - started
                if pass_number == 2:
                    samples[item].append(elapsed)
        gc.collect()
        rss_passes.append(rss_bytes())
    return {
        "sequential_first_hit_ns": sequential,
        "sequential_first_hit_total_ns": sequential_total,
        "rss_loaded_bytes": loaded,
        "rss_workload_pass1_bytes": rss_passes[0],
        "rss_workload_pass2_bytes": rss_passes[1],
        "locales": {
            item: {
                "inputs": len(samples[item]),
                "warm_p50_ns": percentile(samples[item], 0.50),
                "warm_p95_ns": percentile(samples[item], 0.95),
                "errors": errors[item],
            }
            for item in GATE_LOCALES
        },
    }


def _child_command(*arguments: str) -> dict:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(REPO)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), *arguments]
    result = subprocess.run(
        command,
        cwd=REPO,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _median_range(values: Sequence[int]) -> dict[str, int]:
    return {"median": int(statistics.median(values)), "min": min(values), "max": max(values)}


def runtime(nemo_root: Path, out_dir: Path, checked_path: Path, repeat: int) -> dict:
    _refuse_tracemalloc()
    runs = []
    for run_number in range(1, repeat + 1):
        first_hits = {}
        for locale in GATE_LOCALES:
            payload = _child_command(
                "_runtime-child",
                "--kind",
                "first",
                "--locale",
                locale,
                "--nemo-root",
                str(nemo_root),
                "--checked",
                str(checked_path),
            )
            first_hits[locale] = payload["first_hit_ns"]
        combined = _child_command(
            "_runtime-child",
            "--kind",
            "combined",
            "--nemo-root",
            str(nemo_root),
            "--checked",
            str(checked_path),
        )
        combined["run"] = run_number
        combined["first_hit_ns"] = first_hits
        runs.append(combined)
        print(f"runtime {run_number}/{repeat}", file=sys.stderr, flush=True)
    summary = {
        "first_hit_ns": {
            locale: _median_range([run["first_hit_ns"][locale] for run in runs])
            for locale in GATE_LOCALES
        },
        "sequential_first_hit_total_ns": _median_range(
            [run["sequential_first_hit_total_ns"] for run in runs]
        ),
        "rss_loaded_bytes": _median_range([run["rss_loaded_bytes"] for run in runs]),
        "rss_workload_pass1_bytes": _median_range(
            [run["rss_workload_pass1_bytes"] for run in runs]
        ),
        "rss_workload_pass2_bytes": _median_range(
            [run["rss_workload_pass2_bytes"] for run in runs]
        ),
        "locales": {
            locale: {
                "inputs": runs[0]["locales"][locale]["inputs"],
                "warm_p50_ns": _median_range(
                    [run["locales"][locale]["warm_p50_ns"] for run in runs]
                ),
                "warm_p95_ns": _median_range(
                    [run["locales"][locale]["warm_p95_ns"] for run in runs]
                ),
                "errors": [run["locales"][locale]["errors"] for run in runs],
            }
            for locale in GATE_LOCALES
        },
    }
    report = {
        "schema": 2,
        "repeat": repeat,
        "locale_order": list(GATE_LOCALES),
        "tracemalloc": False,
        "budget": asdict(GateBudget()),
        "runs": runs,
        "summary": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runtime.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _unseen_inputs(locale: str, pass_index: int, count: int = 2000) -> list[str]:
    locale_index = GATE_LOCALES.index(locale)
    seed = (locale_index + 1) * 1_000_000_000_000 + pass_index * 100_000_000
    return [str(seed + index * 7919) for index in range(count)]


def _soak_child() -> dict:
    _refuse_tracemalloc()
    for locale in GATE_LOCALES:
        normalize("123", locale=locale)
    rss_passes = []
    for pass_index in range(3):
        for locale in GATE_LOCALES:
            for text in _unseen_inputs(locale, pass_index):
                normalize(text, locale=locale)
        gc.collect()
        rss_passes.append(rss_bytes())
    return {
        "rss_pass1_bytes": rss_passes[0],
        "rss_pass2_bytes": rss_passes[1],
        "rss_pass3_bytes": rss_passes[2],
        "pass3_minus_pass1_bytes": rss_passes[2] - rss_passes[0],
    }


def soak(out_dir: Path, repeat: int, unseen: bool) -> dict:
    _refuse_tracemalloc()
    if not unseen:
        raise ValueError("soak requires --unseen so the workload contract is explicit")
    runs = []
    for run_number in range(1, repeat + 1):
        run = _child_command("_soak-child")
        run["run"] = run_number
        runs.append(run)
        print(f"soak {run_number}/{repeat}", file=sys.stderr, flush=True)
    report = {
        "schema": 1,
        "repeat": repeat,
        "unseen_inputs_per_locale": 2000,
        "passes": 3,
        "runs": runs,
        "summary": {
            key: _median_range([run[key] for run in runs])
            for key in (
                "rss_pass1_bytes",
                "rss_pass2_bytes",
                "rss_pass3_bytes",
                "pass3_minus_pass1_bytes",
            )
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "soak.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


_ALLOWED_UNBOUNDED_PARAMETERS = {
    "locale",
    "currency",
    "plural",
    "ruleset",
    "unit",
    "zone",
    "script",
    "transform",
    "path",
    "path_text",
    "mtime_ns",
    "size",
    "sha256",
    "magnitude",
    "kind",
}


def unbounded_cache_violations(root: Path) -> list[str]:
    """Statically list unbounded caches whose keys include request-shaped inputs."""
    violations = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            unbounded = False
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "cache":
                    unbounded = True
                elif isinstance(decorator, ast.Call):
                    name = decorator.func.id if isinstance(decorator.func, ast.Name) else None
                    if name == "lru_cache":
                        maxsize = next(
                            (item.value for item in decorator.keywords if item.arg == "maxsize"),
                            decorator.args[0] if decorator.args else None,
                        )
                        unbounded = isinstance(maxsize, ast.Constant) and maxsize.value is None
            if not unbounded:
                continue
            parameters = {
                argument.arg
                for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            } - {"self", "cls"}
            unexpected = parameters - _ALLOWED_UNBOUNDED_PARAMETERS
            if unexpected:
                violations.append(f"{path.name}:{node.lineno}:{node.name}:{sorted(unexpected)}")
    return violations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    coverage_parser = subparsers.add_parser("coverage")
    coverage_parser.add_argument("--nemo-root", type=Path, required=True)
    coverage_parser.add_argument("--out-dir", type=Path, required=True)
    coverage_parser.add_argument("--checked", type=Path, default=CHECKED_PT_PT)
    runtime_parser = subparsers.add_parser("runtime")
    runtime_parser.add_argument("--nemo-root", type=Path, required=True)
    runtime_parser.add_argument("--out-dir", type=Path, required=True)
    runtime_parser.add_argument("--checked", type=Path, default=CHECKED_PT_PT)
    runtime_parser.add_argument("--repeat", type=int, default=5)
    soak_parser = subparsers.add_parser("soak")
    soak_parser.add_argument("--out-dir", type=Path, required=True)
    soak_parser.add_argument("--repeat", type=int, default=5)
    soak_parser.add_argument("--unseen", action="store_true")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--base", type=Path, required=True)
    compare_parser.add_argument("--head", type=Path, required=True)
    compare_parser.add_argument("--base-runtime", type=Path)
    compare_parser.add_argument("--head-runtime", type=Path)
    compare_parser.add_argument("--base-soak", type=Path)
    compare_parser.add_argument("--head-soak", type=Path)
    compare_parser.add_argument("--base-fold", type=Path, action="append", default=[])
    compare_parser.add_argument("--head-fold", type=Path, action="append", default=[])
    compare_parser.add_argument("--out", type=Path)
    grep_parser = subparsers.add_parser("fixture-grep")
    grep_parser.add_argument("--nemo-root", type=Path, required=True)
    grep_parser.add_argument("values", nargs="+")
    child = subparsers.add_parser("_runtime-child")
    child.add_argument("--kind", choices=("first", "combined"), required=True)
    child.add_argument("--locale")
    child.add_argument("--nemo-root", type=Path, required=True)
    child.add_argument("--checked", type=Path, required=True)
    subparsers.add_parser("_soak-child")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "coverage":
        coverage(args.nemo_root, args.out_dir, args.checked)
    elif args.command == "runtime":
        runtime(args.nemo_root, args.out_dir, args.checked, args.repeat)
    elif args.command == "soak":
        soak(args.out_dir, args.repeat, args.unseen)
    elif args.command == "compare":
        for label, left, right in (
            ("runtime", args.base_runtime, args.head_runtime),
            ("soak", args.base_soak, args.head_soak),
        ):
            if (left is None) != (right is None):
                raise SystemExit(f"compare requires both --base-{label} and --head-{label}")
        if len(args.base_fold) != len(args.head_fold):
            raise SystemExit(
                "compare requires the same number of --base-fold and --head-fold files"
            )

        def read_optional(path: Path | None) -> dict | None:
            return json.loads(path.read_text(encoding="utf-8")) if path is not None else None

        payload = compare(
            json.loads(args.base.read_text(encoding="utf-8")),
            json.loads(args.head.read_text(encoding="utf-8")),
            base_runtime=read_optional(args.base_runtime),
            head_runtime=read_optional(args.head_runtime),
            base_soak=read_optional(args.base_soak),
            head_soak=read_optional(args.head_soak),
            base_folds=[read_optional(path) for path in args.base_fold],
            head_folds=[read_optional(path) for path in args.head_fold],
        )
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        if payload["resources"]["failures"]:
            return 1
    elif args.command == "fixture-grep":
        payload = {
            value: [str(path) for path in paths]
            for value, paths in fixture_grep(args.nemo_root, args.values).items()
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "_runtime-child":
        print(json.dumps(_runtime_child(args.kind, args.locale, args.nemo_root, args.checked)))
    elif args.command == "_soak-child":
        print(json.dumps(_soak_child()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
