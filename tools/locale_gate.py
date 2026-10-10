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
import hashlib
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
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import icu
from icukit.detectors import detect

from frend import apply_input_fold, compose_choices, normalize, resolve_choices
from frend.lattice import ChoiceGraph
from frend.locale_data import canonical_locale
from frend.normalize import _append_alternative_part, _reading_detectors
from frend.spacing import unit_gap

REPO = Path(__file__).resolve().parents[1]
CHECKED_PT_PT = REPO / "tests" / "data" / "locales" / "pt_PT_checked.tsv"
EXCLUSIONS_PATH = Path(__file__).with_name("nemo_exclusions.tsv")
CONVENTIONS_PATH = Path(__file__).with_name("locale_conventions.json")
DATE_LATENCY_INPUTS = Path(__file__).with_name("date_latency_inputs.json")
FRACTION_LATENCY_INPUTS = Path(__file__).with_name("fraction_latency_inputs.json")

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
_ALL_GATE_LOCALES = GATE_LOCALES


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
    # A row fails only when it exceeds both the ratio and this absolute regression
    # floor (or the wider observed base spread across repeated alternating runs).
    # Sub-millisecond rows otherwise turn scheduler noise into false regressions.
    fold_row_absolute_floor_ns: int = 250_000
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


def offers(graph: ChoiceGraph, wanted: str, *, form: Callable[[str], str]) -> bool:
    """Return whether an exact complete route offers ``wanted`` under ``form``."""
    target = form(wanted)
    outgoing: dict[int, list[tuple[Any, Any]]] = defaultdict(list)
    for edge, unit in zip(graph.lattice.edges, graph.units, strict=True):
        outgoing[edge.start].append((edge, unit))
    states: dict[int, dict[tuple[str, str | None, bool | None], str]] = defaultdict(dict)
    states[0][("", None, None)] = ""
    for position in range(graph.lattice.text_length + 1):
        for (_canonical, previous_text, _previous_surface), rendered in tuple(
            states.get(position, {}).items()
        ):
            for edge, unit in outgoing.get(position, ()):
                for alternative in unit.alternatives:
                    parts = [rendered] if previous_text is not None else []
                    _append_alternative_part(parts, previous_text, alternative)
                    value = "".join(parts)
                    canonical = form(value)
                    if target.startswith(canonical):
                        key = (
                            canonical,
                            alternative.text,
                            alternative.provenance == "surface:passthrough",
                        )
                        states[edge.end].setdefault(key, value)
    return any(
        canonical == target
        for canonical, _previous_text, _previous_surface in states[graph.lattice.text_length]
    )


def _source_record_offers(
    row: Mapping[str, Any],
    wanted: str,
    source_record_id: str,
    *,
    form: Callable[[str], str],
) -> bool:
    """Return whether a complete serialized route offers ``wanted`` through the source.

    A recovery target can be composed from several edges, so requiring one edge to emit
    the complete fixture target rejects real sourced routes in surrounding text and in
    mixed fractions. This repeats :func:`offers` over the byte-complete public offer
    signature, while retaining whether the selected route actually traversed the
    declared normalization record. An unrelated sourced alternative elsewhere in the
    graph therefore cannot witness the route.
    """
    written = row.get("written")
    signature = row.get("offer_signature")
    if not isinstance(written, str) or not isinstance(signature, list):
        return False
    text_length = len(apply_input_fold(written, "typographic"))
    source_marker = f"normalization-record:{source_record_id}"
    target = form(wanted)
    outgoing: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in signature:
        if not isinstance(edge, Mapping):
            continue
        start, end = edge.get("start"), edge.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        outgoing[start].append(edge)
    states: dict[
        int,
        dict[tuple[str, str | None, bool | None, bool], str],
    ] = defaultdict(dict)
    states[0][("", None, None, False)] = ""
    for position in range(text_length + 1):
        for (
            _canonical,
            previous_text,
            _previous_surface,
            carried_source,
        ), rendered in tuple(states.get(position, {}).items()):
            for edge in outgoing.get(position, ()):
                end = edge.get("end")
                for alternative in edge.get("alternatives", ()):
                    if not isinstance(alternative, Mapping):
                        continue
                    text = alternative.get("text")
                    provenance = alternative.get("provenance")
                    if not isinstance(text, str) or not isinstance(provenance, str):
                        continue
                    surface = provenance == "surface:passthrough"
                    part = text if surface else f" {text} "
                    prefix = rendered
                    if previous_text is not None and not unit_gap(previous_text, text):
                        prefix = prefix.rstrip(" ")
                        part = part.lstrip(" ")
                    value = prefix + part
                    canonical = form(value)
                    if not target.startswith(canonical):
                        continue
                    has_source = carried_source or source_marker in provenance.split("+")
                    key = (canonical, text, surface, has_source)
                    states[end].setdefault(key, value)
    return any(
        canonical == target and carried_source
        for canonical, _previous_text, _previous_surface, carried_source in states[text_length]
    )


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


def _offer_signature(graph: ChoiceGraph) -> list[dict]:
    """A byte-output-complete signature of the factored keep-all graph."""
    signature = []
    for edge, unit in zip(graph.lattice.edges, graph.units, strict=True):
        detection = edge.detection or {}
        value = detection.get("value") if isinstance(detection, Mapping) else None
        fields = getattr(value, "fields", ())
        signature.append(
            {
                "start": edge.start,
                "end": edge.end,
                "kind": edge.kind,
                "type": detection.get("type") if isinstance(detection, Mapping) else None,
                "fields": [name for name, _value in fields],
                "alternatives": [
                    {
                        "text": alternative.text,
                        "provenance": alternative.provenance,
                        "prior_provenance": alternative.prior_provenance,
                        "weight": None if alternative.weight is None else str(alternative.weight),
                        "surface": alternative.provenance == "surface:passthrough",
                    }
                    for alternative in unit.alternatives
                ],
            }
        )
    return signature


def _rbnf_language_counts(graph: ChoiceGraph, locale: str) -> tuple[int, int]:
    """Count RBNF-backed alternatives by whether their rules match the language."""
    own_language = not_own_language = 0
    requested_language = canonical_locale(locale).split("_", 1)[0]
    for unit in graph.units:
        for alternative in unit.alternatives:
            sources = alternative.provenance.split("+")
            if not any(source.startswith("icu-rbnf:") for source in sources):
                continue
            fallback_languages = {
                canonical_locale(source.split(":", 1)[1]).split("_", 1)[0]
                for source in sources
                if source.startswith("icu-rbnf-fallback:")
            }
            if fallback_languages - {requested_language}:
                not_own_language += 1
            else:
                own_language += 1
    return own_language, not_own_language


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
        offer_signature = _offer_signature(graph)
        rbnf_own_language, rbnf_not_own_language = _rbnf_language_counts(graph, case.locale)
    except Exception as exc:  # A crash is a measured miss, never a hidden omission.
        hits = {name: [False] * len(targets) for name in ("strict", "presentation", "insensitive")}
        first = ""
        first_strict = first_presentation = False
        mech = "N"
        detection_count = edge_count = max_alternatives = 0
        error = f"{type(exc).__name__}: {exc}"
        junk = []
        offer_signature = []
        rbnf_own_language = rbnf_not_own_language = 0
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
        "offer_signature": offer_signature,
        "first_strict": first_strict,
        "first_presentation": first_presentation,
        "error": error,
        "mechanism": mech,
        "junk": junk,
        "detections": detection_count,
        "max_alternatives": max_alternatives,
        "edges": edge_count,
        "rbnf_own_language": rbnf_own_language,
        "rbnf_not_own_language": rbnf_not_own_language,
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
            "rbnf_own_language": sum(row["rbnf_own_language"] for row in items),
            "rbnf_not_own_language": sum(row["rbnf_not_own_language"] for row in items),
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
    allow_base_spread: bool = True,
) -> dict:
    limit = max(base["median"] * ratio, base["median"] + additive_ns)
    budget = f"{ratio:.3f}x"
    if additive_ns:
        budget += f" or +{additive_ns / 1_000_000:.3f} ms"
    result = _comparison(metric, base, head, limit, budget)
    if not allow_base_spread:
        result["passed"] = not result["breached"]
    return result


def _delta_comparison(
    metric: str,
    base: Mapping[str, int],
    head: Mapping[str, int],
    delta_mib: float,
    *,
    allow_base_spread: bool = True,
) -> dict:
    delta_bytes = delta_mib * 2**20
    result = _comparison(
        metric,
        base,
        head,
        base["median"] + delta_bytes,
        f"+{delta_mib:g} MiB",
    )
    if not allow_base_spread:
        result["passed"] = not result["breached"]
    return result


def _absolute_comparison(
    metric: str,
    base: Mapping[str, int],
    head: Mapping[str, int],
    limit_mib: float,
) -> dict:
    return _comparison(metric, base, head, limit_mib * 2**20, f"{limit_mib:g} MiB")


def _runtime_comparisons(
    base: dict,
    head: dict,
    budget: GateBudget,
    *,
    pr4_improvement: bool = False,
) -> list[dict]:
    base_summary = base["summary"]
    head_summary = head["summary"]
    checks = []
    expected_date_locales = set(head_summary["first_hit_ns"])
    head_runs = head.get("runs", ())
    if not head_runs:
        checks.append(
            {
                "metric": "runtime.date_probe.receipt_contract",
                "passed": False,
                "breached": True,
                "outside_base_range": False,
                "budget": "one receipt per measured locale in every run",
            }
        )
    for run in head_runs:
        date_probes = run.get("date_probes", {})
        observed_date_locales = set(date_probes)
        receipt_contract_passed = observed_date_locales == expected_date_locales
        checks.append(
            {
                "metric": f"runtime.date_probe.receipt_contract.run{run.get('run', 0)}",
                "expected_locales": sorted(expected_date_locales),
                "observed_locales": sorted(observed_date_locales),
                "passed": receipt_contract_passed,
                "breached": not receipt_contract_passed,
                "outside_base_range": False,
                "budget": "one receipt per measured locale in every run",
            }
        )
        for locale, receipt in date_probes.items():
            generated_locale = locale != "en_US"
            events = receipt.get("events", {})
            passed = (
                receipt.get("date_units", 0) >= 1
                and receipt.get("selected_generated") is generated_locale
                and (
                    not generated_locale
                    or events.get("load") == 1
                    and events.get("lower") == 1
                    and events.get("generate", 0) >= 1
                )
            )
            checks.append(
                {
                    "metric": f"runtime.date_probe.{locale}.run{run.get('run', 0)}",
                    "receipt": receipt,
                    "passed": passed,
                    "breached": not passed,
                    "outside_base_range": False,
                    "budget": "selected date-rule provenance and cold load/lower/generate",
                }
            )
        fraction_probes = run.get("fraction_probes", {})
        observed_fraction_locales = set(fraction_probes)
        fraction_contract = observed_fraction_locales == expected_date_locales
        checks.append(
            {
                "metric": f"runtime.fraction_probe.receipt_contract.run{run.get('run', 0)}",
                "expected_locales": sorted(expected_date_locales),
                "observed_locales": sorted(observed_fraction_locales),
                "passed": fraction_contract,
                "breached": not fraction_contract,
                "outside_base_range": False,
                "budget": "one fraction and percent receipt per measured locale",
            }
        )
        for locale, receipt in fraction_probes.items():
            generated_locale = locale != "en_US"
            probes = receipt.get("probes", {})
            events = receipt.get("events", {})
            passed = (
                set(probes) == {"fraction", "percent"}
                and all(item.get("units", 0) >= 1 for item in probes.values())
                and all(
                    item.get("selected_generated") is generated_locale for item in probes.values()
                )
                and receipt.get("bundle_available") is True
                and (
                    not generated_locale
                    or events.get("load") == 1
                    and events.get("lower") == 0
                    and events.get("generate") == 0
                )
                and (generated_locale or events == {"load": 0, "lower": 0, "generate": 0})
            )
            checks.append(
                {
                    "metric": f"runtime.fraction_probe.{locale}.run{run.get('run', 0)}",
                    "receipt": receipt,
                    "passed": passed,
                    "breached": not passed,
                    "outside_base_range": False,
                    "budget": (
                        "selected fraction-rule provenance; compiled recipes avoid runtime "
                        "lower/generate; English zero bundle calls"
                    ),
                }
            )
    for locale in sorted(set(base_summary["first_hit_ns"]) & set(head_summary["first_hit_ns"])):
        checks.append(
            _ratio_comparison(
                f"runtime.first_hit_ns.{locale}",
                base_summary["first_hit_ns"][locale],
                head_summary["first_hit_ns"][locale],
                budget.first_hit_ratio,
                additive_ns=0 if pr4_improvement else 100_000,
                allow_base_spread=not pr4_improvement,
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
    if "prewarm_total_ns" in head_summary:
        base_prewarm = base_summary.get(
            "prewarm_total_ns", base_summary["sequential_first_hit_total_ns"]
        )
        comparison = "base_prewarm" if "prewarm_total_ns" in base_summary else "base_sequential"
        checks.append(
            _ratio_comparison(
                f"runtime.prewarm_total_ns_vs_{comparison}",
                base_prewarm,
                head_summary["prewarm_total_ns"],
                budget.sequential_first_hit_total_ratio,
            )
        )
    checks.append(
        _delta_comparison(
            "runtime.rss_loaded_bytes",
            base_summary["rss_loaded_bytes"],
            head_summary["rss_loaded_bytes"],
            budget.rss_loaded_delta_mib,
            allow_base_spread=not pr4_improvement,
        )
    )
    for name in ("rss_workload_pass1_bytes", "rss_workload_pass2_bytes"):
        checks.append(
            _delta_comparison(
                f"runtime.{name}",
                base_summary[name],
                head_summary[name],
                budget.rss_workload_delta_mib,
                allow_base_spread=not pr4_improvement,
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
                    additive_ns=100_000 if statistic == "warm_p50_ns" else 0,
                    allow_base_spread=statistic == "warm_p50_ns",
                )
            )
    return checks


def _soak_comparisons(base: dict, head: dict, budget: GateBudget) -> list[dict]:
    checks = [
        _absolute_comparison(
            "soak.pass3_minus_pass1_bytes",
            base["summary"]["pass3_minus_pass1_bytes"],
            head["summary"]["pass3_minus_pass1_bytes"],
            budget.rss_unseen_soak_delta_mib,
        )
    ]
    expected_receipts = {
        (locale, pass_index) for locale in GATE_LOCALES for pass_index in range(1, 4)
    }
    head_runs = head.get("runs", ())
    if not head_runs:
        checks.append(
            {
                "metric": "soak.date_probe.receipt_contract",
                "passed": False,
                "breached": True,
                "outside_base_range": False,
                "budget": "one receipt per locale and pass in every run",
            }
        )
    for run in head_runs:
        date_receipts = run.get("date_receipts", ())
        observed_receipts = {
            (receipt.get("locale"), receipt.get("pass")) for receipt in date_receipts
        }
        receipt_contract_passed = observed_receipts == expected_receipts and len(
            date_receipts
        ) == len(expected_receipts)
        checks.append(
            {
                "metric": f"soak.date_probe.receipt_contract.run{run.get('run', 0)}",
                "expected": sorted(expected_receipts),
                "observed": sorted(observed_receipts),
                "passed": receipt_contract_passed,
                "breached": not receipt_contract_passed,
                "outside_base_range": False,
                "budget": "one receipt per locale and pass in every run",
            }
        )
        for receipt in date_receipts:
            count = receipt.get("count", 0)
            generated_expected = receipt.get("locale") != "en_US"
            passed = receipt.get("successful_date_detections") == count and (
                receipt.get("selected_generated") == count if generated_expected else True
            )
            checks.append(
                {
                    "metric": (
                        f"soak.date_probe.{receipt.get('locale')}."
                        f"pass{receipt.get('pass')} .run{run.get('run', 0)}"
                    ).replace(" ", ""),
                    "receipt": receipt,
                    "passed": passed,
                    "breached": not passed,
                    "outside_base_range": False,
                    "budget": "full-span detection and selected date-rule provenance",
                }
            )
        fraction_receipts = run.get("fraction_receipts", ())
        observed_fraction_receipts = {
            (receipt.get("locale"), receipt.get("pass")) for receipt in fraction_receipts
        }
        fraction_contract = observed_fraction_receipts == expected_receipts and len(
            fraction_receipts
        ) == len(expected_receipts)
        checks.append(
            {
                "metric": f"soak.fraction_probe.receipt_contract.run{run.get('run', 0)}",
                "expected": sorted(expected_receipts),
                "observed": sorted(observed_fraction_receipts),
                "passed": fraction_contract,
                "breached": not fraction_contract,
                "outside_base_range": False,
                "budget": "one unseen fraction receipt per locale and pass",
            }
        )
        for receipt in fraction_receipts:
            count = receipt.get("count", 0)
            generated_expected = receipt.get("locale") != "en_US"
            passed = receipt.get("successful_fraction_detections") == count and (
                receipt.get("selected_generated") == count
                if generated_expected
                else receipt.get("selected_generated") == 0
            )
            checks.append(
                {
                    "metric": (
                        f"soak.fraction_probe.{receipt.get('locale')}."
                        f"pass{receipt.get('pass')}.run{run.get('run', 0)}"
                    ),
                    "receipt": receipt,
                    "passed": passed,
                    "breached": not passed,
                    "outside_base_range": False,
                    "budget": "full-span detection and selected fraction-rule provenance",
                }
            )
    return checks


def _fold_key(row: Mapping) -> tuple[str, str, str]:
    return str(row["id"]), str(row["locale"]), str(row["mode"])


def _fold_run_rows(document: dict) -> dict[tuple[str, str, str], int]:
    return {
        _fold_key(row): row["timings_ns"]["resolve_k64"]
        for row in document["rows"]
        if "resolve_k64" in row.get("timings_ns", {})
    }


def _fold_comparisons(
    base_runs: Sequence[dict],
    head_runs: Sequence[dict],
    budget: GateBudget,
    label: str,
    *,
    pr4_improvement: bool = False,
) -> list[dict]:
    base_run_rows = [_fold_run_rows(document) for document in base_runs]
    head_run_rows = [_fold_run_rows(document) for document in head_runs]
    base_keys = set.intersection(*(set(rows) for rows in base_run_rows)) if base_run_rows else set()
    head_keys = set.intersection(*(set(rows) for rows in head_run_rows)) if head_run_rows else set()
    common = sorted(base_keys & head_keys)
    checks = []
    if common:
        base_corpus = _median_range(
            [int(statistics.median(rows[key] for key in common)) for rows in base_run_rows]
        )
        head_corpus = _median_range(
            [int(statistics.median(rows[key] for key in common)) for rows in head_run_rows]
        )
        corpus = _comparison(
            f"fold[{label}].resolve_k64.corpus_median_ns",
            base_corpus,
            head_corpus,
            base_corpus["median"] * budget.fold_median_ratio,
            f"{budget.fold_median_ratio:.3f}x and outside base run spread",
        )
        if pr4_improvement:
            corpus["passed"] = not corpus["breached"]
        corpus["runs"] = {"base": len(base_runs), "head": len(head_runs)}
        checks.append(corpus)
    for key in sorted(base_keys):
        base_value = _median_range([rows[key] for rows in base_run_rows])
        head_values = [rows[key] for rows in head_run_rows if key in rows]
        head_value = _median_range(head_values) if len(head_values) == len(head_run_rows) else None
        ratio_limit = base_value["median"] * budget.fold_row_ratio
        absolute_floor = max(
            budget.fold_row_absolute_floor_ns,
            base_value["max"] - base_value["min"],
        )
        breached = (
            head_value is None
            or head_value["median"] > ratio_limit
            and head_value["median"] - base_value["median"] > absolute_floor
        )
        checks.append(
            {
                "metric": f"fold[{label}].resolve_k64.row",
                "row": {"id": key[0], "locale": key[1], "mode": key[2]},
                "base": base_value,
                "head": head_value,
                "ratio_limit": ratio_limit,
                "absolute_floor_ns": absolute_floor,
                "budget": (
                    f">{budget.fold_row_ratio:.3f}x AND "
                    f">+max({budget.fold_row_absolute_floor_ns} ns, base run spread)"
                ),
                "breached": breached,
                "passed": not breached,
            }
        )
    return checks


def _fold_group(document: dict, index: int) -> str:
    data_dir = document.get("data_dir")
    return Path(data_dir).name if data_dir else f"input-{index}"


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
    require_identical: bool = False,
    expected_recoveries: dict | None = None,
    allowed_changes: dict | None = None,
    require_no_negative_flips: bool = False,
    require_identity_locales: Sequence[str] = (),
    pr4_improvement: bool = False,
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
    identity_mismatches = []
    if require_identical or require_identity_locales:
        for identifier in sorted(set(base_rows) & set(head_rows)):
            before, after = base_rows[identifier], head_rows[identifier]
            if require_identity_locales and before.get("locale") not in require_identity_locales:
                continue
            changed = {
                field: {"base": before.get(field), "head": after.get(field)}
                for field in ("first", "offer_signature", "error")
                if before.get(field) != after.get(field)
            }
            for field in ("first", "offer_signature"):
                missing = [
                    side for side, row in (("base", before), ("head", after)) if field not in row
                ]
                if missing:
                    changed[field] = {"missing_on": missing}
            if changed:
                identity_mismatches.append(
                    {"id": identifier, "written": after["written"], "changes": changed}
                )

    public_fields = (
        "strict",
        "presentation",
        "insensitive",
        "first",
        "first_strict",
        "first_presentation",
        "error",
        "offer_signature",
    )
    correctness_failures: list[dict] = []
    allowed_ids: set[str] = set()
    if allowed_changes is not None:
        manifest_rows = allowed_changes.get("rows") if isinstance(allowed_changes, dict) else None
        if not isinstance(manifest_rows, list):
            correctness_failures.append({"kind": "allowed-changes-schema"})
            manifest_rows = []
        by_id = {str(row.get("id")): row for row in manifest_rows if isinstance(row, dict)}
        if len(by_id) != len(manifest_rows):
            correctness_failures.append({"kind": "allowed-changes-duplicate-id"})
        allowed_ids = set(by_id)
        manifest_family = allowed_changes.get("family", "date")

        def routed(edge: Mapping) -> bool:
            type_name = str(edge.get("type", ""))
            if manifest_family == "date":
                return type_name.startswith("date:") and set(edge.get("fields", ())) in (
                    {"M", "d"},
                    {"y", "M", "d"},
                )
            if manifest_family == "fraction":
                return (
                    type_name.startswith("fraction:")
                    or type_name.startswith("number:fraction")
                    or type_name == "number:percent"
                )
            return False

        if manifest_family not in {"date", "fraction"}:
            correctness_failures.append(
                {"kind": "allowed-changes-family", "family": manifest_family}
            )
        routed_ids = {
            identifier
            for identifier, row in base_rows.items()
            if row.get("locale") != "en_US"
            and any(
                routed(edge) for edge in row.get("offer_signature", ()) if isinstance(edge, Mapping)
            )
        }
        if allowed_ids != routed_ids:
            correctness_failures.append(
                {
                    "kind": "allowed-changes-routing-set",
                    "missing": sorted(routed_ids - allowed_ids),
                    "extra": sorted(allowed_ids - routed_ids),
                }
            )
        for identifier in sorted(set(base_rows) | set(head_rows) | allowed_ids):
            before, after = base_rows.get(identifier), head_rows.get(identifier)
            if before is None or after is None:
                continue
            if identifier in by_id:
                declared = by_id[identifier]
                for side, actual in (("base", before), ("head", after)):
                    expected = declared.get(side)
                    observed = {field: actual.get(field) for field in public_fields}
                    if expected != observed:
                        correctness_failures.append(
                            {
                                "kind": "allowed-change-payload",
                                "id": identifier,
                                "side": side,
                                "expected": expected,
                                "observed": observed,
                            }
                        )
                if manifest_family == "fraction":
                    base_edges = {
                        (edge.get("start"), edge.get("end"), edge.get("type")): edge
                        for edge in before.get("offer_signature", ())
                        if isinstance(edge, Mapping)
                    }
                    head_edges = {
                        (edge.get("start"), edge.get("end"), edge.get("type")): edge
                        for edge in after.get("offer_signature", ())
                        if isinstance(edge, Mapping)
                    }
                    for edge_key in sorted(set(base_edges) & set(head_edges), key=str):
                        base_alternatives = base_edges[edge_key].get("alternatives", ())
                        head_alternatives = head_edges[edge_key].get("alternatives", ())
                        base_by_text = {
                            item.get("text"): item
                            for item in base_alternatives
                            if isinstance(item, Mapping)
                        }
                        head_by_text = {
                            item.get("text"): item
                            for item in head_alternatives
                            if isinstance(item, Mapping)
                        }
                        common = set(base_by_text) & set(head_by_text)
                        base_order = [
                            item.get("text")
                            for item in base_alternatives
                            if isinstance(item, Mapping) and item.get("text") in common
                        ]
                        head_order = [
                            item.get("text")
                            for item in head_alternatives
                            if isinstance(item, Mapping) and item.get("text") in common
                        ]
                        if base_order != head_order:
                            correctness_failures.append(
                                {
                                    "kind": "unchanged-reading-order",
                                    "id": identifier,
                                    "edge": edge_key,
                                    "base": base_order,
                                    "head": head_order,
                                }
                            )
                        for text in common:
                            base_item = base_by_text[text]
                            head_item = head_by_text[text]
                            base_key = base_item.get("prior_provenance") or base_item.get(
                                "provenance"
                            )
                            head_key = head_item.get("prior_provenance") or head_item.get(
                                "provenance"
                            )
                            if base_key != head_key or base_item.get("weight") != head_item.get(
                                "weight"
                            ):
                                correctness_failures.append(
                                    {
                                        "kind": "unchanged-reading-ranking",
                                        "id": identifier,
                                        "edge": edge_key,
                                        "text": text,
                                        "base": {
                                            "effective_prior_provenance": base_key,
                                            "weight": base_item.get("weight"),
                                        },
                                        "head": {
                                            "effective_prior_provenance": head_key,
                                            "weight": head_item.get("weight"),
                                        },
                                    }
                                )
            else:
                changed = {
                    field: {"base": before.get(field), "head": after.get(field)}
                    for field in public_fields
                    if before.get(field) != after.get(field)
                }
                if changed:
                    correctness_failures.append(
                        {"kind": "change-outside-manifest", "id": identifier, "changes": changed}
                    )

    recovery_observations: list[dict] = []
    recovery_ids: set[str] = set()
    if expected_recoveries is not None:
        manifest_rows = (
            expected_recoveries.get("rows") if isinstance(expected_recoveries, dict) else None
        )
        if not isinstance(manifest_rows, list):
            correctness_failures.append({"kind": "expected-recoveries-schema"})
            manifest_rows = []
        witness_bytes = json.dumps(
            manifest_rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        observed_witness_sha256 = hashlib.sha256(witness_bytes).hexdigest()
        if expected_recoveries.get("witness_sha256") != observed_witness_sha256:
            correctness_failures.append(
                {
                    "kind": "expected-recovery-witness-hash",
                    "expected": expected_recoveries.get("witness_sha256"),
                    "observed": observed_witness_sha256,
                }
            )
        seen: set[str] = set()
        for declared in manifest_rows:
            identifier = str(declared.get("id"))
            if identifier in seen:
                correctness_failures.append(
                    {"kind": "expected-recoveries-duplicate-id", "id": identifier}
                )
                continue
            seen.add(identifier)
            recovery_ids.add(identifier)
            before, after = base_rows.get(identifier), head_rows.get(identifier)
            observed = {
                "id": identifier,
                "base": None if before is None else {field: before.get(field) for field in fields},
                "head": None if after is None else {field: after.get(field) for field in fields},
            }
            recovery_observations.append(observed)
            if observed["base"] != declared.get("base") or observed["head"] != declared.get("head"):
                correctness_failures.append(
                    {
                        "kind": "expected-recovery-mismatch",
                        "id": identifier,
                        "expected": {"base": declared.get("base"), "head": declared.get("head")},
                        "observed": {"base": observed["base"], "head": observed["head"]},
                    }
                )
            expected_speech = declared.get("expected_speech")
            targets = () if after is None else after.get("targets", ())
            if not isinstance(expected_speech, str) or expected_speech not in targets:
                correctness_failures.append(
                    {
                        "kind": "expected-recovery-speech",
                        "id": identifier,
                        "expected_speech": expected_speech,
                        "targets": targets,
                    }
                )
            source_record_id = declared.get("source_record_id")
            recovered_forms = (
                form
                for name, form in (
                    ("strict", strict_form),
                    ("presentation", presentation_form),
                    ("insensitive", insensitive_form),
                )
                if before is not None
                and after is not None
                and before.get(name) is False
                and after.get(name) is True
            )
            if (
                not isinstance(source_record_id, str)
                or not isinstance(expected_speech, str)
                or after is None
                or not any(
                    _source_record_offers(
                        after,
                        expected_speech,
                        source_record_id,
                        form=form,
                    )
                    for form in recovered_forms
                )
            ):
                correctness_failures.append(
                    {
                        "kind": "expected-recovery-source-record",
                        "id": identifier,
                        "source_record_id": source_record_id,
                    }
                )
        unexpected_positive = []
        for identifier in sorted(set(base_rows) & set(head_rows) - recovery_ids):
            before, after = base_rows[identifier], head_rows[identifier]
            changed = {
                field: {"base": before[field], "head": after[field]}
                for field in ("strict", "presentation", "insensitive")
                if before[field] is False and after[field] is True
            }
            if changed:
                unexpected_positive.append({"id": identifier, "changes": changed})
        if unexpected_positive:
            correctness_failures.append(
                {"kind": "unexpected-positive-flips", "rows": unexpected_positive}
            )
        strict_total = sum(
            not row["base"]["strict"] and row["head"]["strict"]
            for row in recovery_observations
            if row["base"] is not None and row["head"] is not None
        )
        insensitive_total = sum(
            not row["base"]["insensitive"] and row["head"]["insensitive"]
            for row in recovery_observations
            if row["base"] is not None and row["head"] is not None
        )
        if strict_total < 3:
            correctness_failures.append(
                {
                    "kind": "expected-recovery-minimum",
                    "field": "strict_total",
                    "minimum": 3,
                    "observed": strict_total,
                }
            )
        for name, observed_total in (
            ("strict_total", strict_total),
            ("insensitive_total", insensitive_total),
        ):
            declared_total = expected_recoveries.get(name)
            if observed_total != declared_total:
                correctness_failures.append(
                    {
                        "kind": "expected-recovery-total",
                        "field": name,
                        "expected": declared_total,
                        "observed": observed_total,
                    }
                )
    if require_no_negative_flips and any(row["negative"] for row in flips):
        correctness_failures.append(
            {"kind": "negative-flips", "rows": [row for row in flips if row["negative"]]}
        )
    base_only = sorted(set(base_rows) - set(head_rows))
    head_only = sorted(set(head_rows) - set(base_rows))
    if (allowed_changes is not None or expected_recoveries is not None) and (
        base_only or head_only
    ):
        correctness_failures.append(
            {"kind": "row-set", "base_only": base_only, "head_only": head_only}
        )
    budget = GateBudget()
    if isinstance(allowed_changes, dict) and allowed_changes.get("family") == "fraction":
        generated_bytes = sum(
            path.stat().st_size for path in (REPO / "frend/data/fraction_rules").glob("*.json")
        )
        allocation_bytes = sum(
            max(256 * 2**10, 4 * path.stat().st_size)
            for path in (REPO / "frend/data/fraction_rules").glob("*.json")
        )
        loaded_mib = min(8.0, allocation_bytes / 2**20)
        budget = replace(
            budget,
            rss_loaded_delta_mib=loaded_mib,
            rss_workload_delta_mib=loaded_mib + 2.0,
        )
        if generated_bytes == 0:
            correctness_failures.append({"kind": "fraction-bundle-size-empty"})
    if pr4_improvement:
        budget = replace(
            budget,
            first_hit_ratio=1.0,
            rss_loaded_delta_mib=0.0,
            rss_workload_delta_mib=0.0,
            fold_median_ratio=0.5,
        )
    resource_checks = _max_alternative_comparisons(base, head, budget)
    if base_runtime is not None and head_runtime is not None:
        resource_checks.extend(
            _runtime_comparisons(
                base_runtime,
                head_runtime,
                budget,
                pr4_improvement=pr4_improvement,
            )
        )
    if base_soak is not None and head_soak is not None:
        resource_checks.extend(_soak_comparisons(base_soak, head_soak, budget))
    base_fold_groups: dict[str, list[dict]] = defaultdict(list)
    head_fold_groups: dict[str, list[dict]] = defaultdict(list)
    for index, document in enumerate(base_folds, 1):
        base_fold_groups[_fold_group(document, index)].append(document)
    for index, document in enumerate(head_folds, 1):
        head_fold_groups[_fold_group(document, index)].append(document)
    for label in sorted(set(base_fold_groups) | set(head_fold_groups)):
        resource_checks.extend(
            _fold_comparisons(
                base_fold_groups.get(label, ()),
                head_fold_groups.get(label, ()),
                budget,
                label,
                pr4_improvement=pr4_improvement,
            )
        )
    return {
        "schema": 2,
        "base_only": base_only,
        "head_only": head_only,
        "flips": flips,
        "negative_flips": [row for row in flips if row["negative"]],
        "identity": {
            "required": require_identical or bool(require_identity_locales),
            "locales": list(require_identity_locales),
            "base_only": base_only,
            "head_only": head_only,
            "mismatches": identity_mismatches,
            "passed": not (require_identical or require_identity_locales)
            or not (
                set(base_rows) - set(head_rows)
                or set(head_rows) - set(base_rows)
                or identity_mismatches
            ),
        },
        "correctness": {
            "allowed_change_ids": sorted(allowed_ids),
            "recoveries": recovery_observations,
            "failures": correctness_failures,
            "passed": not correctness_failures,
        },
        "resources": {
            "profile": "pr4-improvement" if pr4_improvement else "common",
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


@cache
def _date_latency_probes() -> dict[str, dict[str, str]]:
    document = json.loads(DATE_LATENCY_INPUTS.read_text(encoding="utf-8"))
    if document.get("schema") != 1 or set(document.get("locales", ())) != set(_ALL_GATE_LOCALES):
        raise ValueError("date latency manifest must name exactly the eleven gate locales")
    return document["locales"]


def _normalize_date_probe(locale: str) -> tuple[int, dict]:
    probe = _date_latency_probes()[locale]["probe"]
    started = time.perf_counter_ns()
    result = normalize(probe, locale=locale, offsets=True)
    elapsed = time.perf_counter_ns() - started
    date_units = [unit for unit in result.units if str(unit.reader or "").startswith("date:")]
    selected_generated = any(
        f"date-rule:{locale}" in unit.provenance.split("+") for unit in date_units
    )
    events = {"load": 0, "lower": 0, "generate": 0}
    try:
        from frend.date_rules import date_rule_events
    except ImportError:
        pass
    else:
        events = dict(date_rule_events())
    return elapsed, {
        "probe": probe,
        "date_units": len(date_units),
        "selected_generated": selected_generated,
        "events": events,
    }


@cache
def _fraction_latency_probes() -> dict[str, dict[str, str]]:
    document = json.loads(FRACTION_LATENCY_INPUTS.read_text(encoding="utf-8"))
    if document.get("schema") != 1 or set(document.get("locales", ())) != set(_ALL_GATE_LOCALES):
        raise ValueError("fraction latency manifest must name exactly the eleven gate locales")
    return document["locales"]


def _normalize_fraction_probes(locale: str) -> tuple[int, dict]:
    elapsed = 0
    receipts = {}
    before = {"load": 0, "lower": 0, "generate": 0}
    try:
        from frend.fraction_rules import fraction_rule_events
    except ImportError:
        fraction_rule_events = None
    if fraction_rule_events is not None:
        before = dict(fraction_rule_events())
    for kind, probe in _fraction_latency_probes()[locale].items():
        started = time.perf_counter_ns()
        result = normalize(probe, locale=locale, offsets=True)
        elapsed += time.perf_counter_ns() - started
        units = [
            unit
            for unit in result.units
            if str(unit.reader or "").startswith("fraction:")
            or str(unit.reader or "").startswith("number:fraction")
            or unit.reader == "number:percent"
        ]
        receipts[kind] = {
            "probe": probe,
            "units": len(units),
            "selected_generated": any(
                f"fraction-rule:{locale}" in unit.provenance.split("+") for unit in units
            ),
        }
    after = before if fraction_rule_events is None else dict(fraction_rule_events())
    return elapsed, {
        "probes": receipts,
        "events": {key: after[key] - before[key] for key in before},
        "bundle_available": fraction_rule_events is not None,
    }


def _runtime_child(kind: str, locale: str | None, nemo_root: Path, checked_path: Path) -> dict:
    _refuse_tracemalloc()
    if kind == "first":
        assert locale is not None
        started = time.perf_counter_ns()
        normalize("123", locale=locale)
        elapsed = time.perf_counter_ns() - started
        date_elapsed, date_receipt = _normalize_date_probe(locale)
        fraction_elapsed, fraction_receipt = _normalize_fraction_probes(locale)
        del date_elapsed, fraction_elapsed
        return {
            "locale": locale,
            "first_hit_ns": elapsed,
            "date_probe": date_receipt,
            "fraction_probe": fraction_receipt,
        }
    if kind == "prewarm":
        try:
            from frend import prewarm
        except ImportError:
            return {"prewarm_available": False}

        started = time.perf_counter_ns()
        order = prewarm(GATE_LOCALES)
        return {
            "prewarm_available": True,
            "prewarm_total_ns": time.perf_counter_ns() - started,
            "order": list(order),
        }
    workloads = _workloads(nemo_root, checked_path)
    sequential: dict[str, int] = {}
    started_total = time.perf_counter_ns()
    for item in GATE_LOCALES:
        started = time.perf_counter_ns()
        normalize("123", locale=item)
        sequential[item] = time.perf_counter_ns() - started
    sequential_total = time.perf_counter_ns() - started_total
    for item in GATE_LOCALES:
        _normalize_date_probe(item)
        _normalize_fraction_probes(item)
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


def _child_command(*arguments: str, repo: Path = REPO) -> dict:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(repo)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), *arguments]
    result = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _median_range(values: Sequence[int]) -> dict[str, int]:
    return {"median": int(statistics.median(values)), "min": min(values), "max": max(values)}


def _runtime_once(
    repo: Path,
    nemo_root: Path,
    checked_path: Path,
    *,
    include_prewarm: bool,
) -> dict:
    first_hits = {}
    date_probes = {}
    fraction_probes = {}
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
            repo=repo,
        )
        first_hits[locale] = payload["first_hit_ns"]
        date_probes[locale] = payload["date_probe"]
        fraction_probes[locale] = payload["fraction_probe"]
    combined = _child_command(
        "_runtime-child",
        "--kind",
        "combined",
        "--nemo-root",
        str(nemo_root),
        "--checked",
        str(checked_path),
        repo=repo,
    )
    combined["first_hit_ns"] = first_hits
    combined["date_probes"] = date_probes
    combined["fraction_probes"] = fraction_probes
    if include_prewarm:
        prewarm_result = _child_command(
            "_runtime-child",
            "--kind",
            "prewarm",
            "--nemo-root",
            str(nemo_root),
            "--checked",
            str(checked_path),
            repo=repo,
        )
        if prewarm_result.get("prewarm_available"):
            combined.update(prewarm_result)
    return combined


def _runtime_report(runs: Sequence[dict]) -> dict:
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
    if all("prewarm_total_ns" in run for run in runs):
        summary["prewarm_total_ns"] = _median_range([run["prewarm_total_ns"] for run in runs])
    return {
        "schema": 2,
        "repeat": len(runs),
        "locale_order": list(GATE_LOCALES),
        "tracemalloc": False,
        "budget": asdict(GateBudget()),
        "runs": list(runs),
        "summary": summary,
    }


def runtime(nemo_root: Path, out_dir: Path, checked_path: Path, repeat: int) -> dict:
    _refuse_tracemalloc()
    runs = []
    for run_number in range(1, repeat + 1):
        combined = _runtime_once(REPO, nemo_root, checked_path, include_prewarm=True)
        combined["run"] = run_number
        runs.append(combined)
        print(f"runtime {run_number}/{repeat}", file=sys.stderr, flush=True)
    report = _runtime_report(runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runtime.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def runtime_pair(
    base_repo: Path,
    head_repo: Path,
    nemo_root: Path,
    out_dir: Path,
    checked_path: Path,
    repeat: int,
) -> tuple[dict, dict]:
    """Measure fresh base/head processes in alternating order."""
    _refuse_tracemalloc()
    runs = {"base": [], "head": []}
    for run_number in range(1, repeat + 1):
        for side, repo in (("base", base_repo), ("head", head_repo)):
            run = _runtime_once(
                repo,
                nemo_root,
                checked_path,
                include_prewarm=True,
            )
            run["run"] = run_number
            runs[side].append(run)
            print(f"runtime {run_number}/{repeat} {side}", file=sys.stderr, flush=True)
    reports = (_runtime_report(runs["base"]), _runtime_report(runs["head"]))
    for side, report in zip(("base", "head"), reports, strict=True):
        destination = out_dir / side
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "runtime.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return reports


def _render_unseen_date(locale: str, value: date) -> str:
    if locale == "en_US":
        return f"{value.month:02d}/{value.day:02d}/{value.year:04d}"
    if locale in {"de_DE", "fr_FR"}:
        return f"{value.day:02d}.{value.month:02d}.{value.year:04d}"
    if locale in {"zh_CN", "ja_JP"}:
        return f"{value.year:04d}/{value.month:02d}/{value.day:02d}"
    if locale == "ko_KR":
        return f"{value.year:04d}. {value.month}. {value.day}."
    return f"{value.day:02d}/{value.month:02d}/{value.year:04d}"


def _unseen_inputs(locale: str, pass_index: int, count: int = 2000) -> list[str]:
    locale_index = GATE_LOCALES.index(locale)
    start = date(2000, 1, 1) + timedelta(days=(locale_index * 3 + pass_index) * count)
    return [_render_unseen_date(locale, start + timedelta(days=index)) for index in range(count)]


def _unseen_fraction_inputs(locale: str, pass_index: int, count: int = 2000) -> list[str]:
    locale_index = GATE_LOCALES.index(locale)
    offset = (locale_index * 3 + pass_index) * count
    result = []
    for index in range(count):
        value = offset + index + 1
        denominator = value % 89 + 2
        if index % 3 == 0:
            result.append(f"{value}/{denominator}")
        elif index % 3 == 1:
            result.append(f"{value} {value % denominator + 1}/{denominator}")
        else:
            result.append(f"{value}%")
    return result


def _soak_child() -> dict:
    _refuse_tracemalloc()
    for locale in GATE_LOCALES:
        normalize("123", locale=locale)
    rss_passes = []
    receipts = []
    fraction_receipts = []
    for pass_index in range(3):
        for locale in GATE_LOCALES:
            inputs = _unseen_inputs(locale, pass_index)
            detections = 0
            selected = 0
            for text in inputs:
                rows = [
                    row
                    for row in detect(text, _reading_detectors(locale))
                    if str(row.get("type", "")).startswith("date:")
                    and row["start"] == 0
                    and row["end"] == len(text)
                ]
                detections += bool(rows)
                result = normalize(text, locale=locale, offsets=True)
                selected += any(
                    f"date-rule:{locale}" in unit.provenance.split("+")
                    for unit in getattr(result, "units", ())
                )
            first = date(2000, 1, 1) + timedelta(
                days=(GATE_LOCALES.index(locale) * 3 + pass_index) * len(inputs)
            )
            receipts.append(
                {
                    "locale": locale,
                    "pass": pass_index + 1,
                    "first_iso": first.isoformat(),
                    "last_iso": (first + timedelta(days=len(inputs) - 1)).isoformat(),
                    "count": len(inputs),
                    "input_shape": _date_latency_probes()[locale]["shape"],
                    "successful_date_detections": detections,
                    "selected_generated": selected,
                }
            )
            fraction_inputs = _unseen_fraction_inputs(locale, pass_index)
            fraction_detections = 0
            fraction_selected = 0
            for text in fraction_inputs:
                rows = [
                    row
                    for row in detect(text, _reading_detectors(locale))
                    if (
                        str(row.get("type", "")).startswith("fraction:")
                        or str(row.get("type", "")).startswith("number:fraction")
                        or row.get("type") == "number:percent"
                    )
                    and row["start"] == 0
                    and row["end"] == len(text)
                ]
                fraction_detections += bool(rows)
                result = normalize(text, locale=locale, offsets=True)
                fraction_selected += any(
                    f"fraction-rule:{locale}" in unit.provenance.split("+")
                    for unit in getattr(result, "units", ())
                )
            fraction_receipts.append(
                {
                    "locale": locale,
                    "pass": pass_index + 1,
                    "count": len(fraction_inputs),
                    "kinds": ["fraction", "mixed", "percent"],
                    "successful_fraction_detections": fraction_detections,
                    "selected_generated": fraction_selected,
                }
            )
        gc.collect()
        rss_passes.append(rss_bytes())
    return {
        "rss_pass1_bytes": rss_passes[0],
        "rss_pass2_bytes": rss_passes[1],
        "rss_pass3_bytes": rss_passes[2],
        "pass3_minus_pass1_bytes": rss_passes[2] - rss_passes[0],
        "date_receipts": receipts,
        "fraction_receipts": fraction_receipts,
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


def soak_pair(
    base_repo: Path,
    head_repo: Path,
    out_dir: Path,
    repeat: int,
    unseen: bool,
) -> tuple[dict, dict]:
    """Measure unseen-input RSS in alternating fresh base/head processes."""
    _refuse_tracemalloc()
    if not unseen:
        raise ValueError("soak-pair requires --unseen so the workload contract is explicit")
    runs = {"base": [], "head": []}
    for run_number in range(1, repeat + 1):
        for side, repo in (("base", base_repo), ("head", head_repo)):
            run = _child_command("_soak-child", repo=repo)
            run["run"] = run_number
            runs[side].append(run)
            print(f"soak {run_number}/{repeat} {side}", file=sys.stderr, flush=True)
    reports = []
    for side in ("base", "head"):
        report = {
            "schema": 1,
            "repeat": repeat,
            "unseen_inputs_per_locale": 2000,
            "passes": 3,
            "runs": runs[side],
            "summary": {
                key: _median_range([run[key] for run in runs[side]])
                for key in (
                    "rss_pass1_bytes",
                    "rss_pass2_bytes",
                    "rss_pass3_bytes",
                    "pass3_minus_pass1_bytes",
                )
            },
        }
        destination = out_dir / side
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "soak.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        reports.append(report)
    return reports[0], reports[1]


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
    runtime_pair_parser = subparsers.add_parser("runtime-pair")
    runtime_pair_parser.add_argument("--base-repo", type=Path, required=True)
    runtime_pair_parser.add_argument("--head-repo", type=Path, required=True)
    runtime_pair_parser.add_argument("--nemo-root", type=Path, required=True)
    runtime_pair_parser.add_argument("--out-dir", type=Path, required=True)
    runtime_pair_parser.add_argument("--checked", type=Path, default=CHECKED_PT_PT)
    runtime_pair_parser.add_argument("--repeat", type=int, default=5)
    soak_parser = subparsers.add_parser("soak")
    soak_parser.add_argument("--out-dir", type=Path, required=True)
    soak_parser.add_argument("--repeat", type=int, default=5)
    soak_parser.add_argument("--unseen", action="store_true")
    soak_pair_parser = subparsers.add_parser("soak-pair")
    soak_pair_parser.add_argument("--base-repo", type=Path, required=True)
    soak_pair_parser.add_argument("--head-repo", type=Path, required=True)
    soak_pair_parser.add_argument("--out-dir", type=Path, required=True)
    soak_pair_parser.add_argument("--repeat", type=int, default=5)
    soak_pair_parser.add_argument("--unseen", action="store_true")
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
    compare_parser.add_argument("--require-identical", action="store_true")
    compare_parser.add_argument("--expected-recoveries", type=Path)
    compare_parser.add_argument("--allowed-changes", type=Path)
    compare_parser.add_argument("--require-no-negative-flips", action="store_true")
    compare_parser.add_argument("--require-identity-locale", action="append", default=[])
    compare_parser.add_argument("--pr4-improvement", action="store_true")
    grep_parser = subparsers.add_parser("fixture-grep")
    grep_parser.add_argument("--nemo-root", type=Path, required=True)
    grep_parser.add_argument("values", nargs="+")
    child = subparsers.add_parser("_runtime-child")
    child.add_argument("--kind", choices=("first", "combined", "prewarm"), required=True)
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
    elif args.command == "runtime-pair":
        runtime_pair(
            args.base_repo,
            args.head_repo,
            args.nemo_root,
            args.out_dir,
            args.checked,
            args.repeat,
        )
    elif args.command == "soak":
        soak(args.out_dir, args.repeat, args.unseen)
    elif args.command == "soak-pair":
        soak_pair(args.base_repo, args.head_repo, args.out_dir, args.repeat, args.unseen)
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
            require_identical=args.require_identical,
            expected_recoveries=read_optional(args.expected_recoveries),
            allowed_changes=read_optional(args.allowed_changes),
            require_no_negative_flips=args.require_no_negative_flips,
            require_identity_locales=args.require_identity_locale,
            pr4_improvement=args.pr4_improvement,
        )
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        if payload["resources"]["failures"]:
            return 1
        if not payload["identity"]["passed"]:
            return 1
        if not payload["correctness"]["passed"]:
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
