#!/usr/bin/env python3
"""Measure when prefix-by-prefix frend readings become equal to full-text readings.

This is an offline evaluator, not a streaming API.  It reuses S0's verified reservoir
sample and writes aggregate-only JSON under the processed-data streaming directory.
Corpus text and per-sentence readings never leave the worker process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = Path(__file__).resolve().parent
for _path in (_REPO, _TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from corpus_inputs import (  # noqa: E402
    store_root,
    verification_receipt,
    verified_inputs,
)
from triage_misses import (  # noqa: E402
    LOCALE,
    MAX_SENTENCES_PER_SHARD,
    POOLS,
    SEED,
    SHARDS,
    SOURCE_ID,
    _reservoir_sample,
)

DEFAULT_SENTENCE_TOKEN_CAP = 64
DEFAULT_CHUNK_SIZE = 32
_PROFILE_NAMES = ("default", "google-tn")
_RIGHT_WORD = re.compile(r"^f_tpg\+(\d+)$")
_RIGHT_CHAR = re.compile(r"^w_(?:wb|gc|sc)\+(\d+)$")

# ProcessPoolExecutor may start fresh interpreters on macOS.  Keep worker imports from
# writing bytecode even when a caller forgets to export the variable as well as using
# ``python -B``.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


@dataclass(frozen=True)
class ContextUse:
    """One selected edge's context span and the features its tree path consulted."""

    start: int
    end: int
    problem: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class TokenReading:
    """A token's observable best-path reading and the full-run trace behind it."""

    signature: tuple[tuple[int, int, str, str], ...]
    readers: tuple[str, ...] = ()
    edge_ends: tuple[int, ...] = ()
    context_uses: tuple[ContextUse, ...] = ()
    multi_token: bool = False


def commit_lookahead(signatures: list[object]) -> int:
    """Smallest lookahead whose value equals the last and never changes again.

    ``signatures[k]`` is the token's reading with ``k`` right tokens present.  Looking
    only for the first equality is wrong for flip-flops; the last mismatch determines
    the commit point.
    """
    if not signatures:
        raise ValueError("at least one prefix signature is required")
    final = signatures[-1]
    last_mismatch = -1
    for index, signature in enumerate(signatures[:-1]):
        if signature != final:
            last_mismatch = index
    return last_mismatch + 1


def validate_output_path(path: Path, *, output_root: Path, repo: Path = _REPO) -> Path:
    """Resolve an output path and require the dedicated external streaming tree."""
    resolved = path.expanduser().resolve()
    root = output_root.expanduser().resolve()
    repository = repo.expanduser().resolve()
    if resolved.is_relative_to(repository):
        raise ValueError("output JSON must not be inside the repository")
    if not resolved.is_relative_to(root):
        raise ValueError(f"output JSON must be under {root}")
    if resolved.suffix != ".json":
        raise ValueError("output must be a .json file")
    return resolved


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _text_and_spans(sentence) -> tuple[str, list[tuple[int, int]], list[int]]:
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    ends: list[int] = []
    at = 0
    for index, row in enumerate(sentence):
        if index:
            pieces.append(" ")
            at += 1
        written = row[1]
        start = at
        pieces.append(written)
        at += len(written)
        spans.append((start, at))
        ends.append(at)
    return "".join(pieces), spans, ends


def _profile_value(name: str) -> str | None:
    return None if name == "default" else name


def _prefix_readings(
    text: str,
    spans,
    profiles: tuple[str, ...],
    *,
    trace_context: bool = False,
) -> dict[str, list[TokenReading]]:
    from icukit.detectors import detect
    from reading_profile import reading_detectors

    from frend import resolve_lattice
    from frend.verbalize import verbalize_lattice

    detections = list(detect(text, reading_detectors(LOCALE))) if text.strip() else []
    lattice = resolve_lattice(detections, source_text=text, locale=LOCALE)
    by_edge = {edge.id: edge for edge in lattice.edges}
    results = {}
    for profile_name in profiles:
        verbalized = verbalize_lattice(lattice, profile=_profile_value(profile_name))
        per_token: list[list[tuple[int, int, str, str]]] = [[] for _ in spans]
        readers: list[set[str]] = [set() for _ in spans]
        edge_ends: list[set[int]] = [set() for _ in spans]
        context_uses: list[list[ContextUse]] = [[] for _ in spans]
        multi_token = [False for _ in spans]
        for unit in verbalized.best_path.units:
            edge = by_edge[unit.edge_id]
            context_use = _context_use(text, edge, unit) if trace_context else None
            provenance = unit.best.provenance
            signature = (edge.start, edge.end, unit.best.text, provenance)
            reader = (
                "surface" if edge.detection is None else str(edge.detection.get("type", "unknown"))
            )
            overlapped = [
                token_index
                for token_index, (start, end) in enumerate(spans)
                if edge.start < end and edge.end > start
            ]
            for token_index in overlapped:
                per_token[token_index].append(signature)
                readers[token_index].add(reader)
                edge_ends[token_index].add(edge.end)
                multi_token[token_index] |= len(overlapped) > 1
                if context_use is not None:
                    context_uses[token_index].append(context_use)
        results[profile_name] = [
            TokenReading(
                tuple(signature),
                tuple(sorted(readers[index])),
                tuple(sorted(edge_ends[index])),
                tuple(context_uses[index]),
                multi_token[index],
            )
            for index, signature in enumerate(per_token)
        ]
    return results


@cache
def _tree(problem: str):
    from frend.context import context_model

    model = context_model(LOCALE)
    if model is None:
        return None
    tree = model.tree(problem)
    return tree


def _tree_names(problem: str, values: dict[str, object]) -> tuple[str, ...]:
    """Feature names consulted on this example's path through ``problem``'s tree."""
    tree = _tree(problem)
    if tree is None:
        return ()
    vector = [values.get(name) for name in tree.names]
    prediction = tree._predictor.predict_path(vector, missing="right")
    steps = prediction["trees"][0]["path"]
    return tuple(dict.fromkeys(str(step["name"]) for step in steps))


def _original_first(unit):
    """The first alternative supplied to context ranking, before any applied move."""
    return unit.alternatives[1] if unit.context.applied else unit.alternatives[0]


def _connector_span(edge) -> tuple[int, int]:
    if edge.kind == "passthrough" or edge.detection is None:
        return edge.start, edge.end
    for capture in edge.detection.get("captures", ()):
        if capture.name == "sign":
            return capture.start, capture.end
    return edge.start, edge.end


def _context_use(text: str, edge, unit) -> ContextUse | None:
    """Recreate the selected unit's values and retain only its evaluated tree path."""
    from frend.context import context_model, features, range_example_features

    choice = unit.context
    if choice is None:
        return None
    model = context_model(LOCALE)
    if model is None:
        return None
    first = _original_first(unit)
    start, end = edge.start, edge.end
    problem = choice.problem
    if problem.startswith("connector:"):
        start, end = _connector_span(edge)
        connector = model.connector(text[start:end])
        if connector is None:
            return None
        problem = connector["problem"]
        values = features(
            text,
            start,
            end,
            first=connector["first"],
            first_weight=connector["first_weight"],
            locale=LOCALE,
            frequent=model.frequent,
            curated=model.curated,
        )
    elif problem.startswith("range:"):
        values = range_example_features(
            text,
            start,
            end,
            edge.detection["value"],
            first=first.provenance,
            first_weight=first.weight,
            locale=LOCALE,
            model=model,
        )
    else:
        values = features(
            text,
            start,
            end,
            first=first.provenance,
            first_weight=first.weight,
            locale=LOCALE,
            frequent=model.frequent,
            curated=model.curated,
        )
    return ContextUse(start, end, choice.problem, _tree_names(problem, values))


@cache
def _all_tree_names(problem: str) -> tuple[str, ...]:
    """All decision features, retained only to compare the superseded audit."""
    tree = _tree(problem)
    if tree is None:
        return ()
    used = sorted({int(decision[0]) for decision in tree._predictor.model["decisions"]})
    return tuple(tree.names[index] for index in used)


def _token_at(char_at: int, spans: list[tuple[int, int]]) -> int:
    for index, (start, end) in enumerate(spans):
        if start <= char_at < end:
            return index
    return len(spans) - 1


def _context_requirement(
    text: str,
    edge_start: int,
    edge_end: int,
    names: tuple[str, ...],
    spans: list[tuple[int, int]],
) -> tuple[int, bool, int, int]:
    """Token endpoint needed by the right-side features on one evaluated tree path."""
    from frend.context import EOS, class_windows, neighbor_words

    words = max(
        [int(match.group(1)) for name in names if (match := _RIGHT_WORD.match(name))]
        + ([1] if "c_rextitle" in names else [0])
    )
    chars = max(
        [int(match.group(1)) for name in names if (match := _RIGHT_CHAR.match(name))] or [0]
    )
    required = max(0, _token_at(max(0, edge_end - 1), spans))
    eos = False
    if words:
        target = neighbor_words(text, edge_start, edge_end, locale=LOCALE, right=words)[1]
        if EOS in target:
            required = len(spans) - 1
            eos = True
        else:
            for index, (_start, token_end) in enumerate(spans):
                if token_end < edge_end:
                    continue
                actual = neighbor_words(
                    text[:token_end], edge_start, edge_end, locale=LOCALE, right=words
                )[1]
                if actual == target:
                    required = max(required, index)
                    break
    if chars:
        char_names = tuple(name for name in names if _RIGHT_CHAR.match(name))
        target = class_windows(text, edge_start, edge_end)
        if any(target[name] == EOS for name in char_names):
            required = len(spans) - 1
            eos = True
        else:
            for index, (_start, token_end) in enumerate(spans):
                if token_end < edge_end:
                    continue
                actual = class_windows(text[:token_end], edge_start, edge_end)
                if all(actual[name] == target[name] for name in char_names):
                    required = max(required, index)
                    break
    return required, eos, words, chars


def _static_bound(
    token_index: int,
    reading: TokenReading,
    text: str,
    spans: list[tuple[int, int]],
) -> tuple[int, bool, int, int]:
    required = token_index
    eos = False
    right_words = 0
    right_chars = 0
    for edge_end in reading.edge_ends:
        required = max(required, _token_at(max(0, edge_end - 1), spans))
    for use in reading.context_uses:
        end, uses_eos, words, chars = _context_requirement(
            text, use.start, use.end, use.names, spans
        )
        required = max(required, end)
        eos |= uses_eos
        right_words = max(right_words, words)
        right_chars = max(right_chars, chars)
    return required - token_index, eos, right_words, right_chars


def _old_context_requirement(
    text: str, edge_end: int, names: tuple[str, ...], spans: list[tuple[int, int]]
) -> int:
    """The superseded all-nodes/all-whitespace bound, for a delta-only recheck."""
    from frend.context import _segments

    words = max(
        [int(match.group(1)) for name in names if (match := _RIGHT_WORD.match(name))]
        + ([1] if "c_rextitle" in names else [0])
    )
    chars = max(
        [int(match.group(1)) for name in names if (match := _RIGHT_CHAR.match(name))] or [0]
    )
    required = max(0, _token_at(max(0, edge_end - 1), spans))
    after = text[edge_end:]
    if words:
        segments = _segments(after, LOCALE)
        if len(segments) < words:
            required = len(spans) - 1
        else:
            required = max(required, _token_at(edge_end + segments[words - 1][1] - 1, spans))
    if chars:
        nonspace = [index for index, char in enumerate(after) if not char.isspace()]
        if len(nonspace) < chars:
            required = len(spans) - 1
        else:
            required = max(required, _token_at(edge_end + nonspace[chars - 1], spans))
    return required


def _old_static_bound(
    token_index: int,
    reading: TokenReading,
    text: str,
    spans: list[tuple[int, int]],
) -> int:
    required = token_index
    problems = {use.problem for use in reading.context_uses}
    for edge_end in reading.edge_ends:
        required = max(required, _token_at(max(0, edge_end - 1), spans))
        for problem in problems:
            required = max(
                required,
                _old_context_requirement(text, edge_end, _all_tree_names(problem), spans),
            )
    return required - token_index


def _reader_group(readers: tuple[str, ...]) -> str:
    if not readers:
        return "none"
    groups = sorted({reader.partition(":")[0] for reader in readers})
    return "+".join(groups)


def _reader_name(readers: tuple[str, ...]) -> str:
    return "+".join(readers) if readers else "none"


def _force_groups(reading: TokenReading, eos: bool) -> tuple[str, ...]:
    groups = []
    if any(reader.startswith("range:") for reader in reading.readers):
        groups.append("open/multi-token range")
    if any(reader.startswith("date:") for reader in reading.readers):
        groups.append("date")
    if eos:
        groups.append("eos-dependent tree")
    if reading.multi_token:
        groups.append("multi-token detection")
    return tuple(groups or ["other"])


def _empty_profile() -> dict[str, object]:
    return {
        "tokens": 0,
        "lookahead_tokens": Counter(),
        "lookahead_code_points": Counter(),
        "commit_delay_tokens": Counter(),
        "by_class": defaultdict(Counter),
        "by_class_code_points": defaultdict(Counter),
        "by_kind": defaultdict(Counter),
        "by_kind_code_points": defaultdict(Counter),
        "by_reader": defaultdict(Counter),
        "by_reader_code_points": defaultdict(Counter),
        "long_forces": Counter(),
        "static": Counter(),
        "static_shortfall": Counter(),
        "static_shortfall_by_class": Counter(),
        "static_shortfall_by_reader": Counter(),
        "tree_right_words": Counter(),
        "tree_right_code_points": Counter(),
    }


def _merge_profile(target: dict, source: dict) -> None:
    target["tokens"] += source["tokens"]
    for key in (
        "lookahead_tokens",
        "lookahead_code_points",
        "commit_delay_tokens",
        "long_forces",
        "static",
        "static_shortfall",
        "static_shortfall_by_class",
        "static_shortfall_by_reader",
        "tree_right_words",
        "tree_right_code_points",
    ):
        target[key].update(source[key])
    for key in (
        "by_class",
        "by_class_code_points",
        "by_kind",
        "by_kind_code_points",
        "by_reader",
        "by_reader_code_points",
    ):
        for name, counts in source[key].items():
            target[key][name].update(counts)


def _measure_sentence(sentence, profiles: tuple[str, ...]) -> dict[str, dict]:
    text, spans, token_ends = _text_and_spans(sentence)
    histories = {profile: [[] for _ in sentence] for profile in profiles}
    full_readings = None
    for prefix_index, prefix_end in enumerate(token_ends):
        prefix_spans = spans[: prefix_index + 1]
        readings = _prefix_readings(
            text[:prefix_end],
            prefix_spans,
            profiles,
            trace_context=prefix_index == len(sentence) - 1,
        )
        for profile in profiles:
            for token_index, reading in enumerate(readings[profile]):
                histories[profile][token_index].append(reading.signature)
        if prefix_index == len(sentence) - 1:
            full_readings = readings
    assert full_readings is not None

    measured = {profile: _empty_profile() for profile in profiles}
    for profile in profiles:
        commit_at = -1
        for token_index, row in enumerate(sentence):
            signatures = histories[profile][token_index]
            lookahead = commit_lookahead(signatures)
            commit_index = token_index + lookahead
            code_points = token_ends[commit_index] - token_ends[token_index]
            commit_at = max(commit_at, commit_index)
            delay = commit_at - token_index
            full = full_readings[profile][token_index]
            static, eos, right_words, right_chars = _static_bound(token_index, full, text, spans)
            result = measured[profile]
            result["tokens"] += 1
            result["lookahead_tokens"][lookahead] += 1
            result["lookahead_code_points"][code_points] += 1
            result["commit_delay_tokens"][delay] += 1
            result["by_class"][row[0]][lookahead] += 1
            result["by_class_code_points"][row[0]][code_points] += 1
            kind = _reader_group(full.readers)
            result["by_kind"][kind][lookahead] += 1
            result["by_kind_code_points"][kind][code_points] += 1
            reader_name = _reader_name(full.readers)
            result["by_reader"][reader_name][lookahead] += 1
            result["by_reader_code_points"][reader_name][code_points] += 1
            result["tree_right_words"][right_words] += 1
            result["tree_right_code_points"][right_chars] += 1
            result["static"]["eos_dependent"] += eos
            result["static"]["measured_exceeds_observed_static"] += lookahead > static
            result["static"]["equal"] += lookahead == static
            result["static"]["observed_static_exceeds_measured"] += static > lookahead
            if lookahead > static:
                result["static_shortfall"][lookahead - static] += 1
                result["static_shortfall_by_class"][row[0]] += 1
                result["static_shortfall_by_reader"][_reader_name(full.readers)] += 1
            if lookahead:
                for force in _force_groups(full, eos):
                    result["long_forces"][force] += 1
    return measured


def _audit_sentence(sentence, profiles: tuple[str, ...]) -> dict[str, dict]:
    """Audit selected full-sentence paths without rerunning prefix lookahead."""
    text, spans, _token_ends = _text_and_spans(sentence)
    readings = _prefix_readings(text, spans, profiles, trace_context=True)
    measured = {profile: _empty_profile() for profile in profiles}
    for profile in profiles:
        for token_index, full in enumerate(readings[profile]):
            _static, eos, right_words, right_chars = _static_bound(token_index, full, text, spans)
            result = measured[profile]
            result["tokens"] += 1
            result["tree_right_words"][right_words] += 1
            result["tree_right_code_points"][right_chars] += 1
            result["static"]["eos_dependent"] += eos
    return measured


def _audit_delta_sentence(sentence, profiles: tuple[str, ...]) -> dict[str, dict]:
    """Recheck L only where the corrected static bound differs from the old one."""
    text, spans, token_ends = _text_and_spans(sentence)
    full_readings = _prefix_readings(text, spans, profiles, trace_context=True)
    candidates = {}
    earliest = len(sentence)
    measured = {profile: _empty_profile() for profile in profiles}
    for profile in profiles:
        for token_index, full in enumerate(full_readings[profile]):
            new = _static_bound(token_index, full, text, spans)[0]
            old = _old_static_bound(token_index, full, text, spans)
            measured[profile]["static"]["tighter_bound_tokens"] += new < old
            measured[profile]["static"]["looser_bound_tokens"] += new > old
            if new != old:
                candidates[profile, token_index] = [new, old, -1, full]
                earliest = min(earliest, token_index + min(new, old))

    for prefix_index in range(earliest, len(sentence) - 1):
        readings = _prefix_readings(
            text[: token_ends[prefix_index]], spans[: prefix_index + 1], profiles
        )
        for (profile, token_index), candidate in candidates.items():
            new, old, _last_mismatch, full = candidate
            if prefix_index >= token_index + min(new, old):
                if readings[profile][token_index].signature != full.signature:
                    candidate[2] = prefix_index - token_index

    for (profile, token_index), (new, old, last_mismatch, full) in candidates.items():
        lookahead = last_mismatch + 1
        if new < lookahead <= old:
            row = sentence[token_index]
            result = measured[profile]
            result["static"]["new_shortfalls_from_tighter_bound"] += 1
            result["static_shortfall"][lookahead - new] += 1
            result["static_shortfall_by_class"][row[0]] += 1
            result["static_shortfall_by_reader"][_reader_name(full.readers)] += 1
        if old < lookahead <= new:
            measured[profile]["static"]["resolved_shortfalls_from_looser_bound"] += 1
        if lookahead > old:
            measured[profile]["static"]["already_short_under_old_bound"] += 1
    return measured


def _measure_chunk(job) -> dict[str, dict]:
    sentences, profiles, mode = job
    total = {profile: _empty_profile() for profile in profiles}
    for sentence in sentences:
        if mode == "static":
            measured = _audit_sentence(sentence, profiles)
        elif mode == "delta":
            measured = _audit_delta_sentence(sentence, profiles)
        else:
            measured = _measure_sentence(sentence, profiles)
        for profile in profiles:
            _merge_profile(total[profile], measured[profile])
    return total


def _percentile(histogram: Counter, fraction: float) -> int | None:
    total = sum(histogram.values())
    if not total:
        return None
    rank = max(1, int((fraction * total) + 0.999999999))
    cumulative = 0
    for value, count in sorted(histogram.items()):
        cumulative += count
        if cumulative >= rank:
            return int(value)
    raise AssertionError("unreachable percentile")


def _summary(histogram: Counter) -> dict[str, int | float | None]:
    count = sum(histogram.values())
    return {
        "n": count,
        "l0_share": histogram[0] / count if count else None,
        "p50": _percentile(histogram, 0.50),
        "p90": _percentile(histogram, 0.90),
        "p99": _percentile(histogram, 0.99),
        "max": max(histogram, default=None),
    }


def _render_profile(profile: dict) -> dict[str, object]:
    return {
        "tokens": profile["tokens"],
        "lookahead_tokens": _summary(profile["lookahead_tokens"]),
        "lookahead_code_points": _summary(profile["lookahead_code_points"]),
        "commit_delay_tokens": _summary(profile["commit_delay_tokens"]),
        "by_class": {
            name: _summary(counts) for name, counts in sorted(profile["by_class"].items())
        },
        "by_class_code_points": {
            name: _summary(counts)
            for name, counts in sorted(profile["by_class_code_points"].items())
        },
        "by_kind": {name: _summary(counts) for name, counts in sorted(profile["by_kind"].items())},
        "by_kind_code_points": {
            name: _summary(counts)
            for name, counts in sorted(profile["by_kind_code_points"].items())
        },
        "by_reader": {
            name: _summary(counts) for name, counts in sorted(profile["by_reader"].items())
        },
        "by_reader_code_points": {
            name: _summary(counts)
            for name, counts in sorted(profile["by_reader_code_points"].items())
        },
        "long_forces": dict(profile["long_forces"].most_common()),
        "static_bound_comparison": dict(profile["static"]),
        "static_shortfall_tokens": _summary(profile["static_shortfall"]),
        "static_shortfall_by_class": dict(profile["static_shortfall_by_class"].most_common()),
        "static_shortfall_by_reader": dict(profile["static_shortfall_by_reader"].most_common()),
        "tree_right_word_offsets": dict(sorted(profile["tree_right_words"].items())),
        "tree_right_code_point_offsets": dict(sorted(profile["tree_right_code_points"].items())),
    }


def _detector_extent_inventory() -> dict[str, object]:
    from reading_profile import reading_detectors

    declared = {}
    detector_types = []
    for detector in reading_detectors(LOCALE):
        name = f"{type(detector).__module__}.{type(detector).__name__}"
        detector_types.append(name)
        for attribute in ("max_extent", "max_width"):
            value = getattr(detector, attribute, None)
            if isinstance(value, int):
                declared[name] = {"attribute": attribute, "code_points": value}
                break
    return {
        "readers": len(detector_types),
        "reader_types": sorted(set(detector_types)),
        "declared_extents": declared,
        "note": (
            "The installed readers expose no common max-extent contract; per-token "
            "static bounds therefore use the selected full-path span plus actual tree "
            "features and are observed lower bounds, not safe detector bounds."
        ),
    }


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sentences-per-shard", type=int, default=MAX_SENTENCES_PER_SHARD)
    parser.add_argument("--sentence-token-cap", type=int, default=DEFAULT_SENTENCE_TOKEN_CAP)
    parser.add_argument(
        "--only-over-sentence-token-cap",
        action="store_true",
        help="measure only sampled sentences longer than the cap",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="run one full-sentence pass for tree-path static statistics, without L",
    )
    parser.add_argument(
        "--static-delta",
        action="store_true",
        help="recheck prefixes only for tokens whose corrected static bound changed",
    )
    parser.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--profile",
        action="append",
        choices=_PROFILE_NAMES,
        dest="profiles",
        help="profile(s) to measure; default is both default and google-tn",
    )
    args = parser.parse_args(argv)
    if args.processed_root is None:
        processed_root = os.environ.get("FREND_PROCESSED")
        if processed_root is None:
            parser.error("--processed-root is required when FREND_PROCESSED is not set")
        args.processed_root = Path(processed_root)
    try:
        output = validate_output_path(
            args.output, output_root=args.processed_root / "frend" / "streaming"
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.sentences_per_shard <= MAX_SENTENCES_PER_SHARD:
        parser.error(f"--sentences-per-shard must be in 1..{MAX_SENTENCES_PER_SHARD}")
    if args.sentence_token_cap < 1 or args.workers < 1 or args.chunk_size < 1:
        parser.error("sentence cap, workers, and chunk size must be positive")
    if args.static_only and args.static_delta:
        parser.error("--static-only and --static-delta are mutually exclusive")
    profiles = tuple(dict.fromkeys(args.profiles or _PROFILE_NAMES))
    mode = "static" if args.static_only else "delta" if args.static_delta else "lookahead"

    # Validate the optional table before the expensive sample scan.
    from evaluate_google_tn import _validate_evaluation_profile

    for profile in profiles:
        _validate_evaluation_profile(_profile_value(profile), LOCALE)

    if args.corpus_dir is None:
        try:
            args.corpus_dir = store_root(SOURCE_ID)
        except ValueError as exc:
            parser.error(str(exc))
    corpus_dir = args.corpus_dir.resolve(strict=True)
    verified = verified_inputs(
        SOURCE_ID,
        [corpus_dir / name for name in SHARDS],
        locale=LOCALE,
        pools=POOLS,
        root=corpus_dir,
    )
    corpus_receipt = verification_receipt(verified, locale=LOCALE, pools=POOLS)
    selected = []
    selected_indexes = {}
    inventory = {}
    skipped = Counter()
    for offset, corpus_input in enumerate(verified):
        sample, available = _reservoir_sample(
            corpus_input, args.sentences_per_shard, args.seed + offset
        )
        selected_indexes[corpus_input.relative_path] = [index for index, _rows in sample]
        inventory[corpus_input.relative_path] = {
            "available_sentences": available,
            "sampled_sentences": len(sample),
        }
        for _index, sentence in sample:
            over_cap = len(sentence) > args.sentence_token_cap
            if args.only_over_sentence_token_cap:
                if over_cap:
                    selected.append(sentence)
                else:
                    skipped["at_or_below_cap_sentences"] += 1
                    skipped["at_or_below_cap_tokens"] += len(sentence)
                continue
            if over_cap:
                skipped["sentences"] += 1
                skipped["tokens"] += len(sentence)
                continue
            selected.append(sentence)

    fingerprint_payload = {
        "corpus_fingerprint": corpus_receipt["fingerprint"],
        "seed": args.seed,
        "sentence_cap_per_shard": args.sentences_per_shard,
        "selected_sentence_indexes": selected_indexes,
    }
    sample_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    totals = {profile: _empty_profile() for profile in profiles}
    jobs = [(chunk, profiles, mode) for chunk in _chunks(selected, args.chunk_size)]
    if args.workers == 1:
        results = map(_measure_chunk, jobs)
        for result in results:
            for profile in profiles:
                _merge_profile(totals[profile], result[profile])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(_measure_chunk, jobs, chunksize=1):
                for profile in profiles:
                    _merge_profile(totals[profile], result[profile])

    payload = {
        "schema_version": 1,
        "subject": {
            "head": _head(),
            "versions": {
                "frend": _version("frend"),
                "icukit": _version("icukit"),
                "tiergraph": _version("tiergraph"),
            },
        },
        "sample": {
            "seed": args.seed,
            "sentence_cap_per_shard": args.sentences_per_shard,
            "shards": list(SHARDS),
            "inventory": inventory,
            "fingerprint": sample_fingerprint,
            "measured_sentences": len(selected),
            "skipped_over_sentence_token_cap": dict(skipped),
            "selection": (
                "over sentence token cap only"
                if args.only_over_sentence_token_cap
                else "at or below sentence token cap"
            ),
        },
        "method": {
            "sentence_token_cap": args.sentence_token_cap,
            "mode": {
                "static": "static full-sentence paths only",
                "delta": "prefix recheck only where the corrected static bound changed",
                "lookahead": "exact prefixes",
            }[mode],
            "prefixes": (
                "not rerun"
                if args.static_only
                else (
                    "only corrected-bound-to-sentence-end whole-token prefixes"
                    if args.static_delta
                    else "every whole-token boundary through the sampled sentence end"
                )
            ),
            "token_reading": (
                "best-path units overlapping the corpus token, including selected span, "
                "spoken text, and provenance"
            ),
            "commit_rule": "equal to full reading at this and every longer prefix",
            "code_points": "from the token end through the committing prefix end",
            "privacy": "aggregate only; no corpus text or per-sentence readings written",
        },
        "corpus_verification": corpus_receipt,
        "detector_extent_inventory": _detector_extent_inventory(),
        "profiles": {profile: _render_profile(totals[profile]) for profile in profiles},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sample": payload["sample"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
