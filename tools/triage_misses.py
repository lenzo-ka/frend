"""Classify first-choice misses on Google TN runtime-eval shards.

Only aggregate counts leave this process. Corpus text is used transiently by workers
and is never written to the report or receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import islice, product
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = Path(__file__).resolve().parent
for _path in (_REPO, _TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import google_tn_rows  # noqa: E402
from bootstrap_intervals import (  # noqa: E402
    CI_LEVEL,
    DEFAULT_INTERVAL_SEED,
    MIN_DEFINED_FRACTION,
    percentile_ratio_intervals,
)
from corpus_inputs import (  # noqa: E402
    VerifiedInput,
    open_verified,
    store_root,
    verification_receipt,
    verified_inputs,
)
from evaluate_google_tn import (  # noqa: E402
    _detectors,
    _in_context,
    _joined,
    _validate_evaluation_profile,
)
from seen_strata import (  # noqa: E402
    SENTENCE_RULE,
    SENTENCE_STRATA,
    STRATA,
    load_vocabulary,
    sentence_strata,
    vocabulary_receipt,
)

SOURCE_ID = "google/tn-en_with_types"
LOCALE = "en_US"
POOLS = ("runtime_eval",)
SHARDS = tuple(f"output-{number:05d}-of-00100" for number in range(90, 95))
SEED = 20260930
MAX_SENTENCES_PER_SHARD = 20_000
ANY_CAP = 64

MISS_CLASSES = ("D", "S", "R", "V", "P", "O", "E")
FAMILIES = (
    "money",
    "fraction",
    "measure",
    "date",
    "time",
    "range",
    "telephone/ID",
    "cardinal/digit",
    "letters",
    "other",
)
FIX_KINDS = ("recognition change", "template-fixable", "new entry", "unknown")
_TEMPLATE_WORDS = frozenset({"and", "minus", "of", "over", "point", "sil", "the", "to"})


@dataclass(frozen=True)
class MissEvidence:
    """The decision surface used by :func:`classify_miss`."""

    corpus_class: str
    written: str
    expected: str
    normalized_surface: str
    readings: tuple[str, ...]
    detection_spans: tuple[tuple[int, int], ...]
    detection_types: tuple[str, ...]
    exact_detection_types: tuple[str, ...]
    verbalizer_exception: bool = False
    leaf_texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Classification:
    miss_class: str
    family: str | None = None
    fix_kind: str | None = None


def _plain_mismatch_class(evidence: MissEvidence) -> str | None:
    """Split corpus-labelled PLAIN rewrites from other surface mismatches.

    A one-for-one sequence of alphabetic words is an orthographic conversion (P),
    including the corpus labeller's UK-to-US respellings.  Expansions, spellings and
    silence are other PLAIN/surface mismatches (O).
    """
    if evidence.corpus_class != "PLAIN" or evidence.expected == evidence.normalized_surface:
        return None
    surface_words = evidence.normalized_surface.split()
    expected_words = evidence.expected.split()
    if (
        surface_words
        and len(surface_words) == len(expected_words)
        and all(word.isalpha() for word in (*surface_words, *expected_words))
    ):
        return "P"
    return "O"


def _family(evidence: MissEvidence) -> str:
    types = evidence.detection_types
    if any(type_.startswith("range:") for type_ in types):
        return "range"
    by_class = {
        "MONEY": "money",
        "FRACTION": "fraction",
        "MEASURE": "measure",
        "DATE": "date",
        "TIME": "time",
        "TELEPHONE": "telephone/ID",
        "ADDRESS": "telephone/ID",
        "CARDINAL": "cardinal/digit",
        "DIGIT": "cardinal/digit",
        "DECIMAL": "cardinal/digit",
        "ORDINAL": "cardinal/digit",
        "LETTERS": "letters",
    }
    family = by_class.get(evidence.corpus_class)
    if family is not None:
        return family
    if any(type_.startswith(("letter", "letters", "abbreviation", "acronym")) for type_ in types):
        return "letters"
    return "other"


def _right_recognition_family(family: str, exact_types: tuple[str, ...]) -> bool:
    prefixes = {
        "money": ("currency", "money"),
        "fraction": ("fraction",),
        "measure": ("measure", "percent", "unit"),
        "date": ("date",),
        "time": ("duration", "time"),
        "range": ("range",),
        "telephone/ID": ("id", "phone", "telephone"),
        "cardinal/digit": ("cardinal", "digit", "number", "ordinal"),
        "letters": ("abbreviation", "acronym", "letter", "letters"),
        "other": (),
    }[family]
    return family == "other" or any(type_.startswith(prefixes) for type_ in exact_types)


def _can_recombine(expected: str, leaf_texts: tuple[str, ...]) -> bool:
    target_words = tuple(word for word in expected.split() if word not in _TEMPLATE_WORDS)
    if not target_words:
        return False
    reachable = {0}
    for text in leaf_texts:
        words = tuple(word for word in text.split() if word not in _TEMPLATE_WORDS)
        if not words:
            continue
        reachable |= {
            at + len(words) for at in reachable if target_words[at : at + len(words)] == words
        }
    return len(target_words) in reachable


def _fix_kind(evidence: MissEvidence, family: str) -> str:
    if not _right_recognition_family(family, evidence.exact_detection_types):
        return "recognition change"
    if _can_recombine(evidence.expected, evidence.leaf_texts):
        return "template-fixable"
    if family in {"money", "fraction", "measure", "letters", "other"}:
        return "new entry"
    return "unknown"


def classify_miss(evidence: MissEvidence) -> Classification:
    """Classify one known first-choice miss from its bounded offered readings."""
    if evidence.verbalizer_exception:
        return Classification("E")
    if evidence.expected in evidence.readings:
        return Classification("R")
    plain_class = _plain_mismatch_class(evidence)
    if plain_class is not None:
        return Classification(plain_class)
    if not evidence.detection_spans:
        return Classification("D")
    full_span = (0, len(evidence.written))
    if full_span not in evidence.detection_spans:
        return Classification("S")
    family = _family(evidence)
    return Classification("V", family, _fix_kind(evidence, family))


def _reservoir_sample(
    corpus_input: VerifiedInput, count: int, seed: int
) -> tuple[list[tuple[int, tuple[tuple[str, str, str], ...]]], int]:
    """Uniformly sample at most ``count`` complete sentences from one verified shard."""
    rng = random.Random(seed)
    sample: list[tuple[int, tuple[tuple[str, str, str], ...]]] = []
    current: list[tuple[str, str, str]] = []
    sentence_index = 0

    def consider(rows: list[tuple[str, str, str]]) -> None:
        nonlocal sentence_index
        item = (sentence_index, tuple(rows))
        if sentence_index < count:
            sample.append(item)
        else:
            replacement = rng.randrange(sentence_index + 1)
            if replacement < count:
                sample[replacement] = item
        sentence_index += 1

    with open_verified(corpus_input) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "<eos>":
                if current:
                    consider(current)
                current = []
            elif len(parts) >= 3:
                current.append((parts[0], parts[1], parts[2]))
    if current:
        consider(current)
    sample.sort(key=lambda item: item[0])
    return sample, sentence_index


def _joined_reading(path, *, alternatives: bool) -> list[str]:
    def passthrough(alternative) -> bool:
        return alternative.provenance == "surface:passthrough"

    if not alternatives:
        return [_joined((unit.best.text, passthrough(unit.best)) for unit in path.units)]
    options = [
        [(alternative.text, passthrough(alternative)) for alternative in unit.alternatives]
        for unit in path.units
    ]
    return [_joined(combination) for combination in islice(product(*options), ANY_CAP)]


def _bounded_readings(path) -> tuple[tuple[str, ...], bool]:
    """The evaluator's first 64 combinations and whether later ones were truncated."""

    def passthrough(alternative) -> bool:
        return alternative.provenance == "surface:passthrough"

    options = [
        [(alternative.text, passthrough(alternative)) for alternative in unit.alternatives]
        for unit in path.units
    ]
    combinations = 1
    for alternatives in options:
        combinations *= len(alternatives)
    readings = tuple(_joined(items) for items in islice(product(*options), ANY_CAP))
    return readings, combinations > ANY_CAP


def _leaf_texts(detections, written: str) -> tuple[str, ...]:
    """All leaf alternatives the unselected choice carrier can already render."""
    from frend import compose_choices, resolve_choices
    from frend.spoken_priors import normalize_spoken

    graph = compose_choices(
        resolve_choices(detections, source_text=written, locale=LOCALE), locale=LOCALE
    )
    return tuple(
        normalize_spoken(alternative.text)
        for unit in graph.units
        if unit.provenance is not None
        for alternative in unit.alternatives
    )


def _score_token(
    row,
    before: str,
    after: str,
    profile: str | None = None,
    classify_misses: bool = True,
    strip_embedded_sil: bool = False,
    case_variant_lookup: bool = False,
) -> tuple[bool, bool, bool, Classification | None]:
    from icukit.detectors import detect

    from frend import resolve_lattice
    from frend.spoken_priors import normalize_spoken
    from frend.verbalize import verbalize_lattice

    corpus_class, written, spoken = row
    target = normalize_spoken(
        google_tn_rows.expected(
            corpus_class, written, spoken, strip_embedded_sil=strip_embedded_sil
        )
    )
    surface = normalize_spoken(written)
    detections = []
    readings: tuple[str, ...] = ()
    capped = False
    verbalizer_exception = False
    try:
        detections = list(detect(written, _detectors(LOCALE))) if written.strip() else []
        verbalized = verbalize_lattice(
            resolve_lattice(detections, source_text=written, locale=LOCALE),
            context=_in_context(written, before, after),
            profile=profile,
            case_variant_lookup=case_variant_lookup,
        )
        first = normalize_spoken(_joined_reading(verbalized.best_path, alternatives=False)[0])
        if first == target:
            return True, True, False, None
        normalized = []
        for path in verbalized.paths:
            path_readings, path_capped = _bounded_readings(path)
            normalized.extend(normalize_spoken(candidate) for candidate in path_readings)
            capped |= path_capped
        readings = tuple(normalized)
    except FileNotFoundError:
        raise
    except Exception:  # noqa: BLE001 - evaluator semantics count a crash as a miss
        verbalizer_exception = True

    if not classify_misses:
        return False, target in readings, capped, None

    spans = tuple(
        (int(detection.get("start", 0)), int(detection.get("end", 0))) for detection in detections
    )
    types = tuple(str(detection.get("type", "")) for detection in detections)
    exact_types = tuple(
        type_ for type_, span in zip(types, spans, strict=True) if span == (0, len(written))
    )
    evidence = MissEvidence(
        corpus_class=corpus_class,
        written=written,
        expected=target,
        normalized_surface=surface,
        readings=readings,
        detection_spans=spans,
        detection_types=types,
        exact_detection_types=exact_types,
        verbalizer_exception=verbalizer_exception,
    )
    if classify_miss(evidence).miss_class == "V":
        try:
            evidence = replace(evidence, leaf_texts=_leaf_texts(detections, written))
        except FileNotFoundError:
            raise
        except Exception:  # noqa: BLE001 - diagnostic resolution failures are E misses
            evidence = replace(evidence, verbalizer_exception=True)
    classification = classify_miss(evidence)
    return False, target in readings, capped, classification


def _score_sentence(item) -> dict[str, object]:
    shard, sentence_index, sentence, token_strata, profile, *options = item
    classify_misses = options[0] if options else True
    strip_embedded_sil = options[1] if len(options) > 1 else False
    case_variant_lookup = options[2] if len(options) > 2 else False
    first_count = 0
    any_count = 0
    capped_count = 0
    miss_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    fix_counts: Counter[str] = Counter()
    cross_counts: Counter[str] = Counter()
    strata_counts = {
        name: {
            "tokens": 0,
            "first": 0,
            "any": 0,
            "capped": 0,
            "miss_counts": Counter(),
            "family_counts": Counter(),
            "fix_counts": Counter(),
            "cross_counts": Counter(),
        }
        for name in STRATA
    }
    for index, row in enumerate(sentence):
        before = " ".join(other[1] for other in sentence[:index])
        after = " ".join(other[1] for other in sentence[index + 1 :])
        target_options = {"strip_embedded_sil": True} if strip_embedded_sil else {}
        if classify_misses:
            first, any_, capped, classification = _score_token(
                row,
                before,
                after,
                profile,
                case_variant_lookup=case_variant_lookup,
                **target_options,
            )
        else:
            first, any_, capped, classification = _score_token(
                row,
                before,
                after,
                profile,
                False,
                case_variant_lookup=case_variant_lookup,
                **target_options,
            )
        first_count += first
        any_count += any_
        capped_count += capped
        stratum_counts = None if token_strata is None else strata_counts[token_strata[index]]
        if stratum_counts is not None:
            stratum_counts["tokens"] += 1
            stratum_counts["first"] += first
            stratum_counts["any"] += any_
            stratum_counts["capped"] += capped
        if classification is None:
            continue
        miss_counts[classification.miss_class] += 1
        if stratum_counts is not None:
            stratum_counts["miss_counts"][classification.miss_class] += 1
        if classification.miss_class == "V":
            assert classification.family is not None and classification.fix_kind is not None
            family_counts[classification.family] += 1
            fix_counts[classification.fix_kind] += 1
            cross_counts[f"{classification.family}\t{classification.fix_kind}"] += 1
            if stratum_counts is not None:
                stratum_counts["family_counts"][classification.family] += 1
                stratum_counts["fix_counts"][classification.fix_kind] += 1
                stratum_counts["cross_counts"][
                    f"{classification.family}\t{classification.fix_kind}"
                ] += 1
    result = {
        "id": f"{shard}:{sentence_index}",
        "tokens": len(sentence),
        "first": first_count,
        "any": any_count,
        "capped": capped_count,
        "capped_sentence": capped_count > 0,
        "sentence_first": first_count == len(sentence),
        "miss_counts": dict(miss_counts),
        "miss_sentences": sorted(miss_counts),
        "family_counts": dict(family_counts),
        "family_sentences": sorted(family_counts),
        "fix_counts": dict(fix_counts),
        "fix_sentences": sorted(fix_counts),
        "cross_counts": dict(cross_counts),
        "cross_sentences": sorted(cross_counts),
    }
    if token_strata is not None:
        result["strata"] = {}
        for name in STRATA:
            counts = strata_counts[name]
            result["strata"][name] = {
                **counts,
                "token_sentence": counts["tokens"] > 0,
                "miss_sentence": bool(counts["miss_counts"]),
                "capped_sentence": counts["capped"] > 0,
                "miss_sentences": sorted(counts["miss_counts"]),
                "family_sentences": sorted(counts["family_counts"]),
                "fix_sentences": sorted(counts["fix_counts"]),
                "cross_sentences": sorted(counts["cross_counts"]),
            }
        result["sentence_strata"] = sentence_strata(sentence, token_strata)
    return result


def _entry(tokens: int, sentences: int, sampled_tokens: int) -> dict[str, int | float]:
    return {
        "tokens": tokens,
        "share_sampled_tokens_pp": 100.0 * tokens / sampled_tokens if sampled_tokens else 0.0,
        "sentences": sentences,
    }


def _stratum_total() -> dict[str, object]:
    return {
        "tokens": 0,
        "token_sentences": 0,
        "first": 0,
        "any": 0,
        "capped": 0,
        "capped_sentences": 0,
        "miss_sentences_total": 0,
        "miss_tokens": Counter(),
        "miss_sentences": Counter(),
        "family_tokens": Counter(),
        "family_sentences": Counter(),
        "fix_tokens": Counter(),
        "fix_sentences": Counter(),
        "cross_tokens": Counter(),
        "cross_sentences": Counter(),
    }


def _add_stratum_result(total: dict[str, object], source: dict[str, object]) -> None:
    total["tokens"] += int(source["tokens"])
    total["token_sentences"] += bool(source["token_sentence"])
    total["first"] += int(source["first"])
    total["any"] += int(source["any"])
    total["capped"] += int(source["capped"])
    total["capped_sentences"] += bool(source["capped_sentence"])
    total["miss_sentences_total"] += bool(source["miss_sentence"])
    for target_name, source_name in (
        ("miss_tokens", "miss_counts"),
        ("miss_sentences", "miss_sentences"),
        ("family_tokens", "family_counts"),
        ("family_sentences", "family_sentences"),
        ("fix_tokens", "fix_counts"),
        ("fix_sentences", "fix_sentences"),
        ("cross_tokens", "cross_counts"),
        ("cross_sentences", "cross_sentences"),
    ):
        total[target_name].update(source[source_name])


def _stratum_report(
    total: dict[str, object], sampled_tokens: int, classify_misses: bool = True
) -> dict[str, object]:
    tokens = total["tokens"]
    misses = tokens - total["first"]
    if classify_misses and sum(total["miss_tokens"].values()) != misses:
        raise AssertionError("stratified miss classes do not partition first-choice misses")
    report = {
        "sample": {"tokens": tokens, "sentences": total["token_sentences"]},
        "accuracy": {
            "first_choice_tokens": total["first"],
            "first_choice": total["first"] / tokens if tokens else 0.0,
            "any_reading_tokens": total["any"],
            "any_reading": total["any"] / tokens if tokens else 0.0,
        },
        "misses": _entry(misses, total["miss_sentences_total"], sampled_tokens),
        "capped": _entry(total["capped"], total["capped_sentences"], sampled_tokens),
        "by_class": {
            name: _entry(total["miss_tokens"][name], total["miss_sentences"][name], sampled_tokens)
            for name in MISS_CLASSES
        },
        "v_by_family": {
            name: _entry(
                total["family_tokens"][name], total["family_sentences"][name], sampled_tokens
            )
            for name in FAMILIES
        },
        "v_by_fix_kind": {
            name: _entry(total["fix_tokens"][name], total["fix_sentences"][name], sampled_tokens)
            for name in FIX_KINDS
        },
        "v_family_by_fix_kind": {
            family: {
                fix: _entry(
                    total["cross_tokens"][f"{family}\t{fix}"],
                    total["cross_sentences"][f"{family}\t{fix}"],
                    sampled_tokens,
                )
                for fix in FIX_KINDS
            }
            for family in FAMILIES
        },
    }
    if not classify_misses:
        for key in ("misses", "by_class", "v_by_family", "v_by_fix_kind", "v_family_by_fix_kind"):
            report.pop(key)
    return report


def _aggregate(
    results,
    sampled_sentences: int,
    intervals: int = 0,
    interval_seed: int = DEFAULT_INTERVAL_SEED,
    record_sink: list[dict] | None = None,
    classify_misses: bool = True,
) -> dict[str, object]:
    interval_metrics = {} if intervals else None
    tokens = first = any_ = capped = sentence_first = miss_sentence_total = capped_sentences = 0
    miss_tokens: Counter[str] = Counter()
    miss_sentences: Counter[str] = Counter()
    family_tokens: Counter[str] = Counter()
    family_sentences: Counter[str] = Counter()
    fix_tokens: Counter[str] = Counter()
    fix_sentences: Counter[str] = Counter()
    cross_tokens: Counter[str] = Counter()
    cross_sentences: Counter[str] = Counter()
    strata_totals = {name: _stratum_total() for name in STRATA}
    sentence_strata = {name: Counter() for name in SENTENCE_STRATA}
    has_strata = False
    for result in results:
        if interval_metrics is not None:
            _append_interval_metrics(interval_metrics, result, classify_misses)
        if record_sink is not None:
            record_sink.append(_per_sentence_record(result))
        tokens += int(result["tokens"])
        first += int(result["first"])
        any_ += int(result["any"])
        capped += int(result["capped"])
        capped_sentences += bool(result["capped_sentence"])
        sentence_first += bool(result["sentence_first"])
        miss_sentence_total += (
            bool(result["miss_counts"]) if classify_misses else not result["sentence_first"]
        )
        miss_tokens.update(result["miss_counts"])
        miss_sentences.update(result["miss_sentences"])
        family_tokens.update(result["family_counts"])
        family_sentences.update(result["family_sentences"])
        fix_tokens.update(result["fix_counts"])
        fix_sentences.update(result["fix_sentences"])
        cross_tokens.update(result["cross_counts"])
        cross_sentences.update(result["cross_sentences"])
        if "strata" in result:
            has_strata = True
            for sentence_name in result["sentence_strata"]:
                sentence_strata[sentence_name]["sentences"] += 1
                sentence_strata[sentence_name]["first"] += bool(result["sentence_first"])
            for name in STRATA:
                _add_stratum_result(strata_totals[name], result["strata"][name])

    misses = tokens - first
    if classify_misses and sum(miss_tokens.values()) != misses:
        raise AssertionError("miss classes do not partition first-choice misses")
    if classify_misses and sum(family_tokens.values()) != miss_tokens["V"]:
        raise AssertionError("V families do not partition V")
    if classify_misses and sum(fix_tokens.values()) != miss_tokens["V"]:
        raise AssertionError("V fix kinds do not partition V")

    report = {
        "sample": {"tokens": tokens, "sentences": sampled_sentences},
        "accuracy": {
            "first_choice_tokens": first,
            "first_choice": first / tokens if tokens else 0.0,
            "any_reading_tokens": any_,
            "any_reading": any_ / tokens if tokens else 0.0,
            "first_choice_sentences": sentence_first,
            "first_choice_sentence_accuracy": (
                sentence_first / sampled_sentences if sampled_sentences else 0.0
            ),
        },
        "misses": _entry(misses, miss_sentence_total, tokens),
        "capped": _entry(capped, capped_sentences, tokens),
        "by_class": {
            name: _entry(miss_tokens[name], miss_sentences[name], tokens) for name in MISS_CLASSES
        },
        "v_by_family": {
            name: _entry(family_tokens[name], family_sentences[name], tokens) for name in FAMILIES
        },
        "v_by_fix_kind": {
            name: _entry(fix_tokens[name], fix_sentences[name], tokens) for name in FIX_KINDS
        },
        "v_family_by_fix_kind": {
            family: {
                fix: _entry(
                    cross_tokens[f"{family}\t{fix}"],
                    cross_sentences[f"{family}\t{fix}"],
                    tokens,
                )
                for fix in FIX_KINDS
            }
            for family in FAMILIES
        },
    }
    if not classify_misses:
        for key in ("by_class", "v_by_family", "v_by_fix_kind", "v_family_by_fix_kind"):
            report.pop(key)
    if has_strata:
        report["strata"] = {
            "sentence_rule": SENTENCE_RULE,
            "tokens": {
                name: _stratum_report(strata_totals[name], tokens, classify_misses)
                for name in STRATA
            },
            "sentences": {
                name: {
                    "sentences": sentence_strata[name]["sentences"],
                    "first_choice_sentences": sentence_strata[name]["first"],
                    "first_choice_sentence_accuracy": (
                        sentence_strata[name]["first"] / sentence_strata[name]["sentences"]
                        if sentence_strata[name]["sentences"]
                        else 0.0
                    ),
                }
                for name in SENTENCE_STRATA
            },
        }
    if intervals:
        _add_triage_intervals(report, interval_metrics, intervals, interval_seed)
    return report


def _add_triage_intervals(
    report: dict[str, object], metrics: dict, replicates: int, seed: int
) -> None:
    bounds = percentile_ratio_intervals(metrics, replicates, seed)
    _attach_triage_intervals(report, bounds, replicates, seed)


def _append_metric(
    metrics: dict, name: str, numerator: int | bool, denominator: int | bool
) -> None:
    numerators, denominators = metrics.setdefault(name, ([], []))
    numerators.append(numerator)
    denominators.append(denominator)


def _append_interval_metrics(metrics: dict, result: dict, classify_misses: bool) -> None:
    tokens = result["tokens"]
    _append_metric(metrics, "first_choice", result["first"], tokens)
    _append_metric(metrics, "any_reading", result["any"], tokens)
    _append_metric(metrics, "first_choice_sentence_accuracy", result["sentence_first"], 1)
    if classify_misses:
        for name in MISS_CLASSES:
            _append_metric(metrics, f"class:{name}", result["miss_counts"].get(name, 0), tokens)
    if "strata" in result:
        _append_strata_interval_metrics(metrics, result, classify_misses)


def _append_strata_interval_metrics(metrics: dict, result: dict, classify_misses: bool) -> None:
    for stratum in STRATA:
        _append_token_stratum_metrics(metrics, result, stratum, classify_misses)
    for stratum in SENTENCE_STRATA:
        _append_sentence_stratum_metric(metrics, result, stratum)


def _append_token_stratum_metrics(
    metrics: dict, result: dict, stratum: str, classify_misses: bool
) -> None:
    row = result["strata"][stratum]
    _append_metric(metrics, f"token_stratum:{stratum}:first", row["first"], row["tokens"])
    _append_metric(metrics, f"token_stratum:{stratum}:any", row["any"], row["tokens"])
    if classify_misses:
        for name in MISS_CLASSES:
            _append_metric(
                metrics,
                f"token_stratum:{stratum}:class:{name}",
                row["miss_counts"].get(name, 0),
                result["tokens"],
            )


def _append_sentence_stratum_metric(metrics: dict, result: dict, stratum: str) -> None:
    member = stratum in result["sentence_strata"]
    _append_metric(
        metrics,
        f"sentence_stratum:{stratum}",
        member and result["sentence_first"],
        member,
    )


def _attach_triage_intervals(report, bounds, replicates, seed) -> None:
    report["intervals"] = {
        "method": "sentence-cluster percentile bootstrap",
        "confidence": CI_LEVEL,
        "replicates": replicates,
        "seed": seed,
        "minimum_defined_fraction": MIN_DEFINED_FRACTION,
    }
    _attach_overall_intervals(report, bounds)
    if "strata" in report:
        _attach_strata_intervals(report["strata"], bounds)


def _attach_overall_intervals(report, bounds) -> None:
    accuracy = report["accuracy"]
    for name in ("first_choice", "any_reading", "first_choice_sentence_accuracy"):
        accuracy[f"{name}_ci95"] = bounds[name]["ci95"]
        accuracy[f"{name}_ci95_defined_replicates"] = bounds[name]["defined_replicates"]
    if "by_class" in report:
        for name in MISS_CLASSES:
            interval = bounds[f"class:{name}"]
            report["by_class"][name]["share_sampled_tokens_pp_ci95"] = _percentage_points(
                interval["ci95"]
            )
            report["by_class"][name]["share_sampled_tokens_pp_ci95_defined_replicates"] = interval[
                "defined_replicates"
            ]


def _attach_strata_intervals(strata, bounds) -> None:
    for stratum in STRATA:
        row = strata["tokens"][stratum]
        for result_name, metric_name in (
            ("first_choice", "first"),
            ("any_reading", "any"),
        ):
            interval = bounds[f"token_stratum:{stratum}:{metric_name}"]
            row["accuracy"][f"{result_name}_ci95"] = interval["ci95"]
            row["accuracy"][f"{result_name}_ci95_defined_replicates"] = interval[
                "defined_replicates"
            ]
        if "by_class" in row:
            for name in MISS_CLASSES:
                interval = bounds[f"token_stratum:{stratum}:class:{name}"]
                row["by_class"][name]["share_sampled_tokens_pp_ci95"] = _percentage_points(
                    interval["ci95"]
                )
                row["by_class"][name]["share_sampled_tokens_pp_ci95_defined_replicates"] = interval[
                    "defined_replicates"
                ]
    for stratum in SENTENCE_STRATA:
        interval = bounds[f"sentence_stratum:{stratum}"]
        strata["sentences"][stratum]["first_choice_sentence_accuracy_ci95"] = interval["ci95"]
        strata["sentences"][stratum]["first_choice_sentence_accuracy_ci95_defined_replicates"] = (
            interval["defined_replicates"]
        )


def _percentage_points(bounds: list[float] | None) -> list[float] | None:
    return None if bounds is None else [100.0 * value for value in bounds]


def _per_sentence_payload(
    records: list[dict], profile: str | None, sample_fingerprint: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "unit": "sentence",
        "profile": profile,
        "sample_fingerprint": sample_fingerprint,
        "records": records,
    }


def _per_sentence_record(result: dict) -> dict[str, object]:
    record = {key: result[key] for key in ("id", "tokens", "first", "any", "sentence_first")}
    if "strata" in result:
        record["strata"] = _per_sentence_strata(result)
    return record


def _per_sentence_strata(result: dict) -> dict[str, object]:
    return {
        "tokens": {
            name: {key: result["strata"][name][key] for key in ("tokens", "first", "any")}
            for name in STRATA
        },
        "sentences": result["sentence_strata"],
    }


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sentences-per-shard", type=int, default=MAX_SENTENCES_PER_SHARD)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--profile", choices=("google-tn",), default=None)
    parser.add_argument(
        "--case-variant-lookup",
        action="store_true",
        help="opt in to unanimous casefold aliases in the spellout dictionary",
    )
    parser.add_argument(
        "--accuracy-only",
        action="store_true",
        help="skip post-miss triage classification while retaining accuracy and strata",
    )
    parser.add_argument("--strata", type=Path, default=None, metavar="VOCAB")
    parser.add_argument("--intervals", type=int, default=0, metavar="N")
    parser.add_argument("--interval-seed", type=int, default=DEFAULT_INTERVAL_SEED)
    parser.add_argument("--per-sentence-out", type=Path, default=None, metavar="PATH")
    parser.add_argument(
        "--strip-embedded-sil",
        action="store_true",
        help="measurement only: remove embedded corpus sil tokens from scored targets",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.sentences_per_shard <= MAX_SENTENCES_PER_SHARD:
        parser.error(f"--sentences-per-shard must be in 1..{MAX_SENTENCES_PER_SHARD}")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.intervals < 0:
        parser.error("--intervals must be nonnegative")
    if args.per_sentence_out is not None and args.per_sentence_out.resolve().is_relative_to(_REPO):
        parser.error("--per-sentence-out must be outside the repository")
    profile = _validate_evaluation_profile(args.profile, LOCALE)

    corpus_dir = (args.corpus_dir or store_root(SOURCE_ID)).resolve(strict=True)
    verified = verified_inputs(
        SOURCE_ID,
        [corpus_dir / name for name in SHARDS],
        locale=LOCALE,
        pools=POOLS,
        root=corpus_dir,
    )
    corpus_receipt = verification_receipt(verified, locale=LOCALE, pools=POOLS)

    selected = []
    inventory = {}
    selected_indexes = {}
    for offset, corpus_input in enumerate(verified):
        sample, available = _reservoir_sample(
            corpus_input, args.sentences_per_shard, args.seed + offset
        )
        inventory[corpus_input.relative_path] = {
            "available_sentences": available,
            "sampled_sentences": len(sample),
        }
        selected_indexes[corpus_input.relative_path] = [index for index, _rows in sample]
        selected.extend((corpus_input.relative_path, index, rows) for index, rows in sample)

    vocabulary = None
    if args.strata is not None:
        vocabulary = load_vocabulary(
            args.strata, (row[1] for _shard, _index, sentence in selected for row in sentence)
        )
    work = [
        (
            shard,
            index,
            sentence,
            (
                None
                if vocabulary is None
                else tuple("SEEN" if row[1] in vocabulary.seen else "UNSEEN" for row in sentence)
            ),
            profile,
            not args.accuracy_only,
            args.strip_embedded_sil,
            args.case_variant_lookup,
        )
        for shard, index, sentence in selected
    ]

    fingerprint_payload = {
        "corpus_fingerprint": corpus_receipt["fingerprint"],
        "seed": args.seed,
        "sentence_cap_per_shard": args.sentences_per_shard,
        "selected_sentence_indexes": selected_indexes,
    }
    sample_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    records = [] if args.per_sentence_out is not None else None
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(_score_sentence, work, chunksize=16)
        report = _aggregate(
            results,
            len(selected),
            args.intervals,
            args.interval_seed,
            records,
            not args.accuracy_only,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts_path = args.output_dir / "counts.json"
    receipt_path = args.output_dir / "receipt.json"
    receipt = {
        "schema_version": 1,
        "head": _head(),
        "versions": {
            "frend": _version("frend"),
            "icukit": _version("icukit"),
            "tiergraph": _version("tiergraph"),
        },
        "sample": {
            "seed": args.seed,
            "sentence_cap_per_shard": args.sentences_per_shard,
            "shards": list(SHARDS),
            "inventory": inventory,
            "fingerprint": sample_fingerprint,
        },
        "corpus_verification": corpus_receipt,
        "outputs": {"counts": counts_path.name},
    }
    if profile is not None:
        receipt["profile"] = profile
    if args.accuracy_only:
        receipt["accuracy_only"] = True
    if args.strip_embedded_sil:
        receipt["strip_embedded_sil"] = True
    if args.case_variant_lookup:
        receipt["case_variant_lookup"] = True
    if vocabulary is not None:
        receipt["strata_vocabulary"] = vocabulary_receipt(vocabulary)
    _write_json(counts_path, report)
    _write_json(receipt_path, receipt)
    if args.per_sentence_out is not None:
        args.per_sentence_out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            args.per_sentence_out,
            _per_sentence_payload(records, profile, sample_fingerprint),
        )
    print(json.dumps({"counts": report, "receipt": receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
