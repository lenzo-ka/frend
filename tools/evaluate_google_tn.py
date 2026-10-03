"""Score frend on the Google text-normalization test set, first choice, as published.

Sproat & Jaitly (2016) test on the first 100,000 lines of the final shard
(``output-00099-of-00100``, about 92,000 tokens with end-of-sentence markers), and
Bakhturina et al. (2022) score whole sentences on the same data. This tool puts every
token of that set through frend's pipeline -- recognition with the spoken-priors
profile, icukit's abbreviations, URLs and frend's written forms; resolution; speech --
and compares frend's FIRST choice with the corpus's spoken form, per class, overall and
per sentence, so the numbers sit beside theirs. It also reports whether ANY reading
frend offers matches, for context. Each token is read alone, with its sentence's
written text either side as its context (``frend.context``: the context trees and the
range connector read it), as running text would give it.

Scoring: both sides are normalized as the spoken priors normalize (lowercase,
punctuation as space); ``<self>`` expects the written token, ``sil`` expects nothing,
and ELECTRONIC's per-letter notation is joined first. One difference remains and is
stated in the report: frend reads each token alone (with its sentence as context), where
the published models read the sentence whole.

Only counts are printed and written; no corpus text.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import islice, product
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import google_tn_rows  # noqa: E402
from bootstrap_intervals import (  # noqa: E402
    CI_LEVEL,
    DEFAULT_INTERVAL_SEED,
    MIN_DEFINED_FRACTION,
    percentile_ratio_intervals,
)
from build_spoken_priors import _default_corpus_dir  # noqa: E402
from corpus_inputs import (  # noqa: E402
    VerifiedInput,
    open_verified,
    verified_inputs,
    write_verification_receipt,
)
from seen_strata import (  # noqa: E402
    SENTENCE_RULE,
    SENTENCE_STRATA,
    STRATA,
    load_vocabulary,
    sentence_strata,
    vocabulary_receipt,
)

_TEST_FILE = "output-00099-of-00100"
_TEST_LINES = 100_000
_ANY_CAP = 64


def _validate_evaluation_profile(profile: str | None, locale: str) -> str | None:
    """Validate the selected profile and its table before entering scoring recovery."""
    from frend.profiles import GOOGLE_TN, validate_profile
    from frend.verbalize import _google_tn_britishisms

    profile = validate_profile(profile)
    if profile == GOOGLE_TN:
        _google_tn_britishisms(locale=locale)
    return profile


def _canonical_held_out_shard(value: str) -> str:
    if value not in google_tn_rows.HELD_OUT_SHARDS:
        raise argparse.ArgumentTypeError(
            "must exactly match a pinned held-out shard name "
            "(output-00090-of-00100 through output-00099-of-00100)"
        )
    return value


def _detectors(locale: str = "en_US"):
    """The evaluator's readers: the shared profile for ``locale``."""
    from reading_profile import reading_detectors

    return reading_detectors(locale)


def _require_detect_k(detect_k: int, detect_function=None) -> None:
    """Refuse the opt-in mode before corpus work when icukit lacks the required API."""
    if detect_function is None:
        from icukit.detectors import detect as detect_function

    try:
        parameters = inspect.signature(detect_function).parameters
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "--detect-k requires an icukit detect() whose signature exposes keyword 'k'"
        ) from error
    if "k" not in parameters:
        raise RuntimeError(
            "--detect-k requires an icukit detect() with keyword 'k'; "
            "the installed icukit does not provide it"
        )
    try:
        detect_function("", (), k=detect_k)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"installed icukit rejects --detect-k {detect_k}: {error}") from error


_expected = google_tn_rows.expected


def _joined(texts_and_passthrough) -> str:
    """Rejoin a path's units: passed-through characters exactly as written (the lattice
    passes text through one character at a time), a spoken reading set off by spaces."""
    out = ""
    for text, passthrough in texts_and_passthrough:
        out += text if passthrough else f" {text} "
    return out


def _score(
    item: tuple[tuple[str, str, str], str, str, str, str | None, int | None],
) -> tuple[str, bool, bool]:
    (corpus_class, written, spoken), before, after, locale, profile, detect_k = item
    first, any_ = _score_text(
        written,
        _expected(corpus_class, written, spoken),
        before,
        after,
        locale,
        profile,
        detect_k,
    )
    return corpus_class, first, any_


def _in_context(written: str, before: str, after: str):
    """The sentence ``written`` is read in: its written neighbors, one space between
    tokens, as frend's context trees read running text (``frend.context``)."""
    from frend.context import TextContext

    head = f"{before} " if before else ""
    tail = f" {after}" if after else ""
    return TextContext(f"{head}{written}{tail}", len(head))


def _score_text(
    written: str,
    target: str,
    before: str = "",
    after: str = "",
    locale: str = "en_US",
    profile: str | None = None,
    detect_k: int | None = None,
) -> tuple[bool, bool]:
    """Whether frend's first reading of ``written``, and whether any reading, says ``target``.

    ``written`` is read alone, as a token of its sentence, and ``before`` and ``after``
    (the sentence's written text either side) are its context.
    """
    from icukit.detectors import detect

    from frend import resolve_lattice
    from frend.spoken_priors import normalize_spoken
    from frend.verbalize import verbalize_lattice

    profile = _validate_evaluation_profile(profile, locale)
    target = normalize_spoken(target)
    try:
        if not written.strip():
            detections = []
        elif detect_k is None:
            # Keep the historical call exactly unchanged unless the option is selected.
            detections = list(detect(written, _detectors(locale)))
        else:
            detections = list(detect(written, _detectors(locale), k=detect_k))
        verbalized = verbalize_lattice(
            resolve_lattice(detections, source_text=written, locale=locale),
            context=_in_context(written, before, after),
            profile=profile,
        )
    except FileNotFoundError:
        raise
    except Exception:  # noqa: BLE001 - a crash is a miss, counted, not hidden
        return False, False

    def passthrough(alternative) -> bool:
        return alternative.provenance == "surface:passthrough"

    best = verbalized.best_path
    first = normalize_spoken(
        _joined((unit.best.text, passthrough(unit.best)) for unit in best.units)
    )
    if first == target:
        return True, True
    for path in verbalized.paths:
        options = [[(a.text, passthrough(a)) for a in unit.alternatives] for unit in path.units]
        for combination in islice(product(*options), _ANY_CAP):
            if normalize_spoken(_joined(combination)) == target:
                return False, True
    return False, False


# The number, separator, number predicate is shared with the builders (one definition);
# the alias keeps the name the evaluator has always had.
_running_text = google_tn_rows.running_text
_range_denominators = google_tn_rows.range_candidate_denominators


def _score_joined(
    item: tuple[tuple[str, str, str, str], tuple[str, str], str, str | None, int | None],
) -> tuple[str, str, bool, bool]:
    (separator, middle, written, target), (before, after), locale, profile, detect_k = item
    return (
        separator,
        middle,
        *_score_text(written, target, before, after, locale, profile, detect_k),
    )


def _rows(corpus_input: VerifiedInput, limit: int | None = _TEST_LINES):
    """Sentences of (class, written, spoken) rows from the first ``limit`` lines of shard
    ``corpus_input`` (every line when ``limit`` is None). The input must already bind
    source, locale pool and bytes through :mod:`corpus_inputs`."""
    sentences, current = [], []
    with open_verified(corpus_input) as handle:
        for line in islice(handle, limit):
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "<eos>":
                if current:
                    sentences.append(current)
                current = []
                continue
            if len(parts) >= 3:
                current.append((parts[0], parts[1], parts[2]))
    if current:
        sentences.append(current)
    return sentences


def _map(function, items, workers: int, chunksize: int) -> list:
    """``function`` over ``items``, across ``workers`` processes when there is more than one."""
    if workers <= 1:
        return [function(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items, chunksize=chunksize))


def _per_token(
    sentences,
    workers: int,
    locale: str = "en_US",
    profile: str | None = None,
    strata_seen: frozenset[str] | None = None,
    detect_k: int | None = None,
    *,
    intervals: int = 0,
    interval_seed: int = DEFAULT_INTERVAL_SEED,
    record_prefix: str = "sample",
    record_sink: list[dict] | None = None,
) -> dict:
    """First-choice, any-reading and whole-sentence accuracy over every token of
    ``sentences``, overall and per class."""
    rows = [
        (
            row,
            " ".join(r[1] for r in sentence[:index]),
            " ".join(r[1] for r in sentence[index + 1 :]),
            locale,
            profile,
            detect_k,
        )
        for sentence in sentences
        for index, row in enumerate(sentence)
    ]
    results = _map(_score, rows, workers, 256)
    by_class: dict[str, Counter] = defaultdict(Counter)
    for corpus_class, first, any_ in results:
        by_class[corpus_class]["tokens"] += 1
        by_class[corpus_class]["first"] += first
        by_class[corpus_class]["any"] += any_
    total = Counter()
    for counts in by_class.values():
        total.update(counts)
    correct_sentences, at = 0, 0
    sentence_records = []
    for sentence_index, sentence in enumerate(sentences):
        outcome = results[at : at + len(sentence)]
        at += len(sentence)
        first_count = sum(first for _, first, _ in outcome)
        any_count = sum(any_ for _, _, any_ in outcome)
        sentence_first = first_count == len(sentence)
        correct_sentences += sentence_first
        sentence_records.append(
            {
                "id": f"{record_prefix}:{sentence_index}",
                "tokens": len(sentence),
                "first": first_count,
                "any": any_count,
                "sentence_first": sentence_first,
            }
        )
    tokens, count = total["tokens"], len(sentences)
    report = {
        "tokens": tokens,
        "sentences": count,
        "first_choice_accuracy": total["first"] / tokens if tokens else 0.0,
        "any_reading_accuracy": total["any"] / tokens if tokens else 0.0,
        "sentence_accuracy": correct_sentences / count if count else 0.0,
        "classes": {
            name: {
                "tokens": counts["tokens"],
                "first_choice": counts["first"] / counts["tokens"],
                "any_reading": counts["any"] / counts["tokens"],
            }
            for name, counts in sorted(by_class.items(), key=lambda kv: -kv[1]["tokens"])
        },
    }
    if strata_seen is None:
        if intervals:
            _add_evaluation_intervals(report, sentence_records, intervals, interval_seed)
        if record_sink is not None:
            record_sink.extend(sentence_records)
        return report

    by_stratum: dict[str, Counter] = {name: Counter() for name in STRATA}
    class_strata: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {name: Counter() for name in STRATA}
    )
    for row, (corpus_class, first, any_) in zip(rows, results, strict=True):
        stratum = "SEEN" if row[0][1] in strata_seen else "UNSEEN"
        counts = by_stratum[stratum]
        counts["tokens"] += 1
        counts["first"] += first
        counts["any"] += any_
        class_counts = class_strata[corpus_class][stratum]
        class_counts["tokens"] += 1
        class_counts["first"] += first
        class_counts["any"] += any_

    def token_entry(counts: Counter) -> dict[str, int | float]:
        count = counts["tokens"]
        return {
            "tokens": count,
            "first_choice_tokens": counts["first"],
            "first_choice_accuracy": counts["first"] / count if count else 0.0,
            "any_reading_tokens": counts["any"],
            "any_reading_accuracy": counts["any"] / count if count else 0.0,
        }

    report["strata"] = {
        "sentence_rule": SENTENCE_RULE,
        "tokens": {name: token_entry(by_stratum[name]) for name in STRATA},
        "sentences": {},
    }
    for name, row in report["classes"].items():
        row["strata"] = {stratum: token_entry(class_strata[name][stratum]) for stratum in STRATA}

    sentence_counts: dict[str, Counter] = {name: Counter() for name in SENTENCE_STRATA}
    at = 0
    for sentence, record in zip(sentences, sentence_records, strict=True):
        outcome = results[at : at + len(sentence)]
        at += len(sentence)
        token_strata = tuple("SEEN" if row[1] in strata_seen else "UNSEEN" for row in sentence)
        record["strata"] = {
            "tokens": {
                name: {
                    "tokens": sum(value == name for value in token_strata),
                    "first": sum(
                        first
                        for value, (_, first, _) in zip(token_strata, outcome, strict=True)
                        if value == name
                    ),
                    "any": sum(
                        any_
                        for value, (_, _, any_) in zip(token_strata, outcome, strict=True)
                        if value == name
                    ),
                }
                for name in STRATA
            },
            "sentences": list(sentence_strata(sentence, token_strata)),
        }
        for stratum in record["strata"]["sentences"]:
            sentence_counts[stratum]["sentences"] += 1
            sentence_counts[stratum]["first"] += all(first for _, first, _ in outcome)
    report["strata"]["sentences"] = {
        name: {
            "sentences": sentence_counts[name]["sentences"],
            "first_choice_sentences": sentence_counts[name]["first"],
            "sentence_accuracy": (
                sentence_counts[name]["first"] / sentence_counts[name]["sentences"]
                if sentence_counts[name]["sentences"]
                else 0.0
            ),
        }
        for name in SENTENCE_STRATA
    }
    if intervals:
        _add_evaluation_intervals(report, sentence_records, intervals, interval_seed)
    if record_sink is not None:
        record_sink.extend(sentence_records)
    return report


def _add_evaluation_intervals(
    report: dict, records: list[dict], replicates: int, seed: int
) -> None:
    ones = [1] * len(records)
    metrics = {
        "first_choice_accuracy": (
            [row["first"] for row in records],
            [row["tokens"] for row in records],
        ),
        "any_reading_accuracy": (
            [row["any"] for row in records],
            [row["tokens"] for row in records],
        ),
        "sentence_accuracy": ([row["sentence_first"] for row in records], ones),
    }
    has_strata = bool(records and "strata" in records[0])
    if has_strata:
        for name in STRATA:
            metrics[f"token_stratum:{name}:first_choice_accuracy"] = (
                [row["strata"]["tokens"][name]["first"] for row in records],
                [row["strata"]["tokens"][name]["tokens"] for row in records],
            )
            metrics[f"token_stratum:{name}:any_reading_accuracy"] = (
                [row["strata"]["tokens"][name]["any"] for row in records],
                [row["strata"]["tokens"][name]["tokens"] for row in records],
            )
        for name in SENTENCE_STRATA:
            membership = [name in row["strata"]["sentences"] for row in records]
            metrics[f"sentence_stratum:{name}:sentence_accuracy"] = (
                [
                    member and row["sentence_first"]
                    for member, row in zip(membership, records, strict=True)
                ],
                membership,
            )
    bounds = percentile_ratio_intervals(metrics, replicates, seed)
    report["intervals"] = {
        "method": "sentence-cluster percentile bootstrap",
        "confidence": CI_LEVEL,
        "replicates": replicates,
        "seed": seed,
        "minimum_defined_fraction": MIN_DEFINED_FRACTION,
        "first_choice_accuracy": bounds["first_choice_accuracy"]["ci95"],
        "any_reading_accuracy": bounds["any_reading_accuracy"]["ci95"],
        "sentence_accuracy": bounds["sentence_accuracy"]["ci95"],
        "defined_replicates": {
            name: bounds[name]["defined_replicates"]
            for name in ("first_choice_accuracy", "any_reading_accuracy", "sentence_accuracy")
        },
    }
    if has_strata:
        for name in STRATA:
            report["strata"]["tokens"][name]["intervals"] = {
                metric: bounds[f"token_stratum:{name}:{metric}"]["ci95"]
                for metric in ("first_choice_accuracy", "any_reading_accuracy")
            }
            report["strata"]["tokens"][name]["intervals"]["defined_replicates"] = {
                metric: bounds[f"token_stratum:{name}:{metric}"]["defined_replicates"]
                for metric in ("first_choice_accuracy", "any_reading_accuracy")
            }
        for name in SENTENCE_STRATA:
            report["strata"]["sentences"][name]["intervals"] = {
                "sentence_accuracy": bounds[f"sentence_stratum:{name}:sentence_accuracy"]["ci95"],
                "defined_replicates": {
                    "sentence_accuracy": bounds[f"sentence_stratum:{name}:sentence_accuracy"][
                        "defined_replicates"
                    ]
                },
            }


def _running_text_rows(
    sentences,
    workers: int,
    locale: str = "en_US",
    profile: str | None = None,
    detect_k: int | None = None,
) -> list[dict]:
    """Each number, separator, number triple of ``sentences`` rejoined as written and
    scored, grouped by separator and the corpus's reading of it."""
    items = list(
        (
            (row, context, locale, profile, detect_k)
            for row, context in zip(
                _running_text(sentences, locale),
                google_tn_rows.running_text_contexts(sentences, locale),
                strict=True,
            )
        )
    )
    joined_results = _map(_score_joined, items, workers, 16)
    by_joint: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for separator, middle, first, any_ in joined_results:
        by_joint[(separator, middle)]["n"] += 1
        by_joint[(separator, middle)]["first"] += first
        by_joint[(separator, middle)]["any"] += any_
    return [
        {
            "separator": separator,
            "corpus_middle": middle,
            "count": counts["n"],
            "first_choice": counts["first"] / counts["n"],
            "any_reading": counts["any"] / counts["n"],
        }
        for (separator, middle), counts in sorted(by_joint.items(), key=lambda kv: -kv[1]["n"])
    ]


def _held_out(
    inputs: dict[str, VerifiedInput],
    name: str,
    workers: int,
    locale: str = "en_US",
    profile: str | None = None,
    detect_k: int | None = None,
    strata_path: Path | None = None,
    intervals: int = 0,
    interval_seed: int = DEFAULT_INTERVAL_SEED,
    record_sink: list[dict] | None = None,
) -> dict:
    """The held-out shard ``name``: per token over its first ``_TEST_LINES`` lines, cut
    as the paper cuts the test shard, and running text over the whole shard."""
    corpus_input = inputs[name]
    whole = _rows(corpus_input, None)
    per_token_sentences = _rows(corpus_input, _TEST_LINES)
    match = None
    if strata_path is not None:
        match = load_vocabulary(
            strata_path, (row[1] for sentence in per_token_sentences for row in sentence)
        )
    report = {
        "shard": name,
        "per_token": {
            "lines": f"first {_TEST_LINES} lines of {name}",
            **_per_token(
                per_token_sentences,
                workers,
                locale,
                profile,
                None if match is None else match.seen,
                detect_k,
                intervals=intervals,
                interval_seed=interval_seed,
                record_prefix=name,
                record_sink=record_sink,
            ),
        },
        "running_text": {
            "lines": f"all lines of {name}",
            "triples": len(_running_text(whole, locale)),
            "rows": _running_text_rows(whole, workers, locale, profile, detect_k),
        },
    }
    if match is not None:
        report["strata_vocabulary"] = vocabulary_receipt(match)
    return report


def evaluate(
    inputs: dict[str, VerifiedInput],
    workers: int,
    held_out_shard: str | None = None,
    *,
    locale: str = "en_US",
    skip_report_shard: bool = False,
    profile: str | None = None,
    detect_k: int | None = None,
    strata_path: Path | None = None,
    intervals: int = 0,
    interval_seed: int = DEFAULT_INTERVAL_SEED,
    record_sink: list[dict] | None = None,
) -> dict:
    profile = _validate_evaluation_profile(profile, locale)
    if skip_report_shard and held_out_shard == _TEST_FILE:
        raise ValueError(
            "skip_report_shard=True requires held_out_shard other than report shard 99"
        )
    if skip_report_shard and _TEST_FILE in inputs:
        raise ValueError("skip_report_shard=True forbids report shard 99 in inputs")
    report = {}
    if not skip_report_shard:
        sentences = _rows(inputs[_TEST_FILE])
        match = None
        if strata_path is not None:
            match = load_vocabulary(
                strata_path, (row[1] for sentence in sentences for row in sentence)
            )
        per_token = _per_token(
            sentences,
            workers,
            locale,
            profile,
            None if match is None else match.seen,
            detect_k,
            intervals=intervals,
            interval_seed=interval_seed,
            record_prefix=_TEST_FILE,
            record_sink=record_sink,
        )
        report = {
            "test_set": f"first {_TEST_LINES} lines of {_TEST_FILE}",
            **{key: per_token[key] for key in ("tokens", "sentences")},
            **{
                key: per_token[key]
                for key in (
                    "first_choice_accuracy",
                    "any_reading_accuracy",
                    "sentence_accuracy",
                )
            },
            "classes": per_token["classes"],
            "running_text": _running_text_rows(sentences, workers, locale, profile, detect_k),
            "note": (
                "frend reads each token alone, its sentence as context; "
                "the published models read the sentence whole"
            ),
        }
        if match is not None:
            report["strata"] = per_token["strata"]
            report["strata_vocabulary"] = vocabulary_receipt(match)
        if intervals:
            report["intervals"] = per_token["intervals"]
    if held_out_shard is not None:
        report["held_out"] = _held_out(
            inputs,
            held_out_shard,
            workers,
            locale,
            profile,
            detect_k,
            strata_path,
            intervals,
            interval_seed,
            record_sink,
        )
    return report


def _render(report: dict) -> str:
    lines = []
    if "test_set" in report:
        lines = [
            f"Test set: {report['test_set']} ({report['tokens']} tokens, "
            f"{report['sentences']} sentences)",
            f"First choice: {100 * report['first_choice_accuracy']:.2f}%   "
            f"any reading: {100 * report['any_reading_accuracy']:.2f}%   "
            f"sentences all right: {100 * report['sentence_accuracy']:.2f}%",
            "",
            f"{'class':12s}{'tokens':>8s}{'first':>9s}{'any':>9s}",
        ]
        if "intervals" in report:
            lines.insert(2, _render_interval_line(report["intervals"]))
        for name, row in report["classes"].items():
            lines.append(
                f"{name:12s}{row['tokens']:>8d}{100 * row['first_choice']:>8.1f}%"
                f"{100 * row['any_reading']:>8.1f}%"
            )
        lines.append("")
        if "strata" in report:
            lines.extend(_render_strata(report["strata"]))
            lines.extend(_render_class_strata(report["classes"]))
            lines.append("")
        lines.append(report["note"])
        lines.append("")
        lines.append('Running text: number, separator, number rejoined as written ("5-10")')
        lines.append(f"{'sep':5s}{'corpus middle':16s}{'count':>7s}{'first':>9s}{'any':>9s}")
        for row in report["running_text"]:
            lines.append(
                f"{row['separator']:5s}{row['corpus_middle']:16s}{row['count']:>7d}"
                f"{100 * row['first_choice']:>8.1f}%{100 * row['any_reading']:>8.1f}%"
            )
    held = report.get("held_out")
    if held:
        tokens = held["per_token"]
        if lines:
            lines.append("")
        lines.append(
            f"Held out: {tokens['lines']} ({tokens['tokens']} tokens, "
            f"{tokens['sentences']} sentences)"
        )
        lines.append(
            f"First choice: {100 * tokens['first_choice_accuracy']:.2f}%   "
            f"any reading: {100 * tokens['any_reading_accuracy']:.2f}%   "
            f"sentences all right: {100 * tokens['sentence_accuracy']:.2f}%"
        )
        if "intervals" in tokens:
            lines.append(_render_interval_line(tokens["intervals"]))
        if "strata" in tokens:
            lines.extend(_render_strata(tokens["strata"]))
        text = held["running_text"]
        lines.append(f"Held-out running text: {text['lines']} ({text['triples']} triples)")
        for row in text["rows"]:
            lines.append(
                f"{row['separator']:5s}{row['corpus_middle']:16s}{row['count']:>7d}"
                f"{100 * row['first_choice']:>8.1f}%{100 * row['any_reading']:>8.1f}%"
            )
    return "\n".join(lines)


def _render_interval_line(intervals: dict) -> str:
    def percent(name: str) -> str:
        bounds = intervals[name]
        if bounds is None:
            defined = intervals["defined_replicates"][name]
            return f"unreliable ({defined}/{intervals['replicates']} draws defined)"
        low, high = bounds
        return f"[{100 * low:.2f}%, {100 * high:.2f}%]"

    return (
        f"95% sentence-bootstrap CI: first {percent('first_choice_accuracy')}   "
        f"any {percent('any_reading_accuracy')}   sentences {percent('sentence_accuracy')}"
    )


def _render_strata(strata: dict) -> list[str]:
    lines = ["Token strata:", f"{'stratum':12s}{'tokens':>9s}{'first':>15s}{'any':>15s}"]
    for name in STRATA:
        row = strata["tokens"][name]
        lines.append(
            f"{name:12s}{row['tokens']:>9d}"
            f"{row['first_choice_tokens']:>8d} {100 * row['first_choice_accuracy']:>5.1f}%"
            f"{row['any_reading_tokens']:>8d} {100 * row['any_reading_accuracy']:>5.1f}%"
        )
        if "intervals" in row:
            lines.append(
                f"  95% CI first {_render_stratum_interval(row, 'first_choice_accuracy')} "
                f"any {_render_stratum_interval(row, 'any_reading_accuracy')}"
            )
    lines.append("Sentence strata:")
    lines.append(f"{'stratum':25s}{'sentences':>11s}{'all right':>15s}")
    for name in SENTENCE_STRATA:
        row = strata["sentences"][name]
        lines.append(
            f"{name:25s}{row['sentences']:>11d}"
            f"{row['first_choice_sentences']:>8d} {100 * row['sentence_accuracy']:>5.1f}%"
        )
        if "intervals" in row:
            lines.append(f"  95% CI {_render_stratum_interval(row, 'sentence_accuracy')}")
    lines.append(strata["sentence_rule"])
    return lines


def _render_stratum_interval(row: dict, name: str) -> str:
    intervals = row["intervals"]
    bounds = intervals[name]
    if bounds is None:
        defined = intervals["defined_replicates"][name]
        return f"unreliable ({defined} draws defined)"
    low, high = bounds
    return f"[{100 * low:.1f}%, {100 * high:.1f}%]"


def _per_sentence_payload(
    records: list[dict], profile: str | None, corpus_fingerprint: str
) -> dict[str, object]:
    sample_identity = {
        "corpus_fingerprint": corpus_fingerprint,
        "sentences": [{"id": row["id"], "tokens": row["tokens"]} for row in records],
    }
    sample_fingerprint = hashlib.sha256(
        json.dumps(sample_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "unit": "sentence",
        "profile": profile,
        "sample_fingerprint": sample_fingerprint,
        "records": records,
    }


def _render_class_strata(classes: dict) -> list[str]:
    lines = ["Class token strata:", f"{'class/stratum':19s}{'tokens':>8s}{'first':>9s}{'any':>9s}"]
    for class_name, class_row in classes.items():
        for stratum in STRATA:
            row = class_row["strata"][stratum]
            lines.append(
                f"{f'{class_name}/{stratum}':19s}{row['tokens']:>8d}"
                f"{100 * row['first_choice_accuracy']:>8.1f}%"
                f"{100 * row['any_reading_accuracy']:>8.1f}%"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pool", action="append", required=True, dest="pools")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--profile", choices=("google-tn",), default=None)
    parser.add_argument(
        "--detect-k",
        type=int,
        default=None,
        metavar="K",
        help="opt in to icukit detect(..., k=K); default keeps the historical call unchanged",
    )
    parser.add_argument(
        "--strata",
        type=Path,
        default=None,
        metavar="VOCAB",
        help="split aggregate metrics by a verified external training vocabulary",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the report here")
    parser.add_argument(
        "--intervals",
        type=int,
        default=0,
        metavar="N",
        help="add sentence-cluster percentile intervals from N bootstrap replicates",
    )
    parser.add_argument("--interval-seed", type=int, default=DEFAULT_INTERVAL_SEED)
    parser.add_argument(
        "--per-sentence-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="write text-free per-sentence sufficient statistics outside the repository",
    )
    parser.add_argument(
        "--held-out-shard",
        default=None,
        metavar="NAME",
        type=_canonical_held_out_shard,
        help="also score shard NAME: per token over its first 100,000 lines, running "
        "text over all of it (acceptance is on output-00095-of-00100)",
    )
    parser.add_argument(
        "--skip-report-shard",
        action="store_true",
        help="do not verify or score published report shard 99; emit only the held-out "
        "section (used for decision runs; requires --held-out-shard naming another shard)",
    )
    args = parser.parse_args(argv)
    if args.skip_report_shard and args.held_out_shard is None:
        parser.error("--skip-report-shard requires --held-out-shard")
    if args.skip_report_shard and args.held_out_shard == _TEST_FILE:
        parser.error("--skip-report-shard requires a held-out shard other than report shard 99")
    if args.intervals < 0:
        parser.error("--intervals must be nonnegative")
    if args.detect_k is not None:
        try:
            _require_detect_k(args.detect_k)
        except RuntimeError as error:
            parser.error(str(error))
    if args.per_sentence_out is not None and args.per_sentence_out.resolve().is_relative_to(_REPO):
        parser.error("--per-sentence-out must be outside the repository")
    corpus_dir = args.corpus_dir or _default_corpus_dir()
    names = [] if args.skip_report_shard else [_TEST_FILE]
    if args.held_out_shard is not None and args.held_out_shard not in names:
        names.append(args.held_out_shard)
    verified = verified_inputs(
        args.source_id,
        [corpus_dir / name for name in names],
        locale=args.locale,
        pools=tuple(args.pools),
        root=corpus_dir,
    )
    corpus_receipt = write_verification_receipt(
        args.receipt, verified, locale=args.locale, pools=tuple(args.pools)
    )
    inputs = {item.relative_path: item for item in verified}
    records = [] if args.per_sentence_out is not None else None
    report = evaluate(
        inputs,
        args.workers,
        args.held_out_shard,
        locale=args.locale,
        skip_report_shard=args.skip_report_shard,
        profile=args.profile,
        detect_k=args.detect_k,
        strata_path=args.strata,
        intervals=args.intervals,
        interval_seed=args.interval_seed,
        record_sink=records,
    )
    print(_render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.per_sentence_out is not None:
        args.per_sentence_out.parent.mkdir(parents=True, exist_ok=True)
        args.per_sentence_out.write_text(
            json.dumps(
                _per_sentence_payload(records, args.profile, corpus_receipt["fingerprint"]),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
