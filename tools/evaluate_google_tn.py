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
from build_spoken_priors import _default_corpus_dir  # noqa: E402
from corpus_inputs import (  # noqa: E402
    VerifiedInput,
    open_verified,
    verified_inputs,
    write_verification_receipt,
)

_TEST_FILE = "output-00099-of-00100"
_TEST_LINES = 100_000
_ANY_CAP = 64


def _detectors(locale: str = "en_US"):
    """The evaluator's readers: the shared profile for ``locale``."""
    from reading_profile import reading_detectors

    return reading_detectors(locale)


_expected = google_tn_rows.expected


def _joined(texts_and_passthrough) -> str:
    """Rejoin a path's units: passed-through characters exactly as written (the lattice
    passes text through one character at a time), a spoken reading set off by spaces."""
    out = ""
    for text, passthrough in texts_and_passthrough:
        out += text if passthrough else f" {text} "
    return out


def _score(item: tuple[tuple[str, str, str], str, str, str]) -> tuple[str, bool, bool]:
    (corpus_class, written, spoken), before, after, locale = item
    first, any_ = _score_text(
        written, _expected(corpus_class, written, spoken), before, after, locale
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
) -> tuple[bool, bool]:
    """Whether frend's first reading of ``written``, and whether any reading, says ``target``.

    ``written`` is read alone, as a token of its sentence, and ``before`` and ``after``
    (the sentence's written text either side) are its context.
    """
    from icukit.detectors import detect

    from frend import resolve_lattice
    from frend.spoken_priors import normalize_spoken
    from frend.verbalize import verbalize_lattice

    target = normalize_spoken(target)
    try:
        detections = list(detect(written, _detectors(locale))) if written.strip() else []
        verbalized = verbalize_lattice(
            resolve_lattice(detections, source_text=written, locale=locale),
            context=_in_context(written, before, after),
        )
    except Exception:  # noqa: BLE001 - a crash is a miss, counted, not hidden
        return target == normalize_spoken(written), False

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
    item: tuple[tuple[str, str, str, str], tuple[str, str], str],
) -> tuple[str, str, bool, bool]:
    (separator, middle, written, target), (before, after), locale = item
    return (separator, middle, *_score_text(written, target, before, after, locale))


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


def _per_token(sentences, workers: int, locale: str = "en_US") -> dict:
    """First-choice, any-reading and whole-sentence accuracy over every token of
    ``sentences``, overall and per class."""
    rows = [
        (
            row,
            " ".join(r[1] for r in sentence[:index]),
            " ".join(r[1] for r in sentence[index + 1 :]),
            locale,
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
    for sentence in sentences:
        outcome = results[at : at + len(sentence)]
        at += len(sentence)
        correct_sentences += all(first for _, first, _ in outcome)
    tokens, count = total["tokens"], len(sentences)
    return {
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


def _running_text_rows(sentences, workers: int, locale: str = "en_US") -> list[dict]:
    """Each number, separator, number triple of ``sentences`` rejoined as written and
    scored, grouped by separator and the corpus's reading of it."""
    items = list(
        (
            (row, context, locale)
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
    inputs: dict[str, VerifiedInput], name: str, workers: int, locale: str = "en_US"
) -> dict:
    """The held-out shard ``name``: per token over its first ``_TEST_LINES`` lines, cut
    as the paper cuts the test shard, and running text over the whole shard."""
    corpus_input = inputs[name]
    whole = _rows(corpus_input, None)
    return {
        "shard": name,
        "per_token": {
            "lines": f"first {_TEST_LINES} lines of {name}",
            **_per_token(_rows(corpus_input, _TEST_LINES), workers, locale),
        },
        "running_text": {
            "lines": f"all lines of {name}",
            "triples": len(_running_text(whole, locale)),
            "rows": _running_text_rows(whole, workers, locale),
        },
    }


def evaluate(
    inputs: dict[str, VerifiedInput],
    workers: int,
    held_out_shard: str | None = None,
    *,
    locale: str = "en_US",
    skip_report_shard: bool = False,
) -> dict:
    report = {}
    if not skip_report_shard:
        sentences = _rows(inputs[_TEST_FILE])
        per_token = _per_token(sentences, workers, locale)
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
            "running_text": _running_text_rows(sentences, workers, locale),
            "note": (
                "frend reads each token alone, its sentence as context; "
                "the published models read the sentence whole"
            ),
        }
    if held_out_shard is not None:
        report["held_out"] = _held_out(inputs, held_out_shard, workers, locale)
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
        for name, row in report["classes"].items():
            lines.append(
                f"{name:12s}{row['tokens']:>8d}{100 * row['first_choice']:>8.1f}%"
                f"{100 * row['any_reading']:>8.1f}%"
            )
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
        text = held["running_text"]
        lines.append(f"Held-out running text: {text['lines']} ({text['triples']} triples)")
        for row in text["rows"]:
            lines.append(
                f"{row['separator']:5s}{row['corpus_middle']:16s}{row['count']:>7d}"
                f"{100 * row['first_choice']:>8.1f}%{100 * row['any_reading']:>8.1f}%"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pool", action="append", required=True, dest="pools")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--json", type=Path, default=None, help="also write the report here")
    parser.add_argument(
        "--held-out-shard",
        default=None,
        metavar="NAME",
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
    write_verification_receipt(args.receipt, verified, locale=args.locale, pools=tuple(args.pools))
    inputs = {item.relative_path: item for item in verified}
    report = evaluate(
        inputs,
        args.workers,
        args.held_out_shard,
        locale=args.locale,
        skip_report_shard=args.skip_report_shard,
    )
    print(_render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
