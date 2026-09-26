"""Score frend on the Google text-normalization test set, first choice, as published.

Sproat & Jaitly (2016) test on the first 100,000 lines of the final shard
(``output-00099-of-00100``, about 92,000 tokens with end-of-sentence markers), and
Bakhturina et al. (2022) score whole sentences on the same data. This tool puts every
token of that set through frend's pipeline -- recognition with the spoken-priors
profile, icukit's abbreviations, URLs and frend's written forms; resolution; speech --
and compares frend's FIRST choice with the corpus's spoken form, per class, overall and
per sentence, so the numbers sit beside theirs. It also reports whether ANY reading
frend offers matches, for context.

Scoring: both sides are normalized as the spoken priors normalize (lowercase,
punctuation as space); ``<self>`` expects the written token, ``sil`` expects nothing,
and ELECTRONIC's per-letter notation is joined first. One difference remains and is
stated in the report: frend reads each token alone, where the published models see
the sentence around it.

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

from build_spoken_priors import _default_corpus_dir  # noqa: E402

_TEST_FILE = "output-00099-of-00100"
_TEST_LINES = 100_000
_ANY_CAP = 64

_DETECTORS = None


def _detectors():
    global _DETECTORS
    if _DETECTORS is None:
        from build_spoken_priors import _detectors as profile
        from icukit.abbreviation_recognize import AbbreviationDetector

        from frend.letters import LettersDetector

        seen, detectors = set(), []
        for group in profile().values():
            for detector in group:
                if id(detector) not in seen:
                    seen.add(id(detector))
                    detectors.append(detector)
        detectors.append(AbbreviationDetector("en_US"))
        detectors.append(LettersDetector("en_US"))
        _DETECTORS = detectors
    return _DETECTORS


def _expected(corpus_class: str, written: str, spoken: str) -> str:
    from frend.electronic import decode_letter_notation
    from frend.spoken_priors import normalize_spoken

    if spoken == "<self>":
        return normalize_spoken(written)
    if spoken == "sil":
        return ""
    if corpus_class == "ELECTRONIC":
        spoken = decode_letter_notation(spoken)
    return normalize_spoken(spoken)


def _joined(texts_and_passthrough) -> str:
    """Rejoin a path's units: passed-through characters exactly as written (the lattice
    passes text through one character at a time), a spoken reading set off by spaces."""
    out = ""
    for text, passthrough in texts_and_passthrough:
        out += text if passthrough else f" {text} "
    return out


def _score(row: tuple[str, str, str]) -> tuple[str, bool, bool]:
    from icukit.detectors import detect

    from frend import resolve_lattice
    from frend.spoken_priors import normalize_spoken
    from frend.verbalize import verbalize_lattice

    corpus_class, written, spoken = row
    target = _expected(corpus_class, written, spoken)
    try:
        detections = list(detect(written, _detectors())) if written.strip() else []
        verbalized = verbalize_lattice(resolve_lattice(detections, source_text=written))
    except Exception:  # noqa: BLE001 - a crash is a miss, counted, not hidden
        return corpus_class, target == normalize_spoken(written), False

    def passthrough(unit) -> bool:
        return unit.best.provenance == "surface:passthrough"

    best = verbalized.best_path
    first = normalize_spoken(_joined((unit.best.text, passthrough(unit)) for unit in best.units))
    if first == target:
        return corpus_class, True, True
    for path in verbalized.paths:
        options = [[(a.text, passthrough(unit)) for a in unit.alternatives] for unit in path.units]
        for combination in islice(product(*options), _ANY_CAP):
            if normalize_spoken(_joined(combination)) == target:
                return corpus_class, False, True
    return corpus_class, False, False


def _rows(corpus_dir: Path):
    """The test set: sentences of (class, written, spoken) rows, as the paper cuts it."""
    sentences, current = [], []
    with (corpus_dir / _TEST_FILE).open(encoding="utf-8") as handle:
        for line in islice(handle, _TEST_LINES):
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


def evaluate(corpus_dir: Path, workers: int) -> dict:
    sentences = _rows(corpus_dir)
    rows = [row for sentence in sentences for row in sentence]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_score, rows, chunksize=256))
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
    return {
        "test_set": f"first {_TEST_LINES} lines of {_TEST_FILE}",
        "tokens": total["tokens"],
        "sentences": len(sentences),
        "first_choice_accuracy": total["first"] / total["tokens"],
        "any_reading_accuracy": total["any"] / total["tokens"],
        "sentence_accuracy": correct_sentences / len(sentences),
        "classes": {
            name: {
                "tokens": counts["tokens"],
                "first_choice": counts["first"] / counts["tokens"],
                "any_reading": counts["any"] / counts["tokens"],
            }
            for name, counts in sorted(by_class.items(), key=lambda kv: -kv[1]["tokens"])
        },
        "note": "frend reads each token alone; the published models see its sentence",
    }


def _render(report: dict) -> str:
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
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--json", type=Path, default=None, help="also write the report here")
    args = parser.parse_args(argv)
    report = evaluate(args.corpus_dir or _default_corpus_dir(), args.workers)
    print(_render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
