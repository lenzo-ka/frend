"""Build contextual counts distinguishing bare inches from prepositional ``in``.

All Google TN training shards (00--89) are streamed. For every MEASURE row, the
following written token is classified with the shipped context reranker's learned word
classes. The comparison population is every CARDINAL, DECIMAL, or ORDINAL row whose
next row is the as-written PLAIN token ``in``; for those pairs, the token after ``in``
is classified. Held-out shards are never opened.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
for _path in (_REPO, _TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from google_tn_rows import (  # noqa: E402
    TRAINING_SHARDS,
    corpus_label,
    full_training_set,
    training_shards,
)

from frend.context import EOS, _word_class, context_model  # noqa: E402

DEFAULT_OUT = _REPO / "frend" / "data" / "en" / "inch_context_priors.json"
_NUMERIC_CLASSES = frozenset({"CARDINAL", "DECIMAL", "ORDINAL"})
_DECISION = (
    "keep bare N in as a measure only with positive class-specific measure evidence "
    "greater than its preposition count; the heterogeneous other class is not evidence; "
    "preserve terminal measures and one trailing modifier"
)


def _default_corpus_dir() -> Path:
    from build_spoken_priors import _default_corpus_dir as spoken_default

    return spoken_default()


def _shards(corpus_dir: Path, inputs=None) -> list:
    if inputs is not None:
        paths = sorted(inputs, key=lambda item: item.relative_path)
    else:
        available = sorted(Path(corpus_dir).glob("output-*-of-*"))
        paths = full_training_set(available) or training_shards(available)
    if not paths:
        raise FileNotFoundError(f"no training shards under {corpus_dir}")
    return paths


def _count_file(job) -> tuple[Counter, Counter]:
    item, locale, frequent, verified = job
    if verified:
        from corpus_inputs import open_verified

        handle_context = open_verified(item)
    else:
        handle_context = Path(item).open(encoding="utf-8")
    measure: Counter = Counter()
    preposition: Counter = Counter()
    sentence: list[tuple[str, str]] = []

    def word_class(index: int) -> str:
        return EOS if index >= len(sentence) else _word_class(sentence[index][1], locale, frequent)

    def count_sentence() -> None:
        for index, (corpus_class, _written) in enumerate(sentence):
            if corpus_class == "MEASURE":
                measure[word_class(index + 1)] += 1
            if (
                corpus_class in _NUMERIC_CLASSES
                and index + 1 < len(sentence)
                and sentence[index + 1] == ("PLAIN", "in")
            ):
                preposition[word_class(index + 2)] += 1

    with handle_context as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "<eos>":
                count_sentence()
                sentence.clear()
            elif len(parts) >= 3:
                sentence.append((parts[0], parts[1]))
    if sentence:
        count_sentence()
    return measure, preposition


def build_document(
    corpus_dir: Path,
    *,
    locale: str = "en_US",
    inputs=None,
    workers: int = 1,
) -> dict:
    paths = _shards(Path(corpus_dir), inputs)
    model = context_model(locale)
    if model is None:
        raise ValueError(f"{locale} has no shipped context model")
    jobs = [(item, locale, model.frequent, inputs is not None) for item in paths]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            totals = list(pool.map(_count_file, jobs))
    else:
        totals = list(map(_count_file, jobs))
    measure: Counter = Counter()
    preposition: Counter = Counter()
    for measure_part, preposition_part in totals:
        measure.update(measure_part)
        preposition.update(preposition_part)
    names = [item.relative_path if inputs is not None else Path(item).name for item in paths]
    return {
        "provenance": {
            "attribution": "derived from Sproat & Jaitly (2016) Google TN corpus",
            "locale": locale.partition("_")[0],
            "corpus": corpus_label(Path(corpus_dir)),
            "license": "CC BY-SA 4.0",
            "shards": sorted(names),
            "feature": "f_tpg+1 from the shipped context reranker",
            "measure_population": "every MEASURE token in the training shards",
            "preposition_population": (
                "every CARDINAL, DECIMAL, or ORDINAL token immediately followed by "
                "the as-written PLAIN token in"
            ),
            "decision": _DECISION,
            "builder": "tools/build_inch_context_priors.py",
        },
        "counts": {
            "measure": dict(sorted(measure.items())),
            "preposition": dict(sorted(preposition.items())),
        },
    }


def render(document: dict) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pool", action="append", required=True, dest="pools")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    corpus_dir = args.corpus_dir or _default_corpus_dir()
    # Name the training split directly: do not enumerate, stat, or open any held-out
    # shard merely because it sits beside the build inputs.
    candidates = [corpus_dir / name for name in sorted(TRAINING_SHARDS)]
    missing = [path.name for path in candidates if not path.is_file()]
    if missing:
        raise SystemExit(
            f"the shipped table requires the complete 00--89 training set; missing {missing[0]}"
        )

    from corpus_inputs import verified_inputs, write_verification_receipt

    verified = verified_inputs(
        args.source_id,
        candidates,
        locale=args.locale,
        pools=tuple(args.pools),
        root=corpus_dir,
    )
    write_verification_receipt(args.receipt, verified, locale=args.locale, pools=tuple(args.pools))
    text = render(
        build_document(corpus_dir, locale=args.locale, inputs=verified, workers=args.workers)
    )
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"out of date: {args.out}")
            return 1
        print(f"up to date: {args.out}")
        return 0
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
