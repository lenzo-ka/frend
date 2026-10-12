"""Build contextual counts distinguishing bare inches from prepositional ``in``.

All Google TN training shards (00--89) are streamed. For every MEASURE row, the
following written token is classified with the shipped context reranker's learned word
classes. The comparison population is every CARDINAL, DECIMAL, or ORDINAL row whose
next row is the as-written PLAIN token ``in``; for those pairs, the token after ``in``
is classified. Held-out shards are never opened.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
for _path in (_REPO, _TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from corpus_builder import (  # noqa: E402
    corpus_provenance,
    open_corpus_input,
    render_document,
    run_builder,
    training_inputs,
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


def _count_file(job) -> tuple[Counter, Counter]:
    item, locale, frequent = job
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

    with open_corpus_input(item) as handle:
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
    paths = training_inputs(Path(corpus_dir), inputs)
    model = context_model(locale)
    if model is None:
        raise ValueError(f"{locale} has no shipped context model")
    jobs = [(item, locale, model.frequent) for item in paths]
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
    source = corpus_provenance(corpus_dir, paths)
    return {
        "provenance": {
            "attribution": "derived from Sproat & Jaitly (2016) Google TN corpus",
            "locale": locale.partition("_")[0],
            "corpus": source.corpus,
            "license": "CC BY-SA 4.0",
            "shards": source.shards,
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


render = partial(render_document, indent=2)


def main(argv: list[str] | None = None) -> int:
    return run_builder(
        argv,
        description=__doc__.splitlines()[0],
        default_out=DEFAULT_OUT,
        default_corpus_dir=_default_corpus_dir,
        build=lambda corpus, workers, inputs, locale: build_document(
            corpus, locale=locale, inputs=inputs, workers=workers
        ),
        render=render,
        parallel_option="--workers",
        parallel_default=max(1, (os.cpu_count() or 2) // 2),
    )


if __name__ == "__main__":
    raise SystemExit(main())
