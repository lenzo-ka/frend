"""Build the measured prior for bare four-digit DATE/CARDINAL/DIGIT choices.

All Google TN training shards (00--89) are streamed.  A row is evidence only when its
written token is exactly four ASCII digits and its class is DATE, CARDINAL, or DIGIT.
Counts are stored by the token's first two digits (its century bucket) and in an
all-century row.  Held-out shards are never opened.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
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

DEFAULT_OUT = _REPO / "frend" / "data" / "en" / "number_priors.json"
CLASSES = frozenset({"DATE", "CARDINAL", "DIGIT"})


def _default_corpus_dir() -> Path:
    from build_spoken_priors import _default_corpus_dir as spoken_default

    return spoken_default()


def _counts(path) -> tuple[Counter, dict[str, Counter]]:
    total: Counter = Counter()
    buckets: dict[str, Counter] = defaultdict(Counter)
    with open_corpus_input(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] not in CLASSES:
                continue
            written = parts[1]
            if len(written) != 4 or not written.isascii() or not written.isdigit():
                continue
            choice = parts[0].lower()
            total[choice] += 1
            buckets[written[:2]][choice] += 1
    return total, buckets


def build_document(corpus_dir: Path, jobs: int = 1, *, inputs=None) -> dict:
    paths = training_inputs(Path(corpus_dir), inputs)
    if jobs <= 1:
        pieces = map(_counts, paths)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pieces = list(pool.map(_counts, paths))
    total: Counter = Counter()
    buckets: dict[str, Counter] = defaultdict(Counter)
    for part_total, part_buckets in pieces:
        total.update(part_total)
        for key, counts in part_buckets.items():
            buckets[key].update(counts)
    names = ("date", "cardinal", "digit")

    def row(counts: Counter) -> dict[str, int]:
        return {name: counts[name] for name in names if counts[name]}

    source = corpus_provenance(corpus_dir, paths)
    return {
        "locale": "en",
        "schema_version": 1,
        "provenance": {
            "locale": "en",
            "corpus": source.corpus,
            "shards": source.shards,
            "unit": (
                "Google TN rows whose written token is exactly four ASCII digits and "
                "whose corpus class is DATE, CARDINAL, or DIGIT"
            ),
            "bucket": "the token's first two digits; raw counts, with no year window",
            "builder": "tools/build_number_priors.py",
        },
        "all": row(total),
        "centuries": {key: row(buckets[key]) for key in sorted(buckets)},
    }


render = render_document


def main(argv: list[str] | None = None) -> int:
    return run_builder(
        argv,
        description=__doc__.splitlines()[0],
        default_out=DEFAULT_OUT,
        default_corpus_dir=_default_corpus_dir,
        build=lambda corpus, jobs, inputs, _locale: build_document(corpus, jobs, inputs=inputs),
    )


if __name__ == "__main__":
    raise SystemExit(main())
