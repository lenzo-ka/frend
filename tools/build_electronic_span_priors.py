"""Build aggregate whole-token electronic-span support from Google-TN training."""

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

from frend.artifact_policy import ELECTRONIC_SPAN_POLICY, policy_document  # noqa: E402
from frend.electronic import _span_features  # noqa: E402

DEFAULT_OUT = _REPO / "frend" / "data" / "en" / "electronic_span_priors.json"


def _default_corpus_dir() -> Path:
    from build_spoken_priors import _default_corpus_dir as spoken_default

    return spoken_default()


def _piece(path):
    counts: dict[str, Counter] = defaultdict(Counter)
    with open_corpus_input(path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            corpus_class, written = parts[:2]
            for feature in _span_features(written):
                counts[feature][corpus_class] += 1
    return counts


def build_document(corpus_dir: Path, jobs: int = 1, *, inputs=None) -> dict:
    paths = training_inputs(Path(corpus_dir), inputs)
    if jobs <= 1:
        pieces = map(_piece, paths)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pieces = list(pool.map(_piece, paths))
    counts: dict[str, Counter] = defaultdict(Counter)
    for piece in pieces:
        for feature, classes in piece.items():
            counts[feature].update(classes)
    source = corpus_provenance(corpus_dir, paths)
    provenance = {
        "builder": "tools/build_electronic_span_priors.py",
        "corpus": source.corpus,
        "locale": "en",
        "shards": source.shards,
        "policy": policy_document(ELECTRONIC_SPAN_POLICY),
        "privacy": "aggregate feature and corpus-class counts only; no corpus text",
    }
    return {
        "schema_version": 1,
        "locale": "en",
        "provenance": provenance,
        "features": {
            feature: {"classes": dict(sorted(classes.items()))}
            for feature, classes in sorted(counts.items())
        },
    }


render = render_document


def main(argv: list[str] | None = None) -> int:
    return run_builder(
        argv,
        description=__doc__,
        default_out=DEFAULT_OUT,
        default_corpus_dir=_default_corpus_dir,
        build=lambda corpus, jobs, inputs, _locale: build_document(corpus, jobs, inputs=inputs),
    )


if __name__ == "__main__":
    raise SystemExit(main())
