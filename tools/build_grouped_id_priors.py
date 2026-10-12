"""Build measured grouped-ID shape and reading counts from Google-TN training."""

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

from frend.grouped_ids import _groups, grouped_id_shape  # noqa: E402
from frend.letters import letter_names  # noqa: E402
from frend.spoken_priors import normalize_spoken  # noqa: E402
from frend.verbalize import _spoken_digits  # noqa: E402
from frend.written_forms import DigitsValue  # noqa: E402

DEFAULT_OUT = _REPO / "frend" / "data" / "en" / "grouped_id_priors.json"


def _default_corpus_dir() -> Path:
    from build_spoken_priors import _default_corpus_dir as spoken_default

    return spoken_default()


def _spoken_forms(group: str) -> set[str]:
    if group == "X":
        names = letter_names(group, "en_US")
        return set() if names is None else {normalize_spoken(" ".join(names.spoken))}
    return {normalize_spoken(item.text) for item in _spoken_digits(DigitsValue(group), "en_US")}


def _is_grouped(groups: tuple[str, ...], spoken: str) -> bool:
    chunks = spoken.split(" sil ")
    return len(groups) == len(chunks) and all(
        normalize_spoken(chunk) in _spoken_forms(group)
        for group, chunk in zip(groups, chunks, strict=True)
    )


def _piece(path):
    classes: dict[str, Counter] = defaultdict(Counter)
    readings: dict[str, Counter] = defaultdict(Counter)
    with open_corpus_input(path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            corpus_class, written, spoken = parts[:3]
            groups = _groups(written)
            if groups is None:
                continue
            shape = grouped_id_shape(written)
            classes[shape][corpus_class] += 1
            key = "grouped" if _is_grouped(groups, spoken) else "other"
            readings[shape][key] += 1
    return classes, readings


def build_document(corpus_dir: Path, jobs: int = 1, *, inputs=None) -> dict:
    paths = training_inputs(Path(corpus_dir), inputs)
    if jobs <= 1:
        pieces = map(_piece, paths)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pieces = list(pool.map(_piece, paths))
    classes: dict[str, Counter] = defaultdict(Counter)
    readings: dict[str, Counter] = defaultdict(Counter)
    for part_classes, part_readings in pieces:
        for shape, counts in part_classes.items():
            classes[shape].update(counts)
        for shape, counts in part_readings.items():
            readings[shape].update(counts)
    selected = {}
    for shape, counts in sorted(readings.items()):
        grouped = counts["grouped"]
        if grouped < 3 or 2 * grouped <= sum(counts.values()):
            continue
        selected[shape] = {
            "classes": dict(sorted(classes[shape].items())),
            "readings": {"grouped": grouped, "other": counts["other"]},
        }
    source = corpus_provenance(corpus_dir, paths)
    provenance = {
        "builder": "tools/build_grouped_id_priors.py",
        "corpus": source.corpus,
        "locale": "en",
        "shards": source.shards,
        "selection": (
            "exact ASCII digit-group shape with at least 3 grouped-reading rows "
            "and a strict grouped-reading majority over all corpus rows"
        ),
        "reading": (
            "one ICU digit name per written digit, locale lexical o for zero, "
            "and literal sil between written groups"
        ),
    }
    return {
        "schema_version": 1,
        "locale": "en",
        "provenance": provenance,
        "shapes": selected,
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
