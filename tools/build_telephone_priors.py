"""Build measured phone-shape and grouped-reading counts from Google-TN training shards."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
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

from frend.artifact_policy import (  # noqa: E402
    TELEPHONE_CONTEXT_POLICY,
    TELEPHONE_READING_POLICY,
    TELEPHONE_SHAPE_POLICY,
    policy_document,
    strict_majority,
)
from frend.spoken_priors import normalize_spoken  # noqa: E402
from frend.telephone import (  # noqa: E402
    _is_left_context_shape,
    _left_context_shape,
    _phone_groups,
    telephone_reading_key,
    telephone_shape,
)
from frend.verbalize import _number_leaf, _spoken_digits  # noqa: E402
from frend.written_forms import DigitsValue  # noqa: E402

DEFAULT_OUT = _REPO / "frend" / "data" / "en" / "telephone_priors.json"


def _default_corpus_dir() -> Path:
    from build_spoken_priors import _default_corpus_dir as spoken_default

    return spoken_default()


def _mode(group: str, spoken: str) -> str | None:
    target = normalize_spoken(spoken)
    digit_forms = {
        normalize_spoken(alternative.text): alternative.text
        for alternative in _spoken_digits(DigitsValue(group), "en_US")
    }
    if target in digit_forms:
        if "0" not in group:
            return "digits"
        return "digits-o" if " o " in f" {target} " else "digits-zero"
    if target in {
        normalize_spoken(alternative.text)
        for alternative in _number_leaf(Decimal(group), "cardinal", "en_US")
    }:
        return "cardinal"
    return None


def _piece(path):
    classes: dict[str, Counter] = defaultdict(Counter)
    readings: dict[str, Counter] = defaultdict(Counter)
    left_contexts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    unclassified = Counter()
    previous = None
    with open_corpus_input(path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                previous = None
                continue
            corpus_class, written, spoken = parts[:3]
            groups = _phone_groups(written)
            if groups is None:
                previous = (corpus_class, written)
                continue
            shape = telephone_shape(written)
            pattern = telephone_reading_key(groups)
            classes[shape][corpus_class] += 1
            if previous is not None:
                previous_class, previous_written = previous
                context_shape = _left_context_shape(previous_written)
                if context_shape is not None:
                    left_contexts[(shape, context_shape)][previous_class] += 1
            if corpus_class != "TELEPHONE":
                previous = (corpus_class, written)
                continue
            chunks = spoken.split(" sil ")
            modes = tuple(_mode(group, chunk) for group, chunk in zip(groups, chunks, strict=False))
            if len(chunks) != len(groups) or any(mode is None for mode in modes):
                unclassified[(shape, pattern)] += 1
            else:
                readings[(shape, pattern)][modes] += 1
            previous = (corpus_class, written)
    return classes, readings, left_contexts, unclassified


def build_document(corpus_dir: Path, jobs: int = 1, *, inputs=None) -> dict:
    paths = training_inputs(Path(corpus_dir), inputs)
    if jobs <= 1:
        pieces = map(_piece, paths)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pieces = list(pool.map(_piece, paths))
    classes: dict[str, Counter] = defaultdict(Counter)
    readings: dict[str, Counter] = defaultdict(Counter)
    left_contexts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    unclassified = Counter()
    for part_classes, part_readings, part_left_contexts, part_unclassified in pieces:
        for shape, counts in part_classes.items():
            classes[shape].update(counts)
        for key, counts in part_readings.items():
            readings[key].update(counts)
        for key, counts in part_left_contexts.items():
            left_contexts[key].update(counts)
        unclassified.update(part_unclassified)
    selected = {}
    for shape in sorted(classes):
        counts = classes[shape]
        patterns = {}
        for (reading_shape, pattern), pattern_readings in readings.items():
            if reading_shape != shape or sum(pattern_readings.values()) < int(
                TELEPHONE_READING_POLICY["minimum_support"]
            ):
                continue
            patterns[pattern] = {
                "readings": [
                    {"modes": list(modes), "count": count}
                    for modes, count in sorted(
                        pattern_readings.items(), key=lambda item: (-item[1], item[0])
                    )
                ],
                "unclassified_telephone": unclassified[(shape, pattern)],
            }
        if (
            not strict_majority(
                counts,
                str(TELEPHONE_SHAPE_POLICY["label"]),
                minimum_support=int(TELEPHONE_SHAPE_POLICY["minimum_support"]),
            )
            or not patterns
        ):
            continue
        selected[shape] = {
            "classes": dict(sorted(counts.items())),
            "patterns": dict(sorted(patterns.items())),
        }
    selected_contexts: dict[str, Counter] = defaultdict(Counter)
    for (shape, context), counts in left_contexts.items():
        if shape in selected:
            selected_contexts[context].update(counts)
    vetoes = {}
    for context, counts in sorted(selected_contexts.items()):
        prediction = counts.most_common(1)[0][0]
        if (
            strict_majority(
                counts,
                prediction,
                minimum_observations=int(TELEPHONE_CONTEXT_POLICY["minimum_observations"]),
            )
            and prediction not in TELEPHONE_CONTEXT_POLICY["excluded_predictions"]
        ):
            vetoes[context] = {
                "classes": dict(sorted(counts.items())),
                "prediction": prediction,
            }
    if not all(_is_left_context_shape(context) for context in vetoes):
        raise ValueError("left-context prior contains a token outside the fixed shape alphabet")
    source = corpus_provenance(corpus_dir, paths)
    return {
        "schema_version": 3,
        "locale": "en",
        "provenance": {
            "builder": "tools/build_telephone_priors.py",
            "corpus": source.corpus,
            "locale": "en",
            "shards": source.shards,
            "shape_policy": policy_document(TELEPHONE_SHAPE_POLICY),
            "reading_policy": policy_document(TELEPHONE_READING_POLICY),
            "context_policy": policy_document(TELEPHONE_CONTEXT_POLICY),
            "reading": (
                "raw counts of whole grouped readings classified against ICU cardinal and "
                "digit spellout, with zero/o from the locale lexical table"
            ),
        },
        "left_contexts": vetoes,
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
