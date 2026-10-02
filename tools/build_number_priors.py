"""Build the measured prior for bare four-digit DATE/CARDINAL/DIGIT choices.

All Google TN training shards (00--89) are streamed.  A row is evidence only when its
written token is exactly four ASCII digits and its class is DATE, CARDINAL, or DIGIT.
The default counts are stored by century. The off-by-default ``google-tn`` profile
selects an aggregate key on shards 80--89 after fitting candidates on 00--79, then
stores the selected key's raw counts from all 00--89. Held-out shards are never opened.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
for _path in (_REPO, _TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from google_tn_rows import (  # noqa: E402
    corpus_label,
    expected,
    full_training_set,
    training_shards,
)

from frend.number_priors import _profile_keys  # noqa: E402
from frend.spoken_priors import normalize_spoken  # noqa: E402
from frend.verbalize import _spoken_digits, _year_leaf  # noqa: E402
from frend.written_forms import DigitsValue  # noqa: E402

DEFAULT_OUT = _REPO / "frend" / "data" / "en" / "number_priors.json"
CLASSES = frozenset({"DATE", "CARDINAL", "DIGIT"})
NAMES = ("date", "cardinal", "digit")
PROFILE_KEYS = tuple(_profile_keys("2012", "", ""))
SELECTION_TRAINING = frozenset(f"output-{index:05d}-of-00100" for index in range(80))
DEVELOPMENT = frozenset(f"output-{index:05d}-of-00100" for index in range(80, 90))


@lru_cache(maxsize=10000)
def _reading_choices(written: str) -> dict[str, str]:
    """Normalized offered text -> year/cardinal/digit class, in runtime order."""
    alternatives = [
        *_year_leaf(Decimal(written), "en_US"),
        *_spoken_digits(DigitsValue(written), "en_US"),
    ]
    choices: dict[str, str] = {}
    for item in alternatives:
        if "numbering-year" in item.provenance:
            choice = "date"
        elif item.provenance.startswith("icu-rbnf:%spellout-cardinal"):
            choice = "digit"
        else:
            choice = "cardinal"
        choices.setdefault(normalize_spoken(item.text), choice)
    return choices


def _reading_choice(corpus_class: str, written: str, spoken: str) -> str | None:
    return _reading_choices(written).get(expected(corpus_class, written, spoken))


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


def _counts(path):
    if hasattr(path, "relative_path"):
        from corpus_inputs import open_verified

        context = open_verified(path)
        name = path.relative_path
    else:
        context = Path(path).open(encoding="utf-8")
        name = Path(path).name
    total: Counter = Counter()
    buckets: dict[str, Counter] = defaultdict(Counter)
    all_models = {model: defaultdict(Counter) for model in PROFILE_KEYS}
    training_models = {model: defaultdict(Counter) for model in PROFILE_KEYS}
    training_centuries: dict[str, Counter] = defaultdict(Counter)
    development_models = {model: defaultdict(Counter) for model in PROFILE_KEYS}

    def count_sentence(sentence: list[tuple[str, str, str]]) -> None:
        for index, (corpus_class, written, spoken) in enumerate(sentence):
            if (
                corpus_class not in CLASSES
                or len(written) != 4
                or not written.isascii()
                or not written.isdigit()
            ):
                continue
            choice = corpus_class.lower()
            before = sentence[index - 1][1] if index else ""
            after = sentence[index + 1][1] if index + 1 < len(sentence) else ""
            century = written[:2]
            total[choice] += 1
            buckets[century][choice] += 1
            profile_choice = _reading_choice(corpus_class, written, spoken)
            if profile_choice is None:
                continue
            keys = _profile_keys(written, before, after)
            for model, key in keys.items():
                all_models[model][key][profile_choice] += 1
                if name in SELECTION_TRAINING:
                    training_models[model][key][profile_choice] += 1
                elif name in DEVELOPMENT:
                    development_models[model][(century, key)][profile_choice] += 1
            if name in SELECTION_TRAINING:
                training_centuries[century][profile_choice] += 1

    sentence: list[tuple[str, str, str]] = []
    with context as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "<eos>":
                count_sentence(sentence)
                sentence = []
            elif len(parts) >= 3:
                sentence.append((parts[0], parts[1], parts[2]))
    count_sentence(sentence)
    return (
        total,
        buckets,
        all_models,
        training_models,
        training_centuries,
        development_models,
    )


def build_document(corpus_dir: Path, jobs: int = 1, *, inputs=None) -> dict:
    paths = _shards(Path(corpus_dir), inputs)
    if jobs <= 1:
        pieces = map(_counts, paths)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pieces = list(pool.map(_counts, paths))
    total: Counter = Counter()
    buckets: dict[str, Counter] = defaultdict(Counter)
    all_models = {model: defaultdict(Counter) for model in PROFILE_KEYS}
    training_models = {model: defaultdict(Counter) for model in PROFILE_KEYS}
    training_centuries: dict[str, Counter] = defaultdict(Counter)
    development_models = {model: defaultdict(Counter) for model in PROFILE_KEYS}
    for (
        part_total,
        part_buckets,
        part_all_models,
        part_training_models,
        part_training_centuries,
        part_development_models,
    ) in pieces:
        total.update(part_total)
        for key, counts in part_buckets.items():
            buckets[key].update(counts)
        for century, counts in part_training_centuries.items():
            training_centuries[century].update(counts)
        for target, source in (
            (all_models, part_all_models),
            (training_models, part_training_models),
            (development_models, part_development_models),
        ):
            for model, model_rows in source.items():
                for key, counts in model_rows.items():
                    target[model][key].update(counts)

    def row(counts: Counter) -> dict[str, int]:
        return {name: counts[name] for name in NAMES if counts[name]}

    def first(counts: Counter) -> str | None:
        if not counts:
            return None
        return max(NAMES, key=lambda name: (counts[name], -NAMES.index(name)))

    development_total = sum(counts.total() for counts in development_models["century"].values())
    scores = {}
    for model in PROFILE_KEYS:
        correct = 0
        unseen = 0
        for (century, key), actual in development_models[model].items():
            measured = training_models[model].get(key)
            if measured is None:
                unseen += actual.total()
                measured = training_centuries.get(century)
            prediction = first(measured or Counter())
            if prediction is not None:
                correct += actual[prediction]
        scores[model] = {"correct": correct, "unseen": unseen}
    selected = max(PROFILE_KEYS, key=lambda model: scores[model]["correct"])

    return {
        "locale": "en",
        "schema_version": 1,
        "provenance": {
            "locale": "en",
            "corpus": corpus_label(Path(corpus_dir)),
            "shards": [p.relative_path if inputs is not None else p.name for p in paths],
            "unit": (
                "Google TN rows whose written token is exactly four ASCII digits and "
                "whose corpus class is DATE, CARDINAL, or DIGIT"
            ),
            "bucket": "the token's first two digits; raw counts, with no year window",
            "profile_bucket": (
                "development-selected decade, numeric structure and neighboring context "
                "classes; raw counts, with century fallback"
            ),
            "profile_unit": (
                "eligible rows classified by the year, cardinal or digit alternative "
                "whose normalized text matches the corpus spoken form"
            ),
            "builder": "tools/build_number_priors.py",
        },
        "all": row(total),
        "centuries": {key: row(buckets[key]) for key in sorted(buckets)},
        "google_tn_profile": {
            "selected_key": selected,
            "selection": {
                "objective": "maximize first-choice DATE/CARDINAL/DIGIT labels",
                "selection_training_shards": sorted(SELECTION_TRAINING),
                "development_shards": sorted(DEVELOPMENT),
                "development_tokens": development_total,
                "candidates": {model: scores[model] for model in PROFILE_KEYS},
            },
            "keys": {key: row(all_models[selected][key]) for key in sorted(all_models[selected])},
        },
    }


def render(document: dict) -> str:
    return json.dumps(document, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pool", action="append", required=True, dest="pools")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    corpus_dir = args.corpus_dir or _default_corpus_dir()
    from corpus_inputs import verified_inputs, write_verification_receipt

    paths = _shards(corpus_dir)
    verified = verified_inputs(
        args.source_id,
        paths,
        locale=args.locale,
        pools=tuple(args.pools),
        root=corpus_dir,
    )
    write_verification_receipt(args.receipt, verified, locale=args.locale, pools=tuple(args.pools))
    text = render(build_document(corpus_dir, args.jobs, inputs=verified))
    if args.check:
        shipped = args.out.read_text(encoding="utf-8") if args.out.exists() else None
        if shipped != text:
            print(f"--check: {args.out} differs from a rebuild")
            return 1
        print(f"--check: {args.out} matches a rebuild byte for byte")
        return 0
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
