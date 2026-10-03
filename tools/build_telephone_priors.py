"""Build measured phone-shape and grouped-reading counts from Google-TN training shards."""

from __future__ import annotations

import argparse
import json
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

from google_tn_rows import corpus_label, full_training_set, training_shards  # noqa: E402

from frend.spoken_priors import normalize_spoken  # noqa: E402
from frend.telephone import _phone_groups, telephone_reading_key, telephone_shape  # noqa: E402
from frend.verbalize import _number_leaf, _spoken_digits  # noqa: E402
from frend.written_forms import DigitsValue  # noqa: E402

DEFAULT_OUT = _REPO / "frend" / "data" / "en" / "telephone_priors.json"


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
    if hasattr(path, "relative_path"):
        from corpus_inputs import open_verified

        context = open_verified(path)
    else:
        context = Path(path).open(encoding="utf-8")
    classes: dict[str, Counter] = defaultdict(Counter)
    readings: dict[str, Counter] = defaultdict(Counter)
    unclassified = Counter()
    with context as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            corpus_class, written, spoken = parts[:3]
            groups = _phone_groups(written)
            if groups is None:
                continue
            shape = telephone_shape(written)
            pattern = telephone_reading_key(groups)
            classes[shape][corpus_class] += 1
            if corpus_class != "TELEPHONE":
                continue
            chunks = spoken.split(" sil ")
            modes = tuple(_mode(group, chunk) for group, chunk in zip(groups, chunks, strict=False))
            if len(chunks) != len(groups) or any(mode is None for mode in modes):
                unclassified[(shape, pattern)] += 1
            else:
                readings[(shape, pattern)][modes] += 1
    return classes, readings, unclassified


def build_document(corpus_dir: Path, jobs: int = 1, *, inputs=None) -> dict:
    paths = _shards(Path(corpus_dir), inputs)
    if jobs <= 1:
        pieces = map(_piece, paths)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pieces = list(pool.map(_piece, paths))
    classes: dict[str, Counter] = defaultdict(Counter)
    readings: dict[str, Counter] = defaultdict(Counter)
    unclassified = Counter()
    for part_classes, part_readings, part_unclassified in pieces:
        for shape, counts in part_classes.items():
            classes[shape].update(counts)
        for key, counts in part_readings.items():
            readings[key].update(counts)
        unclassified.update(part_unclassified)
    selected = {}
    for shape in sorted(classes):
        counts = classes[shape]
        telephone = counts["TELEPHONE"]
        patterns = {}
        for (reading_shape, pattern), pattern_readings in readings.items():
            if reading_shape != shape or sum(pattern_readings.values()) < 3:
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
        if telephone < 3 or 2 * telephone <= sum(counts.values()) or not patterns:
            continue
        selected[shape] = {
            "classes": dict(sorted(counts.items())),
            "patterns": dict(sorted(patterns.items())),
        }
    return {
        "schema_version": 1,
        "locale": "en",
        "provenance": {
            "builder": "tools/build_telephone_priors.py",
            "corpus": corpus_label(Path(corpus_dir)),
            "locale": "en",
            "shards": [path.relative_path if inputs is not None else path.name for path in paths],
            "selection": (
                "phone-shaped NANP digit groups; at least 3 TELEPHONE rows and a "
                "strict TELEPHONE majority among all corpus classes for the exact shape"
            ),
            "reading": (
                "raw counts of whole grouped readings classified against ICU cardinal and "
                "digit spellout, with zero/o from the locale lexical table"
            ),
        },
        "shapes": selected,
    }


def render(document: dict) -> str:
    return json.dumps(document, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--locale", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pool", action="append", required=True, dest="pools")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    corpus_dir = (args.corpus_dir or _default_corpus_dir()).resolve(strict=True)

    from corpus_inputs import verified_inputs, write_verification_receipt

    verified = verified_inputs(
        args.source_id,
        _shards(corpus_dir),
        locale=args.locale,
        pools=tuple(args.pools),
        root=corpus_dir,
    )
    write_verification_receipt(args.receipt, verified, locale=args.locale, pools=tuple(args.pools))
    rendered = render(build_document(corpus_dir, args.jobs, inputs=verified))
    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != rendered:
            print(
                f"drift: {args.out} is out of date; rerun build_telephone_priors.py",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {args.out} is up to date")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
