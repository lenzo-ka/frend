"""Build the measured prior for bare four-digit DATE/CARDINAL/DIGIT choices.

All Google TN training shards (00--89) are streamed.  A row is evidence only when its
written token is exactly four ASCII digits and its class is DATE, CARDINAL, or DIGIT.
Counts are stored by the token's first two digits (its century bucket) and in an
all-century row.  Held-out shards are never opened.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
for _path in (_REPO, _TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from google_tn_rows import corpus_label, full_training_set, training_shards  # noqa: E402

DEFAULT_OUT = _REPO / "frend" / "data" / "en" / "number_priors.json"
CLASSES = frozenset({"DATE", "CARDINAL", "DIGIT"})


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


def _counts(path) -> tuple[Counter, dict[str, Counter]]:
    if hasattr(path, "relative_path"):
        from corpus_inputs import open_verified

        context = open_verified(path)
    else:
        context = Path(path).open(encoding="utf-8")
    total: Counter = Counter()
    buckets: dict[str, Counter] = defaultdict(Counter)
    with context as handle:
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
    paths = _shards(Path(corpus_dir), inputs)
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
            "builder": "tools/build_number_priors.py",
        },
        "all": row(total),
        "centuries": {key: row(buckets[key]) for key in sorted(buckets)},
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
