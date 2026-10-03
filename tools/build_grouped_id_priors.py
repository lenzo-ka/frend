"""Build measured grouped-ID shape and reading counts from Google-TN training."""

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

from frend.grouped_ids import _groups, grouped_id_shape  # noqa: E402
from frend.letters import letter_names  # noqa: E402
from frend.spoken_priors import normalize_spoken  # noqa: E402
from frend.verbalize import _spoken_digits  # noqa: E402
from frend.written_forms import DigitsValue  # noqa: E402

DEFAULT_OUT = _REPO / "frend" / "data" / "en" / "grouped_id_priors.json"


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
    if hasattr(path, "relative_path"):
        from corpus_inputs import open_verified

        context = open_verified(path)
    else:
        context = Path(path).open(encoding="utf-8")
    classes: dict[str, Counter] = defaultdict(Counter)
    readings: dict[str, Counter] = defaultdict(Counter)
    with context as handle:
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
    paths = _shards(Path(corpus_dir), inputs)
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
    provenance = {
        "builder": "tools/build_grouped_id_priors.py",
        "corpus": corpus_label(Path(corpus_dir)),
        "locale": "en",
        "shards": [path.relative_path if inputs is not None else path.name for path in paths],
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
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else None
        if current != rendered:
            print(f"drift: {args.out} is out of date", file=sys.stderr)
            return 1
        print(f"ok: {args.out} is up to date")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
