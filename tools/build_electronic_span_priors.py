"""Build aggregate whole-token electronic-span support from Google-TN training."""

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

from frend.electronic import _span_features  # noqa: E402

DEFAULT_OUT = _REPO / "frend" / "data" / "en" / "electronic_span_priors.json"


def _default_corpus_dir() -> Path:
    from build_spoken_priors import _default_corpus_dir as spoken_default

    return spoken_default()


def _shards(corpus_dir: Path, inputs=None) -> list:
    if inputs is not None:
        paths = sorted(inputs, key=lambda item: item.relative_path)
    else:
        available = sorted(corpus_dir.glob("output-*-of-*"))
        paths = full_training_set(available) or training_shards(available)
    if not paths:
        raise FileNotFoundError(f"no training shards under {corpus_dir}")
    return paths


def _piece(path):
    if hasattr(path, "relative_path"):
        from corpus_inputs import open_verified

        context = open_verified(path)
    else:
        context = Path(path).open(encoding="utf-8")
    counts: dict[str, Counter] = defaultdict(Counter)
    with context as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            corpus_class, written = parts[:2]
            for feature in _span_features(written):
                counts[feature][corpus_class] += 1
    return counts


def build_document(corpus_dir: Path, jobs: int = 1, *, inputs=None) -> dict:
    paths = _shards(Path(corpus_dir), inputs)
    if jobs <= 1:
        pieces = map(_piece, paths)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pieces = list(pool.map(_piece, paths))
    counts: dict[str, Counter] = defaultdict(Counter)
    for piece in pieces:
        for feature, classes in piece.items():
            counts[feature].update(classes)
    provenance = {
        "builder": "tools/build_electronic_span_priors.py",
        "corpus": corpus_label(Path(corpus_dir)),
        "locale": "en",
        "shards": [path.relative_path if inputs is not None else path.name for path in paths],
        "selection": (
            "feature enabled with at least 3 ELECTRONIC rows and a strict ELECTRONIC "
            "majority over all corpus rows"
        ),
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
