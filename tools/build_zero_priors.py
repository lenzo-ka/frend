"""Build or verify how the corpus says a zero digit, "o", "oh" or "zero", by reading kind.

frend offers each spoken zero both ways where both are said ("3.05": "three point zero
five", "three point o five"; "1905": ICU's "nineteen oh-five" and "nineteen o five");
this table says which comes first. It counts the zero words in the corpus's spoken
form of each token whose written form has a "0": after "point" for a decimal, measure
or money (the integer part is always "zero"), anywhere for a date, time, digit run or
electronic token. The sampled shards are the spoken priors' (every tenth). Only counts
are stored.

``--check`` repeats the sample and compares the JSON byte for byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from build_spoken_priors import _default_corpus_dir, _files  # noqa: E402
from google_tn_rows import corpus_label  # noqa: E402

_OUT = _REPO / "frend" / "data" / "en" / "zero_priors.json"
_ZEROS = ("o", "oh", "zero")
_DIGIT_WORDS = frozenset(
    {*_ZEROS, "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
)
# The corpus class and the kind frend measures it under; a fractional kind counts only
# the zeros after "point".
_KINDS = {
    "DECIMAL": ("decimal", True),
    "MEASURE": ("measure", True),
    "MONEY": ("money", True),
    "DATE": ("date", False),
    "TIME": ("time", False),
    "DIGIT": ("digit", False),
    "ELECTRONIC": ("electronic", False),
}


def _zero_words(spoken: str, fractional: bool) -> list[str]:
    words = spoken.split(" ")
    if not fractional:
        return [word for word in words if word in _ZEROS]
    found, after_point = [], False
    for word in words:
        if word == "point":
            after_point = True
        elif after_point and word in _ZEROS:
            found.append(word)
        elif word not in _DIGIT_WORDS:
            after_point = False
    return found


def build_document(corpus_dir: Path, *, inputs=None) -> dict:
    counts: dict[str, Counter] = defaultdict(Counter)
    files = list(inputs) if inputs is not None else _files(corpus_dir)
    for path in files:
        if inputs is None:
            handle_context = path.open(encoding="utf-8")
        else:
            from corpus_inputs import open_verified

            handle_context = open_verified(path)
        with handle_context as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3 or parts[0] not in _KINDS or "0" not in parts[1]:
                    continue
                kind, fractional = _KINDS[parts[0]]
                counts[kind].update(_zero_words(parts[2], fractional))
    return {
        "provenance": {
            "attribution": "derived from Sproat & Jaitly (2016) Google TN corpus",
            "locale": "en",
            "corpus": corpus_label(corpus_dir),
            "license": "CC BY-SA 4.0",
            "shards": [
                path.relative_path if inputs is not None else path.name for path in files
            ],
            "rule": (
                "zero words (o, oh, zero) in the spoken form of a token written with 0: "
                "after 'point' for DECIMAL, MEASURE, MONEY; anywhere for DATE, TIME, "
                "DIGIT, ELECTRONIC"
            ),
        },
        "kinds": {kind: dict(sorted(words.items())) for kind, words in sorted(counts.items())},
    }


def _render(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pool", action="append", required=True, dest="pools")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    corpus_dir = args.corpus_dir or _default_corpus_dir()
    from corpus_inputs import verified_inputs, write_verification_receipt

    verified = verified_inputs(
        args.source_id,
        _files(corpus_dir),
        locale=args.locale,
        pools=tuple(args.pools),
        root=corpus_dir,
    )
    write_verification_receipt(
        args.receipt, verified, locale=args.locale, pools=tuple(args.pools)
    )
    rendered = _render(build_document(corpus_dir, inputs=verified))
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != rendered:
            print(f"out of date: {args.out}")
            return 1
        print(f"up to date: {args.out}")
        return 0
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
