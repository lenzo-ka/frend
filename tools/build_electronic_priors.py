"""Build or verify how URL and email runs are said, measured from the corpus.

Each ELECTRONIC row of the sampled shards is split into runs as frend's detector splits
it (``frend.electronic.runs``), and its spoken form (the corpus's per-letter notation
joined into words) is aligned run by run: a letter run is said as a word or spelled, a
digit run as one of frend's digit readings, a separator as one spoken word. Rows that
do not align are counted and left out. The table stores integer counts only: letter
runs by shape (``letter_key``) and top-level domains by the domain itself, digit runs
by length, and each separator character's spoken words. No corpus text is stored.

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

from frend.electronic import (  # noqa: E402
    decode_letter_notation,
    digit_forms,
    digit_key,
    letter_key,
    runs,
    tld_positions,
    tld_version,
)
from frend.spoken_priors import normalize_spoken  # noqa: E402

_OUT = _REPO / "frend" / "data" / "electronic_priors.json"


def _align(written: str, spoken: str):
    parts = runs(written)
    words = normalize_spoken(decode_letter_notation(spoken)).split()
    tlds = tld_positions(parts)
    at = 0
    events = []
    for index, (kind, text) in enumerate(parts):
        if kind == "letters":
            lower = text.lower()
            if words[at : at + 1] == [lower]:
                label, width = "word", 1
            elif words[at : at + len(lower)] == list(lower):
                label, width = "spelled", len(lower)
            else:
                return None
            key = f"tld:{lower}" if index in tlds else letter_key(text)
            events.append(("letters", key, label))
        elif kind == "digits":
            for form, readings in digit_forms(text).items():
                tokens = normalize_spoken(readings[0][0]).split()
                if words[at : at + len(tokens)] == tokens:
                    events.append(("digits", digit_key(text), form))
                    width = len(tokens)
                    break
            else:
                return None
        else:
            if at >= len(words):
                return None
            events.append(("separators", text, words[at]))
            width = 1
        at += width
    return events if at == len(words) else None


def build_document(corpus_dir: Path) -> dict:
    counts = {
        "letters": defaultdict(Counter),
        "digits": defaultdict(Counter),
        "separators": defaultdict(Counter),
    }
    files = _files(corpus_dir)
    rows = aligned = 0
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3 or parts[0] != "ELECTRONIC" or parts[2] in {"<self>", "sil"}:
                    continue
                rows += 1
                events = _align(parts[1].rstrip(" ,"), parts[2])
                if events is None:
                    continue
                aligned += 1
                for table, key, label in events:
                    counts[table][key][label] += 1
                    if table != "separators":
                        counts[table]["*"][label] += 1
    return {
        "provenance": {
            "attribution": "derived from Sproat & Jaitly (2016) Google TN corpus",
            "license": "CC BY-SA 4.0",
            "shards": [path.name for path in files],
            "rows": rows,
            "rows_aligned": aligned,
            "tld_list": tld_version(),
            "keys": {
                "letters": "letter_key(run), or tld:<domain> for a top-level domain; * pools all",
                "digits": "digit_key(run); * pools all",
                "separators": "the written character",
            },
        },
        **{
            table: {key: dict(sorted(labels.items())) for key, labels in sorted(entries.items())}
            for table, entries in counts.items()
        },
    }


def _render(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rendered = _render(build_document(args.corpus_dir or _default_corpus_dir()))
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
