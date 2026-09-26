"""Build or verify whether an acronym is spelled or said as a word, measured from the corpus.

An all-capitals token of two or more letters is spelled when the corpus files it as
LETTERS ("FBI" "f b i") and said as written when it files it as PLAIN ("NASA"). The
sampled shards are the spoken priors' (every tenth). Counts are kept by the token's
shape (``frend.electronic.letter_key``: case, length, whether a vowel letter occurs), by
its consonant-vowel pattern up to seven letters (``frend.letters.cv_pattern``: "cvc" is
mostly a word, "ccc" spelled), and for an acronym icukit's lexicon lists, by its own
surface ("NASA" is a word, "FBI" spelled; the surfaces are icukit's, not the corpus's);
``*`` pools all. A token icukit also reads as a Roman numeral ("II", "CD") is counted a
third way, as a numeral when the corpus files it CARDINAL or ORDINAL, under
``roman:<surface>`` for a numeral seen at least ``_ROMAN_FLOOR`` times and ``roman:*``
for all, so frend reads "II" as two and "CD" as letters. Only counts are stored.

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

from frend.electronic import letter_key  # noqa: E402
from frend.letters import cv_pattern, is_roman  # noqa: E402


def _lexicon_acronyms() -> frozenset[str]:
    """The all-capitals acronyms icukit's en_US lexicon lists (its surfaces, not corpus text)."""
    from icukit.abbreviation_compile import compile_lexicon

    compiled = compile_lexicon("en_US")
    return frozenset(
        entry.surface
        for entry in compiled.lexicon.entries
        if len(entry.surface) >= 2 and entry.surface.isalpha() and entry.surface.isupper()
    )


_OUT = _REPO / "frend" / "data" / "acronym_priors.json"
_LABELS = {"LETTERS": "spelled", "PLAIN": "word"}
_NUMERAL = frozenset({"CARDINAL", "ORDINAL"})
_ROMAN_FLOOR = 20


def build_document(corpus_dir: Path) -> dict:
    counts: dict[str, Counter] = defaultdict(Counter)
    romans: dict[str, Counter] = defaultdict(Counter)
    acronyms = _lexicon_acronyms()
    files = _files(corpus_dir)
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3 or parts[0] not in _LABELS.keys() | _NUMERAL:
                    continue
                written = parts[1]
                if len(written) < 2 or not (written.isalpha() and written.isupper()):
                    continue
                roman = is_roman(written)
                label = "numeral" if parts[0] in _NUMERAL else _LABELS[parts[0]]
                if roman:
                    romans[written][label] += 1
                if label == "numeral":
                    continue
                counts[letter_key(written)][label] += 1
                counts["*"][label] += 1
                if (pattern := cv_pattern(written)) is not None:
                    counts[f"cv:{pattern}"][label] += 1
                if written in acronyms:
                    counts[f"surface:{written}"][label] += 1
    for surface, labels in romans.items():
        counts["roman:*"].update(labels)
        if labels.total() >= _ROMAN_FLOOR:
            counts[f"roman:{surface}"].update(labels)
    return {
        "provenance": {
            "attribution": "derived from Sproat & Jaitly (2016) Google TN corpus",
            "license": "CC BY-SA 4.0",
            "shards": [path.name for path in files],
            "rule": (
                "all-capitals letter tokens of two or more letters: LETTERS spelled, PLAIN word"
            ),
            "keys": (
                "letter_key(token); cv:<consonant-vowel pattern> up to seven letters; "
                "surface:<acronym> for an acronym icukit's en_US lexicon "
                "lists; roman:<numeral> for a Roman numeral icukit reads, seen at least "
                f"{_ROMAN_FLOOR} times, numeral counting CARDINAL and ORDINAL; roman:* pools "
                "the numerals; * pools all"
            ),
        },
        "keys": {key: dict(sorted(labels.items())) for key, labels in sorted(counts.items())},
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
