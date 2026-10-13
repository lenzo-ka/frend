"""Build or verify spell-or-say exceptions for short letter tokens.

The population is exactly :func:`frend.letters.is_spelled_token`. A row is relevant
only when its class and literal normalized reading agree. All training shards 00--89
are measured. The shipped lexicon keeps only majority outcomes that disagree with
the cheap rule (a token with an a/e/i/o/u is said; otherwise it is spelled).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from build_spoken_priors import _default_corpus_dir  # noqa: E402
from google_tn_rows import corpus_label, full_training_set  # noqa: E402

from frend.letters import is_spelled_token  # noqa: E402
from tools.literal_spell_say import literal_spell_say, normalize_literal  # noqa: E402

_OUT = _REPO / "frend" / "data" / "en" / "spelled_token_priors.json"
_ACRONYM_CASES = frozenset({"SOS", "USA", "ATM", "FBI", "NATO", "NASA", "SCUBA"})


def _count_file(
    job,
) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, dict[str, int]]]:
    item, locale, verified = job
    if verified:
        from corpus_inputs import open_verified

        handle_context = open_verified(item)
    else:
        handle_context = Path(item).open(encoding="utf-8")
    counts: dict[str, Counter] = defaultdict(Counter)
    outcomes = Counter()
    cases: dict[str, Counter] = defaultdict(Counter)
    with handle_context as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if parts[1] in _ACRONYM_CASES:
                case = _literal_reading(parts[0], parts[1], parts[2], locale, bounded=False)
                if case is not None:
                    cases[parts[1]][case] += 1
            label = literal_outcome(parts[0], parts[1], parts[2], locale)
            if label is None:
                continue
            token = normalize_literal(parts[1])
            counts[token][label] += 1
            outcomes[label] += 1
    return (
        {token: dict(labels) for token, labels in counts.items()},
        dict(outcomes),
        {token: dict(labels) for token, labels in cases.items()},
    )


def _literal_reading(
    corpus_class: str, written: str, spoken: str, locale: str, *, bounded: bool
) -> str | None:
    """Literal ``spell``/``say`` label, optionally restricted to the new population."""
    return literal_spell_say(
        corpus_class,
        written,
        spoken,
        locale,
        eligible=(lambda token, language: is_spelled_token(token, language))
        if bounded
        else (lambda _token, _language: True),
    )


def literal_outcome(corpus_class: str, written: str, spoken: str, locale: str) -> str | None:
    """Return ``spell`` or ``say`` only for a literal relevant corpus row."""
    return _literal_reading(corpus_class, written, spoken, locale, bounded=True)


def rule_outcome(token: str) -> str:
    """The selected cheap decision: say an AEIOU token, spell a consonant-only one."""
    return "say" if any(letter in "aeiou" for letter in token.casefold()) else "spell"


def build_document(
    corpus_dir: Path,
    *,
    locale: str = "en_US",
    inputs=None,
    workers: int = 1,
) -> dict:
    if inputs is not None:
        files = list(inputs)
    else:
        available = sorted(Path(corpus_dir).glob("output-*-of-*"))
        files = full_training_set(available) or available
    counts: dict[str, Counter] = defaultdict(Counter)
    outcomes = Counter()
    cases: dict[str, Counter] = defaultdict(Counter)
    jobs = [(item, locale, inputs is not None) for item in files]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            partials = pool.map(_count_file, jobs)
            for partial, labels, partial_cases in partials:
                outcomes.update(labels)
                for token, token_counts in partial.items():
                    counts[token].update(token_counts)
                for token, token_counts in partial_cases.items():
                    cases[token].update(token_counts)
    else:
        for partial, labels, partial_cases in map(_count_file, jobs):
            outcomes.update(labels)
            for token, token_counts in partial.items():
                counts[token].update(token_counts)
            for token, token_counts in partial_cases.items():
                cases[token].update(token_counts)
    tokens = {}
    for token, labels in sorted(counts.items()):
        total = labels.total()
        spell_count, say_count = labels["spell"], labels["say"]
        majority = "spell" if spell_count > say_count else "say"
        if majority == rule_outcome(token):
            continue
        tokens[token] = {
            "counts": {"spell": spell_count, "say": say_count},
            "shares": {"spell": spell_count / total, "say": say_count / total},
        }
    names = [item.relative_path if inputs is not None else Path(item).name for item in files]
    return {
        "provenance": {
            "attribution": "derived from Sproat & Jaitly (2016) Google TN corpus",
            "locale": locale.partition("_")[0],
            "corpus": corpus_label(corpus_dir),
            "license": "CC BY-SA 4.0",
            "training_shards": sorted(names),
            "shards": sorted(names),
            "population": "2..6 locale-script letters; all-uppercase acronym runs excluded",
            "rule": "say iff token contains a/e/i/o/u; y remains a consonant",
            "relevance": (
                "LETTERS iff normalized spoken equals authoritative letter names; "
                "PLAIN iff normalized spoken equals normalized written; all else dropped"
            ),
            "outcomes": dict(sorted(outcomes.items())),
            "full_measured_tokens": len(counts),
            "exceptions": len(tokens),
            "acronym_cases": {
                token: dict(sorted(cases[token].items())) for token in sorted(_ACRONYM_CASES)
            },
        },
        "tokens": tokens,
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
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args(argv)
    corpus_dir = args.corpus_dir or _default_corpus_dir()
    from corpus_inputs import verified_inputs, write_verification_receipt

    candidates = full_training_set(sorted(corpus_dir.glob("output-*-of-*")))
    if candidates is None:
        raise SystemExit("the shipped table requires the complete 00--89 training set")
    verified = verified_inputs(
        args.source_id,
        candidates,
        locale=args.locale,
        pools=tuple(args.pools),
        root=corpus_dir,
    )
    write_verification_receipt(args.receipt, verified, locale=args.locale, pools=tuple(args.pools))
    rendered = _render(
        build_document(
            corpus_dir,
            locale=args.locale,
            inputs=verified,
            workers=args.workers,
        )
    )
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
