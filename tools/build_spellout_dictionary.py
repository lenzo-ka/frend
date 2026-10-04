"""Build or verify the compact case-preserved spell-or-say dictionary.

The table contains only Google TN decisions that differ from the AEIOU fallback for
that exact surface. Counts preserve the measured strength without repeating provenance
on every row; no sentence or context enters the table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import icu

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from build_spoken_priors import _default_corpus_dir  # noqa: E402
from google_tn_rows import corpus_label, full_training_set  # noqa: E402

from frend.letters import is_spelled_token, spelled_token_rule, split_acronym_surface  # noqa: E402

_NFC = icu.Normalizer2.getNFCInstance()
_OUT = _REPO / "frend" / "data" / "en" / "spellout_dictionary.json"

GOOGLE_SOURCE = "google/tn-en_with_types"
MINIMUM_SUPPORT = 5
MINIMUM_PURITY = 0.99


def _eligible(surface: str, locale: str = "en_US") -> bool:
    return split_acronym_surface(surface) is not None or is_spelled_token(surface, locale)


def _count_file(job) -> dict[str, dict[str, int]]:
    item, locale, verified = job
    if verified:
        from corpus_inputs import open_verified

        handle_context = open_verified(item)
    else:
        handle_context = Path(item).open(encoding="utf-8")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with handle_context as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] not in {"LETTERS", "PLAIN"}:
                continue
            surface = _NFC.normalize(parts[1])
            if _eligible(surface, locale):
                counts[surface]["spell" if parts[0] == "LETTERS" else "say"] += 1
    return {surface: dict(labels) for surface, labels in counts.items()}


def google_counts(corpus_dir: Path, *, locale="en_US", inputs=None, workers=1):
    if inputs is None:
        available = sorted(Path(corpus_dir).glob("output-*-of-*"))
        files = full_training_set(available) or available
    else:
        files = list(inputs)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    jobs = [(item, locale, inputs is not None) for item in files]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for partial in pool.map(_count_file, jobs):
                for surface, labels in partial.items():
                    counts[surface].update(labels)
    else:
        for partial in map(_count_file, jobs):
            for surface, labels in partial.items():
                counts[surface].update(labels)
    return counts


def _decision(counts: Counter[str]) -> str | None:
    total = counts.total()
    if total < MINIMUM_SUPPORT or counts["spell"] == counts["say"]:
        return None
    decision = "spell" if counts["spell"] > counts["say"] else "say"
    return decision if counts[decision] / total >= MINIMUM_PURITY else None


def build_document(
    corpus_dir: Path,
    *,
    locale="en_US",
    inputs=None,
    workers=1,
) -> dict:
    raw_counts = google_counts(corpus_dir, locale=locale, inputs=inputs, workers=workers)
    rows = []
    for surface, counts in sorted(raw_counts.items()):
        decision = _decision(counts)
        if decision is None or decision == spelled_token_rule(surface):
            continue
        rows.append([surface, decision, counts["say"], counts["spell"]])
    names = [
        item.relative_path if inputs is not None else Path(item).name for item in (inputs or [])
    ]
    return {
        "provenance": {
            "columns": ["surface", "decision", "say_count", "spell_count"],
            "corpus": corpus_label(corpus_dir),
            "locale": locale.partition("_")[0],
            "privacy": "rows retain only token, decision, and aggregate counts",
            "selection": {
                "development_shards": [f"output-{index:05d}-of-00100" for index in range(90, 95)],
                "minimum_purity": MINIMUM_PURITY,
                "minimum_support": MINIMUM_SUPPORT,
                "objective": "maximize S0 first-choice spell-versus-say labels",
                "rule": "retain only decisions differing from the exact surface AEIOU fallback",
            },
            "license": "CC-BY-SA-4.0",
            "source": GOOGLE_SOURCE,
            "training_shards": sorted(names),
        },
        "tokens": rows,
    }


def _render(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


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
        raise SystemExit("the shipped dictionary requires the complete 00--89 training set")
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
