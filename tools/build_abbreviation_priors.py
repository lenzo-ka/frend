"""Build or verify how the corpus says each abbreviation, by its fold and written case.

The keys are ``frend.abbreviation_variants.measured_keys``: each variant source's (a
lexicon abbreviation ending in ".", with a lowercase letter, no inner period and a fold
longer than one letter; 135 in icukit's en_US lexicon) and each dotted chain's with an
expansion ("e.g.", "U.S."). A key is the written token's ICU lower case, one trailing
period removed. Every corpus row whose written token folds to a key is counted under
the token's written case (``lower``, ``title`` or ``upper``; a token in none of the
three is not counted) by what the corpus says (``abbreviation_variants.category``):
``spelled`` for its letters one by one, ``as-written`` for the token itself, the
lexicon expansion it says ("saint", "street"), else ``other``. Rows of every class are
counted. The sampled shards are the spoken priors' (every tenth training shard). Only
counts are stored.

``--check`` repeats the count and compares the JSON byte for byte.
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
from google_tn_rows import corpus_label, expected  # noqa: E402

from frend.abbreviation_variants import (  # noqa: E402
    category,
    fold_key,
    measured_keys,
    written_case,
)
from frend.spoken_priors import SUB_KEY_PRIOR_STRENGTH  # noqa: E402

_OUT = _REPO / "frend" / "data" / "en" / "abbreviation_priors.json"
_LOCALE = "en_US"


def build_document(corpus_dir: Path, *, inputs=None) -> dict:
    expansions = measured_keys(_LOCALE)
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
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
                if len(parts) < 3:
                    continue
                corpus_class, written, spoken = parts[0], parts[1], parts[2]
                stem = written[:-1] if written.endswith(".") else written
                if stem.lower() not in expansions:
                    continue
                key = fold_key(written, _LOCALE)
                if key not in expansions:
                    continue
                case = written_case(written, _LOCALE)
                if case is None:
                    continue
                said = expected(corpus_class, written, spoken)
                counts[key][case][category(written, said, expansions[key], _LOCALE)] += 1
    return {
        "provenance": {
            "attribution": "derived from Sproat & Jaitly (2016) Google TN corpus",
            "locale": "en",
            "corpus": corpus_label(corpus_dir),
            "license": "CC BY-SA 4.0",
            "shards": [path.relative_path if inputs is not None else path.name for path in files],
            "builder": "tools/build_abbreviation_priors.py",
            "sources": (
                "icukit en_US lexicon surfaces ending in '.', holding a lowercase letter, "
                "with no inner '.', folding to more than one letter; and its dotted chains "
                "of single letters with an expansion"
            ),
            "key": "ICU toLower(en) of the written token, one trailing period removed",
            "sub_key": (
                "written case: lower, title, upper; blended toward the key share with "
                f"strength {SUB_KEY_PRIOR_STRENGTH}; a case the corpus never writes for a "
                "key has no row, and a surface in that case reads the key"
            ),
            "categories": (
                "spelled when the corpus says the letters one by one (a dotted chain said "
                "as written is spelled); as-written when it says the token itself; the "
                "lexicon expansion it says; other for anything else"
            ),
            "normalization": (
                "the corpus writes no variant key with a period, and some keys (st, dr, mr, "
                "...) in lower and upper case only; period is therefore unmeasured, and "
                "title case is measured only where the corpus writes it"
            ),
        },
        "keys": {
            key: {case: dict(sorted(labels.items())) for case, labels in sorted(cases.items())}
            for key, cases in sorted(counts.items())
        },
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
    write_verification_receipt(args.receipt, verified, locale=args.locale, pools=tuple(args.pools))
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
