"""Build the private seen/unseen vocabulary from verified Google-TN training shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = Path(__file__).resolve().parent
for _path in (_REPO, _TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from build_spoken_priors import _default_corpus_dir  # noqa: E402
from corpus_inputs import verified_inputs  # noqa: E402
from google_tn_rows import TRAINING_SHARDS  # noqa: E402
from seen_strata import LOCALE, SOURCE_ID, build_vocabulary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--locale", choices=(LOCALE,), default=LOCALE)
    parser.add_argument("--source-id", choices=(SOURCE_ID,), default=SOURCE_ID)
    args = parser.parse_args(argv)
    corpus_dir = (args.corpus_dir or _default_corpus_dir()).resolve(strict=True)
    verified = verified_inputs(
        args.source_id,
        [corpus_dir / name for name in sorted(TRAINING_SHARDS)],
        locale=args.locale,
        pools=("training",),
        root=corpus_dir,
    )
    receipt = build_vocabulary(args.out, verified)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
