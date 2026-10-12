"""Shared execution boundary for corpus-derived Google-TN table builders."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, TextIO

from corpus_inputs import VerifiedInput, open_verified, verified_inputs, write_verification_receipt
from google_tn_rows import HELD_OUT_SHARDS, corpus_label, full_training_set, training_shards


class CorpusProvenance(NamedTuple):
    corpus: str
    shards: list[str]


def _input_name(item: Path | VerifiedInput) -> str:
    return item.relative_path if isinstance(item, VerifiedInput) else Path(item).name


def training_inputs(corpus_dir: Path, inputs: Sequence[Path | VerifiedInput] | None = None) -> list:
    """Return training inputs in name order and reject any explicitly supplied holdout."""
    corpus_dir = Path(corpus_dir)
    if inputs is None:
        available = sorted(corpus_dir.glob("output-*-of-*"))
        paths = full_training_set(available) or training_shards(available)
    else:
        paths = sorted(inputs, key=_input_name)
        held_out = [_input_name(item) for item in paths if _input_name(item) in HELD_OUT_SHARDS]
        if held_out:
            raise ValueError(f"held-out shard is not a training input: {held_out[0]}")
    if not paths:
        raise FileNotFoundError(f"no training shards under {corpus_dir}")
    return list(paths)


@contextmanager
def open_corpus_input(item: Path | VerifiedInput) -> Iterator[TextIO]:
    """Open a direct fixture path or re-verify a production input at the read boundary."""
    if isinstance(item, VerifiedInput):
        with open_verified(item) as handle:
            yield handle
        return
    with Path(item).open(encoding="utf-8") as handle:
        yield handle


def corpus_provenance(corpus_dir: Path, paths: Sequence[Path | VerifiedInput]) -> CorpusProvenance:
    """Return the common corpus label and stable shard names for a builder document."""
    return CorpusProvenance(
        corpus_label(Path(corpus_dir)),
        [_input_name(item) for item in paths],
    )


def render_document(document: dict, *, indent: int = 1) -> str:
    return json.dumps(document, indent=indent, ensure_ascii=False) + "\n"


def run_builder(
    argv: list[str] | None,
    *,
    description: str,
    default_out: Path,
    default_corpus_dir: Callable[[], Path],
    build: Callable[[Path, int, tuple[VerifiedInput, ...], str], dict],
    render: Callable[[dict], str] = render_document,
    parallel_option: str = "--jobs",
    parallel_default: int = 1,
) -> int:
    """Verify, receipt, build, and either write or byte-check one corpus table."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pool", action="append", required=True, dest="pools")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument(parallel_option, type=int, default=parallel_default, dest="parallelism")
    parser.add_argument("--check", action="store_true", help="rebuild and compare byte for byte")
    args = parser.parse_args(argv)

    corpus_dir = (args.corpus_dir or default_corpus_dir()).resolve(strict=True)
    paths = training_inputs(corpus_dir)
    verified = verified_inputs(
        args.source_id,
        paths,
        locale=args.locale,
        pools=tuple(args.pools),
        root=corpus_dir,
    )
    write_verification_receipt(args.receipt, verified, locale=args.locale, pools=tuple(args.pools))
    rendered = render(build(corpus_dir, args.parallelism, verified, args.locale))
    encoded = rendered.encode("utf-8")
    if args.check:
        if not args.out.exists() or args.out.read_bytes() != encoded:
            print(f"--check: {args.out} differs from a rebuild", file=sys.stderr)
            return 1
        print(f"--check: {args.out} matches a rebuild byte for byte")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(encoded)
    print(f"wrote {args.out}")
    return 0
