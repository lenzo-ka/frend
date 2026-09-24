"""Build (or verify) the vendored corpus prior table ``irn/data/type_priors.json``.

The table is ``shape -> class -> count``: for every ``(written_surface, class)``
pair a corpus yields, increment ``counts[shape(written)][class]``. Runtime derives
``P(class | shape)`` and ``n(shape)`` from these raw counts, so the artifact
carries counts (not just ratios) plus provenance naming the source.

The corpus is reached through one small seam -- an iterable of
``(written_surface, corpus_class)`` pairs:

* ``google_tn_pairs`` (the default) streams the Google/Sproat "en_with_types"
  text-normalization corpus (Sproat & Jaitly 2016; CC BY-SA 4.0). This is the
  basis the vendored table is built from.
Point ``--corpus`` at another Google-TN-format shard directory, use
``--corpus-dir`` to override the default corpus location, or import
``build_counts`` with a different pair source to swap the corpus without touching
the shape signature, the JSON format, the runtime, or the resolver.

The Google-TN corpus is ~20 GB uncompressed and is never vendored; it is streamed
line by line at build time. ``--check`` therefore re-derives from it and so needs
the corpus present at the ``--corpus-dir`` path (default from the
``IRN_TN_CORPUS_DIR`` environment variable, else the sibling ``tn-corpus``
checkout).

Usage::

    python tools/build_type_priors.py            # rebuild from Google-TN (default)
    python tools/build_type_priors.py --check     # verify it is up to date (CI)
    python tools/build_type_priors.py --corpus tests/data/google_tn  # fixture build
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "irn" / "data" / "type_priors.json"


def _ensure_repo_importable() -> None:
    """Put the repo root on ``sys.path`` so ``import irn.*`` works.

    Executing this file as a script puts tools/ -- not the repo root -- on
    ``sys.path``, so ``irn.shape`` is unimportable in a clean environment (no
    implicit PYTHONPATH). It is called before every ``irn`` import, including inside
    the multiprocessing workers, which do not reliably inherit the parent's
    ``sys.path`` under the spawn start method. Idempotent.
    """
    repo = str(_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)


_ensure_repo_importable()
from irn.shape import is_single_uppercase  # noqa: E402

# A fixed build date is used deliberately (no wall clock) so a rebuild from the
# same corpus is byte-identical and ``--check`` stays a clean drift guard.
_BUILD_DATE = "2026-08-23"


def _default_corpus_dir() -> Path:
    """Resolve the default Google-TN corpus directory without a baked-in absolute
    path.

    ``IRN_TN_CORPUS_DIR`` wins if set; otherwise look for a ``tn-corpus/
    en_with_types`` tree beside the repo (walking ancestors), so a sibling checkout
    is found relatively. When nothing is discoverable a non-existent path is
    returned, so the shard glob raises a clear error naming the knobs to set. The
    raw corpus lives outside any repo and is never shipped.
    """
    env = os.environ.get("IRN_TN_CORPUS_DIR")
    if env:
        return Path(env)
    for base in (_REPO, *_REPO.parents):
        candidate = base / "tn-corpus" / "en_with_types"
        if candidate.is_dir():
            return candidate
    return _REPO.parent / "tn-corpus" / "en_with_types"


# ---------------------------------------------------------------------------
# Provenance profiles. ``main`` identifies path-based fixture/custom builds while
# retaining the stable provenance of the shipped Google-TN table.

_PAIRS_NOTE = (
    "Raw shape->class counts from caller-provided pairs. Runtime derives "
    "P(class|shape) and n(shape); no smoothing, no ratios stored."
)

_GOOGLE_TN_NOTE = (
    "Raw shape->class counts over the Google/Sproat en_with_types text-"
    "normalization corpus. Only surfaces containing at least one Unicode-Nd "
    "digit are counted (the numeric-TN focus; the PLAIN alphabetic mass is "
    "dropped). Runtime derives P(class|shape) and n(shape); no smoothing, no "
    "ratios stored. Classes measure/telephone/address/digit are counted into "
    "each shape's total even though no reading type maps to them yet, so n(shape) "
    "reflects the full attested competition; the reflective ICU-generated shape "
    "backfill and blend are a separate later pass. Single Unicode-Lu letter "
    "surfaces lift the digit filter and class drops so their distribution includes "
    "the corpus's LETTERS and PLAIN classifications; per-letter counts are recorded "
    "separately."
)

_GOOGLE_TN_PROFILE: dict[str, str] = {
    "source": "google-tn-en_with_types",
    "license": "CC BY-SA 4.0",
    "attribution": ("derived from Sproat & Jaitly (2016) Google TN corpus; CC BY-SA 4.0"),
    "note": _GOOGLE_TN_NOTE,
}

_SOURCE = "provided-pairs"
_NOTE = _PAIRS_NOTE

# ---------------------------------------------------------------------------
# Corpus class -> our class. DATE/TIME/MONEY/FRACTION/ORDINAL/CARDINAL/DECIMAL are
# the classes the runtime maps reading types onto; MEASURE/TELEPHONE/ADDRESS/DIGIT
# are real numeric readings kept in each shape's total even though no reading type
# maps to them yet. PLAIN/PUNCT/LETTERS/VERBATIM/ELECTRONIC are dropped as noise /
# non-numeric.

_GOOGLE_TN_CLASS_MAP: dict[str, str] = {
    "CARDINAL": "cardinal",
    "DECIMAL": "decimal",
    "DATE": "date",
    "TIME": "time",
    "MONEY": "money",
    "FRACTION": "fraction",
    "ORDINAL": "ordinal",
    "MEASURE": "measure",
    "TELEPHONE": "telephone",
    "ADDRESS": "address",
    "DIGIT": "digit",
}


def _has_digit(surface: str) -> bool:
    """True if ``surface`` contains at least one Unicode decimal digit (category
    ``Nd``) -- the same digit notion :mod:`irn.shape` uses to collapse ``N`` runs."""
    return any(unicodedata.category(ch) == "Nd" for ch in surface)


def _google_tn_file_pairs(path: Path) -> Iterator[tuple[str, str]]:
    """Yield mapped, digit-bearing ``(surface, class)`` pairs from one corpus file.

    The corpus is TAB-separated ``(class, written, spoken)``; sentence boundaries
    are 2-column ``<eos>`` lines. Column 3 (spoken, with its ``<self>``/``sil``
    sentinels) is ignored entirely. Only mapped classes whose surface carries a
    digit are yielded.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:  # <eos> (2 columns) or malformed: skip
                continue
            surface = parts[1]
            single_uppercase = is_single_uppercase(surface)
            mapped = parts[0].lower() if single_uppercase else _GOOGLE_TN_CLASS_MAP.get(parts[0])
            if mapped is None:  # PLAIN/PUNCT/LETTERS/VERBATIM/ELECTRONIC: drop
                continue
            if not single_uppercase and not _has_digit(surface):
                continue
            yield surface, mapped


def _google_tn_files(corpus_dir: Path) -> list[Path]:
    """The sorted ``output-NNNNN-of-NNNNN`` shards under ``corpus_dir``."""
    files = sorted(corpus_dir.glob("output-*-of-*"))
    if not files:
        raise FileNotFoundError(
            f"no Google-TN corpus shards (output-*-of-*) under {corpus_dir}; "
            f"set IRN_TN_CORPUS_DIR or pass --corpus-dir"
        )
    return files


def google_tn_pairs(corpus_dir: Path | None = None) -> Iterator[tuple[str, str]]:
    """Stream ``(written_surface, corpus_class)`` from the Google-TN corpus.

    The default seam. Streams every shard line by line (never loading a file
    whole), mapping corpus classes to our class names, dropping non-numeric noise
    classes and any surface without a digit.
    """
    corpus_dir = Path(corpus_dir) if corpus_dir is not None else _default_corpus_dir()
    for path in _google_tn_files(corpus_dir):
        yield from _google_tn_file_pairs(path)


def _tally(
    pairs: Iterable[tuple[str, str]],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Sum ``counts[shape(written)][class]`` over a stream of corpus pairs."""
    _ensure_repo_importable()  # workers may not inherit the parent's sys.path
    from irn.shape import shape  # noqa: PLC0415 -- avoid import cost when only --check diffs

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    letters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for written, corpus_class in pairs:
        counts[shape(written)][corpus_class] += 1
        if is_single_uppercase(written):
            letters[written][corpus_class] += 1
    return counts, letters


def _canonical(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Sorted, plain dicts so the emitted JSON is canonical and diff-stable."""
    return {
        shape_key: dict(sorted(by_class.items())) for shape_key, by_class in sorted(counts.items())
    }


def build_counts(pairs: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Tally ``counts[shape(written)][class]`` over a stream of corpus pairs."""
    counts, _letters = _tally(pairs)
    return _canonical(counts)


def _count_shard(
    path: Path,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Tally one Google-TN shard (a picklable worker for parallel builds)."""
    counts, letters = _tally(_google_tn_file_pairs(path))
    return (
        {key: dict(by_class) for key, by_class in counts.items()},
        {key: dict(by_class) for key, by_class in letters.items()},
    )


def _merge(
    partials: Iterable[tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Sum per-shard tallies into one canonical table.

    Summation is order-independent, so a parallel build is byte-identical to the
    serial ``build_counts(google_tn_pairs(...))``.
    """
    merged: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    merged_letters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for partial, letters in partials:
        for shape_key, by_class in partial.items():
            for class_name, count in by_class.items():
                merged[shape_key][class_name] += count
        for letter, by_class in letters.items():
            for class_name, count in by_class.items():
                merged_letters[letter][class_name] += count
    return _canonical(merged), _canonical(merged_letters)


def build_google_tn_counts(
    corpus_dir: Path | None = None, jobs: int | None = None
) -> dict[str, dict[str, int]]:
    """Build the canonical Google-TN counts, optionally across ``jobs`` processes.

    With ``jobs`` > 1 each shard is tallied in its own process and the partials
    are summed; the result is identical to the serial stream.
    """
    return _build_google_tn_material(corpus_dir, jobs)[0]


def _build_google_tn_material(
    corpus_dir: Path | None = None, jobs: int | None = None
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Build aggregate and per-letter counts in one corpus pass."""
    corpus_dir = Path(corpus_dir) if corpus_dir is not None else _default_corpus_dir()
    files = _google_tn_files(corpus_dir)
    if jobs is None:
        jobs = os.cpu_count() or 1
    if jobs <= 1 or len(files) <= 1:
        counts, letters = _tally(google_tn_pairs(corpus_dir))
        return _canonical(counts), _canonical(letters)
    import multiprocessing as mp  # noqa: PLC0415 -- only when parallelizing

    with mp.Pool(processes=min(jobs, len(files))) as pool:
        return _merge(pool.imap_unordered(_count_shard, files))


def build_document(
    pairs: Iterable[tuple[str, str]],
    source: str = _SOURCE,
    note: str = _NOTE,
    license: str | None = None,
    attribution: str | None = None,
    counts: dict[str, dict[str, int]] | None = None,
    single_uppercase_letters: dict[str, dict[str, int]] | None = None,
) -> dict:
    """Assemble the full artifact: provenance plus raw counts.

    ``source``/``note``/``license``/``attribution`` name the provenance and default
    to a generic caller-provided-pairs profile; a corpus build passes its profile without
    touching the format or the runtime. ``counts`` may be supplied pre-built (a
    parallel build); otherwise it is tallied from ``pairs``.
    """
    provenance: dict[str, str] = {"source": source, "generated": _BUILD_DATE, "note": note}
    if license is not None:
        provenance["license"] = license
    if attribution is not None:
        provenance["attribution"] = attribution
    document = {
        "provenance": provenance,
        "counts": build_counts(pairs) if counts is None else _canonical(counts),
    }
    if single_uppercase_letters is not None:
        document["single_uppercase_letters"] = _canonical(single_uppercase_letters)
    return document


def _serialize(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _build(corpus: str, corpus_dir: Path | None, jobs: int | None) -> dict:
    """Assemble the artifact for the selected corpus seam."""
    if corpus == "google-tn":
        counts, letters = _build_google_tn_material(corpus_dir, jobs)
        return build_document(
            [], counts=counts, single_uppercase_letters=letters, **_GOOGLE_TN_PROFILE
        )
    path = Path(corpus)
    counts, letters = _build_google_tn_material(path, 1 if jobs is None else jobs)
    profile = {**_GOOGLE_TN_PROFILE, "source": f"google-tn-en_with_types:{path}"}
    return build_document([], counts=counts, single_uppercase_letters=letters, **profile)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the checked-in JSON matches a fresh build; exit nonzero on drift",
    )
    parser.add_argument(
        "--corpus",
        default="google-tn",
        help="Google-TN-format shard directory (default: google-tn, the vendored basis)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="override the default Google-TN en_with_types corpus directory",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="google-tn: parallel worker processes (default: os.cpu_count())",
    )
    parser.add_argument("--out", type=Path, default=_OUT, help="output path")
    args = parser.parse_args(argv)

    document = _build(args.corpus, args.corpus_dir, args.jobs)
    rendered = _serialize(document)

    if args.check:
        if not args.out.exists():
            print(f"drift: {args.out} does not exist; run build_type_priors.py", file=sys.stderr)
            return 1
        current = args.out.read_text(encoding="utf-8")
        if current != rendered:
            print(
                f"drift: {args.out} is out of date; rerun build_type_priors.py",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {args.out} is up to date")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
