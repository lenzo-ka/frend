"""Build or verify whether an acronym is spelled or said as a word, measured from the corpus.

An all-capitals token of two or more letters is spelled when the corpus files it as
LETTERS ("FBI" "f b i") and said as written when it files it as PLAIN ("NASA"). A token
is counted when frend's letters reader would match it whole
(``frend.letters.is_letter_run``: after NFC, capitals of general category Lu with their
combining marks, in one script as UAX #24 resolves a run), so the counts and the reader
cover one population, except that the reader leaves a bare run the corpus mostly reads
as a Roman numeral ("II") to icukit, while this counts it, by shape and under
``roman:``. Shape counts use the spoken priors' sample (every tenth). The optional
``google-tn`` conformance-profile table counts every training shard by the
case-preserved NFC surface and the detector's suffix subkey (bare, plural,
possessive). It is written to a user-local file, never under ``frend/data``. Counts in
the packaged table are kept by the token's shape
(``frend.electronic.letter_key``: case, length, whether a vowel letter
occurs), by its consonant-vowel pattern up to seven letters
(``frend.letters.cv_pattern``: "cvc" is mostly a word, "ccc" spelled), and for an
acronym icukit's lexicon lists, by its own surface ("NASA" is a word, "FBI" spelled; the
surfaces are icukit's, not the corpus's); ``*`` pools all. A token icukit also reads as
a Roman numeral ("II", "CD") is counted a third way, as a numeral when the corpus files
it CARDINAL or ORDINAL, under ``roman:<surface>`` for a numeral seen at least
``_ROMAN_FLOOR`` times and ``roman:*`` for all, so frend reads "II" as two and "CD" as
letters. Only counts are stored.

``--check`` repeats both populations and compares both JSON files byte for byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import icu

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from build_spoken_priors import _default_corpus_dir, _files  # noqa: E402
from google_tn_rows import (  # noqa: E402
    TRAINING_SHARDS,
    corpus_label,
    full_training_set,
    training_shards,
)

from frend.electronic import letter_key  # noqa: E402
from frend.letters import cv_pattern, is_letter_run, is_roman, split_acronym_surface  # noqa: E402
from frend.profiles import GOOGLE_TN, google_tn_profile_path  # noqa: E402


def _lexicon_acronyms() -> frozenset[str]:
    """The all-capitals acronyms icukit's en_US lexicon lists (its surfaces, not corpus text)."""
    from icukit.abbreviation_compile import compile_lexicon

    compiled = compile_lexicon("en_US")
    return frozenset(
        entry.surface for entry in compiled.lexicon.entries if is_letter_run(entry.surface)
    )


_NFC = icu.Normalizer2.getNFCInstance()
_OUT = _REPO / "frend" / "data" / "en" / "acronym_priors.json"
_LABELS = {"LETTERS": "spelled", "PLAIN": "word"}
_NUMERAL = frozenset({"CARDINAL", "ORDINAL"})
_ROMAN_FLOOR = 20
SURFACE_MINIMUM_SUPPORT = 1
SURFACE_PARENT_STRENGTH = 1


def _training_files(corpus_dir: Path) -> list[Path]:
    paths = sorted(path for path in corpus_dir.iterdir() if path.is_file())
    return full_training_set(paths) or training_shards(paths)


def _sampled_inputs(inputs) -> list:
    """Every tenth verified training input, preserving fixture behavior."""
    items = list(inputs)
    if any(item.relative_path.endswith("-of-00100") for item in items):
        return [
            item
            for item in items
            if int(item.relative_path.removeprefix("output-").split("-", 1)[0]) % 10 == 0
        ]
    return items[::10]


def _profile_source_shards(inputs) -> list[dict[str, str]]:
    """The verified identities used for the external profile, when supplied.

    Direct-path fixture builds have no verification claim. Production builds receive
    :class:`corpus_inputs.VerifiedInput` objects and record every pinned digest.
    """
    return [
        {"relative_path": item.relative_path, "sha256": item.sha256}
        for item in inputs
        if isinstance(getattr(item, "sha256", None), str)
    ]


def _require_training_inputs(inputs) -> None:
    outside = sorted(
        item.relative_path for item in inputs if item.relative_path not in TRAINING_SHARDS
    )
    if outside:
        raise ValueError(
            f"profile inputs must be Google TN training shards 00-89; got {outside[0]!r}"
        )


def build_documents(
    corpus_dir: Path, *, inputs=None, minimum_support: int = SURFACE_MINIMUM_SUPPORT
) -> tuple[dict, dict]:
    counts: dict[str, Counter] = defaultdict(Counter)
    romans: dict[str, Counter] = defaultdict(Counter)
    surfaces: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    acronyms = _lexicon_acronyms()
    all_files = list(inputs) if inputs is not None else _training_files(corpus_dir)
    if inputs is not None:
        _require_training_inputs(all_files)
    sampled_files = _sampled_inputs(all_files) if inputs is not None else _files(corpus_dir)
    sampled_names = {
        path.relative_path if inputs is not None else path.name for path in sampled_files
    }
    for path in all_files:
        if inputs is None:
            handle_context = path.open(encoding="utf-8")
            path_name = path.name
        else:
            from corpus_inputs import open_verified

            handle_context = open_verified(path)
            path_name = path.relative_path
        with handle_context as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3 or parts[0] not in _LABELS.keys() | _NUMERAL:
                    continue
                # Keyed in NFC, as the reader reads a run: "E\u0301CO" is "ÉCO".
                written = _NFC.normalize(parts[1])
                # PLAIN dominates the corpus. The cheap Unicode casing guard avoids
                # running the script-aware parser on ordinary words.
                maybe_run = written
                if maybe_run.endswith(("'s", "’s", "s'", "s’")):
                    maybe_run = maybe_run[:-2]
                elif maybe_run.endswith("s"):
                    maybe_run = maybe_run[:-1]
                if (
                    parts[0] in _LABELS
                    and len(maybe_run) >= 2
                    and maybe_run.isupper()
                    and (parsed := split_acronym_surface(written))
                ):
                    surface, subkey = parsed
                    surfaces[surface][subkey][_LABELS[parts[0]]] += 1
                if path_name not in sampled_names:
                    continue
                if not is_letter_run(written):
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
    packaged = {
        "provenance": {
            "attribution": "derived from Sproat & Jaitly (2016) Google TN corpus",
            "locale": "en",
            "corpus": corpus_label(corpus_dir),
            "license": "CC BY-SA 4.0",
            "shards": sorted(sampled_names),
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
    profile = {
        "schema_version": 1,
        "profile": GOOGLE_TN,
        "locale": "en",
        "provenance": {
            "attribution": "derived from Sproat & Jaitly (2016) Google TN corpus",
            "corpus": corpus_label(corpus_dir),
            "license": "CC BY-SA 4.0",
            "source_shards": _profile_source_shards(all_files),
            "shards": [
                path.relative_path if inputs is not None else path.name for path in all_files
            ],
            "rule": (
                "case-preserved all-capitals surfaces of two or more letters: "
                "LETTERS spelled, PLAIN word; suffixes separated"
            ),
        },
        "selection": {
            "baseline_correct": 1034857,
            "candidate_minimum_support": [1, 2, 3, 5, 10, 20, 30, 50, 100],
            "candidate_parent_strength": [0, 1, 2, 5, 10, 20],
            "development_correct": 1187738,
            "development_shards": [f"output-{index:05d}-of-00100" for index in range(80, 90)],
            "development_tokens": 1203782,
            "minimum_support": minimum_support,
            "objective": "maximize first-choice spelled-versus-word labels",
            "parent_strength": SURFACE_PARENT_STRENGTH,
            "selection_training_shards": [f"output-{index:05d}-of-00100" for index in range(80)],
        },
        "surfaces": {
            surface: {
                subkey: dict(sorted(labels.items()))
                for subkey, labels in sorted(subkeys.items())
                if labels.total() >= minimum_support
            }
            for surface, subkeys in sorted(surfaces.items())
            if any(labels.total() >= minimum_support for labels in subkeys.values())
        },
    }
    return packaged, profile


def build_document(corpus_dir: Path, *, inputs=None) -> dict:
    """The package-safe shape/CV table (no corpus-derived surface inventory)."""
    return build_documents(corpus_dir, inputs=inputs)[0]


def build_profile_document(
    corpus_dir: Path, *, inputs=None, minimum_support: int = SURFACE_MINIMUM_SUPPORT
) -> dict:
    """The external ``google-tn`` exact-surface profile table."""
    return build_documents(corpus_dir, inputs=inputs, minimum_support=minimum_support)[1]


def _render(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _external_profile_output(path: Path) -> Path:
    """Resolve ``path`` and refuse any profile artifact inside this repository."""
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(_REPO.resolve()):
        raise ValueError(
            f"--profile-out must be outside the repository and package tree: {resolved}"
        )
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pool", action="append", required=True, dest="pools")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=_OUT)
    parser.add_argument(
        "--profile-out",
        type=Path,
        default=None,
        help="external google-tn profile table (default: environment or user cache)",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    profile_out = _external_profile_output(args.profile_out or google_tn_profile_path())
    corpus_dir = args.corpus_dir or _default_corpus_dir()
    from corpus_inputs import verified_inputs, write_verification_receipt

    verified = verified_inputs(
        args.source_id,
        _training_files(corpus_dir),
        locale=args.locale,
        pools=tuple(args.pools),
        root=corpus_dir,
    )
    write_verification_receipt(args.receipt, verified, locale=args.locale, pools=tuple(args.pools))
    document, profile_document = build_documents(corpus_dir, inputs=verified)
    rendered = _render(document)
    profile_rendered = _render(profile_document)
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        profile_current = profile_out.read_text(encoding="utf-8") if profile_out.exists() else ""
        stale = False
        if current != rendered:
            print(f"out of date: {args.out}")
            stale = True
        else:
            print(f"up to date: {args.out}")
        if profile_current != profile_rendered:
            print(f"out of date: {profile_out}")
            stale = True
        else:
            print(f"up to date: {profile_out}")
        return int(stale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}")
    profile_out.parent.mkdir(parents=True, exist_ok=True)
    profile_out.write_text(profile_rendered, encoding="utf-8")
    print(f"wrote {profile_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
