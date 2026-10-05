"""Build or verify the compact case-preserved spell-or-say dictionary.

The table contains only Google TN decisions that differ from the AEIOU fallback for
that exact surface. Counts preserve the measured strength without repeating provenance
on every row; no sentence or context enters the table.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import icu

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from build_spoken_priors import _default_corpus_dir  # noqa: E402
from google_tn_rows import corpus_label, expected, full_training_set  # noqa: E402

from frend.letters import (  # noqa: E402
    is_spelled_token,
    spelled,
    spelled_token_rule,
    split_acronym_surface,
)
from frend.spoken_priors import normalize_spoken  # noqa: E402

_NFC = icu.Normalizer2.getNFCInstance()
_OUT = _REPO / "frend" / "data" / "en" / "spellout_dictionary.json"
GOOGLE_SOURCE = "google/tn-en_with_types"
MINIMUM_SUPPORT = 5
MINIMUM_PURITY = 0.99
_ADJUDICATED_NONREADINGS = frozenset({"zijn", "échec", "échecs"})
_NONLEXICAL_POS = frozenset({"abbrev", "acronym", "initialism", "letter", "symbol"})
_NONLEXICAL_TAGS = frozenset({"abbreviation", "acronym", "initialism", "letter-name", "symbol"})
_LETTER_NAME = re.compile(r"\b(?:letter name|name of (?:the )?.*\bletter)\b", re.IGNORECASE)


def _eligible(surface: str, locale: str = "en_US") -> bool:
    return split_acronym_surface(surface) is not None or is_spelled_token(surface, locale)


def literal_outcome(corpus_class: str, written: str, spoken: str, locale: str) -> str | None:
    """Return a label only when the corpus row literally says or spells ``written``."""
    if corpus_class not in {"LETTERS", "PLAIN"}:
        return None
    written = _NFC.normalize(written)
    if not _eligible(written, locale):
        return None
    target = expected(corpus_class, written, spoken)
    if corpus_class == "PLAIN" and target == normalize_spoken(written):
        return "say"
    letter_form = spelled(written, locale)
    if (
        corpus_class == "LETTERS"
        and letter_form is not None
        and target == normalize_spoken(letter_form.text)
    ):
        return "spell"
    return None


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
            if len(parts) < 3:
                continue
            surface = _NFC.normalize(parts[1])
            if (label := literal_outcome(parts[0], surface, parts[2], locale)) is not None:
                counts[surface][label] += 1
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


def _casefold_entries(
    raw_counts: dict[str, Counter[str]], exact_surfaces: set[str]
) -> list[list[str]]:
    """Compatibility aliases for unanimous observed case families.

    Exact lookup always wins. Families such as ``IT``/``it`` and ``OR``/``or``
    abstain because their attested variants disagree.
    """
    families: dict[str, list[str]] = defaultdict(list)
    for surface in raw_counts:
        families[surface.casefold()].append(surface)
    folded = []
    for key, variants in sorted(families.items()):
        decisions = set()
        combined = Counter()
        tied = False
        for surface in variants:
            counts = raw_counts[surface]
            combined.update(counts)
            tied |= counts["spell"] == counts["say"]
            if counts["spell"] != counts["say"]:
                decisions.add("spell" if counts["spell"] > counts["say"] else "say")
        candidates = [surface for surface in variants if surface in exact_surfaces]
        if (
            len(variants) >= 2
            and not tied
            and len(decisions) == 1
            and _decision(combined) is not None
            and candidates
        ):
            target = max(candidates, key=lambda surface: (raw_counts[surface].total(), surface))
            folded.append([key, target])
    return folded


def _lexicon_abbreviations(path: Path) -> set[str]:
    """Exact forms for which lex/1 contains any abbreviation-like sense."""
    forms = set()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.reader((line for line in handle if not line.startswith("#")), delimiter="\t")
        for row in rows:
            if len(row) != 6:
                continue
            form, kind, _case, _sense, _expansion, tags = row
            labels = {kind, *(tag.casefold() for tag in tags.split(","))}
            if labels & {"abbrev", "abbreviation", "acronym", "initialism"}:
                forms.add(_NFC.normalize(form))
    return forms


def _wiktionary_artifact(path: Path) -> Path:
    receipt = path.parents[1] / "sources" / "wiktionary" / "RECEIPT"
    for line in receipt.read_text(encoding="utf-8").splitlines():
        if line.startswith("artifact: "):
            return Path(line.removeprefix("artifact: ")).resolve(strict=True)
    raise ValueError(f"no artifact path in {receipt}")


def _mixed_case_technical(surface: str) -> bool:
    cased = [character for character in surface if character.isalpha()]
    return (
        any(character.islower() for character in cased)
        and any(character.isupper() for character in cased)
        and not surface.istitle()
    )


def _letter_name_sense(sense: dict) -> bool:
    glosses = [str(gloss) for gloss in sense.get("glosses") or ()]
    return any(_LETTER_NAME.search(gloss) for gloss in glosses)


def wiktionary_ordinary_words(path: Path, candidates: set[str]) -> set[str]:
    """Return exact candidate headwords with unambiguous lexical evidence.

    Translation words are deliberately ignored. A surface is ordinary only when an
    exact Wiktionary headword supplies a lexical sense and no entry or sense for that
    surface is abbreviation-like, symbolic, or a letter name. Inflection-only senses
    do not establish lexical evidence: this conservatively avoids treating plurals of
    abbreviations as words. Non-titlecase mixed-case forms are technical spellings.
    """
    abbreviation_forms = _lexicon_abbreviations(path)
    wanted = {
        word
        for word in candidates - abbreviation_forms
        if not word.isupper() and not _mixed_case_technical(word)
    }
    evidence = {word: {"lexical": False, "blocked": False, "related": set()} for word in wanted}
    artifact = _wiktionary_artifact(path)
    handle_context = (
        gzip.open(artifact, "rt", encoding="utf-8")
        if artifact.suffix == ".gz"
        else artifact.open("r", encoding="utf-8")
    )
    with handle_context as handle:
        for line in handle:
            entry = json.loads(line)
            word = _NFC.normalize(str(entry.get("word", "")))
            if word not in evidence:
                continue
            state = evidence[word]
            entry_pos = str(entry.get("pos", "")).casefold()
            entry_labels = {entry_pos} | {str(tag).casefold() for tag in entry.get("tags") or ()}
            if entry_labels & (_NONLEXICAL_POS | _NONLEXICAL_TAGS):
                state["blocked"] = True
            senses = entry.get("senses") or ({},)
            for sense in senses:
                labels = {str(tag).casefold() for tag in sense.get("tags") or ()}
                related = {
                    _NFC.normalize(str(link.get("word", "")))
                    for field in ("form_of", "alt_of")
                    for link in sense.get(field) or ()
                    if isinstance(link, dict)
                }
                if entry_pos != "name":
                    state["related"].update(related)
                else:
                    state["related"].update(
                        target for target in related if target.rstrip(".") == word
                    )
                if labels & _NONLEXICAL_TAGS or _letter_name_sense(sense):
                    state["blocked"] = True
                    continue
                if "form-of" not in labels:
                    state["lexical"] = True
    found = {
        word
        for word, state in evidence.items()
        if state["lexical"] and not state["blocked"] and not state["related"] & abbreviation_forms
    }
    return found | (candidates & _ADJUDICATED_NONREADINGS)


def _build_documents(
    corpus_dir: Path,
    *,
    locale="en_US",
    inputs=None,
    workers=1,
    wiktionary: Path | None = None,
) -> dict:
    raw_counts = google_counts(corpus_dir, locale=locale, inputs=inputs, workers=workers)
    ordinary_words = (
        set() if wiktionary is None else wiktionary_ordinary_words(wiktionary, set(raw_counts))
    )
    ordinary_casefolds = {word.casefold() for word in ordinary_words}
    rows = []
    ordinary_word_spell_rows_dropped = 0
    for surface, counts in sorted(raw_counts.items()):
        decision = _decision(counts)
        if decision is None or decision == spelled_token_rule(surface):
            continue
        row = [surface, decision, counts["say"], counts["spell"]]
        if decision == "spell" and surface in ordinary_words:
            ordinary_word_spell_rows_dropped += 1
            continue
        rows.append(row)
    exact_surfaces = {row[0] for row in rows}
    aliases = [
        alias
        for alias in _casefold_entries(raw_counts, exact_surfaces)
        if alias[0] not in ordinary_casefolds
    ]
    names = [
        item.relative_path if inputs is not None else Path(item).name for item in (inputs or [])
    ]
    provenance = {
        "columns": ["surface", "decision", "say_count", "spell_count"],
        "corpus": corpus_label(corpus_dir),
        "locale": locale.partition("_")[0],
        "privacy": "rows retain only token, decision, and aggregate counts",
        "selection": {
            "development_shards": [f"output-{index:05d}-of-00100" for index in range(90, 95)],
            "minimum_purity": MINIMUM_PURITY,
            "minimum_support": MINIMUM_SUPPORT,
            "objective": "maximize S0 first-choice spell-versus-say labels",
            "ordinary_word_filter": (
                "drop spell decisions with unambiguous exact-headword lexical evidence"
            ),
            "ordinary_word_spell_rows_dropped": ordinary_word_spell_rows_dropped,
            "rule": "retain only decisions differing from the exact surface AEIOU fallback",
        },
        "license": "CC-BY-SA-4.0",
        "source": GOOGLE_SOURCE,
        "training_shards": sorted(names),
    }
    return {"casefold": aliases, "provenance": provenance, "tokens": rows}


def build_document(
    corpus_dir: Path,
    *,
    locale="en_US",
    inputs=None,
    workers=1,
    wiktionary: Path | None = None,
) -> dict:
    return _build_documents(
        corpus_dir,
        locale=locale,
        inputs=inputs,
        workers=workers,
        wiktionary=wiktionary,
    )


def _render(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pool", action="append", required=True, dest="pools")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--wiktionary",
        type=Path,
        default=None,
        help="lex/1 Wiktionary TSV whose receipt identifies the source wiktextract JSONL",
    )
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
    exact = _build_documents(
        corpus_dir,
        locale=args.locale,
        inputs=verified,
        workers=args.workers,
        wiktionary=args.wiktionary,
    )
    outputs = [(args.out, _render(exact))]
    if args.check:
        stale = []
        for path, rendered in outputs:
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != rendered:
                stale.append(path)
        if stale:
            for path in stale:
                print(f"out of date: {path}")
            return 1
        for path, _rendered in outputs:
            print(f"up to date: {path}")
        return 0
    for path, rendered in outputs:
        path.write_text(rendered, encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
