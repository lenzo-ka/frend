"""Measure frend's range table (``frend/data/en/range_priors.json``) from the Google-TN
corpus's training shards: whether a range written in running text is read as one span
(E), and how its readings are ordered (J).

**E, the emit decision** (``readings``), over all 90 training shards (00-89). A
positive is a corpus triple (``google_tn_rows.range_triples``: a token ending in an ASCII
digit, a joiner, a token holding one) that the range rules can emit on
(``frend.ranges.emit_relevant``: R1-R3) and that is no punctuation dash
(``frend.ranges.punctuation_dash``: R6, kal's ruling A). A negative is a single corpus
token written ``<ASCII digits><separator><ASCII digits>`` (a TELEPHONE "555-1212", a TIME
"10:30"), counted by its class. Both are keyed by the emit key (``frend.ranges.emit_key``:
the class, then the two ends' digit lengths; a ratio icukit reads as a clock is
``ratio:clock``) and summed per class. Triples the rules cannot emit on and R6's dashes
are counted apart and never enter a row.

**J, the joint reading prior** (``kinds``), over every tenth training shard (00, 10, ...,
80), as the spoken priors sample. Each relevant triple (R1-R3, both ends read by R7's
end types, kept by R6) is read as ``frend.ranges.RangeDetector`` reads its ends and said
as ``frend.verbalize`` says a range in builder mode (P5's order, no priors; the ends
labeled as R8 and R11 say); it credits the joint source (``range:<left kind>+<connector
id>+<right kind>``) of the first reading whose normalized text is the corpus's reading of
the three tokens. Each class's row and each sub-key's row (``frend.ranges.range_sub_key``)
count those sources, in the spoken priors' schema.

``--check`` rebuilds and compares with the shipped table byte for byte. Every shard is
read through ``google_tn_rows.full_training_set`` / ``training_shards``, so no held-out
shard (90-99) is ever opened.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
for _path in (_REPO, _TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from google_tn_rows import (  # noqa: E402
    corpus_label,
    expected,
    full_training_set,
    range_candidate_denominators,
    range_triples,
    training_shards,
)

_range_denominators = range_candidate_denominators

LOCALE = "en_US"
DEFAULT_OUT = _REPO / "frend" / "data" / "en" / "range_priors.json"
_SAMPLE_STEP = 10


def _default_corpus_dir() -> Path:
    from build_spoken_priors import _default_corpus_dir as spoken_default

    return spoken_default()


def _shards(corpus_dir: Path, inputs=None) -> tuple[list, list]:
    """(E's shards, J's shards): the 90 training shards and every tenth of them for the
    full corpus; for another (a fixture), its shards less any held-out one, and every
    tenth of those by position."""
    if inputs is not None:
        available = sorted(inputs, key=lambda item: item.relative_path)
        if not available:
            raise FileNotFoundError(f"no corpus shards under {corpus_dir}")
        return available, available[::_SAMPLE_STEP]
    available = sorted(Path(corpus_dir).glob("output-*-of-*"))
    if not available:
        raise FileNotFoundError(f"no corpus shards under {corpus_dir}")
    full = full_training_set(available)
    if full is not None:
        return full, full[::_SAMPLE_STEP]
    emit = training_shards(available)
    sample = training_shards(available[::_SAMPLE_STEP])
    if not emit or not sample:
        raise FileNotFoundError(f"no training shards under {corpus_dir} (only held-out ones)")
    return emit, sample


def _sentences(path: Path):
    sentences, current = [], []
    if hasattr(path, "relative_path"):
        from corpus_inputs import open_verified

        handle_context = open_verified(path)
    else:
        handle_context = Path(path).open(encoding="utf-8")
    with handle_context as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "<eos>":
                if current:
                    sentences.append(current)
                current = []
                continue
            if len(parts) >= 3:
                current.append((parts[0], parts[1], parts[2]))
    if current:
        sentences.append(current)
    return sentences


def _single_token_pattern() -> re.Pattern:
    from frend.ranges import range_separator_classes

    separators = sorted(set().union(*range_separator_classes(LOCALE).values()))
    return re.compile("([0-9]+)([" + re.escape("".join(separators)) + "])([0-9]+)")


def _class_of(separator: str) -> str | None:
    from frend.ranges import range_separator_classes

    for cls, members in range_separator_classes(LOCALE).items():
        if separator in members:
            return cls
    return None


# ---------------------------------------------------------------------------------------
# E: one shard's emission evidence.


def emission_counts(path: Path) -> Counter:
    """(what, emit key, detail) -> count for one shard: ``("range", key, "")`` for a
    positive, ``("single", key, class)`` for a negative, ``("punctuation_dash", key,
    "")`` and ``("not_emittable", key, "")`` for the triples set apart."""
    from frend.ranges import emit_key, emit_relevant, punctuation_dash, written_sub_key

    counts: Counter = Counter()
    sentences = _sentences(path)
    for left, middle, right in range_triples(sentences):
        cls = _class_of(middle[1])
        if cls is None:
            continue
        key = emit_key(written_sub_key(cls, left[1], middle[1], right[1], LOCALE))
        if punctuation_dash(left, middle, right):
            counts[("punctuation_dash", key, "")] += 1
        elif not emit_relevant(left[1], middle[1], right[1], LOCALE):
            counts[("not_emittable", key, "")] += 1
        else:
            counts[("range", key, "")] += 1
    single = _single_token_pattern()
    for sentence in sentences:
        for corpus_class, written, _ in sentence:
            found = single.fullmatch(written)
            if found is None:
                continue
            cls = _class_of(found.group(2))
            key = emit_key(
                written_sub_key(cls, found.group(1), found.group(2), found.group(3), LOCALE)
            )
            counts[("single", key, corpus_class)] += 1
    return counts


# ---------------------------------------------------------------------------------------
# J: one shard's credited joint sources.

_ENDS = None


def _end_reader():
    global _ENDS
    if _ENDS is None:
        from reading_profile import reading_detectors

        from frend.ranges import RangeDetector

        readers = [d for d in reading_detectors(LOCALE) if not isinstance(d, RangeDetector)]
        _ENDS = RangeDetector(LOCALE, endpoints=readers, table=None)
    return _ENDS


def credit(left, middle, right) -> tuple[str, str | None, str | None]:
    """(outcome, sub-key, credited source) for one triple: outcome ``credited``,
    ``punctuation_dash``, ``not_emittable``, ``no_end``, ``unmatched`` or ``error``."""
    from frend.ranges import RangeValue, emit_relevant, punctuation_dash
    from frend.spoken_priors import normalize_spoken
    from frend.verbalize import _range_candidates

    separator = middle[1]
    cls = _class_of(separator)
    if cls is None:
        return "not_emittable", None, None
    reader = _end_reader()
    sub_key = reader.sub_key(cls, left[1], separator, right[1])
    if punctuation_dash(left, middle, right):
        return "punctuation_dash", sub_key, None
    if not emit_relevant(left[1], separator, right[1], LOCALE):
        return "not_emittable", sub_key, None
    ends = reader.ends(left[1], right[1])
    if ends is None:
        return "no_end", sub_key, None
    left_end, right_end = ends
    target = normalize_spoken(" ".join(expected(*row) for row in (left, middle, right)))
    try:
        value = RangeValue((left_end,), separator, cls, (right_end,))
        candidates = _range_candidates(value, LOCALE, apply_source_priors=False, measured=True)
    except Exception:  # noqa: BLE001 - a failure is counted, not hidden
        return "error", sub_key, None
    source = next(
        (item.provenance for item in candidates if normalize_spoken(item.text) == target), None
    )
    return ("credited" if source else "unmatched"), sub_key, source


def joint_counts(path: Path) -> Counter:
    """(outcome, sub-key, source) -> count for one shard."""
    counts: Counter = Counter()
    for left, middle, right in range_triples(_sentences(path)):
        counts[credit(left, middle, right)] += 1
    return counts


def _map(function, paths, jobs: int):
    if jobs <= 1:
        return [function(path) for path in paths]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(function, paths))


def build_document(corpus_dir: Path, jobs: int = 1, *, inputs=None) -> dict:
    from frend.ranges import RANGE_TYPES, range_class_key

    emit_paths, sample_paths = _shards(Path(corpus_dir), inputs)
    emission: Counter = Counter()
    for counts in _map(emission_counts, emit_paths, jobs):
        emission.update(counts)
    joint: Counter = Counter()
    for counts in _map(joint_counts, sample_paths, jobs):
        joint.update(counts)

    classes = [range_class_key(cls) for cls in RANGE_TYPES]
    readings: dict[str, dict] = {}
    for (what, key, detail), n in sorted(emission.items()):
        if what not in ("range", "single"):
            continue
        for row_key in (key, key.split(":", 1)[0]):
            row = readings.setdefault(row_key, {"range": 0, "single_token": {}})
            if what == "range":
                row["range"] += n
            else:
                row["single_token"][detail] = row["single_token"].get(detail, 0) + n
    readings = {
        key: {"range": row["range"], "single_token": dict(sorted(row["single_token"].items()))}
        for key, row in sorted(readings.items())
    }
    kinds: dict[str, dict] = {}
    outcomes: Counter = Counter()
    for (outcome, sub_key, source), n in joint.items():
        outcomes[outcome] += n
        if sub_key is None or outcome not in ("credited", "unmatched"):
            continue
        cls, _, rest = sub_key.partition(":")
        kind = kinds.setdefault(
            cls, {"total": 0, "matched": 0, "unmatched": 0, "source_matched": {}, "sub_keys": {}}
        )
        kind["total"] += n
        if outcome == "unmatched":
            kind["unmatched"] += n
            continue
        kind["matched"] += n
        kind["source_matched"][source] = kind["source_matched"].get(source, 0) + n
        sub = kind["sub_keys"].setdefault(rest, {"matched": 0, "source_matched": {}})
        sub["matched"] += n
        sub["source_matched"][source] = sub["source_matched"].get(source, 0) + n

    def ordered(counts: dict) -> dict:
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    kinds = {
        cls: {
            "total": kind["total"],
            "matched": kind["matched"],
            "unmatched": kind["unmatched"],
            "source_matched": ordered(kind["source_matched"]),
            "sub_keys": {
                key: {"matched": sub["matched"], "source_matched": ordered(sub["source_matched"])}
                for key, sub in sorted(kind["sub_keys"].items())
            },
        }
        for cls, kind in sorted(kinds.items())
        if cls in classes
    }
    set_apart = Counter()
    for (what, _key, _detail), n in emission.items():
        if what in ("punctuation_dash", "not_emittable"):
            set_apart[what] += n
    return {
        "locale": "en",
        "schema_version": 1,
        "provenance": {
            "locale": "en",
            "corpus": corpus_label(Path(corpus_dir)),
            "shards": [
                path.relative_path if inputs is not None else path.name for path in sample_paths
            ],
            "emit_shards": [
                path.relative_path if inputs is not None else path.name for path in emit_paths
            ],
            "sample_rule": {
                "rule": "full_training_set(corpus)[::10], as tools/build_spoken_priors.py",
                "shards": [
                    path.relative_path if inputs is not None else path.name
                    for path in sample_paths
                ],
            },
            "unit": (
                "tools/google_tn_rows.range_triples: a written token ending in an ASCII "
                "digit, a joiner, a written token holding one; a superset of the "
                "evaluator's running_text triple"
            ),
            "single_token": (
                "a single corpus token written <ASCII digits><separator><ASCII digits>, "
                "counted by its class"
            ),
            "sub_key_rules": (
                "R4a/R4b (frend.ranges): the class (dash: '-' and the en dash pooled; "
                "ratio; dimension), then '<L>+<R>' ASCII digit lengths when both ends are "
                "ASCII digits only, ':0' added for a right end of two or more digits "
                "written with a leading zero; 'ratio:clock' where icukit's time:flexible "
                "reads the written span whole; else 'other'. E reads the key without ':0'"
            ),
            "credit": (
                "the joint source (range:<left kind>+<connector id>+<right kind>) of the "
                "first builder-mode reading (P5's order, no priors; ends labeled by R8 "
                "and R11) whose normalized text equals the triple's corpus reading"
            ),
            "emission": (
                "RangeDetector emits a span where range >= EMIT_RATIO x single_token for "
                "the sub-key's emit key; a key never observed reads its class row"
            ),
            "relevance": {
                "emit": (
                    "range_triples passing R1-R3 on the 90 training shards, R6 rows "
                    "dropped; negatives: single tokens <digits><sep><digits> by class"
                ),
                "joint": (
                    "range_triples passing R1-R3 and R7 on the sample shards, R6 rows "
                    "dropped; credited by the first matching builder-mode candidate"
                ),
                "dropped_filter": (
                    "punctuation_dash: middle VERBATIM|PUNCT written '-'|'–' said "
                    "'sil', and (left CARDINAL written [12][0-9]{3} or a MONEY end)"
                ),
                "wider_not_counted": {
                    "not_emittable": set_apart["not_emittable"],
                    "punctuation_dash": set_apart["punctuation_dash"],
                },
                "joint_outcomes": dict(sorted(outcomes.items())),
            },
            "builder": "tools/build_range_priors.py",
        },
        "readings": readings,
        "kinds": kinds,
    }


def render(document: dict) -> str:
    return json.dumps(document, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pool", action="append", required=True, dest="pools")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--check", action="store_true", help="rebuild and compare byte for byte")
    args = parser.parse_args(argv)
    corpus_dir = args.corpus_dir or _default_corpus_dir()
    from corpus_inputs import verified_inputs, write_verification_receipt

    emit_paths, _sample_paths = _shards(corpus_dir)
    verified = verified_inputs(
        args.source_id,
        emit_paths,
        locale=args.locale,
        pools=tuple(args.pools),
        root=corpus_dir,
    )
    write_verification_receipt(
        args.receipt, verified, locale=args.locale, pools=tuple(args.pools)
    )
    text = render(build_document(corpus_dir, args.jobs, inputs=verified))
    if args.check:
        shipped = args.out.read_text(encoding="utf-8") if args.out.exists() else None
        if shipped != text:
            print(f"--check: {args.out} differs from a rebuild")
            return 1
        print(f"--check: {args.out} matches a rebuild byte for byte")
        return 0
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
