"""What every Google-TN builder and the evaluator share: the corpus's split and the
running-text triple.

The en_with_types corpus's README splits its 100 shards into three pools, named here:

* :data:`TRAINING_SHARDS`, ``output-00000-of-00100`` .. ``output-00089-of-00100``:
  the only shards a shipped table counts;
* :data:`RUNTIME_EVAL_SHARDS`, ``output-00090-of-00100`` .. ``output-00094-of-00100``:
  held out, a second pool to measure on; and
* :data:`TEST_SHARDS`, ``output-00095-of-00100`` .. ``output-00099-of-00100``: held
  out. Shard 99 is the published test shard (Sproat & Jaitly 2016 test on its first
  100,000 lines), which ``tools/evaluate_google_tn.py`` reports; shard 95 is the one a
  change is accepted and selected on (``--held-out-shard``).

:data:`HELD_OUT_SHARDS` is the runtime-eval and test pools together. Every builder reads
its shards through :func:`training_shards`, so no held-out shard can reach a table
however the builder slices the corpus. :func:`corpus_label` names the corpus a table
records, from the directory the builder read. :func:`running_text` is the one number,
separator, number predicate the evaluator scores; :func:`range_triples` is the wider
one the range builders count (a left token that ends in a digit, such as "$15,000").
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def _pool(first: int, last: int) -> frozenset[str]:
    return frozenset(f"output-{index:05d}-of-00100" for index in range(first, last + 1))


# The corpus README's split: training 00-89, runtime eval 90-94, test 95-99.
TRAINING_SHARDS = _pool(0, 89)
RUNTIME_EVAL_SHARDS = _pool(90, 94)
TEST_SHARDS = _pool(95, 99)
HELD_OUT_SHARDS = RUNTIME_EVAL_SHARDS | TEST_SHARDS


_FULL_SUFFIX = "-of-00100"


def full_training_set(paths: Iterable[Path]) -> list[Path] | None:
    """For the full 100-shard corpus, its training shards in order, refusing a partial
    listing (a missing shard, or a stalled mount that lists only some): a table built
    from fewer shards than it names would be silently wrong. ``None`` for any other
    corpus (a fixture), whose shards are taken as found."""
    paths = [Path(path) for path in paths]
    if not any(path.name.endswith(_FULL_SUFFIX) for path in paths):
        return None
    by_name = {path.name: path for path in paths}
    missing = sorted(TRAINING_SHARDS - by_name.keys())
    if len(missing) == len(TRAINING_SHARDS):
        raise FileNotFoundError("no training shards in the corpus (only held-out ones)")
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} training shard(s) missing, first {missing[0]}; "
            "refusing to build from a partial corpus"
        )
    return [by_name[name] for name in sorted(TRAINING_SHARDS)]


def corpus_label(corpus_dir: Path) -> str:
    """The corpus a table was counted from, named by the directory actually read:
    ``google-tn:en_with_types`` for the shipped corpus, ``google-tn:<name>`` for any other
    Google-TN-format shard directory (a fixture, a custom slice)."""
    return f"google-tn:{Path(corpus_dir).resolve().name}"


def training_shards(paths: Iterable[Path]) -> list[Path]:
    """``paths`` in order, less any shard whose name is in :data:`HELD_OUT_SHARDS` (a
    shard of another split, such as a fixture's ``output-00000-of-00002``, is kept)."""
    return [path for path in paths if Path(path).name not in HELD_OUT_SHARDS]


def expected(corpus_class: str, written: str, spoken: str) -> str:
    """The corpus's spoken form of one row, normalized as the spoken priors normalize:
    ``<self>`` expects the written token, ``sil`` expects nothing, and ELECTRONIC's
    per-letter notation is joined first."""
    from frend.electronic import decode_letter_notation
    from frend.spoken_priors import normalize_spoken

    if spoken == "<self>":
        return normalize_spoken(written)
    if spoken == "sil":
        return ""
    if corpus_class == "ELECTRONIC":
        spoken = decode_letter_notation(spoken)
    return normalize_spoken(spoken)


# Running text writes a range or a dimension as one word ("5-10", "3x4", "3:2"), where the
# corpus splits it into three tokens; these are rejoined by their written form alone.
JOINERS = frozenset({"-", "–", "x", ":"})


def _triples(sentences):
    """(sentence, index of the triple's left row) for each number, separator, number
    triple, left to right, never overlapping: the one predicate."""
    for sentence in sentences:
        at = 1
        while at < len(sentence) - 1:
            left, middle, right = sentence[at - 1], sentence[at], sentence[at + 1]
            if middle[1] in JOINERS and left[1][:1].isdigit() and right[1][:1].isdigit():
                yield sentence, at - 1
                at += 3
            else:
                at += 1


def running_text(sentences) -> list[tuple[str, str, str, str]]:
    """(separator, the middle's corpus reading, joined written, joined target) for each
    number, separator, number triple, left to right, never overlapping."""
    found = []
    for sentence, at in _triples(sentences):
        left, middle, right = sentence[at : at + 3]
        parts = [expected(*row) for row in (left, middle, right)]
        found.append(
            (
                middle[1],
                parts[1] or "(silence)",
                left[1] + middle[1] + right[1],
                " ".join(parts),
            )
        )
    return found


def running_text_contexts(sentences) -> list[tuple[str, str]]:
    """For each triple of :func:`running_text`, in the same order, the sentence's written
    text before it and after it (tokens joined by one space): the running text the joined
    triple is read in."""
    return [
        (
            " ".join(row[1] for row in sentence[:at]),
            " ".join(row[1] for row in sentence[at + 3 :]),
        )
        for sentence, at in _triples(sentences)
    ]


def range_triple_positions(sentences):
    """(sentence, index of the triple's left row) for each triple of
    :func:`range_triples`, in the same order."""
    for sentence in sentences:
        at = 1
        while at < len(sentence) - 1:
            left, middle, right = sentence[at - 1], sentence[at], sentence[at + 1]
            if (
                middle[1] in JOINERS
                and left[1][-1:] in _ASCII_DIGITS
                and any(char in _ASCII_DIGITS for char in right[1])
            ):
                yield sentence, at - 1
                at += 3
            else:
                at += 1


_ASCII_DIGITS = frozenset("0123456789")


def range_triples(sentences):
    """(left, middle, right) corpus rows, each (class, written, spoken), for each triple
    whose left written token ends in an ASCII digit, whose middle is a joiner and whose
    right written token holds an ASCII digit; left to right, never overlapping. A
    superset of :func:`running_text`'s predicate (the evaluator's, unchanged): it also
    reaches "$15,000 - $25,000" and "Oct. 29, 1951 - April 28, 1953", which the range
    rules then judge (``frend.ranges.emit_relevant``)."""
    for sentence, at in range_triple_positions(sentences):
        yield tuple(sentence[at : at + 3])
