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
separator, number predicate the evaluator scores (and the planned range builder will
count).
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


def running_text(sentences) -> list[tuple[str, str, str, str]]:
    """(separator, the middle's corpus reading, joined written, joined target) for each
    number, separator, number triple, left to right, never overlapping."""
    found = []
    for sentence in sentences:
        at = 1
        while at < len(sentence) - 1:
            left, middle, right = sentence[at - 1], sentence[at], sentence[at + 1]
            if middle[1] in JOINERS and left[1][:1].isdigit() and right[1][:1].isdigit():
                parts = [expected(*row) for row in (left, middle, right)]
                found.append(
                    (
                        middle[1],
                        parts[1] or "(silence)",
                        left[1] + middle[1] + right[1],
                        " ".join(parts),
                    )
                )
                at += 3
            else:
                at += 1
    return found
