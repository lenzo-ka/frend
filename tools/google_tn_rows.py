"""What every Google-TN builder and the evaluator share: the held-out shards and the
running-text triple.

Two shards of the en_with_types corpus are never counted by a shipped table:

* ``output-00099-of-00100``, the published test shard (Sproat & Jaitly 2016 test on
  its first 100,000 lines), which ``tools/evaluate_google_tn.py`` reports; and
* ``output-00095-of-00100``, the held-out shard a change is accepted and selected on
  (``--held-out-shard``).

Every builder reads its shards through :func:`training_shards`, so neither can reach
a table however the builder slices the corpus. :func:`running_text` is the one
number, separator, number predicate the evaluator scores and a range builder counts.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

HELD_OUT_SHARDS = frozenset({"output-00095-of-00100", "output-00099-of-00100"})


def training_shards(paths: Iterable[Path]) -> list[Path]:
    """``paths`` in order, less any shard whose name is in :data:`HELD_OUT_SHARDS`."""
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
