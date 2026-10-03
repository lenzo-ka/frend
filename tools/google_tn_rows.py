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
  change is accepted and selected on (``--held-out-shard --skip-report-shard`` keeps
  decision runs from verifying or scoring shard 99).

:data:`HELD_OUT_SHARDS` is the runtime-eval and test pools together. Every builder reads
its shards through :func:`training_shards`, so no held-out shard can reach a table
however the builder slices the corpus. :func:`corpus_label` names the corpus a table
records, from the directory the builder read. :func:`running_text` is the one number,
separator, number predicate the evaluator scores; :func:`range_triples` is the wider
one the range builders count (a left token that ends in a digit, such as "$15,000").
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from frend.locale_data import canonical_locale


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


def expected(
    corpus_class: str, written: str, spoken: str, *, strip_embedded_sil: bool = False
) -> str:
    """The corpus's spoken form of one row, normalized as the spoken priors normalize:
    ``<self>`` expects the written token, ``sil`` expects nothing, and ELECTRONIC's
    per-letter notation is joined first."""
    from frend.electronic import decode_letter_notation
    from frend.spoken_priors import normalize_spoken

    if spoken == "<self>":
        return normalize_spoken(written)
    if spoken == "sil":
        return ""
    if strip_embedded_sil:
        spoken = " ".join(word for word in spoken.split() if word != "sil")
    if corpus_class == "ELECTRONIC":
        spoken = decode_letter_notation(spoken)
    return normalize_spoken(spoken)


type CorpusRow = tuple[str, str, str]
RANGE_CANDIDATE_SEPARATORS = frozenset({"-", "–", "—", "x", ":"})


@dataclass(frozen=True)
class RangeCandidate:
    left: CorpusRow
    middle: CorpusRow
    right: CorpusRow
    separator: str


def _candidate_positions(
    sentences,
    separators,
    *,
    endpoints_start_digit: bool = False,
    exclude_chains: bool = True,
):
    for sentence in sentences:
        at = 1
        while at < len(sentence) - 1:
            left, middle, right = sentence[at - 1], sentence[at], sentence[at + 1]
            endpoints_match = (
                left[1][:1] in _ASCII_DIGITS and right[1][:1] in _ASCII_DIGITS
                if endpoints_start_digit
                else left[1][-1:] in _ASCII_DIGITS
                and any(char in _ASCII_DIGITS for char in right[1])
            )
            if (
                middle[1] in separators
                and endpoints_match
                and (not exclude_chains or not _chained(sentence, at, separators))
            ):
                yield RangeCandidate(left, middle, right, middle[1]), sentence, at - 1
                at += 3
            else:
                at += 1


def range_candidates(
    sentences,
    *,
    separators=RANGE_CANDIDATE_SEPARATORS,
    endpoints_start_digit: bool = False,
    exclude_chains: bool = True,
):
    """Yield the shared, non-overlapping corpus range-candidate inventory."""
    separators = frozenset(separators)
    for candidate, _sentence, _at in _candidate_positions(
        sentences,
        separators,
        endpoints_start_digit=endpoints_start_digit,
        exclude_chains=exclude_chains,
    ):
        yield candidate


def range_candidate_denominators(sentences) -> Counter:
    return Counter(
        (item.separator, item.middle[0], item.middle[2]) for item in range_candidates(sentences)
    )


def _operational_separators(locale: str) -> frozenset[str]:
    """Range separators scored and trained for ``locale``.

    English is the frozen published population, which predates em-dash support. Other
    locales use the complete locale candidate inventory.
    """
    canonical = canonical_locale(locale)
    return (
        RANGE_CANDIDATE_SEPARATORS - {"—"}
        if canonical.partition("_")[0] == "en"
        else RANGE_CANDIDATE_SEPARATORS
    )


def running_text(sentences, locale: str = "en_US") -> list[tuple[str, str, str, str]]:
    """(separator, the middle's corpus reading, joined written, joined target) for each
    number, separator, number triple, left to right, never overlapping."""
    found = []
    # Preserve the published English evaluator's frozen population while sourcing its
    # positions from the shared extractor: the historical report used four separators,
    # endpoints beginning with digits, and did not remove chains.
    for candidate in range_candidates(
        sentences,
        separators=_operational_separators(locale),
        endpoints_start_digit=True,
        exclude_chains=False,
    ):
        left, middle, right = candidate.left, candidate.middle, candidate.right
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


def running_text_contexts(sentences, locale: str = "en_US") -> list[tuple[str, str]]:
    """For each triple of :func:`running_text`, in the same order, the sentence's written
    text before it and after it (tokens joined by one space): the running text the joined
    triple is read in."""
    return [
        (
            " ".join(row[1] for row in sentence[:at]),
            " ".join(row[1] for row in sentence[at + 3 :]),
        )
        for _candidate, sentence, at in _candidate_positions(
            sentences,
            _operational_separators(locale),
            endpoints_start_digit=True,
            exclude_chains=False,
        )
    ]


def range_triple_positions(sentences, locale: str = "en_US"):
    """(sentence, index of the triple's left row) for each triple of
    :func:`range_triples`, in the same order."""
    for _candidate, sentence, at in _candidate_positions(
        sentences, _operational_separators(locale)
    ):
        yield sentence, at


_ASCII_DIGITS = frozenset("0123456789")


def _chained(sentence, at: int, separators=RANGE_CANDIDATE_SEPARATORS) -> bool:
    """Whether the triple around ``sentence[at]`` is a fragment of a chain ("1 - 2 - 3",
    "2008 - 09 - 30"): a joiner then a number after its right end, or a number then a
    joiner before its left."""
    after = sentence[at + 2 : at + 4]
    before = sentence[max(0, at - 3) : at - 1]
    return (len(after) == 2 and after[0][1] in separators and after[1][1][:1] in _ASCII_DIGITS) or (
        len(before) == 2 and before[1][1] in separators and before[0][1][-1:] in _ASCII_DIGITS
    )


def range_triples(sentences, locale: str = "en_US"):
    """(left, middle, right) corpus rows, each (class, written, spoken), for each triple
    whose left written token ends in an ASCII digit, whose middle is a joiner and whose
    right written token holds an ASCII digit, and that is no fragment of a chain ("1 - 2
    - 3"); left to right, never overlapping. A
    superset of :func:`running_text`'s predicate (the evaluator's, unchanged): it also
    reaches "$15,000 - $25,000" and "Oct. 29, 1951 - April 28, 1953", which the range
    rules then judge (``frend.ranges.emit_relevant``)."""
    for candidate in range_candidates(sentences, separators=_operational_separators(locale)):
        yield candidate.left, candidate.middle, candidate.right
