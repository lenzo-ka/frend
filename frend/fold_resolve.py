"""Resolve an icukit candidate forest into descending non-overlapping covers,
as a tiergraph ranked path fold.

icukit deposits a *universe* of overlapping candidate detections (`1/3/2026`
arrives as a whole `date` span alongside the digit fragments `1`, `3`, `2026`).
The resolution question -- pick the maximum-weight set of pairwise
non-overlapping candidates -- is weighted interval scheduling, equivalently the
maximum-weight path through a position lattice. That path is a semiring fold, so
the resolver is a tiergraph fold parameter rather than bespoke machinery.

The lattice, per the H4-resolution design note:

  - nodes are code-point positions ``0..N`` (one ``positions`` tier item each);
  - each candidate is an edge ``start -> end`` carrying its weight, plus a unit
    ``skip`` edge ``p -> p+1`` of weight 0 so a full ``0..N`` path always exists
    even across uncovered text;
  - the fold uses ``PATH``: a position OR-combines the candidates offered there,
    and a candidate AND-requires its end position. Since PATH minimizes cost,
    candidate weights are negated so its ascending ranked witnesses are covers
    in descending score order.

Each real candidate's lattice weight is an exact mixed-radix integer Decimal.
Thus a cover's encoded fold weight is ``coverage * S * C - span_count * C +
capture_count``, where ``S`` is one greater than the furthest span end and ``C``
is one greater than the total captures in all valid candidates. Coverage is
strictly primary, fewer spans are secondary, and more captures break only ties
on both higher axes. This scalar is local to a lattice, not a portable candidate
score. Exposed scores and margins are recomputed structural facts, never decoded
from it.
"""

from __future__ import annotations

import heapq
from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from decimal import Context, Decimal, getcontext, localcontext
from functools import lru_cache
from math import prod
from typing import Any, cast

from tiergraph import (
    AttributeDeclaration,
    AttributeDomain,
    AttributeValuation,
    AttributeValue,
    BipartiteRelationDeclaration,
    ChildCombination,
    FoldDeclaration,
    FoldTransition,
    Graph,
    Item,
    ItemRef,
    NamespaceDeclaration,
    QualifiedName,
    RelationInstance,
    SimpleRelationDeclaration,
    Tier,
    TierDeclaration,
    XsdType,
)
from tiergraph.semiring import COUNTING, DECIMAL_TROPICAL, PATH, LexicographicSemiring

from frend.input_limits import DEFAULT_MAX_INPUT_CHARS, validate_input
from frend.locale_data import LOCALE_CACHE, canonical_locale
from frend.shape import shape
from frend.type_priors import (
    LOCALE_NEUTRAL,
    BlendedPrior,
    CorpusPrior,
    FeatureSource,
    ReadingPrior,
    ResolveContext,
)

__all__ = [
    "Candidate",
    "BlendedPrior",
    "CorpusPrior",
    "Cover",
    "CoverMargin",
    "CoverScore",
    "FeatureSource",
    "ReadingPrior",
    "Resolution",
    "ResolveContext",
    "SpanReading",
    "SpanResolution",
    "candidate_weight",
    "resolve",
    "resolve_cover",
]

NS = "https://ogion.org/frend/resolve"
_POS = QualifiedName(NS, "positions")
_POS_T = QualifiedName(NS, "position")
_CAND = QualifiedName(NS, "candidates")
_CAND_T = QualifiedName(NS, "candidate")
_OFFERS = QualifiedName(NS, "offers")  # position -> candidate (OR: alternatives)
_SPANS = QualifiedName(NS, "spans")  # candidate -> end position (AND: requirement)
_WEIGHT = QualifiedName(NS, "weight")

# A detection is a Mapping (icukit's ValueDetection is a TypedDict). Structural
# scoring reads start/end and the number of captures; the whole detection is
# carried through untouched.
Detection = Mapping[str, Any]
DEFAULT_EPSILON = 1


def _detection_span(detection: Detection) -> tuple[int, int]:
    start = detection["start"]
    end = detection["end"]
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        raise ValueError(f"detection offsets must be integers, got start={start!r} and end={end!r}")
    return start, end


def candidate_weight(detection: Detection) -> Decimal:
    """Return the portable geometric length of ``detection``.

    This compatibility helper is not the fold weight. The fold encoding also
    depends on the lattice's furthest span end and is computed in `_candidates`.
    """
    start, end = _detection_span(detection)
    length = end - start
    return Decimal(length)


@dataclass(frozen=True)
class Candidate:
    """One interval and its lattice-local fold-encoding weight."""

    index: int
    start: int
    end: int
    weight: Decimal
    detection: Detection


@dataclass(frozen=True)
class CoverScore:
    """Structural facts about a cover."""

    coverage: int
    span_count: int
    capture_count: int


@dataclass(frozen=True)
class CoverMargin:
    """Top cover's structural advantage over the runner-up, per axis.

    ``coverage`` and ``capture_count`` are the top's advantage (top minus
    runner-up); ``span_count`` is the runner-up's extra spans (runner-up minus
    top). All zero is a dead structural tie.
    """

    coverage: int
    span_count: int
    capture_count: int


@dataclass(frozen=True)
class Cover:
    """The resolved 1-best non-overlapping cover.

    ``best`` is the chosen detections in span order; ``score`` contains facts
    recomputed directly from those detections. ``priors`` is parallel to ``best``:
    the corpus :class:`~frend.type_priors.ReadingPrior` for each detection (carrying
    its ``supported`` state), or ``None`` for a type mapping to no corpus class."""

    best: tuple[Detection, ...]
    score: CoverScore
    priors: tuple[ReadingPrior | None, ...] = ()


@dataclass(frozen=True)
class SpanReading:
    """One candidate reading at a span, paired with its corpus prior.

    The unit of the per-span alternatives view: a detection that reads a given
    ``(start, end)`` span, and the :class:`~frend.type_priors.ReadingPrior` the
    corpus attests for it (``None`` for a type mapping to no corpus class)."""

    detection: Detection
    prior: ReadingPrior | None


@dataclass(frozen=True)
class SpanResolution:
    """The independent type choice at one span of the canonical structure ``s*``.

    Within a single identical-span equivalence class the type at each span is
    chosen independently of the others. ``readings`` are the candidate detections
    with maximal capture count at exactly ``(start, end)``, ranked: supported
    readings by base rate ``p`` descending, placed above all (mutually tied)
    unsupported readings. ``winner`` is the top-ranked reading. ``ambiguous`` is
    True unless there is a unique best -- exactly one supported reading with
    strictly-highest ``p``, or a single candidate; it is True when two-or-more
    supported readings share the maximum ``p``, or there is no supported reading
    and more than one candidate."""

    start: int
    end: int
    winner: Detection
    ambiguous: bool
    readings: tuple[SpanReading, ...]


@dataclass(frozen=True)
class Resolution:
    """The weighed reading of a candidate universe, resolved per span.

    Geometry ``(coverage, span_count, capture_count)`` and the full
    ``span_signature`` are the structural axes: they decide which spans form a
    cover and are never reordered by the prior. Within the top geometry level the
    covers are grouped by span signature; the canonical structure ``s* = min`` of
    those signatures is the representative, and each of its spans is resolved
    independently in ``spans`` (a :class:`SpanResolution` per span). ``best`` is
    the per-span winners of ``s*``.

    ``covers`` is the ``n``-best covers across geometry levels, descending: ordered
    by geometry (coverage, parsimony, then capture count), then span signature,
    then each reading's per-span rank, then a canonical content key, and capped at
    the requested ``n``.
    ``covers[0]`` is always ``best``. Only this legacy list depends on ``n``; the
    1-best cover, the ambiguity flags, and the per-span alternatives do not.
    ``margin`` compares the top cover to the runner-up ``covers[1]`` (top minus
    second coverage; second minus top span count) -- purely geometric facts,
    unchanged by Layer 2.

    ``priors`` is parallel to ``covers``: for each cover, the per-detection
    :class:`~frend.type_priors.ReadingPrior` (or ``None``). ``structural_ambiguous``
    is True when the top geometry level holds more than one span signature -- an
    ambiguity the prior is not allowed to resolve. ``semantic_ambiguous`` is True
    when any span of ``s*`` has no unique best reading. ``ambiguous`` is their
    disjunction."""

    best: tuple[Detection, ...]
    covers: tuple[tuple[Detection, ...], ...]
    margin: CoverMargin
    ambiguous: bool
    priors: tuple[tuple[ReadingPrior | None, ...], ...] = ()
    spans: tuple[SpanResolution, ...] = ()
    structural_ambiguous: bool = False
    semantic_ambiguous: bool = False


def _content_key(detection: Detection) -> tuple:
    """A canonical content key identifying a detection independent of object
    identity, matching icukit ``resolve.py``: two candidates that read the same
    span the same way collapse to one, so equal readings never split a cover's
    weight (which would otherwise show up as a spurious tie)."""
    captures = tuple(
        (getattr(c, "name", None), getattr(c, "start", None), getattr(c, "end", None))
        for c in detection.get("captures", ())
    )
    start, end = _detection_span(detection)
    return (
        start,
        end,
        detection.get("type"),
        repr(detection.get("value")),
        captures,
    )


def _dedupe(detections: Sequence[Detection]) -> list[Detection]:
    seen: set = set()
    unique: list[Detection] = []
    for detection in detections:
        key = _content_key(detection)
        if key not in seen:
            seen.add(key)
            unique.append(detection)
    return unique


def _span_priors(
    detections: Sequence[Detection],
    sources: Sequence[FeatureSource],
    context: ResolveContext,
) -> tuple[ReadingPrior | None, ...]:
    """Return each detection's prior, parallel to ``detections``.

    This is what :func:`_select` computes under ``collect_edge_priors`` for any
    input with at least one valid candidate, computed without it -- ``_select``
    early-returns an empty tuple when no cover exists at all, and this does not.
    That branch ranks the readings at each span of ``detections`` and never
    consults a cover, so no cover is enumerated here either -- which is what makes
    the prior available on inputs whose cover set is intractable.
    :func:`frend.lattice.resolve_choices` is pinned against ``_select``'s own value
    rather than trusted to agree with it.

    On detections that collide under ``_content_key``, this deliberately differs
    from ``_select``: object identity keeps each retained carrier reading paired
    with its own prior instead of the collision's first prior.

    Every reading at a span is ranked, including one whose capture count is below
    its span-mates'. That filter belongs to canonical selection, not to the prior,
    and a choice carrier must not inherit it.

    Ranking is span-local only insofar as the feature sources are: ``context``
    carries ``all_detections``, so a source may read the whole universe, and the
    icu-backfill tier renormalizes over the readings present at the span --
    changing not only ``p`` but ``tier`` and ``supported`` with it. Those fields
    are conditional on the detections supplied; ``generated_p`` is not, but it is
    a different conditional and exists only at the backfill tier.
    """
    corpus = _corpus_source(sources)
    ranked_by_span: dict[tuple[int, int], list[_RankedReading]] = {}
    for span in dict.fromkeys((int(det["start"]), int(det["end"])) for det in detections):
        at_span = [d for d in detections if (int(d["start"]), int(d["end"])) == span]
        ranked_by_span[span] = _rank_span_readings(at_span, sources, context, corpus)
    return tuple(
        next(
            reading.prior
            for reading in ranked_by_span[(int(det["start"]), int(det["end"]))]
            if reading.detection is det
        )
        for det in detections
    )


def _candidates(detections: Sequence[Detection]) -> tuple[list[Candidate], int]:
    valid: list[tuple[int, int, int, Detection]] = []
    span_end = 0
    for i, det in enumerate(detections):
        start, end = _detection_span(det)
        if end <= start or start < 0:
            # Precondition: detections are valid spans over nonnegative positions
            # (0 <= start < end), as detector output always is. Zero-length,
            # reversed, or negative-position spans are dropped here; on such
            # out-of-contract input this diverges from icukit's resolve, which
            # would keep them as degenerate covers. Dropping negative starts keeps
            # the lexicographic invariant below airtight (a span at a negative
            # position would not be one of the span_end covered positions).
            continue
        length = end - start
        assert length >= 1
        valid.append((i, start, end, det))
        span_end = max(span_end, end)
    span_radix = span_end + 1
    capture_radix = 1 + sum(len(det.get("captures", ())) for _, _, _, det in valid)
    cands = [
        Candidate(
            i,
            start,
            end,
            Decimal(
                (end - start) * span_radix * capture_radix
                - capture_radix
                + len(det.get("captures", ()))
            ),
            det,
        )
        for i, start, end, det in valid
    ]
    # Every stored span covers at least one distinct position, so a cover has at
    # most span_end spans. Its capture count is at most the total over all valid
    # candidates, hence strictly below capture_radix. Therefore one fewer span is
    # worth capture_radix and dominates every capture difference. One more unit
    # of coverage is worth (span_end + 1) * capture_radix, which exceeds the
    # maximum span disadvantage (span_end * capture_radix) plus the maximum
    # capture disadvantage (capture_radix - 1).
    return cands, span_end


def build_lattice(
    detections: Sequence[Detection],
    *,
    boundaries_only: bool = False,
) -> tuple[Graph, tuple[ItemRef, ...], dict[str, int]]:
    """Build the position lattice for ``detections``. Returns the graph, the
    root refs (``p0``), and a map from a candidate item's durable id to the
    index of the detection it stands for (skip edges are absent from the map).

    With ``boundaries_only``, the positions are only ``0``, the lattice's end and the
    candidates' starts and ends, and one skip edge joins each to the next. Text no
    candidate bounds is passed in one way whatever its length, so the covers, their
    count and their order are the same, over a lattice whose size follows the
    candidates rather than the text (a long URL is one skip, not one per character).
    The candidate weights are unchanged: they are computed from the real spans."""
    cands, span_end = _candidates(detections)
    if boundaries_only:
        points = sorted({0, span_end} | {c.start for c in cands} | {c.end for c in cands})
    else:
        points = list(range(span_end + 1))
    at = {point: index for index, point in enumerate(points)}

    pos_items = tuple(
        Item(f"p{p}", (AttributeValue(_WEIGHT, XsdType.DECIMAL, "0"),)) for p in range(len(points))
    )

    # candidates tier = real detections + skip edges. Durable ids are distinct
    # across both kinds ("c{i}" vs "skip{p}") and across the position tier
    # ("p{n}"), so a witness path is unambiguous to decode.
    cand_items: list[Item] = []
    offers: list[RelationInstance] = []
    spans: list[RelationInstance] = []
    id_to_index: dict[str, int] = {}

    def add(cand_id: str, start: int, end: int, weight: Decimal) -> None:
        ref = ItemRef(_CAND, len(cand_items))
        cand_items.append(Item(cand_id, (AttributeValue(_WEIGHT, XsdType.DECIMAL, str(weight)),)))
        offers.append(RelationInstance(_OFFERS, ItemRef(_POS, start), ref))
        spans.append(RelationInstance(_SPANS, ref, ItemRef(_POS, end)))

    for cand in cands:
        add(f"c{cand.index}", at[cand.start], at[cand.end], cand.weight)
        id_to_index[f"c{cand.index}"] = cand.index
    for p in range(len(points) - 1):
        add(f"skip{p}", p, p + 1, Decimal(0))

    graph = Graph(
        (NamespaceDeclaration("frend", NS),),
        (
            Tier(TierDeclaration(_POS, "Positions"), pos_items),
            Tier(TierDeclaration(_CAND, "Candidates"), tuple(cand_items)),
        ),
        (
            SimpleRelationDeclaration(QualifiedName(NS, "pos-membership"), _POS, _POS_T),
            SimpleRelationDeclaration(QualifiedName(NS, "cand-membership"), _CAND, _CAND_T),
            BipartiteRelationDeclaration(_OFFERS, _POS_T, _CAND_T, acyclic=True),
            BipartiteRelationDeclaration(_SPANS, _CAND_T, _POS_T, acyclic=True),
        ),
        tuple(offers) + tuple(spans),
        (AttributeDeclaration(_WEIGHT, AttributeDomain.ITEM, XsdType.DECIMAL),),
    )
    return graph, (ItemRef(_POS, 0),), id_to_index


def _span_signature(cover: Sequence[Detection]) -> tuple[tuple[int, int], ...]:
    """The full, canonical set of span boundaries a cover occupies.

    Two covers whose spans differ at all carry different signatures. Signatures
    partition the top geometry level into identical-span equivalence classes: the
    prior resolves the type choice *within* one class (a shared span set) and is
    never allowed to choose between classes -- that is a structural ambiguity."""
    return tuple(sorted((int(d["start"]), int(d["end"])) for d in cover))


def _geometry_rank(cover: Sequence[Detection]) -> tuple[int, int, int]:
    """The geometry equivalence class: ``(-coverage, span_count, -capture_count)``.

    Under the weight encoding of :func:`_candidates`, this rank and the fold's
    witness cost correspond one to one, in the same order. A cover's cost is
    ``-(coverage * span_radix * capture_radix - span_count * capture_radix +
    capture_count)``, and the bounds argued at the end of :func:`_candidates` --
    captures strictly below ``capture_radix``, spans at most ``span_end`` -- make
    that a mixed-radix numeral whose digits are exactly this triple. So two covers
    share a cost exactly when they share this rank, and the lower cost is the
    better rank. The level-whole gathering in :func:`_gather_top_geometry` relies
    on this: it reads a geometry level's boundary off the fold's cost.

    It is recomputed from the cover rather than read off the cost because the
    triple is what callers compare and report; the encoding is what keeps the two
    in agreement, and a test holds them to it."""
    score = _cover_score(cover)
    return (-score.coverage, score.span_count, -score.capture_count)


_COVER_TRANSITIONS = (
    FoldTransition(_OFFERS, ChildCombination.OR),
    FoldTransition(_SPANS, ChildCombination.AND),
)


def _exact_digits(graph: Graph) -> int:
    """The decimal digits a fold over ``graph`` needs to stay exact.

    A cover's cost is a sum of at most one weight per position, so its magnitude is
    below the largest weight times the number of positions. The weights are exact
    mixed-radix integers (:func:`_candidates`) whose size grows with the square of
    the lattice's furthest span end; Decimal arithmetic rounds past the context's
    precision (28 digits by default), which would merge distinct geometries."""
    positions, candidates = graph.tiers
    largest = max(
        (
            abs(int(Decimal(attribute.lexical)))
            for item in candidates.items
            for attribute in item.attributes
        ),
        default=0,
    )
    return len(str(largest * max(len(positions.items), 1))) + 2


def _exact(graph: Graph) -> AbstractContextManager[Context]:
    """A decimal context precise enough for every fold over ``graph``."""
    context = getcontext().copy()
    context.prec = max(context.prec, _exact_digits(graph))
    return localcontext(context)


def _count_covers(graph: Graph, roots: tuple[ItemRef, ...]) -> int:
    """Count the covers the lattice admits, exactly, without ranking any of them.

    The same lattice and transitions as the ranked fold, over ``COUNTING``: every
    item lifts to one, so a position sums its candidates' counts and a candidate
    carries its end position's. This is linear in the lattice, where the ranked fold
    pays for every witness it keeps, so the count is known before any witness is
    ranked. It is an exact int however large (2**96 and more); nothing converts it
    to a float or sizes anything by it."""
    fold = FoldDeclaration(
        "cover-count",
        graph,
        AttributeValuation("weight", _WEIGHT, (_POS, _CAND)),
        COUNTING,
        lambda _value, _label: COUNTING.one,
        _COVER_TRANSITIONS,
        roots=roots,
    )
    with _exact(graph):
        return cast(int, fold.run().value)


# The ranked fold's geometry, with every cover at the best geometry counted: the
# first component picks the least cost exactly as PATH's does, and on a tie the
# covers' counts add.
_TOP_LEVEL = LexicographicSemiring(DECIMAL_TROPICAL, COUNTING)


def _count_top_level(graph: Graph, roots: tuple[ItemRef, ...]) -> int:
    """Count the covers of the best geometry level, exactly, without ranking any.

    The ranked fold's lattice and transitions over ``(cost, count)``: an item lifts
    to its PATH cost with a count of one, alternatives keep the least cost and add
    the counts of those that tie at it, and a requirement adds costs and multiplies
    counts. The count is the size of the top level the ranked fold would have to
    emit whole, known before it emits any; like the total, it is an exact int."""
    fold = FoldDeclaration(
        "top-level-count",
        graph,
        AttributeValuation("weight", _WEIGHT, (_POS, _CAND)),
        _TOP_LEVEL,
        lambda value, _label: (-cast(Decimal, value), 1),
        _COVER_TRANSITIONS,
        roots=roots,
    )
    with _exact(graph):
        return cast(tuple[Decimal, int], fold.run().value)[1]


def _fold_covers(
    detections: Sequence[Detection],
    graph: Graph,
    roots: tuple[ItemRef, ...],
    id_to_index: Mapping[str, int],
    output_cap: int,
) -> tuple[list[tuple[Decimal, tuple[Detection, ...]]], bool]:
    """Run the geometry PATH fold and decode its ranked witnesses into covers.

    Returns ``(scored, truncated)``, where each entry pairs the fold's own witness
    value with the cover it decodes to, in the fold's ranked order. The value is
    carried rather than discarded because it is what actually produced the order:
    recomputing an equivalent here would be a second statement of the ranking that
    nothing keeps in agreement with the first. ``truncated`` is True when more
    witnesses exist than the cap emitted.
    The fold itself is unchanged -- geometry alone; the prior never enters here."""
    fold = FoldDeclaration(
        "cover",
        graph,
        AttributeValuation("weight", _WEIGHT, (_POS, _CAND)),
        PATH,
        lambda value, label: (-cast(Decimal, value), ((label,),)),
        _COVER_TRANSITIONS,
        roots=roots,
        output_cap=output_cap,
        ranked_output=True,
    )
    with _exact(graph):
        result = fold.run()
    scored: list[tuple[Decimal, tuple[Detection, ...]]] = []
    for value, labels in result.ranked_witnesses or ():
        indices = [id_to_index[label] for label in labels if label in id_to_index]
        indices.sort(key=lambda i: (int(detections[i]["start"]), int(detections[i]["end"])))
        scored.append((value, tuple(detections[i] for i in indices)))
    return scored, result.truncated


def _cover_score(cover: Sequence[Detection]) -> CoverScore:
    return CoverScore(
        coverage=sum(int(d["end"]) - int(d["start"]) for d in cover),
        span_count=len(cover),
        capture_count=sum(len(d.get("captures", ())) for d in cover),
    )


# Whole geometry levels are wanted, never a ranked prefix that ends inside one:
# canonical ``s*`` selection needs the top level whole, and the cover list and the
# margin need every level down to the one holding their last cover. A prefix
# returned as though it were complete is the failure to avoid; it looks exactly like
# an answer, so a level is used only once it is shown whole.
#
# The sentence is resolved per component of overlapping spans (``_components``), so
# this applies to one component's lattice. Its total covers are not bounded: they
# are counted exactly, as an int that may be astronomically large, never converted
# to a float and never sizing anything. What the ranked fold must emit whole is
# bounded instead, since its cost grows with the witnesses it keeps (about 4x in
# time and memory for every doubling of a tied top level). The component's top level
# is counted exactly first and refused past this bound, before any witness is
# ranked; a request that would still have to widen past it to show a lower needed
# level whole is refused too.
#
# Set from a measurement: over 3.57M sentences of the runtime-eval shards 90-94 no
# component's top level held more than 4 covers (p99.99 3), and a tied top level of
# 256 costs about 0.2 s and 17 MB to rank whole (tiergraph 0.4.1), 1,024 about 1.3 s
# and 100 MB, 4,096 about 8 s and 530 MB: 64 times the largest observed, at a cost
# that stays interactive.
_RANKED_LEVEL_BOUND = 256
#
# The ranked fold's first request when fewer covers are needed than the lattice
# admits. Its cost grows with the witnesses it keeps (measured on tiergraph 0.4.0:
# 0.4 s for 64 witnesses of a 3,240-cover sentence, 17 s for all of them), so it
# asks small and widens only when the geometry level it needs is not yet whole.
_FIRST_RANKED_REQUEST = 16


def _gather_top_geometry(
    detections: Sequence[Detection],
    needed: int,
) -> tuple[list[tuple[Decimal, tuple[Detection, ...]]], int]:
    """Return every cover of the best geometry levels, in the fold's ranked order,
    each paired with the fold's own witness value, and the count of all covers.

    See :func:`_gather_levels`, which also returns the first cover past them."""
    scored, count, _beyond = _gather_levels(detections, needed)
    return scored, count


def _gather_levels(
    detections: Sequence[Detection],
    needed: int,
    *,
    boundaries_only: bool = False,
) -> tuple[list[tuple[Decimal, tuple[Detection, ...]]], int, tuple[Detection, ...] | None]:
    """Return the whole best geometry levels in the fold's ranked order, the count of
    all covers, and the first cover past the returned levels (``None`` when they are
    every cover).

    The covers are counted exactly first, without ranking any; the total is never
    bounded and only caps each ranked request. The top level is counted exactly next,
    and past ``_RANKED_LEVEL_BOUND`` this refuses before ranking anything; a widening
    that would pass the bound refuses too. The returned levels are whole and are
    the fewest that together hold at least ``needed`` covers (every cover when the
    lattice admits no more than that). The fold ranks by geometry, and each geometry
    level is one witness value, so a level is shown whole by a ranked witness after it
    with a greater value: the fold's order is ascending and exact, so nothing of the
    level lies beyond that witness. That witness is the best cover of the next level,
    returned so a caller can read the next geometry without ranking that level. Until
    the fold emits one, it is asked for four times as many, up to the count itself.
    ``boundaries_only`` builds the lattice over candidate boundaries only
    (:func:`build_lattice`): the same covers, in a lattice sized by the candidates."""
    graph, roots, id_to_index = build_lattice(detections, boundaries_only=boundaries_only)
    if not id_to_index:
        return [], 1, None
    count = _count_covers(graph, roots)
    top = _count_top_level(graph, roots)
    if top > _RANKED_LEVEL_BOUND:
        raise ValueError(
            f"the top geometry level holds {top} covers (of {count} in all) over "
            f"{len(detections)} detections, past the {_RANKED_LEVEL_BOUND} the ranked "
            "fold is allowed to emit whole, so canonical selection cannot be shown "
            "complete and would run over a prefix"
        )
    # One witness past the ``needed`` shows at once whether the level holding the
    # last needed cover ends there. The bound limits the top level's size (above) and
    # how far the level holding the last needed cover may run past it, never the
    # number of covers asked for.
    ceiling = needed + _RANKED_LEVEL_BOUND + 1
    request = min(count, max(needed + 1, _FIRST_RANKED_REQUEST))
    while True:
        scored, truncated = _fold_covers(detections, graph, roots, id_to_index, request)
        if not truncated:
            return scored, count, None
        # A PATH witness value is (cost, paths); the cost alone is the geometry.
        last_level = scored[needed - 1][0][0]
        if scored[-1][0][0] != last_level:
            whole = [entry for entry in scored if entry[0][0] <= last_level]
            return whole, count, scored[len(whole)][1]
        # A level is shown whole by one witness past it, so ``ceiling`` witnesses
        # show whole levels holding one fewer covers.
        if request >= ceiling:
            raise ValueError(
                f"the geometry level holding cover {needed} of the cover order runs more than "
                f"{_RANKED_LEVEL_BOUND} covers past it (of {count} covers over "
                f"{len(detections)} detections), past what the ranked fold is allowed "
                "to emit whole, so the cover list cannot be shown complete and would "
                "be a prefix"
            )
        request = min(count, request * 4, ceiling)


# A component admitting at most this many covers is enumerated directly in Python;
# a larger one is gathered by the tiergraph fold (count first, top level bounded).
_DIRECT_ENUMERATION_MAX = 512


def _components(candidates: Sequence[Candidate]) -> list[list[Candidate]]:
    """Split the candidates into components of mutually overlapping spans.

    A cut falls only at a position no candidate crosses, so every cover of the
    sentence is exactly one local cover of each component (the empty one included),
    chosen independently, and the components lie in text order."""
    components: list[list[Candidate]] = []
    end = -1
    for candidate in sorted(candidates, key=lambda c: (c.start, c.end, c.index)):
        if not components or candidate.start >= end:
            components.append([])
        components[-1].append(candidate)
        end = max(end, candidate.end)
    return components


def _count_local(candidates: Sequence[Candidate]) -> int:
    """Count a component's local covers (non-overlapping subsets), exactly."""
    ordered = sorted(candidates, key=lambda c: (c.start, c.end, c.index))
    starts = [c.start for c in ordered]
    counts = [1] * (len(ordered) + 1)
    for i in range(len(ordered) - 1, -1, -1):
        counts[i] = counts[i + 1] + counts[bisect_left(starts, ordered[i].end, i + 1)]
    return counts[0]


def _enumerate_local(candidates: Sequence[Candidate]) -> list[tuple[Detection, ...]]:
    """Every local cover of a component, each in span order, the empty one included."""
    ordered = sorted(candidates, key=lambda c: (c.start, c.end, c.index))
    found: list[tuple[Detection, ...]] = []
    chosen: list[Detection] = []

    def extend(i: int, free_from: int) -> None:
        if i == len(ordered):
            found.append(tuple(chosen))
            return
        extend(i + 1, free_from)
        candidate = ordered[i]
        if candidate.start >= free_from:
            chosen.append(candidate.detection)
            extend(i + 1, candidate.end)
            chosen.pop()

    extend(0, 0)
    return found


@dataclass(frozen=True)
class _Local:
    """One component's local covers in cover order: a prefix of whole geometry
    levels holding at least the covers asked for (every cover when ``complete``),
    the count of all of them, and the geometry of the level after the top."""

    covers: tuple[tuple[Detection, ...], ...]
    count: int
    top_size: int
    second_geometry: tuple[int, int, int]


def _resolve_component(
    candidates: Sequence[Candidate],
    needed: int,
    cover_key: Callable[[tuple[Detection, ...]], tuple],
) -> _Local:
    count = _count_local(candidates)
    if count <= _DIRECT_ENUMERATION_MAX:
        covers = sorted(_enumerate_local(candidates), key=cover_key)
        beyond = None
    else:
        offset = min(c.start for c in candidates)
        originals = [c.detection for c in candidates]
        proxies = [
            {
                "start": int(d["start"]) - offset,
                "end": int(d["end"]) - offset,
                "captures": d.get("captures", ()),
            }
            for d in originals
        ]
        back = {id(proxy): original for proxy, original in zip(proxies, originals, strict=True)}
        scored, _count, beyond_proxy = _gather_levels(proxies, needed, boundaries_only=True)
        covers = sorted(
            (tuple(back[id(d)] for d in cover) for _value, cover in scored), key=cover_key
        )
        beyond = None if beyond_proxy is None else tuple(back[id(d)] for d in beyond_proxy)
    top_geometry = _geometry_rank(covers[0])
    top_size = sum(1 for cover in covers if _geometry_rank(cover) == top_geometry)
    below = covers[top_size] if top_size < len(covers) else beyond
    # A component has a candidate, so its empty cover lies strictly below the top.
    assert below is not None
    return _Local(tuple(covers), count, top_size, _geometry_rank(below))


def _merged_covers(
    locals_: Sequence[_Local],
    n: int,
    cover_key: Callable[[tuple[Detection, ...]], tuple],
) -> list[tuple[Detection, ...]]:
    """The ``n`` best sentence covers in cover order, drawn lazily from the
    components' ordered local covers without forming their product.

    A sentence cover is one local cover per component, concatenated in text order,
    and it is ordered by the same ``cover_key`` as any cover. Advancing one component
    to its next local cover never lowers that key: a worse local geometry worsens the
    summed geometry, and an equal one keeps that component's part of the signature,
    ranks and content keys the same length, so the concatenation orders as the part
    does. A best-first walk over index vectors therefore yields covers in exact key
    order, and it visits at most ``n`` covers and their successors."""

    def cover(vector: tuple[int, ...]) -> tuple[Detection, ...]:
        return tuple(d for c, i in enumerate(vector) for d in locals_[c].covers[i])

    start = (0,) * len(locals_)
    first = cover(start)
    heap: list[tuple[tuple, tuple[int, ...], tuple[Detection, ...]]] = [
        (cover_key(first), start, first)
    ]
    seen = {start}
    found: list[tuple[Detection, ...]] = []
    while heap and len(found) < n:
        _key, vector, best = heapq.heappop(heap)
        found.append(best)
        for c in range(len(vector)):
            following = vector[:c] + (vector[c] + 1,) + vector[c + 1 :]
            if following[c] < len(locals_[c].covers) and following not in seen:
                seen.add(following)
                candidate = cover(following)
                heapq.heappush(heap, (cover_key(candidate), following, candidate))
    return found


def _validated_sources(
    canonical: str, sources: tuple[FeatureSource, ...]
) -> tuple[FeatureSource, ...]:
    for source in sources:
        source_locale = getattr(source, "locale", None)
        if source_locale not in (canonical, LOCALE_NEUTRAL):
            raise ValueError(
                f"feature source {type(source).__name__} locale {source_locale!r} "
                f"does not match request locale {canonical!r}"
            )
    return sources


@lru_cache(maxsize=LOCALE_CACHE)
def _default_sources(locale: str) -> tuple[FeatureSource, ...]:
    """Load and validate the immutable runtime tables once for one canonical locale."""
    return _validated_sources(locale, (BlendedPrior(locale=locale),))


def _resolve_sources(
    locale: str,
    feature_sources: Sequence[FeatureSource] | None,
    class_prior: Mapping[str, Decimal | int] | None = None,
    class_prior_source: str | None = None,
) -> tuple[FeatureSource, ...]:
    """Default to the measured-first, ICU-backfilled runtime prior."""
    canonical = canonical_locale(locale)
    if feature_sources is None and class_prior is None and class_prior_source is None:
        return _default_sources(canonical)
    sources = (
        (
            BlendedPrior(
                locale=canonical,
                class_prior=class_prior,
                class_prior_source=class_prior_source,
            ),
        )
        if feature_sources is None
        else tuple(feature_sources)
    )
    return _validated_sources(canonical, sources)


def _corpus_source(sources: Sequence[FeatureSource]) -> CorpusPrior | BlendedPrior | None:
    for source in sources:
        if isinstance(source, (CorpusPrior, BlendedPrior)):
            return source
    return None


def _feature_support(
    detection: Detection,
    sources: Sequence[FeatureSource],
    context: ResolveContext,
) -> tuple[bool, Decimal]:
    """Return ``(supported, contribution_sum)`` from the feature sources.

    Every feature a source emits for the reading is a *supported* observation
    carrying a finite log weight; an unsupported reading emits none. ``supported``
    is True when at least one feature was emitted. Non-finite contributions are
    rejected at this boundary rather than left to corrupt the ordering (no
    ``-Infinity``, no fabrication). The summed contribution is the additive seam
    for Layer-3 cues and is *not* used for the Layer-2 per-span decision, which
    ranks on exact empirical probability (see :func:`_rank_span_readings`); it is
    returned only so the finiteness gate has one place."""
    supported = False
    contribution_sum = Decimal(0)
    for source in sources:
        for feature in source.features(detection, context):
            contribution = feature.contribution
            if not (isinstance(contribution, Decimal) and contribution.is_finite()):
                raise ValueError(
                    f"feature {feature.name!r} contributed a non-finite value "
                    f"{contribution!r}; prior contributions must be finite"
                )
            supported = True
            contribution_sum += contribution
    return supported, contribution_sum


@dataclass(frozen=True)
class _RankedReading:
    """A candidate reading at a span with everything the ranking derived from it.

    ``strength`` is the exact ordering key among supported readings: the empirical
    probability ``p`` (a Decimal) when the corpus attests one, else the additive
    feature contribution for a non-corpus source. It is never routed through
    ``float``, so exact probabilities that differ only in their least significant
    digits still order and compare correctly."""

    detection: Detection
    supported: bool
    strength: Decimal
    prior: ReadingPrior | None


_TIER_RANK = {"measured": 0, "icu-backfill": 1, "unsupported": 2}


def _rank_span_readings(
    readings: Sequence[Detection],
    sources: Sequence[FeatureSource],
    context: ResolveContext,
    corpus: CorpusPrior | BlendedPrior | None,
) -> list[_RankedReading]:
    """Rank the candidate readings at one span.

    Supported readings come first, ordered by exact base rate ``p`` descending; all
    unsupported readings follow, mutually tied. The strength compared is the EXACT
    Decimal probability from the corpus reading prior -- never ``float(p)``, which
    would round distinct probabilities to an equal log and mis-report a tie. A
    canonical content key gives genuine ties a deterministic, base-rate-neutral
    order so indexing is stable; semantic ties are detected by comparing exact
    strengths, not indices. The feature sources are still evaluated (finiteness
    gate, and the support flag / additive seam for a non-corpus source)."""
    ranked: list[_RankedReading] = []
    for det in readings:
        feature_supported, contribution_sum = _feature_support(det, sources, context)
        prior = corpus.reading_prior(det) if corpus is not None else None
        if prior is not None and prior.tier == "measured" and prior.p is not None:
            # Exact empirical probability drives the per-span decision; no float.
            supported, strength = True, prior.p
        elif prior is not None:
            supported = prior.supported
            strength = prior.p or Decimal(0)
        else:
            supported, strength = feature_supported, contribution_sum
        ranked.append(
            _RankedReading(detection=det, supported=supported, strength=strength, prior=prior)
        )
    backfilled = [r for r in ranked if r.prior is not None and r.prior.tier == "icu-backfill"]
    if backfilled:
        weighted_by_group: dict[str, Decimal] = {}
        for reading in backfilled:
            assert reading.prior is not None
            weighted_by_group.setdefault(
                reading.prior.group,
                reading.strength
                * (
                    corpus.class_weight(reading.prior.group)
                    if isinstance(corpus, BlendedPrior)
                    else Decimal(1)
                ),
            )
        denominator = sum(weighted_by_group.values(), Decimal(0))
        for index, item in enumerate(ranked):
            if item.prior is None or item.prior.tier != "icu-backfill":
                continue
            if denominator == 0:
                ranked[index] = replace(
                    item,
                    supported=False,
                    strength=Decimal(0),
                    prior=replace(item.prior, p=None, supported=False, tier="unsupported"),
                )
            else:
                posterior = weighted_by_group[item.prior.group] / denominator
                ranked[index] = replace(
                    item, strength=posterior, prior=replace(item.prior, p=posterior)
                )

    def tier(reading: _RankedReading) -> str:
        if reading.prior is not None:
            return reading.prior.tier
        return "measured" if reading.supported else "unsupported"

    ranked.sort(key=lambda r: (_TIER_RANK[tier(r)], -r.strength, _content_key(r.detection)))
    return ranked


def _span_is_ambiguous(ranked: Sequence[_RankedReading]) -> bool:
    """Whether the span has no unique best reading.

    False only when there is a single candidate, or exactly one supported reading
    holds the strictly-highest exact strength. True when two-or-more supported
    readings share the maximum strength, or there is no supported reading and more
    than one candidate (no basis to choose). Comparison is exact Decimal equality,
    so a genuine probability tie is reported and a rounding artifact is not."""
    if len(ranked) == 1:
        return False
    first = ranked[0]
    first_tier = (
        first.prior.tier
        if first.prior is not None
        else ("measured" if first.supported else "unsupported")
    )
    return any(
        (r.prior.tier if r.prior is not None else ("measured" if r.supported else "unsupported"))
        == first_tier
        and r.strength == first.strength
        for r in ranked[1:]
    )


@dataclass(frozen=True)
class _Selection:
    best: tuple[Detection, ...]
    covers: tuple[tuple[Detection, ...], ...]
    spans: tuple[SpanResolution, ...]
    structural_ambiguous: bool
    semantic_ambiguous: bool
    ambiguous: bool
    margin: CoverMargin
    corpus: CorpusPrior | BlendedPrior | None
    priors: tuple[tuple[ReadingPrior | None, ...], ...]
    edge_priors: tuple[ReadingPrior | None, ...]
    truncated: bool


def _select(
    detections: Sequence[Detection],
    sources: Sequence[FeatureSource],
    context: ResolveContext,
    n: int,
    *,
    projection_probe: bool = False,
    collect_edge_priors: bool = False,
) -> _Selection:
    """Resolve the candidate universe by per-span selection.

    Geometry and the span signature are primary and untouched by the prior. The
    complete top geometry level is grouped by span signature; the canonical
    structure ``s* = min`` of the signatures is the representative. Each span of
    ``s*`` is resolved independently -- the type choice at one span does not depend
    on the others -- so the 1-best cover is the per-span winners, and ambiguity is
    the disjunction of a structural ambiguity (more than one span signature) and a
    semantic one (any span without a unique best). The 1-best cover, the ambiguity
    flags, and the per-span alternatives derive entirely from the (complete) top
    geometry level, so they do not depend on ``n``; ``n`` only caps the legacy
    ``covers`` list, which additionally ranks the lower geometry levels."""
    corpus = _corpus_source(sources)
    candidates = _candidates(detections)[0]
    if not candidates:
        # No valid candidates: the sole reading is the empty cover.
        return _Selection(
            (),
            ((),),
            (),
            False,
            False,
            False,
            CoverMargin(0, 0, 0),
            corpus,
            ((),),
            (),
            False,
        )

    # Rank the structurally maximal-capture readings at every span a cover can
    # occupy, so both the per-span resolution (over s*) and the cover ordering can
    # index into a stable ranking. Reading sets are gathered from ALL detections at
    # each span, independent of any cover, so the per-span view over s* is complete.
    all_span_readings: dict[tuple[int, int], list[_RankedReading]] = {}
    span_readings: dict[tuple[int, int], list[_RankedReading]] = {}
    span_sequence = (
        ((int(det["start"]), int(det["end"])) for det in detections)
        if collect_edge_priors
        # Every valid candidate lies on some cover, so these are the spans of all covers.
        else ((candidate.start, candidate.end) for candidate in candidates)
    )
    spans_to_rank = dict.fromkeys(span_sequence)
    for span in spans_to_rank:
        at_span = [d for d in detections if (int(d["start"]), int(d["end"])) == span]
        all_ranked = _rank_span_readings(at_span, sources, context, corpus)
        all_span_readings[span] = all_ranked
        max_capture_count = max(len(d.get("captures", ())) for d in at_span)
        span_readings[span] = [
            reading
            for reading in all_ranked
            if len(reading.detection.get("captures", ())) == max_capture_count
        ]

    # A long cover list used to rebuild every detection's structural key and every
    # span's rank list for every comparison.  Both are invariant for this resolve.
    content_keys = {id(detection): _content_key(detection) for detection in detections}
    rank_by_span = {
        span: {content_keys[id(reading.detection)]: rank for rank, reading in enumerate(readings)}
        for span, readings in span_readings.items()
    }

    # Rank the covers deterministically: geometry first (never touched by the prior),
    # then span signature, then each reading's per-span rank (so the per-span winners
    # sort first within a signature), then a canonical content key.
    def cover_key(cover: tuple[Detection, ...]) -> tuple:
        geometry = _geometry_rank(cover)
        ranks: list[int] = []
        for det in cover:
            span = (int(det["start"]), int(det["end"]))
            key = content_keys[id(det)]
            ranks.append(rank_by_span[span].get(key, len(rank_by_span[span])))
        return (
            geometry,
            _span_signature(cover),
            tuple(ranks),
            tuple(content_keys[id(d)] for d in cover),
        )

    # Each component of mutually overlapping spans is resolved alone; a sentence
    # cover is one local cover of each. The top geometry level is the product of the
    # components' top levels (geometry adds, and a sum is least only where every
    # part is), so it is read per component and never formed.
    locals_ = [_resolve_component(component, n, cover_key) for component in _components(candidates)]

    # Within a component's top level every cover has the same span count, so the
    # signatures there have one length and the least sentence signature is the
    # components' least ones in text order.
    s_star_parts: list[tuple[tuple[int, int], ...]] = []
    structural_ambiguous = False
    for local in locals_:
        signatures = sorted({_span_signature(cover) for cover in local.covers[: local.top_size]})
        structural_ambiguous = structural_ambiguous or len(signatures) > 1
        s_star_parts.append(signatures[0])
    s_star = tuple(span for part in s_star_parts for span in part)

    span_resolutions: list[SpanResolution] = []
    semantic_ambiguous = False
    for span in s_star:
        ranked = span_readings[span]
        ambiguous = _span_is_ambiguous(ranked)
        semantic_ambiguous = semantic_ambiguous or ambiguous
        span_resolutions.append(
            SpanResolution(
                start=span[0],
                end=span[1],
                winner=ranked[0].detection,
                ambiguous=ambiguous,
                readings=tuple(SpanReading(r.detection, r.prior) for r in ranked),
            )
        )
    # s_star is a sorted tuple of spans, so the winners are already in span order.
    # This is precisely the all-rank-0 cover of s*, so it is also ordered[0].
    best = tuple(sr.winner for sr in span_resolutions)
    ambiguous = structural_ambiguous or semantic_ambiguous

    # The runner-up shares the top geometry when the top level holds two or more
    # covers; otherwise it is the top with the one component that loses least moved
    # to its next level. In rank terms that loss is exactly the margin's triple.
    if prod(local.top_size for local in locals_) > 1:
        margin = CoverMargin(0, 0, 0)
    else:
        losses = []
        for local in locals_:
            top_rank = _geometry_rank(local.covers[0])
            losses.append(
                tuple(b - t for b, t in zip(local.second_geometry, top_rank, strict=True))
            )
        margin = CoverMargin(*min(losses))

    ordered = _merged_covers(locals_, n, cover_key)
    truncated = prod(local.count for local in locals_) > n
    edge_priors = (
        tuple(
            next(
                reading.prior
                for reading in all_span_readings[(int(det["start"]), int(det["end"]))]
                if _content_key(reading.detection) == _content_key(det)
            )
            for det in detections
        )
        if collect_edge_priors
        else ()
    )
    return _Selection(
        best=best,
        covers=tuple(ordered),
        spans=tuple(span_resolutions),
        structural_ambiguous=structural_ambiguous,
        semantic_ambiguous=semantic_ambiguous,
        ambiguous=ambiguous,
        margin=margin,
        corpus=corpus,
        priors=tuple(_cover_priors(cover, all_span_readings) for cover in ordered),
        edge_priors=edge_priors,
        truncated=truncated,
    )


def _cover_priors(
    cover: Sequence[Detection],
    span_readings: Mapping[tuple[int, int], Sequence[_RankedReading]],
) -> tuple[ReadingPrior | None, ...]:
    priors: list[ReadingPrior | None] = []
    for detection in cover:
        span = (int(detection["start"]), int(detection["end"]))
        content = _content_key(detection)
        ranked = span_readings.get(span, ())
        match = next(reading for reading in ranked if _content_key(reading.detection) == content)
        if match.prior is not None:
            priors.append(match.prior)
            continue
        type_ = str(detection.get("type", ""))
        priors.append(
            ReadingPrior(
                group=type_.partition(":")[0] or shape(str(detection.get("text", ""))),
                shape=shape(str(detection.get("text", ""))),
                p=None,
                n=None,
                supported=False,
                tier="unsupported",
                provenance="unsupported",
            )
        )
    return tuple(priors)


def resolve_cover(
    detections: Sequence[Detection],
    *,
    locale: str = "en_US",
    feature_sources: Sequence[FeatureSource] | None = None,
    source_text: str | None = None,
    class_prior: Mapping[str, Decimal | int] | None = None,
    class_prior_source: str | None = None,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
) -> Cover:
    """Resolve ``detections`` to a maximum-weight non-overlapping cover via a
    tiergraph ``PATH`` fold over negated weights, then per-span type selection within the canonical
    structure. Detections may overlap and nest on input (that is icukit's
    deposited universe); the returned ``best`` is pairwise non-overlapping and in
    span order. Use :func:`resolve` to see the per-span alternatives and the
    ambiguity flags."""
    validate_input(source_text, max_input_chars=max_input_chars)
    sources = _resolve_sources(locale, feature_sources, class_prior, class_prior_source)
    if not detections:
        return Cover(best=(), score=CoverScore(0, 0, 0))
    unique = _dedupe(detections)
    context = ResolveContext(source_text=source_text, all_detections=tuple(unique))
    selection = _select(unique, sources, context, n=1)
    return Cover(
        best=selection.best,
        score=_cover_score(selection.best),
        priors=selection.priors[0],
    )


def resolve(
    detections: Sequence[Detection],
    *,
    locale: str = "en_US",
    n: int = 8,
    epsilon: int = DEFAULT_EPSILON,
    feature_sources: Sequence[FeatureSource] | None = None,
    source_text: str | None = None,
    class_prior: Mapping[str, Decimal | int] | None = None,
    class_prior_source: str | None = None,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
) -> Resolution:
    """Resolve detections into a per-span reading of the candidate universe.

    Geometry (coverage, then parsimony, then capture count) and the full span
    signature decide which spans form a cover. Within the top geometry level the
    type at each span is chosen independently by ``feature_sources`` -- by default
    the corpus base-rate prior only (Layer 2). ``source_text`` is offered to
    feature sources through :class:`ResolveContext` for future context cues; the
    corpus prior ignores it.

    The correctness-critical outputs -- the 1-best cover, the ``ambiguous`` flag,
    and the per-span alternatives (``spans``) -- do not depend on ``n``, which only
    caps the ``covers`` list (the n-best covers across geometry levels, descending).
    ``ambiguous`` is True when the top geometry level holds more than one span
    signature (a structural ambiguity the prior may not resolve) or when any span
    has no unique best reading. ``epsilon`` is deprecated and ignored.

    ``n`` must be a positive integer; a non-positive or non-integer ``n`` is
    rejected so the ``covers[0] == best`` contract can never be broken by an empty
    ``covers`` list sitting beside a populated ``best``.
    """
    del epsilon
    validate_input(source_text, max_input_chars=max_input_chars)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    sources = _resolve_sources(locale, feature_sources, class_prior, class_prior_source)
    if not detections:
        return Resolution(best=(), covers=((),), margin=CoverMargin(0, 0, 0), ambiguous=False)
    unique = _dedupe(detections)
    context = ResolveContext(source_text=source_text, all_detections=tuple(unique))
    selection = _select(unique, sources, context, n=n)
    return Resolution(
        best=selection.best,
        covers=selection.covers,
        margin=selection.margin,
        ambiguous=selection.ambiguous,
        priors=selection.priors,
        spans=selection.spans,
        structural_ambiguous=selection.structural_ambiguous,
        semantic_ambiguous=selection.semantic_ambiguous,
    )
