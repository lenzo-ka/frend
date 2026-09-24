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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
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
from tiergraph.semiring import PATH

from frend.shape import shape
from frend.type_priors import (
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


def candidate_weight(detection: Detection) -> Decimal:
    """Return the portable geometric length of ``detection``.

    This compatibility helper is not the fold weight. The fold encoding also
    depends on the lattice's furthest span end and is computed in `_candidates`.
    """
    length = int(detection["end"]) - int(detection["start"])
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
    return (
        int(detection["start"]),
        int(detection["end"]),
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
        start = int(det["start"])
        end = int(det["end"])
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
) -> tuple[Graph, tuple[ItemRef, ...], dict[str, int]]:
    """Build the position lattice for ``detections``. Returns the graph, the
    root refs (``p0``), and a map from a candidate item's durable id to the
    index of the detection it stands for (skip edges are absent from the map)."""
    cands, span_end = _candidates(detections)

    pos_items = tuple(
        Item(f"p{p}", (AttributeValue(_WEIGHT, XsdType.DECIMAL, "0"),)) for p in range(span_end + 1)
    )

    # candidates tier = real detections + unit skip edges. Durable ids are
    # distinct across both kinds ("c{i}" vs "skip{p}") and across the position
    # tier ("p{n}"), so a witness path is unambiguous to decode.
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
        add(f"c{cand.index}", cand.start, cand.end, cand.weight)
        id_to_index[f"c{cand.index}"] = cand.index
    for p in range(span_end):
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

    This is a COARSENING of the fold's own witness value, not a restatement of it.
    The fold ranks by a weight that varies per position and candidate, so two covers
    can share this rank and still carry different fold values -- grouping by the
    fold value instead splits classes that structural ambiguity is defined over,
    which is what makes two span signatures at one geometry a structural ambiguity
    rather than a ranking. Measured: doing so drops structural ambiguity and changes
    a resolved winner.

    Ordering agrees with the fold's, so the ranked emission is still in this order;
    only equality is coarser."""
    score = _cover_score(cover)
    return (-score.coverage, score.span_count, -score.capture_count)


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
        (
            FoldTransition(_OFFERS, ChildCombination.OR),
            FoldTransition(_SPANS, ChildCombination.AND),
        ),
        roots=roots,
        output_cap=output_cap,
        ranked_output=True,
    )
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


# Every cover the lattice admits is wanted, not a ranked prefix: canonical ``s*``
# selection needs the top geometry level whole, and the lower levels carry the
# geometric margin. tiergraph's fold has no spelling for "all witnesses" -- its
# ``output_cap`` must be a positive integer -- so this asks for a bound far above
# any input this pipeline meets, and refuses when the fold reports the bound was
# reached. A prefix returned as though it were complete is the failure to avoid;
# it looks exactly like an answer.
_COVER_ENUMERATION_CAP = 1 << 16


def _gather_top_geometry(
    detections: Sequence[Detection],
) -> list[tuple[Decimal, tuple[Detection, ...]]]:
    """Return every cover the lattice admits, in the fold's ranked order, each
    paired with the fold's own witness value.

    The PATH fold ranks witnesses by geometry alone. This asks once, for more
    witnesses than any practical input produces, and refuses if the fold says it
    truncated -- rather than discovering completeness by doubling a cap and
    inferring it from where the ranking fell. The result is complete, so the top
    same-span equivalence classes never depend on any requested ``n``, and the
    lower geometry levels the margin needs are present by construction."""
    graph, roots, id_to_index = build_lattice(detections)
    if not id_to_index:
        return []
    scored, truncated = _fold_covers(detections, graph, roots, id_to_index, _COVER_ENUMERATION_CAP)
    if truncated:
        raise ValueError(
            f"cover enumeration reached the {_COVER_ENUMERATION_CAP} witness bound over "
            f"{len(detections)} detections, so the top geometry level cannot be shown "
            "complete and canonical selection would run over a prefix"
        )
    return scored


def _resolve_sources(
    feature_sources: Sequence[FeatureSource] | None,
    class_prior: Mapping[str, Decimal | int] | None = None,
    class_prior_source: str | None = None,
) -> tuple[FeatureSource, ...]:
    """Default to the measured-first, ICU-backfilled runtime prior."""
    return (
        (BlendedPrior(class_prior=class_prior, class_prior_source=class_prior_source),)
        if feature_sources is None
        else tuple(feature_sources)
    )


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


def _margin(ordered: Sequence[tuple[Detection, ...]]) -> CoverMargin:
    """Coverage advantage of the top cover over the runner-up.

    The runner-up is simply the second cover in ranked order: its geometry is the
    top geometry when the top level holds two-or-more covers (a dead ``(0, 0, 0)``
    tie), and the best strictly-worse geometry otherwise. Purely geometric; the
    prior never enters here."""
    if len(ordered) < 2:
        return CoverMargin(0, 0, 0)
    top = _cover_score(ordered[0])
    second = _cover_score(ordered[1])
    return CoverMargin(
        top.coverage - second.coverage,
        second.span_count - top.span_count,
        top.capture_count - second.capture_count,
    )


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
    scored = _gather_top_geometry(detections)
    if not scored:
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

    covers = [cover for _value, cover in scored]
    best_geo = min(_geometry_rank(cover) for cover in covers)
    top = [cover for cover in covers if _geometry_rank(cover) == best_geo]

    # Rank the structurally maximal-capture readings at every span occupied by any
    # gathered cover, so
    # both the per-span resolution (over s*) and the legacy cover ordering can index
    # into a stable ranking. Reading sets are gathered from ALL detections at each
    # span, independent of the fold's emission order or the requested n, so the
    # per-span view over s* is complete.
    all_span_readings: dict[tuple[int, int], list[_RankedReading]] = {}
    span_readings: dict[tuple[int, int], list[_RankedReading]] = {}
    span_sequence = (
        ((int(det["start"]), int(det["end"])) for det in detections)
        if collect_edge_priors
        else ((int(det["start"]), int(det["end"])) for cover in covers for det in cover)
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

    signatures = sorted({_span_signature(cover) for cover in top})
    structural_ambiguous = len(signatures) > 1
    s_star = signatures[0]

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
    # This is precisely the all-rank-0 cover of s*, so it is also ranked_covers[0].
    best = tuple(sr.winner for sr in span_resolutions)
    ambiguous = structural_ambiguous or semantic_ambiguous

    # Rank the covers deterministically: geometry first (never touched by the prior),
    # then span signature, then each reading's per-span rank (so the per-span winners
    # sort first within a signature), then a canonical content key. Capped at n.
    def cover_key(cover: tuple[Detection, ...]) -> tuple:
        geometry = _geometry_rank(cover)
        ranks: list[int] = []
        for det in cover:
            span = (int(det["start"]), int(det["end"]))
            order = [_content_key(r.detection) for r in span_readings[span]]
            key = _content_key(det)
            ranks.append(order.index(key) if key in order else len(order))
        return (
            geometry,
            _span_signature(cover),
            tuple(ranks),
            tuple(_content_key(d) for d in cover),
        )

    ordered = sorted(covers, key=cover_key)
    margin = _margin(ordered)
    truncated = len(ordered) > n
    ordered = ordered[:n]
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
    feature_sources: Sequence[FeatureSource] | None = None,
    source_text: str | None = None,
    class_prior: Mapping[str, Decimal | int] | None = None,
    class_prior_source: str | None = None,
) -> Cover:
    """Resolve ``detections`` to a maximum-weight non-overlapping cover via a
    tiergraph ``PATH`` fold over negated weights, then per-span type selection within the canonical
    structure. Detections may overlap and nest on input (that is icukit's
    deposited universe); the returned ``best`` is pairwise non-overlapping and in
    span order. Use :func:`resolve` to see the per-span alternatives and the
    ambiguity flags."""
    if not detections:
        return Cover(best=(), score=CoverScore(0, 0, 0))
    unique = _dedupe(detections)
    sources = _resolve_sources(feature_sources, class_prior, class_prior_source)
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
    n: int = 8,
    epsilon: int = DEFAULT_EPSILON,
    feature_sources: Sequence[FeatureSource] | None = None,
    source_text: str | None = None,
    class_prior: Mapping[str, Decimal | int] | None = None,
    class_prior_source: str | None = None,
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
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not detections:
        return Resolution(best=(), covers=((),), margin=CoverMargin(0, 0, 0), ambiguous=False)
    unique = _dedupe(detections)
    sources = _resolve_sources(feature_sources, class_prior, class_prior_source)
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
