"""Immutable, consumer-facing readings distilled from the resolver lattice.

This module deliberately does not expose the tiergraph execution graph or its
lattice-local mixed-radix weights. Geometry remains primary. Semantic priors are
only choices among readings of the canonical top span signature, as in
:func:`frend.fold_resolve.resolve`; they are not a global geometry/prior score.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from frend.fold_resolve import (
    CoverScore,
    Detection,
    _candidates,
    _content_key,
    _cover_score,
    _dedupe,
    _resolve_sources,
    _select,
    _span_priors,
)
from frend.input_folds import InputFold, apply_input_fold
from frend.input_limits import (
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_UNIT_CHARS,
    validate_input,
    validate_unit_length,
)
from frend.locale_data import canonical_locale
from frend.type_priors import FeatureSource, ReadingPrior, ResolveContext

__all__ = [
    "LatticeNode",
    "ChoiceGraph",
    "ChoiceLattice",
    "PriorSummary",
    "ReadingEdge",
    "ReadingLattice",
    "ReadingPath",
    "ReadingRank",
    "SemanticRank",
    "compose_choices",
    "resolve_choices",
    "resolve_lattice",
    "route_geometry",
]

if TYPE_CHECKING:
    from frend.verbalize import CuratedSupplements, VerbalizedUnit


@dataclass(frozen=True)
class LatticeNode:
    """One code-point boundary in the source text."""

    position: int


@dataclass(frozen=True)
class SemanticRank:
    """Portable, exact semantic components attached to a reading edge."""

    tier: Literal["measured", "icu-backfill", "unsupported"]
    p: Decimal | None
    generated_p: Decimal | None


@dataclass(frozen=True)
class ReadingRank:
    """Decomposed edge rank, independent of tiergraph's scalar encoding."""

    coverage: int
    span_count: int
    capture_count: int
    semantic: SemanticRank


@dataclass(frozen=True)
class ReadingEdge:
    """A candidate reading or unit passthrough transition."""

    id: str
    start: int
    end: int
    detection: Detection | None
    kind: Literal["reading", "passthrough"]
    geometry: CoverScore
    prior: ReadingPrior | None
    rank: ReadingRank


@dataclass(frozen=True)
class PriorSummary:
    """Per-reading priors for one path, in reading order."""

    priors: tuple[ReadingPrior | None, ...]
    supported_count: int


@dataclass(frozen=True)
class ReadingPath:
    """One complete, ranked route through reading and passthrough edges."""

    edge_ids: tuple[str, ...]
    readings: tuple[Detection, ...]
    geometry: CoverScore
    prior_summary: PriorSummary
    rank: int


@dataclass(frozen=True)
class ReadingLattice:
    """A distilled lattice and its capped public ranked-path projection.

    ``paths`` is bounded by the requested output cap, but the resolver must read
    the complete top-geometry equivalence class to compute canonical-structure
    selection and ambiguity correctly. It reads it per component of overlapping
    spans, so what it materializes is each component's top level, never their
    product; a component of more than 512 covers whose top level passes 256 covers
    is refused. :class:`ChoiceLattice` carries
    every reading without any enumeration, at the cost of selecting nothing.
    """

    locale: str = field(default="en_US", kw_only=True, repr=False)
    text_length: int
    nodes: tuple[LatticeNode, ...]
    edges: tuple[ReadingEdge, ...]
    paths: tuple[ReadingPath, ...]
    best_path: ReadingPath
    truncated: bool
    structural_ambiguous: bool
    semantic_ambiguous: bool
    ambiguous: bool
    source_text: str | None = None
    raw_source_text: str | None = field(default=None, repr=False)
    fold: InputFold | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ChoiceLattice:
    """Every distinct reading as a scored edge, with nothing selected.

    This is the reading layer of the hand-off. ``edges`` holds one ``"reading"``
    edge per valid detection that is distinct under this carrier's identity, and
    one ``"passthrough"`` edge per code point, over ``nodes`` ``0..text_length``.
    Each edge records its own geometry and its own prior; no edge is preferred
    over another, and readings that compete for the same span sit side by side.

    The identity is total over mappings, sequences, sets, dataclasses, and
    built-in scalars, with recursive type tags. A leaf outside those forms is
    identified by its concrete type and ``repr``; two objects of that type can
    therefore collapse if they deliberately share a representation. Within the
    supported forms this is stricter than the identity used by
    :func:`resolve_lattice`, which does not read capture values or text.

    Geometry is recorded and never applied. It composes over a route: a set of
    strictly forward edges tiling the text with no gap and no overlap. A route's
    ``CoverScore`` is the componentwise sum over its edges, and
    :func:`route_geometry` checks the route before summing. The result reproduces
    the resolver's geometry rank only. Per-span reading rank and content key are
    not recoverable, so routes that tie on geometry are unordered here by design.

    There are no paths, best path, truncation flag, or ambiguity flags. Each
    would be a fact about a selection this carrier does not make, and a constant
    standing in for one would be fabricated. ``dropped`` holds snapshots of
    distinct detections rejected for invalid geometry, in input order.

    The composed graph is the hand-off. :func:`compose_choices` joins these edges
    to their spoken forms. ``source_text`` is required for that composition
    because a passthrough edge carries no detection from which to recover surface
    text.
    """

    locale: str = field(default="en_US", kw_only=True, repr=False)
    text_length: int
    nodes: tuple[LatticeNode, ...]
    edges: tuple[ReadingEdge, ...]
    dropped: tuple[Detection, ...] = ()
    source_text: str | None = None


@dataclass(frozen=True)
class ChoiceGraph:
    """A choice carrier joined to one verbalized unit for every edge."""

    lattice: ChoiceLattice
    units: tuple[VerbalizedUnit, ...]


def _semantic_rank(prior: ReadingPrior | None) -> SemanticRank:
    if prior is None:
        return SemanticRank("unsupported", None, None)
    return SemanticRank(prior.tier, prior.p, prior.generated_p)


def _edge_rank(geometry: CoverScore, prior: ReadingPrior | None) -> ReadingRank:
    return ReadingRank(
        geometry.coverage,
        geometry.span_count,
        geometry.capture_count,
        _semantic_rank(prior),
    )


def _freeze(value: object, active: set[int] | None = None) -> object:
    """Recursively freeze an already-owned detection value.

    Dataclasses retain their concrete type while every init field is replaced by
    its frozen value. Cycles, unusual dataclasses that cannot be replaced, and
    set members that become unhashable are left as their already-deep-copied
    owned value as a last resort rather than making lattice construction fail.
    """
    active = set() if active is None else active
    compound = isinstance(value, (Mapping, list, tuple, set)) or (
        is_dataclass(value) and not isinstance(value, type)
    )
    identity = id(value)
    if compound and identity in active:
        return value
    if compound:
        active.add(identity)
    try:
        if isinstance(value, Mapping):
            return MappingProxyType({key: _freeze(item, active) for key, item in value.items()})
        if isinstance(value, (list, tuple)):
            return tuple(_freeze(item, active) for item in value)
        if isinstance(value, set):
            frozen_items = (_freeze(item, active) for item in value)
            try:
                return frozenset(frozen_items)
            except TypeError:
                return value
        if is_dataclass(value) and not isinstance(value, type):
            replacements = {
                field.name: _freeze(getattr(value, field.name), active)
                for field in fields(value)
                if field.init
            }
            try:
                return replace(value, **replacements)
            except (TypeError, ValueError):
                try:
                    return type(value)(**replacements)
                except (TypeError, ValueError):
                    return value
        return value
    finally:
        if compound:
            active.remove(identity)


def _snapshot(detection: Detection) -> Detection:
    """Return a deeply owned detection mapping, recursively frozen where possible.

    Deep copying is the ownership boundary: the snapshot is always isolated from
    later caller mutation. Freezing is then complete for every standard detection
    payload -- scalars, strings, mappings, sequences, and frozen dataclasses with
    freezable fields (which is all a real detection carries). The unavoidable
    exceptions are cyclic references (a cyclic Python structure cannot be made
    immutable) and unhashable set members; those are deep-copied (still
    caller-isolated) but left mutable as the documented last resort rather than
    failing lattice construction. Real detections do not contain such payloads.
    """
    source = dict(detection)
    try:
        owned = deepcopy(source)
    except Exception:  # An arbitrary user payload may implement a failing copy hook.
        owned = source
    frozen = _freeze(owned)
    assert isinstance(frozen, Mapping)
    return frozen


def _snapshot_spelled_token(detection: Detection) -> Detection:
    """Own frend's scalar/frozen spell-token payload without generic deep copying."""
    from frend.letters import LettersValue

    value = detection.get("value")
    captures = detection.get("captures", ())
    frozen_captures = isinstance(captures, tuple) and all(
        (params := getattr(type(item), "__dataclass_params__", None)) is not None and params.frozen
        for item in captures
    )
    if not isinstance(value, LettersValue) or not frozen_captures:
        return _snapshot(detection)
    return MappingProxyType(
        {
            "text": str(detection["text"]),
            "start": int(detection["start"]),
            "end": int(detection["end"]),
            "type": "letters:token",
            "value": value,
            "captures": captures,
        }
    )


def _content_identity(detection: Detection) -> Hashable:
    """Return a hashable, type-tagged identity for all snapshot content.

    Mapping and set order are canonicalized by sorting the serialized items, so
    two mappings whose items are distinguishable share one identity whatever
    their insertion order, including under keys of heterogeneous types.
    Dataclasses retain their concrete type and serialize by fields recursively.
    A leaf outside the supported built-in and dataclass
    forms falls back to its type-tagged ``repr``, which can over-collapse objects
    of the same type that deliberately share a representation.
    """

    def serialize(value: object) -> Hashable:
        type_name = (type(value).__module__, type(value).__qualname__)
        if isinstance(value, Mapping):
            items = [(serialize(key), serialize(item)) for key, item in value.items()]
            items.sort(key=repr)
            return ("mapping", tuple(items))
        if is_dataclass(value) and not isinstance(value, type):
            return (
                "dataclass",
                type_name,
                tuple(
                    (field.name, serialize(getattr(value, field.name))) for field in fields(value)
                ),
            )
        if isinstance(value, (tuple, list)):
            return ("sequence", type_name, tuple(serialize(item) for item in value))
        if isinstance(value, (set, frozenset)):
            items = [serialize(item) for item in value]
            items.sort(key=repr)
            return ("set", type_name, tuple(items))
        if value is None or isinstance(value, (bool, int, float, str, bytes, Decimal)):
            return ("scalar", type_name, value)
        return ("repr", type_name, repr(value))

    source = dict(detection)
    try:
        owned = deepcopy(source)
    except Exception:
        owned = source
    return serialize(_freeze(owned))


_CHOICE_READING_CAP = 1 << 16


def resolve_choices(
    detections: Sequence[Detection],
    *,
    locale: str = "en_US",
    reading_cap: int = _CHOICE_READING_CAP,
    feature_sources: Sequence[FeatureSource] | None = None,
    source_text: str | None = None,
    class_prior: Mapping[str, Decimal | int] | None = None,
    class_prior_source: str | None = None,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
    max_unit_chars: int | None = DEFAULT_MAX_UNIT_CHARS,
) -> ChoiceLattice:
    """Carry every distinct reading as a scored edge, filtering none of them.

    No cover is enumerated, which is why this is defined on inputs where
    :func:`resolve_lattice` is not, and it is the only cost claim available here.
    It does not make that function cheaper. What it does cost is a deep copy and
    recursive freeze of each detection, once for its identity and once for its
    snapshot, so a call scales with the payload the detections carry; it also
    builds one passthrough edge and one node per code point, so it scales with
    the length of the text as well. Geometry is recorded rather than applied. Every
    reading edge carries its own :class:`~frend.fold_resolve.CoverScore`; a route's
    score is their componentwise sum when the edges strictly tile the text.

    Identity is total over mappings, sequences, sets, dataclasses, and built-in
    scalars, with recursive type tags. A leaf outside those forms is identified
    by its concrete type and ``repr``; two objects of that type can therefore
    collapse if they share a representation. The identity is hashable,
    insensitive to mapping order when serialized items are distinguishable, and
    uses a set-membership scan.

    ``prior`` is metadata, not an ordering. Nothing is sorted by it, and no rule
    is offered for combining it along a route because there is none. At
    ``tier="measured"``, ``p`` is ``P(class | shape)``, independent of the rest
    of the input, and ``generated_p`` is ``None``. At ``tier="icu-backfill"``,
    ``p`` is a posterior renormalized over readings at that edge's span, so it
    changes when a span-mate is added or removed; ``generated_p`` is
    ``P(shape | class)``, a different conditional rather than a base rate.
    ``tier`` and ``supported`` can shift with span-mates too. No probability
    field is comparable across tiers or calls. The comparable metadata is the
    three geometry integers and ``provenance``, ``shape``, ``group``, and ``n``.

    ``reading_cap`` bounds readings only and refuses when exceeded. The check is
    before edge snapshots and prior work, but identity computation has already
    copied and traversed every detection payload, so a refused call is not free.
    A ranked list may return a prefix and flag it because rank gives the prefix
    meaning. A prefix of an unordered complete set has no meaning and is
    indistinguishable from the whole set, so this refuses rather than truncates.
    It does not bound copied payload size, passthrough edges, nodes, or spoken
    forms. Passthroughs and nodes are outside the cap: there is one passthrough
    per code point, which the caller already holds, and counting them would make
    a carrier refuse a long document for its length rather than its ambiguity.
    The composed graph has a separate bound in different units. Composition
    through :func:`compose_choices` requires ``source_text``.
    """
    validate_input(source_text, max_input_chars=max_input_chars)
    if source_text is not None:
        validate_unit_length(len(source_text), max_unit_chars=max_unit_chars)
    if not isinstance(reading_cap, int) or isinstance(reading_cap, bool) or reading_cap < 1:
        raise ValueError(f"reading_cap must be a positive integer, got {reading_cap!r}")

    unique: list[Detection] = []
    seen: set[Hashable] = set()
    for detection in detections:
        identity = _content_identity(detection)
        if identity not in seen:
            seen.add(identity)
            unique.append(detection)
    candidates, detected_length = _candidates(unique)
    text_length = len(source_text) if source_text is not None else detected_length
    validate_unit_length(
        text_length,
        max_unit_chars=max_unit_chars,
        max_input_chars=max_input_chars,
    )
    if text_length < detected_length:
        raise ValueError(
            f"source_text length {text_length} is shorter than detection extent {detected_length}"
        )
    if len(candidates) > reading_cap:
        raise ValueError(
            f"this carries {len(candidates)} readings over the {reading_cap} reading "
            f"bound; an unordered reading set has no meaningful prefix, so a truncated "
            f"carrier would claim a completeness it does not have"
        )
    canonical = canonical_locale(locale)
    sources = _resolve_sources(canonical, feature_sources, class_prior, class_prior_source)
    context = ResolveContext(source_text=source_text, all_detections=tuple(unique))
    priors = _span_priors(unique, sources, context) if unique else ()
    snapshots = tuple(_snapshot(detection) for detection in unique)
    reading_edges: list[ReadingEdge] = []
    valid_indices = {candidate.index for candidate in candidates}
    for candidate in candidates:
        source_detection = candidate.detection
        prior = priors[candidate.index]
        geometry = CoverScore(
            candidate.end - candidate.start,
            1,
            len(source_detection.get("captures", ())),
        )
        reading_edges.append(
            ReadingEdge(
                f"c{candidate.index}",
                candidate.start,
                candidate.end,
                snapshots[candidate.index],
                "reading",
                geometry,
                prior,
                _edge_rank(geometry, prior),
            )
        )
    zero = CoverScore(0, 0, 0)
    passthrough_edges = [
        ReadingEdge(
            f"skip{position}",
            position,
            position + 1,
            None,
            "passthrough",
            zero,
            None,
            _edge_rank(zero, None),
        )
        for position in range(text_length)
    ]
    dropped = tuple(
        snapshot for index, snapshot in enumerate(snapshots) if index not in valid_indices
    )
    return ChoiceLattice(
        text_length,
        tuple(LatticeNode(position) for position in range(text_length + 1)),
        tuple(reading_edges + passthrough_edges),
        dropped,
        source_text,
        locale=canonical,
    )


def route_geometry(lattice: ChoiceLattice, edge_ids: Sequence[str]) -> CoverScore:
    """Return the :class:`~frend.fold_resolve.CoverScore` of one route, or refuse.

    A route is a set of strictly forward edges tiling ``0..text_length`` with no
    gap and no overlap; the order they are named in does not matter. On such a set
    the componentwise sum of the edges' geometries is the geometry
    :func:`resolve_lattice` would report for the corresponding cover, and hence
    its geometry rank. It says nothing about routes that tie.

    Off such a set the sum is arithmetic without meaning, and this refuses rather
    than returning it. Two conditions carry the precondition: every edge moves
    forward, and the edges tile the text. A backward edge can satisfy contiguity
    while covering coordinates twice.

    This is a convenience with its precondition enforced, not a property of
    :class:`ChoiceLattice`, which validates nothing. It builds a map over every
    carrier edge before sorting the named edges, so naming a handful of edges on
    a large carrier still touches the whole edge set. The sort adds a log-linear
    term in the number of named edges.
    """
    by_id = {edge.id: edge for edge in lattice.edges}
    walk = []
    for edge_id in edge_ids:
        edge = by_id.get(edge_id)
        if edge is None:
            raise ValueError(f"edge {edge_id!r} is not in this lattice")
        if edge.end <= edge.start:
            raise ValueError(
                f"edge {edge_id!r} does not move forward: {edge.start} -> {edge.end}; "
                "a route is built from strictly forward edges"
            )
        walk.append(edge)
    walk.sort(key=lambda edge: (edge.start, edge.end, edge.id))
    coverage = span_count = capture_count = 0
    position = 0
    for edge in walk:
        if edge.start < position:
            raise ValueError(
                f"edge {edge.id!r} starts at {edge.start}, inside the span already covered to "
                f"{position}: two edges of this set overlap, so summing them would count a "
                "coordinate twice"
            )
        if edge.start > position:
            raise ValueError(
                f"edge {edge.id!r} starts at {edge.start}, leaving 0..{lattice.text_length} "
                f"uncovered from {position}: a route tiles the text with no gap"
            )
        coverage += edge.geometry.coverage
        span_count += edge.geometry.span_count
        capture_count += edge.geometry.capture_count
        position = edge.end
    if position != lattice.text_length:
        raise ValueError(
            f"route ends at {position}, not at {lattice.text_length}; a partial walk has no "
            "cover geometry"
        )
    return CoverScore(coverage, span_count, capture_count)


_COMPOSED_SPOKEN_CAP = 1 << 16


def compose_choices(
    lattice: ChoiceLattice,
    *,
    locale: str | None = None,
    supplements: CuratedSupplements | None = None,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
    max_unit_chars: int | None = DEFAULT_MAX_UNIT_CHARS,
) -> ChoiceGraph:
    """Join every choice edge to its spoken forms without selecting a route.

    The bound counts spoken forms summed over every edge, not readings; it uses
    the same numeral as the reading bound in different units. The graph refuses
    rather than returning an incomplete prefix. The running total is checked
    after each edge is verbalized, so this cannot bound the cost of verbalizing
    one intrinsically pathological edge once.
    """
    validate_input(lattice.source_text, max_input_chars=max_input_chars)
    validate_unit_length(
        lattice.text_length,
        max_unit_chars=max_unit_chars,
        max_input_chars=max_input_chars,
    )
    effective = lattice.locale
    if locale is not None and canonical_locale(locale) != effective:
        raise ValueError(f"locale {locale!r} does not match lattice locale {effective!r}")
    if lattice.source_text is None:
        raise ValueError("source_text is required to compose passthrough spoken forms")
    from frend.verbalize import _verbalize_edge

    units = []
    spoken_total = 0
    for edge in lattice.edges:
        unit = _verbalize_edge(
            edge, source_text=lattice.source_text, locale=effective, supplements=supplements
        )
        units.append(unit)
        spoken_total += len(unit.alternatives)
        if spoken_total > _COMPOSED_SPOKEN_CAP:
            raise ValueError(
                f"composed graph exceeds the {_COMPOSED_SPOKEN_CAP} spoken forms bound; "
                "this counts spoken forms, not readings, and an unordered possibilia set "
                "has no meaningful prefix, so truncation would claim completeness"
            )
    return ChoiceGraph(lattice, tuple(units))


def resolve_lattice(
    detections: Sequence[Detection] | None = None,
    *,
    locale: str = "en_US",
    output_cap: int = 1,
    feature_sources: Sequence[FeatureSource] | None = None,
    source_text: str | None = None,
    fold: InputFold | None = "typographic",
    class_prior: Mapping[str, Decimal | int] | None = None,
    class_prior_source: str | None = None,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
    max_unit_chars: int | None = DEFAULT_MAX_UNIT_CHARS,
) -> ReadingLattice:
    """Resolve detections into an immutable, distilled reading lattice.

    With ``detections=None``, recognize ``source_text`` with frend's public detector
    profile after applying the declared input ``fold``. A lattice reports that fold
    only on this path. Caller-supplied detections cannot prove what text they saw, so
    their lattice reports ``fold=None`` even if the default argument was not overridden.

    ``output_cap=1`` is the winner-take-all public projection; a larger cap
    exposes more ranked paths over the same candidate and passthrough edges.
    The cap bounds only the paths exposed in :attr:`ReadingLattice.paths`. The
    resolver reads the complete top-geometry equivalence class, regardless of
    ``output_cap``, because canonical ``s*`` selection and ambiguity require it, but
    it reads it per component of overlapping spans: the sentence's top level is the
    product of the components' and is never formed. A component admitting at most
    512 covers is enumerated directly and has no bound. A larger one is gathered by
    the ranked fold, and there, a top level of more than 256 covers is refused (a
    ``ValueError``) rather than ranked, as is a cover list whose last needed level
    runs more than 256 covers past the ``output_cap``-th cover; the ``output_cap``
    itself is never refused. :func:`resolve_choices` carries every
    reading as a scored edge without enumerating any cover, and is defined there
    too. This API does not silently claim a bound it does not have.
    """
    validate_input(source_text, max_input_chars=max_input_chars)
    raw_source_text = source_text
    applied_fold: InputFold | None = None
    if detections is None:
        if raw_source_text is None:
            raise ValueError("source_text is required when detections is None")
        source_text = apply_input_fold(raw_source_text, fold)
        applied_fold = fold
        from icukit.detectors import detect

        from frend.normalize import _reading_detectors

        detections = list(detect(source_text, _reading_detectors(canonical_locale(locale))))
    return _resolve_lattice_validated(
        detections,
        locale=locale,
        output_cap=output_cap,
        feature_sources=feature_sources,
        source_text=source_text,
        raw_source_text=raw_source_text,
        fold=applied_fold,
        class_prior=class_prior,
        class_prior_source=class_prior_source,
        max_input_chars=max_input_chars,
        max_unit_chars=max_unit_chars,
    )


def _resolve_lattice_validated(
    detections: Sequence[Detection],
    *,
    locale: str = "en_US",
    output_cap: int = 1,
    feature_sources: Sequence[FeatureSource] | None = None,
    source_text: str | None = None,
    raw_source_text: str | None = None,
    fold: InputFold | None = None,
    class_prior: Mapping[str, Decimal | int] | None = None,
    class_prior_source: str | None = None,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
    max_unit_chars: int | None = DEFAULT_MAX_UNIT_CHARS,
) -> ReadingLattice:
    """Resolve text whose enclosing document API has already validated it."""
    if source_text is not None:
        validate_unit_length(len(source_text), max_unit_chars=max_unit_chars)
    if raw_source_text is None:
        raw_source_text = source_text
    if (
        source_text is not None
        and raw_source_text is not None
        and len(source_text) != len(raw_source_text)
    ):
        raise ValueError("folded and raw source text must have equal code-point lengths")
    if not isinstance(output_cap, int) or isinstance(output_cap, bool) or output_cap < 1:
        raise ValueError(f"output_cap must be a positive integer, got {output_cap!r}")

    unique = _dedupe(detections)
    candidates, detected_length = _candidates(unique)
    key_cache: dict[int, tuple] = {}

    def content_key(detection: Detection) -> tuple:
        identity = id(detection)
        if identity not in key_cache:
            key_cache[identity] = _content_key(detection)
        return key_cache[identity]

    text_length = len(source_text) if source_text is not None else detected_length
    validate_unit_length(
        text_length,
        max_unit_chars=max_unit_chars,
        max_input_chars=max_input_chars,
    )
    if text_length < detected_length:
        raise ValueError(
            f"source_text length {text_length} is shorter than detection extent {detected_length}"
        )

    canonical = canonical_locale(locale)
    sources = _resolve_sources(canonical, feature_sources, class_prior, class_prior_source)
    context = ResolveContext(source_text=source_text, all_detections=tuple(unique))
    # A bounded spelled-token candidate that overlaps no other valid candidate is
    # structurally forced into every maximum-coverage cover. Keep it in the public
    # lattice, but do not make the generic tiergraph fold rediscover that fact for
    # every ordinary short word in running text. This type has no type-prior mapping;
    # its spell/say ranking is internal to verbalization.
    # Detector output is already start-ordered, so the common path is one linear
    # sweep. Preserve the public API's behavior for an arbitrarily ordered sequence
    # with a sort only when its caller did not supply that invariant.
    ordered = candidates
    if any(left.start > right.start for left, right in zip(ordered, ordered[1:], strict=False)):
        ordered = sorted(ordered, key=lambda candidate: candidate.start)
    isolated_keys = set()
    prior_max_end = -1
    for index, candidate in enumerate(ordered):
        next_start = ordered[index + 1].start if index + 1 < len(ordered) else candidate.end
        if (
            candidate.detection.get("type") == "letters:token"
            and prior_max_end <= candidate.start
            and next_start >= candidate.end
        ):
            isolated_keys.add(content_key(candidate.detection))
        prior_max_end = max(prior_max_end, candidate.end)
    selected_unique = [item for item in unique if content_key(item) not in isolated_keys]
    isolated = sorted(
        (item for item in unique if content_key(item) in isolated_keys),
        key=lambda item: (int(item["start"]), int(item["end"]), content_key(item)),
    )
    selection = (
        _select(
            selected_unique,
            sources,
            context,
            output_cap,
            projection_probe=True,
            collect_edge_priors=True,
        )
        if selected_unique
        else None
    )

    prior_by_key = {key: None for key in isolated_keys}
    if selection is not None and selection.edge_priors:
        prior_by_key.update(
            {
                content_key(det): prior
                for det, prior in zip(selected_unique, selection.edge_priors, strict=True)
            }
        )
    snapshot_by_key = {
        content_key(det): (
            _snapshot_spelled_token(det) if det.get("type") == "letters:token" else _snapshot(det)
        )
        for det in unique
    }
    reading_edges: list[ReadingEdge] = []
    id_by_key: dict[tuple, str] = {}
    for candidate in candidates:
        source_detection = candidate.detection
        key = content_key(source_detection)
        detection = snapshot_by_key[key]
        prior = prior_by_key[key]
        geometry = CoverScore(
            candidate.end - candidate.start,
            1,
            len(source_detection.get("captures", ())),
        )
        edge_id = f"c{candidate.index}"
        id_by_key[key] = edge_id
        reading_edges.append(
            ReadingEdge(
                edge_id,
                candidate.start,
                candidate.end,
                detection,
                "reading",
                geometry,
                prior,
                _edge_rank(geometry, prior),
            )
        )

    zero = CoverScore(0, 0, 0)
    passthrough_edges = [
        ReadingEdge(
            f"skip{position}",
            position,
            position + 1,
            None,
            "passthrough",
            zero,
            None,
            _edge_rank(zero, None),
        )
        for position in range(text_length)
    ]

    selected_covers = selection.covers if selection is not None else ((),)
    selected_priors = selection.priors if selection is not None else ((),)
    covers = []
    cover_priors = []
    for cover, priors in zip(selected_covers, selected_priors, strict=True):
        by_key = {content_key(item): prior for item, prior in zip(cover, priors, strict=True)}
        merged = tuple(
            sorted(
                (*cover, *isolated),
                key=lambda item: (int(item["start"]), int(item["end"]), content_key(item)),
            )
        )
        covers.append(merged)
        cover_priors.append(tuple(by_key.get(content_key(item)) for item in merged))
    paths: list[ReadingPath] = []
    for rank, (cover, priors) in enumerate(zip(covers, cover_priors, strict=True)):
        edge_ids: list[str] = []
        position = 0
        for detection in cover:
            start = int(detection["start"])
            edge_ids.extend(f"skip{offset}" for offset in range(position, start))
            edge_ids.append(id_by_key[content_key(detection)])
            position = int(detection["end"])
        edge_ids.extend(f"skip{offset}" for offset in range(position, text_length))
        paths.append(
            ReadingPath(
                tuple(edge_ids),
                tuple(snapshot_by_key[content_key(detection)] for detection in cover),
                _cover_score(cover),
                PriorSummary(
                    priors,
                    sum(prior is not None and prior.supported for prior in priors),
                ),
                rank,
            )
        )

    ranked_paths = tuple(paths)
    return ReadingLattice(
        text_length,
        tuple(LatticeNode(position) for position in range(text_length + 1)),
        tuple(reading_edges + passthrough_edges),
        ranked_paths,
        ranked_paths[0],
        selection.truncated if selection is not None else False,
        selection.structural_ambiguous if selection is not None else False,
        selection.semantic_ambiguous if selection is not None else False,
        selection.ambiguous if selection is not None else False,
        source_text,
        raw_source_text,
        fold,
        locale=canonical,
    )
