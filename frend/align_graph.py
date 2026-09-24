"""Word-level alignment DAGs over the complete choice carrier.

Plan posteriors mean acoustics plus measured prior, never a path probability
model. Unscored alternatives contribute unit weight per branch. This module
neither selects a reading nor changes the exact Decimal 1-best resolver.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
from types import MappingProxyType

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
from tiergraph.pathplan import PathPlan
from tiergraph.semiring import COUNTING, LOG_PROBABILITY

from frend.lattice import ChoiceGraph, ReadingEdge
from frend.spoken_priors import spoken_tokens
from frend.verbalize import SpokenAlternative

__all__ = ["AlignGraph", "AlignItem", "ArcWeight", "arc_weight", "build_align_graph"]

NS = "https://ogion.org/frend/alignment"
_TIER = QualifiedName(NS, "alignment")
_TYPE = QualifiedName(NS, "item")
_NEXT = QualifiedName(NS, "next")
_LOG_WEIGHT = QualifiedName(NS, "plan-log-weight")
_COUNT = QualifiedName(NS, "count")
_SCORED = QualifiedName(NS, "scored")
_PRIOR_KIND = QualifiedName(NS, "prior-kind")
_REASON = QualifiedName(NS, "prior-reason")
_P = QualifiedName(NS, "measured-p")
_N = QualifiedName(NS, "measured-n")
_ROLE = QualifiedName(NS, "role")
_TOKEN = QualifiedName(NS, "token")
_ALIGN_ITEM_CAP = 1 << 20


@dataclass(frozen=True)
class ArcWeight:
    """A plan factor together with its evidence marker and exact measurement."""

    scored: bool
    reason: str
    log_weight: float = 0.0
    p: Decimal | None = None
    n: int | None = None


def _scored(edge: ReadingEdge) -> bool:
    prior = edge.prior
    return (
        edge.kind == "reading"
        and prior is not None
        and prior.tier == "measured"
        and prior.supported
        and prior.p is not None
        and prior.p > 0
    )


def arc_weight(edge: ReadingEdge, span_mates: Sequence[ReadingEdge]) -> ArcWeight:
    """Use Decimal log(p / p_max), over all scored readings at the exact span.

    Generated backfill estimates and spoken-source shares are not measurements.
    The maximum is across corpus groups; extent competition has no measured rule.
    """
    prior = edge.prior
    if _scored(edge):
        assert prior is not None and prior.p is not None
        maximum = max(
            mate.prior.p
            for mate in (edge, *span_mates)
            if (mate.start, mate.end) == (edge.start, edge.end) and _scored(mate)
        )
        return ArcWeight(True, "measured", float((prior.p / maximum).ln()), prior.p, prior.n)
    if edge.kind == "passthrough":
        reason = "passthrough"
    elif prior is None:
        reason = "no-corpus-source"
    elif prior.tier == "icu-backfill":
        reason = "icu-backfill"
    elif prior.tier == "measured" and prior.n is None:
        reason = "sparse-shape"
    elif prior.tier == "measured" and (prior.p is None or prior.p <= 0):
        reason = "attested-zero"
    else:
        reason = "unsupported"
    return ArcWeight(False, reason)


@dataclass(frozen=True)
class AlignItem:
    """One graph item with its emission and trace to the carrier.

    Form exits retain all alternatives that normalize to the same token tuple;
    their source shares remain metadata and never become plan factors.
    """

    id: str
    role: str
    weight: ArcWeight
    tokens: tuple[str, ...] = ()
    reading_id: str | None = None
    span: tuple[int, int] | None = None
    alternatives: tuple[SpokenAlternative, ...] = ()


@dataclass(frozen=True)
class AlignGraph:
    """A single-root, single-accepting-sink DAG with reusable log and count plans.

    Log-plan posteriors mean acoustics plus measured prior, not a path probability
    model. Replacement value vectors follow the plan's canonical item order.
    """

    graph: Graph
    items: Mapping[str, AlignItem]
    root: ItemRef
    sink: ItemRef
    kept_positions: frozenset[int]
    unalignable: tuple[tuple[str, SpokenAlternative], ...] = ()

    @cached_property
    def _log_plan(self) -> PathPlan[float]:
        return self._prepare(LOG_PROBABILITY, _LOG_WEIGHT)

    @cached_property
    def _count_plan(self) -> PathPlan[int]:
        return self._prepare(COUNTING, _COUNT)

    def _prepare(self, algebra, attribute):
        return PathPlan.prepare(
            FoldDeclaration(
                "alignment",
                self.graph,
                AttributeValuation("alignment", attribute, (_TIER,)),
                algebra,
                lambda value, _label: int(value) if algebra is COUNTING else float(value),
                (FoldTransition(_NEXT, ChildCombination.OR),),
                roots=(self.root,),
            )
        )

    def log_plan(self) -> PathPlan[float]:
        """Return the cached LOG_PROBABILITY plan."""
        return self._log_plan

    def count_plan(self) -> PathPlan[int]:
        """Return the cached COUNTING plan, valued by an integer attribute."""
        return self._count_plan


def _separator(character: str) -> bool:
    return character.isspace() or unicodedata.category(character).startswith("P")


def _check_sinks(items, links, root, sink):
    children = defaultdict(list)
    for left, right in links:
        children[left].append(right)
    reachable = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        if current != sink and not children[current]:
            raise ValueError(f"reachable non-sink item {current!r} has no child")
        pending.extend(children[current])
    if sink not in reachable:
        raise ValueError("no complete alignment route remains")
    return reachable


def build_align_graph(
    choices: ChoiceGraph,
    *,
    pronounceable: Callable[[str], bool] | None = None,
    item_cap: int = _ALIGN_ITEM_CAP,
) -> AlignGraph:
    """Encode carrier tilings times distinct token forms, refusing whole at the cap.

    Tries share prefixes only within one reading. Passthrough uses word runs and
    pre-reading nodes so that a skip run cannot split into duplicate token paths.
    Item growth is linear in spoken words plus quadratic in reading endpoints
    within each word run. The cap counts graph items and is checked on each
    insertion before any Graph is constructed, including before lexicon trimming.
    A declared lexicon filter removes complete alternatives, records them, and
    trims items that cannot lie on a complete retained route.
    """
    lattice = choices.lattice
    text = lattice.source_text
    if text is None:
        raise ValueError("source_text is required for alignment")
    if isinstance(item_cap, bool) or not isinstance(item_cap, int) or item_cap < 1:
        raise ValueError("item_cap must be a positive integer in graph-item units")
    if len(text) != lattice.text_length:
        raise ValueError("source_text length differs from carrier extent")
    readings = [edge for edge in lattice.edges if edge.kind == "reading"]
    if len({edge.id for edge in readings}) != len(readings):
        raise ValueError("duplicate reading identity")
    if any(not 0 <= edge.start < edge.end <= len(text) for edge in readings):
        raise ValueError("reading must move forward within source_text")
    units = {unit.edge_id: unit for unit in choices.units}
    kept = {0, len(text)} | {p for edge in readings for p in (edge.start, edge.end)}
    starts = {edge.start for edge in readings}
    ends = {edge.end for edge in readings}
    groups = defaultdict(list)
    for edge in readings:
        groups[edge.start, edge.end].append(edge)
    runs = []
    offset = 0
    while offset < len(text):
        end = offset + 1
        separator = _separator(text[offset])
        while end < len(text) and _separator(text[end]) == separator:
            end += 1
        runs.append((offset, end, separator))
        offset = end
    positions = kept | {p for a, b, _separator_run in runs for p in (a, b)}
    items = {}
    links = []
    unalignable = []

    def add(id, role, *, tokens=(), reading_id=None, span=None, alternatives=(), weight=None):
        if id in items:
            raise ValueError(f"duplicate alignment item {id!r}")
        if len(items) >= item_cap:
            raise ValueError(f"alignment exceeds {item_cap} graph-item bound; refusing whole")
        reason = {"token": "passthrough", "separator": "separator", "word": "spoken-form"}
        items[id] = AlignItem(
            id,
            role,
            weight or ArcWeight(False, reason.get(role, "structure")),
            tokens,
            reading_id,
            span,
            alternatives,
        )

    def link(left, right):
        links.append((left, right))

    for position in sorted(positions):
        add(f"B{position}", "position")
    for a, b, separator in runs:
        if separator:
            junctions = sorted({a, b} | ({p for p in kept if a <= p <= b}))
            for i, j in zip(junctions, junctions[1:], strict=False):
                id = f"sep{i}_{j}"
                add(id, "separator", span=(i, j))
                link(f"B{i}", id)
                link(id, f"B{j}")
        else:
            for i in sorted({a} | {p for p in ends if a < p < b}):
                for j in sorted({b} | {p for p in starts if a < p < b}):
                    if i >= j:
                        continue
                    tokens = spoken_tokens(text[i:j])
                    if pronounceable is not None and not all(map(pronounceable, tokens)):
                        unalignable.append(
                            (f"tok{i}_{j}", SpokenAlternative(text[i:j], "passthrough"))
                        )
                        continue
                    id = f"tok{i}_{j}"
                    add(id, "token", tokens=tokens, span=(i, j))
                    link(f"B{i}", id)
                    if j == b:
                        link(id, f"B{b}")
                    else:
                        pre = f"pre{j}"
                        if pre not in items:
                            add(pre, "pre-reading")
                        link(id, pre)
    for edge in readings:
        if edge.id not in units:
            raise ValueError(f"missing spoken unit for reading {edge.id!r}")
        add(
            edge.id,
            "reading",
            reading_id=edge.id,
            span=(edge.start, edge.end),
            weight=arc_weight(edge, groups[edge.start, edge.end]),
        )
        link(f"B{edge.start}", edge.id)
        if f"pre{edge.start}" in items:
            link(f"pre{edge.start}", edge.id)
        forms = defaultdict(list)
        for alternative in units[edge.id].alternatives:
            tokens = spoken_tokens(alternative.text)
            if pronounceable is not None and not all(map(pronounceable, tokens)):
                unalignable.append((edge.id, alternative))
                continue
            forms[tokens].append(alternative)
        prefixes = {(): edge.id}
        for index, (tokens, alternatives) in enumerate(forms.items()):
            for length in range(1, len(tokens) + 1):
                prefix = tokens[:length]
                if prefix not in prefixes:
                    id = f"{edge.id}w{len(prefixes) - 1}"
                    add(id, "word", tokens=(prefix[-1],), reading_id=edge.id)
                    link(prefixes[prefix[:-1]], id)
                    prefixes[prefix] = id
            exit_id = f"{edge.id}x{index}"
            add(
                exit_id,
                "form-exit",
                reading_id=edge.id,
                span=(edge.start, edge.end),
                alternatives=tuple(alternatives),
            )
            link(prefixes[tokens], exit_id)
            link(exit_id, f"B{edge.end}")
    root, sink = "B0", f"B{len(text)}"
    if pronounceable is not None:
        parents = defaultdict(list)
        for left, right in links:
            parents[right].append(left)
        viable = set()
        pending = [sink]
        while pending:
            current = pending.pop()
            if current not in viable:
                viable.add(current)
                pending.extend(parents[current])
        if root not in viable:
            raise ValueError("no complete alignment route remains after lexicon filtering")
        items = {id: item for id, item in items.items() if id in viable}
        links = [(left, right) for left, right in links if left in viable and right in viable]
    reachable = _check_sinks(items, links, root, sink)
    items = {id: item for id, item in items.items() if id in reachable}
    links = [(left, right) for left, right in links if left in reachable]
    refs = {id: ItemRef(_TIER, index) for index, id in enumerate(items)}
    graph_items = []
    for item in items.values():
        weight = item.weight
        attributes = [
            AttributeValue(_LOG_WEIGHT, XsdType.DOUBLE, repr(weight.log_weight)),
            AttributeValue(_COUNT, XsdType.INTEGER, "1"),
            AttributeValue(_SCORED, XsdType.BOOLEAN, "true" if weight.scored else "false"),
            AttributeValue(_PRIOR_KIND, XsdType.STRING, "scored" if weight.scored else "unscored"),
            AttributeValue(_REASON, XsdType.STRING, weight.reason),
            AttributeValue(_ROLE, XsdType.STRING, item.role),
        ]
        if item.tokens:
            attributes.append(AttributeValue(_TOKEN, XsdType.STRING, item.tokens[0]))
        if weight.scored:
            attributes.append(AttributeValue(_P, XsdType.DECIMAL, format(weight.p, "f")))
            if weight.n is not None:
                attributes.append(AttributeValue(_N, XsdType.INTEGER, str(weight.n)))
        graph_items.append(Item(item.id, tuple(attributes)))
    declarations = (
        (_LOG_WEIGHT, XsdType.DOUBLE),
        (_COUNT, XsdType.INTEGER),
        (_SCORED, XsdType.BOOLEAN),
        (_PRIOR_KIND, XsdType.STRING),
        (_REASON, XsdType.STRING),
        (_P, XsdType.DECIMAL),
        (_N, XsdType.INTEGER),
        (_ROLE, XsdType.STRING),
        (_TOKEN, XsdType.STRING),
    )
    graph = Graph(
        (NamespaceDeclaration("frend", NS),),
        (Tier(TierDeclaration(_TIER, "Alignment"), tuple(graph_items)),),
        (
            SimpleRelationDeclaration(QualifiedName(NS, "membership"), _TIER, _TYPE),
            BipartiteRelationDeclaration(_NEXT, _TYPE, _TYPE, acyclic=True),
        ),
        tuple(RelationInstance(_NEXT, refs[left], refs[right]) for left, right in links),
        tuple(
            AttributeDeclaration(name, AttributeDomain.ITEM, kind) for name, kind in declarations
        ),
    )
    return AlignGraph(
        graph, MappingProxyType(items), refs[root], refs[sink], frozenset(kept), tuple(unalignable)
    )
