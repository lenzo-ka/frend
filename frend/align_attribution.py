"""Attribute one decoder token sequence back to alignment choices.

The decoder has already used acoustics to choose ``aligned``.  Conditioning the
alignment graph on that one sequence then combines the surviving structure with
the measured reading prior.  The result is not a path probability model.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from tiergraph import Emissions, OutputPlan

from frend.align_graph import AlignGraph, AlignItem
from frend.spoken_priors import spoken_tokens

__all__ = [
    "Attribution",
    "ClassMember",
    "OutputClass",
    "PriorShare",
    "PriorSplit",
    "ReadingPosterior",
    "SpanAttribution",
    "attribute",
]

MEANING = "acoustics plus measured prior"
PRIOR_SPLIT_MEANING = "prior, not audio evidence"


@dataclass(frozen=True)
class ReadingPosterior:
    """Conditioned mass and posterior for one reading."""

    reading_id: str
    log_mass: float
    mass: float
    posterior: float


@dataclass(frozen=True)
class ClassMember:
    """One reading form or passthrough route in an output-equivalent class."""

    id: str
    item_id: str
    reading_id: str | None
    role: str
    log_mass: float
    mass: float
    posterior: float


@dataclass(frozen=True)
class PriorShare:
    """An exact measured-prior share, never an inference from the audio.

    ``share`` is a rational constructed directly from the measured Decimal
    ``p`` without a lossy float conversion.
    """

    member: str
    p: Decimal
    share: Fraction


@dataclass(frozen=True)
class PriorSplit:
    """The exact prior-only label split among scored class members."""

    meaning: str
    shares: tuple[PriorShare, ...]
    tied: tuple[str, ...]


@dataclass(frozen=True)
class OutputClass:
    """Members indistinguishable to audio because they emit the same tokens."""

    tokens: tuple[str, ...]
    members: tuple[ClassMember, ...]
    log_mass: float
    mass: float
    posterior: float
    prior_split: PriorSplit
    no_evidence: tuple[str, ...]


@dataclass(frozen=True)
class SpanAttribution:
    """Reading and output-class attribution at one source span."""

    span: tuple[int, int]
    readings: tuple[ReadingPosterior, ...]
    output_classes: tuple[OutputClass, ...]


@dataclass(frozen=True)
class Attribution:
    """Attribution conditioned on one complete decoder output."""

    aligned: tuple[str, ...]
    meaning: str
    prior: bool
    prior_scale: float
    zero_mass: bool
    total_log_mass: float | None
    total_mass: float | None
    spans: tuple[SpanAttribution, ...]


def _tokens(aligned: Sequence[str]) -> tuple[str, ...]:
    if isinstance(aligned, (str, bytes)) or not isinstance(aligned, Sequence):
        raise TypeError("aligned must be a sequence of token strings")
    result = tuple(aligned)
    for token in result:
        if type(token) is not str:
            raise TypeError(f"aligned token {token!r} is not a string")
    return result


def _values(alignment: AlignGraph, *, prior: bool, prior_scale: float) -> tuple[float, ...]:
    plan = alignment.log_plan()
    values = []
    for index, label in enumerate(plan.labels):
        item = alignment.items[label]
        if item.weight.scored:
            values.append(plan.values[index] * prior_scale if prior else 0.0)
        else:
            values.append(plan.values[index])
    return tuple(values)


def _first_divergence(alignment: AlignGraph, aligned: tuple[str, ...]) -> str:
    """Name the first token outside every graph-output prefix."""
    plan = alignment.log_plan()
    reached: list[set[int]] = [set() for _label in plan.labels]
    best = 0
    for root in plan.roots:
        reached[root].add(0)
    indegree = [0] * len(plan.labels)
    for children in plan.children:
        for child in children:
            indegree[child] += 1
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    ordered = []
    while ready:
        index = ready.pop(0)
        ordered.append(index)
        for child in plan.children[index]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    for index in ordered:
        label = plan.labels[index]
        emission = alignment.items[label].tokens
        for start in reached[index]:
            position = start
            for token in emission:
                if position == len(aligned) or aligned[position] != token:
                    best = max(best, position)
                    break
                position += 1
                best = max(best, position)
            else:
                for child in plan.children[index]:
                    reached[child].add(position)
    if best < len(aligned):
        return repr(aligned[best])
    return "end of decoder output"


def _exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _posterior(log_mass: float, total: float) -> float:
    if log_mass == -math.inf:
        return 0.0
    return math.exp(log_mass - total)


def _log_sum(values: Sequence[float]) -> float:
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in finite))


def _member(
    item: AlignItem, reading: AlignItem | None, log_mass: float, total: float
) -> ClassMember:
    return ClassMember(
        item.reading_id or item.id,
        item.id,
        item.reading_id,
        "reading" if reading is not None else "passthrough",
        log_mass,
        _exp(log_mass),
        _posterior(log_mass, total),
    )


def _prior_split(
    members: Sequence[ClassMember], alignment: AlignGraph
) -> tuple[PriorSplit, tuple[str, ...]]:
    scored = []
    no_evidence = []
    for member in members:
        item = alignment.items[member.reading_id] if member.reading_id is not None else None
        if item is None or not item.weight.scored:
            no_evidence.append(member.id)
        else:
            if item.weight.p is None:
                raise ValueError(f"scored class member {member.id!r} has no measured prior")
            scored.append((member.id, item.weight.p))
    denominator = sum((Fraction(p) for _member_id, p in scored), Fraction())
    shares = tuple(PriorShare(member_id, p, Fraction(p) / denominator) for member_id, p in scored)
    maximum = max((p for _member_id, p in scored), default=None)
    tied = tuple(member_id for member_id, p in scored if p == maximum)
    return PriorSplit(PRIOR_SPLIT_MEANING, shares, tied), tuple(no_evidence)


def attribute(
    graph: AlignGraph,
    aligned: Sequence[str],
    *,
    prior_scale: float = 1.0,
    prior: bool = True,
) -> Attribution:
    """Attribute a complete decoder token sequence to readings and output classes.

    The caller must strip decoder filler and silence tokens before calling.  With
    ``prior=True``, ``prior_scale`` multiplies only measured log weights; a scale
    of zero therefore removes the measured prior.  With ``prior=False``, every
    scored weight is set to zero, leaving only graph structure and the acoustic
    choice already represented by ``aligned``.

    A structurally foreign decoder output is refused with its first divergent
    token.  A structurally accepted candidate whose supplied values have zero
    mass returns ``zero_mass=True`` and no attribution values.
    """
    if type(prior) is not bool:
        raise TypeError("prior must be a bool")
    if isinstance(prior_scale, bool) or not isinstance(prior_scale, (int, float)):
        raise TypeError("prior_scale must be a finite real number")
    scale = float(prior_scale)
    if not math.isfinite(scale):
        raise ValueError("prior_scale must be a finite real number")
    tokens = _tokens(aligned)
    plan = graph.log_plan()
    emissions = Emissions.bind(
        plan,
        {label: graph.items[label].tokens for label in plan.labels if graph.items[label].tokens},
    )
    output = OutputPlan.prepare(plan, emissions, (tokens,))
    if not output.accepted[0]:
        divergent = _first_divergence(graph, tokens)
        raise ValueError(
            "decoder output is not a path of the exported graph; "
            f"first divergent token: {divergent}"
        )
    values = _values(graph, prior=prior, prior_scale=scale)
    marginals = output.item_marginals(0, values)
    marginal_values = marginals.values
    if marginal_values is None:
        if not marginals.zero_mass:
            raise ValueError("marginal invariant violated: nonzero mass has no item values")
        return Attribution(tokens, MEANING, prior, scale, True, None, None, ())
    if marginals.zero_mass:
        raise ValueError("marginal invariant violated: zero mass has item values")
    total = marginals.total
    by_span_readings: dict[tuple[int, int], list[ReadingPosterior]] = defaultdict(list)
    by_class: dict[tuple[tuple[int, int], tuple[str, ...]], list[ClassMember]] = defaultdict(list)
    for index, label in enumerate(plan.labels):
        item = graph.items[label]
        log_mass = marginal_values[index]
        if item.role == "reading":
            if item.span is None or item.reading_id is None:
                raise ValueError(
                    f"reading item invariant violated: {item.id!r} requires span and reading_id"
                )
            by_span_readings[item.span].append(
                ReadingPosterior(
                    item.reading_id,
                    log_mass,
                    _exp(log_mass),
                    _posterior(log_mass, total),
                )
            )
        elif item.role == "form-exit":
            if item.span is None or item.reading_id is None or not item.alternatives:
                raise ValueError(
                    "form-exit item invariant violated: "
                    f"{item.id!r} requires span, reading_id, and alternatives"
                )
            form_tokens = spoken_tokens(item.alternatives[0].text)
            by_class[item.span, form_tokens].append(
                _member(item, graph.items[item.reading_id], log_mass, total)
            )
        elif item.role == "token" and item.span is not None:
            by_class[item.span, item.tokens].append(_member(item, None, log_mass, total))
    all_spans = sorted(set(by_span_readings) | {span for span, _tokens in by_class})
    spans = []
    for span in all_spans:
        classes = []
        for (class_span, class_tokens), members in by_class.items():
            if class_span != span:
                continue
            log_mass = _log_sum([member.log_mass for member in members])
            split, no_evidence = _prior_split(members, graph)
            classes.append(
                OutputClass(
                    class_tokens,
                    tuple(members),
                    log_mass,
                    _exp(log_mass),
                    math.fsum(member.posterior for member in members),
                    split,
                    no_evidence,
                )
            )
        classes.sort(key=lambda output_class: output_class.tokens)
        spans.append(
            SpanAttribution(
                span,
                tuple(by_span_readings[span]),
                tuple(classes),
            )
        )
    return Attribution(tokens, MEANING, prior, scale, False, total, _exp(total), tuple(spans))
