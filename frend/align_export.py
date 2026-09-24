"""Decoder-neutral text exports for a retained alignment graph.

Both formats preserve the retained token language and never preserve path sums.
Factors, products, and max-merges use exact fractions built from the graph's
Decimal values.  FSG probabilities are rounded with ROUND_HALF_EVEN at the
recorded precision; text-level best-path products agree with exact products
within the computed route tolerance recorded in the manifest.  The manifest
also records every transition's exact rational factor.  AT&T weights are
Decimal natural logarithms of those exact fractions, rounded once at the
recorded precision; products recovered from the written logs have their own
computed manifest tolerance.  Finite decimal logarithms are not claimed to
round-trip exactly.
Reader-specific score conversion, language weighting, and search arithmetic are
outside this module's contract.
"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from frend.align_graph import AlignGraph

__all__ = [
    "ArcManifest",
    "ExportBudgetError",
    "ManifestTransition",
    "to_att_text",
    "to_fsg_text",
    "write_att",
    "write_fsg",
]

SEMANTICS = (
    "exact retained token language; measured-prior-only factors; unscored arcs carry "
    "weight 1; not a path probability model"
)
NUMERIC_DISCLAIMER = (
    "No claim is made about reader integer scores, score shift, language weight, additional "
    "rounding, or search-time accumulation."
)
FSG_REPRESENTATION = (
    "exact Fraction probabilities rounded once to decimal_precision significant digits "
    "with ROUND_HALF_EVEN when written"
)
FSG_GUARANTEE = (
    "best-path products from written probabilities differ from exact-Fraction graph "
    "best-path products by at most product_tolerance; exact transition factors are "
    "recorded as integer numerators and denominators"
)
ATT_REPRESENTATION = (
    "weights written as negated Decimal natural logs of exact Fraction probabilities at "
    "decimal_precision with ROUND_HALF_EVEN"
)
ATT_GUARANTEE = (
    "products recovered from written logs in decimal.Context at decimal_precision differ "
    "from graph best-path products by at most product_tolerance"
)
DEFAULT_PRECISION = 50
DEFAULT_WORK_BUDGET = 1 << 20
MAX_FRACTION_BITS = 4096


class ExportBudgetError(ValueError):
    """The explicit export work budget was exhausted before output was produced."""


@dataclass(frozen=True)
class TransitionOrigin:
    """One graph path contracted into a written transition."""

    items: tuple[str, ...]


@dataclass(frozen=True)
class ManifestTransition:
    """Trace, written weight, and exact rational factor for one transition."""

    id: int
    source: int
    target: int
    word: str | None
    probability: str
    exact_numerator: int
    exact_denominator: int
    origins: tuple[TransitionOrigin, ...]
    scored: bool
    reason: str
    readings: tuple[dict[str, object], ...]
    alternatives: tuple[dict[str, object], ...]
    tokens: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ArcManifest:
    """Machine-readable contract and provenance for one text export."""

    format: str
    semantics: str
    numeric_disclaimer: str
    weight_representation: str
    best_product_guarantee: str
    product_tolerance: str
    longest_weighted_path: int
    accepts_empty: bool
    decimal_precision: int
    budget_limit: int
    budget_used: dict[str, int]
    start_state: int
    final_state: int
    transitions: tuple[ManifestTransition, ...]
    symbols: tuple[str, ...]

    def to_json(self) -> str:
        """Return deterministic JSON suitable for a sidecar file."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class _Candidate:
    source: str
    target: str
    word: str | None
    probability: Fraction
    items: tuple[str, ...]


class _Budget:
    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("work_budget must be a positive integer in export-work units")
        self.limit = limit
        self.topological_steps = 0
        self.log_steps = 0
        self.factor_steps = 0
        self.provenance_steps = 0
        self.source_states_scanned = 0
        self.pairs_examined = 0
        self.transitions_merged = 0
        self.transitions_written = 0

    def charge(self, field: str, units: int = 1) -> None:
        setattr(self, field, getattr(self, field) + units)
        if self.used > self.limit:
            raise ExportBudgetError(
                f"export exceeds {self.limit} work units while charging {field}; refusing whole"
            )

    @property
    def used(self) -> int:
        return sum(
            (
                self.topological_steps,
                self.log_steps,
                self.factor_steps,
                self.provenance_steps,
                self.source_states_scanned,
                self.pairs_examined,
                self.transitions_merged,
                self.transitions_written,
            )
        )

    def record(self) -> dict[str, int]:
        return {
            "topological_steps": self.topological_steps,
            "log_steps": self.log_steps,
            "factor_steps": self.factor_steps,
            "provenance_steps": self.provenance_steps,
            "source_states_scanned": self.source_states_scanned,
            "pairs_examined": self.pairs_examined,
            "transitions_merged": self.transitions_merged,
            "transitions_written": self.transitions_written,
            "total": self.used,
        }


def _children(alignment: AlignGraph, budget: _Budget):
    # A cold count_plan() prepares the complete plan by visiting every graph
    # item and edge.  Both counts are already present in the immutable graph,
    # so reserve the full preparation cost before invoking it.
    budget.charge("topological_steps", len(alignment.items) + len(alignment.graph.relations))
    plan = alignment.count_plan()
    # Refuse before scanning or allocating the complete child mapping.  The
    # vertex and edge units account explicitly for that construction.
    budget.charge("topological_steps", len(plan.labels))
    edge_count = sum(map(len, plan.children))
    budget.charge("topological_steps", edge_count)
    return (
        {
            label: tuple(plan.labels[child] for child in plan.children[index])
            for index, label in enumerate(plan.labels)
        },
        plan,
    )


def _topological_labels(
    plan, children: dict[str, tuple[str, ...]], budget: _Budget
) -> tuple[str, ...]:
    """Return the DAG's labels in deterministic parent-before-child order."""
    # Reserve the vertex and edge visits before starting the topological sort.
    edge_count = sum(map(len, children.values()))
    budget.charge("topological_steps", len(plan.labels) + edge_count)
    rank = {label: index for index, label in enumerate(plan.labels)}
    indegree = {label: 0 for label in plan.labels}
    for successors in children.values():
        for child in successors:
            indegree[child] += 1
    ready = sorted((label for label, degree in indegree.items() if degree == 0), key=rank.get)
    ordered = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=rank.get)
    if len(ordered) != len(plan.labels):
        raise ValueError("alignment graph must be acyclic")
    return tuple(ordered)


def _bounded_fraction(value: Fraction, description: str) -> Fraction:
    """Refuse exact values whose numerator or denominator exceeds 4096 bits."""
    if (
        value.numerator.bit_length() > MAX_FRACTION_BITS
        or value.denominator.bit_length() > MAX_FRACTION_BITS
    ):
        raise ExportBudgetError(
            f"{description} exceeds the {MAX_FRACTION_BITS}-bit exact Fraction bound"
        )
    return value


def _decimal_ratio_bit_bounds(value: Decimal) -> tuple[int, int]:
    """Return conservative numerator/denominator bit bounds without a Fraction."""
    sign, digits, exponent = value.as_tuple()
    del sign
    if not value.is_finite() or not digits or all(digit == 0 for digit in digits):
        return 1, 1
    significant = list(digits)
    while exponent < 0 and significant[-1] == 0:
        significant.pop()
        exponent += 1
    coefficient_digits = len(significant)

    # 3322/1000 is a strict upper bound for log2(10).  Thus this calculation
    # bounds the integers represented by the coefficient and exponent while
    # touching only the Decimal tuple, never constructing 10**abs(exponent).
    def decimal_digits_to_bits(count: int) -> int:
        return max(1, (3322 * count + 999) // 1000)

    def power_of_ten_to_bits(power: int) -> int:
        return max(1, (3322 * power + 999) // 1000)

    if exponent >= 0:
        return decimal_digits_to_bits(coefficient_digits + exponent), 1
    return (
        decimal_digits_to_bits(coefficient_digits),
        power_of_ten_to_bits(-exponent),
    )


def _preflight_factor_ratio(numerator: Decimal, denominator: Decimal, description: str) -> None:
    numerator_bits, numerator_denominator_bits = _decimal_ratio_bit_bounds(numerator)
    denominator_bits, denominator_denominator_bits = _decimal_ratio_bit_bounds(denominator)
    factor_numerator_bits = numerator_bits + denominator_denominator_bits
    factor_denominator_bits = numerator_denominator_bits + denominator_bits
    if max(factor_numerator_bits, factor_denominator_bits) > MAX_FRACTION_BITS:
        raise ExportBudgetError(
            f"{description} exceeds the {MAX_FRACTION_BITS}-bit exact Fraction bound"
        )


def _factors(alignment: AlignGraph, budget: _Budget) -> dict[str, Fraction]:
    # The maxima and normalization comprehensions each scan every item.
    budget.charge("factor_steps", 2 * len(alignment.items))
    maxima: dict[tuple[int, int], Decimal] = {}
    for item in alignment.items.values():
        if item.weight.scored:
            assert item.span is not None and item.weight.p is not None
            maxima[item.span] = max(maxima.get(item.span, Decimal(0)), item.weight.p)
    factors = {}
    for item in alignment.items.values():
        if item.weight.scored:
            assert item.weight.p is not None and item.span is not None
            description = f"factor for item {item.id!r}"
            _preflight_factor_ratio(item.weight.p, maxima[item.span], description)
            factor = Fraction(item.weight.p) / Fraction(maxima[item.span])
        else:
            factor = Fraction(1)
        factors[item.id] = _bounded_fraction(factor, f"factor for item {item.id!r}")
    return factors


def _round_fraction(value: Fraction, context: Context) -> Decimal:
    """Round an exact fraction once under the export's stated decimal rule."""
    with localcontext(context):
        return Decimal(value.numerator) / Decimal(value.denominator)


def _decimal_ulp(value: Decimal, precision: int) -> Decimal:
    """Return one unit in the last place for a precision-digit value."""
    return Decimal(1).scaleb(value.adjusted() - precision + 1)


def _fraction_cost(value: Fraction, context: Context, budget: _Budget) -> Decimal:
    """Return -ln(value), correctly rounded once at the recorded precision."""
    if value == 1:
        return Decimal(0)
    work_precision = context.prec + 3
    while True:
        # Two correctly rounded logarithms are the charged units of this adaptive
        # computation.  Refusal therefore happens before either expensive call.
        budget.charge("log_steps", 2 * work_precision)
        work_context = Context(prec=work_precision, rounding=ROUND_HALF_EVEN)
        with localcontext(work_context):
            denominator_log = Decimal(value.denominator).ln()
            numerator_log = Decimal(value.numerator).ln()

        # Decimal.ln is correctly rounded in its context, so each logarithm's
        # error is below one ULP.  Exact subtraction of their finite Decimal
        # approximations therefore differs from the exact log ratio by less
        # than two of the larger ULPs.
        exact_subtract_precision = (
            max(len(denominator_log.as_tuple().digits), len(numerator_log.as_tuple().digits))
            + abs(denominator_log.as_tuple().exponent - numerator_log.as_tuple().exponent)
            + 1
        )
        with localcontext(Context(prec=exact_subtract_precision, rounding=ROUND_HALF_EVEN)):
            approximation = denominator_log - numerator_log
        error = 2 * max(
            _decimal_ulp(denominator_log, work_precision),
            _decimal_ulp(numerator_log, work_precision),
        )
        with localcontext(context):
            lower = +(approximation - error)
            upper = +(approximation + error)
        # Ziv's strategy: if the complete rigorous interval has one recorded
        # rounding, the exact value necessarily has that rounding too.
        if lower == upper:
            return lower
        work_precision *= 2


def _validate_probability(probability: Decimal) -> None:
    if not probability.is_finite() or probability <= 0 or probability > 1:
        raise ValueError(f"written probability must be finite and in (0, 1], got {probability}")
    try:
        binary32 = struct.unpack("f", struct.pack("f", float(probability)))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(
            f"written probability is not representable as float32: {probability}"
        ) from error
    if binary32 == 0.0:
        raise ValueError(f"written probability underflows float32: {probability}")


def _candidates(alignment: AlignGraph, context: Context, budget: _Budget):
    """Use max-product DAG closure for null-only routes between visible states."""
    children, plan = _children(alignment, budget)
    topological = _topological_labels(plan, children, budget)
    factors = _factors(alignment, budget)
    sink = next(
        plan.labels[index] for index, successors in enumerate(plan.children) if not successors
    )
    root = plan.labels[plan.roots[0]]
    reachable = {label: False for label in plan.labels}
    word_reachable = {label: False for label in plan.labels}
    empty_reachable = {label: False for label in plan.labels}
    reachable[root] = True
    empty_reachable[root] = True
    for current in topological:
        item = alignment.items[current]
        if not reachable[current]:
            continue
        seen_word = word_reachable[current] or bool(item.tokens)
        seen_empty = empty_reachable[current] and not item.tokens
        for child in children[current]:
            budget.charge("pairs_examined")
            reachable[child] = True
            word_reachable[child] = word_reachable[child] or seen_word
            empty_reachable[child] = empty_reachable[child] or seen_empty
    accepts_empty = empty_reachable[sink] and not alignment.items[sink].tokens
    accepts_nonempty = reachable[sink] and (
        word_reachable[sink] or bool(alignment.items[sink].tokens)
    )
    if accepts_empty and not accepts_nonempty:
        raise ValueError("cannot export a grammar whose only accepted token sequence is empty")

    candidates: list[_Candidate] = []
    frontier = {root}
    for item_id in topological:
        item = alignment.items[item_id]
        if not item.tokens:
            continue
        if len(item.tokens) != 1:
            raise ValueError(f"alignment item {item_id!r} emits more than one token")
        for child in children[item_id]:
            budget.charge("pairs_examined")
            candidates.append(
                _Candidate(item_id, child, item.tokens[0], factors[item_id], (item_id,))
            )
            frontier.add(child)

    order = {label: index for index, label in enumerate(topological)}
    for source in sorted(frontier, key=order.__getitem__):
        if alignment.items[source].tokens:
            continue
        # Membership is tested once for every state in this suffix, reachable or not.
        budget.charge("source_states_scanned", len(topological) - order[source])
        best: dict[str, tuple[Fraction, tuple[str, ...]]] = {source: (Fraction(1), ())}
        for current in topological[order[source] :]:
            if current not in best:
                continue
            probability, path = best[current]
            item = alignment.items[current]
            if item.tokens:
                budget.charge("pairs_examined")
                candidates.append(_Candidate(source, current, None, probability, path))
                continue
            product = _bounded_fraction(
                probability * factors[current], f"null-chain product at item {current!r}"
            )
            here = (*path, current)
            if current == sink:
                budget.charge("pairs_examined")
                candidates.append(_Candidate(source, "__final__", None, product, here))
                continue
            for child in children[current]:
                budget.charge("pairs_examined")
                previous = best.get(child)
                if (
                    previous is None
                    or product > previous[0]
                    or (product == previous[0] and here < previous[1])
                ):
                    best[child] = (product, here)
    return candidates, accepts_empty, topological, plan


def _provenance_indices(alignment: AlignGraph, plan, budget: _Budget):
    # Reserve the two item scans used only to size the subsequent passes.
    budget.charge("provenance_steps", 2 * len(plan.labels))
    edge_count = sum(map(len, plan.children))
    alternative_count = sum(len(item.alternatives) for item in alignment.items.values())
    # Parent construction and recursive parent inspection cost at most two edge
    # scans; building parents, token indexing, and alternative indexing each scan
    # all items, with one additional unit for every indexed alternative.
    budget.charge(
        "provenance_steps",
        2 * edge_count + 3 * len(plan.labels) + alternative_count,
    )
    parents = defaultdict(list)
    for index, successors in enumerate(plan.children):
        for child in successors:
            parents[plan.labels[child]].append(plan.labels[index])
    token_indices = {}

    def token_index(item_id):
        if item_id in token_indices:
            return token_indices[item_id]
        item = alignment.items[item_id]
        if item.reading_id is None:
            result = 0
        else:
            word_parents = [
                parent for parent in parents[item_id] if alignment.items[parent].role == "word"
            ]
            result = token_index(word_parents[0]) + 1 if word_parents else 0
        token_indices[item_id] = result
        return result

    for item_id, item in alignment.items.items():
        if item.tokens:
            token_index(item_id)
    alternative_indices = {}
    next_alternative = defaultdict(int)
    for item_id, item in alignment.items.items():
        for local_index, _alternative in enumerate(item.alternatives):
            alternative_indices[item_id, local_index] = next_alternative[item.reading_id]
            next_alternative[item.reading_id] += 1
    return token_indices, alternative_indices


def _origin_metadata(
    alignment: AlignGraph,
    item_ids: tuple[str, ...],
    token_indices,
    alternative_indices,
):
    readings = []
    alternatives = []
    tokens = []
    reasons = set()
    for item_id in item_ids:
        item = alignment.items[item_id]
        reasons.add(item.weight.reason)
        if item.role == "reading":
            record: dict[str, object] = {
                "edge_id": item.reading_id,
                "scored": item.weight.scored,
                "reason": item.weight.reason,
            }
            if item.weight.scored:
                record["p"] = format(item.weight.p, "f")
                record["n"] = item.weight.n
            readings.append(record)
        if item.alternatives:
            alternatives.extend(
                {
                    "reading_edge_id": item.reading_id,
                    "alternative_index": alternative_indices[item_id, index],
                    "text": alternative.text,
                    "provenance": alternative.provenance,
                }
                for index, alternative in enumerate(item.alternatives)
            )
        if item.tokens:
            tokens.append(
                {
                    "reading_edge_id": item.reading_id,
                    "token_index": token_indices[item_id],
                    "token": item.tokens[0],
                    "structure": None if item.reading_id else item.role,
                }
            )
    return tuple(readings), tuple(alternatives), tuple(tokens), reasons


def _prepare(
    alignment: AlignGraph, format_name: str, precision: int, work_budget: int
) -> tuple[
    tuple[ManifestTransition, ...],
    ArcManifest,
    dict[str, int],
    tuple[Fraction, ...],
    _Budget,
]:
    if isinstance(precision, bool) or not isinstance(precision, int) or precision < 1:
        raise ValueError("decimal_precision must be a positive integer")
    context = Context(prec=precision, rounding=ROUND_HALF_EVEN)
    budget = _Budget(work_budget)
    raw, accepts_empty, topological, plan = _candidates(alignment, context, budget)
    merged: dict[tuple[str, str, str | None], _Candidate] = {}
    for candidate in raw:
        budget.charge("transitions_merged")
        key = (candidate.source, candidate.target, candidate.word)
        current = merged.get(key)
        if (
            current is None
            or candidate.probability > current.probability
            or (candidate.probability == current.probability and candidate.items < current.items)
        ):
            merged[key] = candidate
    root = plan.labels[plan.roots[0]]
    state_labels = (root, *(label for label in topological if label != root), "__final__")
    states = {label: index for index, label in enumerate(state_labels)}
    ordered = sorted(
        merged.items(),
        key=lambda entry: (states[entry[0][0]], states[entry[0][1]], entry[0][2] or ""),
    )
    token_indices, alternative_indices = _provenance_indices(alignment, plan, budget)
    transitions = []
    for transition_id, (_key, candidate) in enumerate(ordered):
        budget.charge("transitions_written")
        budget.charge("provenance_steps", len(candidate.items))
        written_probability = _round_fraction(candidate.probability, context)
        _validate_probability(written_probability)
        reading_records = []
        alternative_records = []
        token_records = []
        item_reasons = set()
        readings, alternatives, tokens, reasons = _origin_metadata(
            alignment, candidate.items, token_indices, alternative_indices
        )
        reading_records.extend(readings)
        alternative_records.extend(alternatives)
        token_records.extend(tokens)
        item_reasons.update(reasons)
        scored = bool(reading_records) and any(record["scored"] for record in reading_records)
        transitions.append(
            ManifestTransition(
                transition_id,
                states[candidate.source],
                states[candidate.target],
                candidate.word,
                format(written_probability, "f"),
                candidate.probability.numerator,
                candidate.probability.denominator,
                (TransitionOrigin(candidate.items),),
                scored,
                ",".join(sorted(item_reasons)) if item_reasons else "structure",
                tuple(reading_records),
                tuple(alternative_records),
                tuple(token_records),
            )
        )
    symbols = tuple(
        sorted({transition.word for transition in transitions if transition.word is not None})
    )
    longest_weighted = [0] * len(states)
    for transition, (_key, candidate) in zip(transitions, ordered, strict=True):
        longest_weighted[transition.target] = max(
            longest_weighted[transition.target],
            longest_weighted[transition.source] + (candidate.probability != 1),
        )
    if format_name == "finite-state grammar text":
        representation = FSG_REPRESENTATION
        guarantee = FSG_GUARANTEE
    else:
        representation = ATT_REPRESENTATION
        guarantee = ATT_GUARANTEE
    manifest = ArcManifest(
        format_name,
        SEMANTICS,
        NUMERIC_DISCLAIMER,
        representation,
        guarantee,
        "0",
        max(longest_weighted),
        accepts_empty,
        precision,
        work_budget,
        budget.record(),
        states[root],
        states["__final__"],
        tuple(transitions),
        symbols,
    )
    return (
        tuple(transitions),
        manifest,
        states,
        tuple(candidate.probability for _key, candidate in ordered),
        budget,
    )


def _minimum_fsg_precision(longest_weighted_path: int) -> int:
    precision = 1
    while longest_weighted_path >= 2 * 10**precision:
        precision += 1
    return precision


def _fsg_tolerance(
    transitions: tuple[ManifestTransition, ...],
    probabilities: tuple[Fraction, ...],
    precision: int,
    longest_weighted_path: int,
) -> Decimal:
    # ROUND_HALF_EVEN at p significant digits changes a factor in (0, 1] by
    # at most 0.5 * 10**-p (the widest applicable decimal bin is below 1).
    # Products of [0, 1] factors are 1-Lipschitz in every factor, and max is
    # 1-Lipschitz too, so multiplying that limit by the greatest number of
    # weighted transitions bounds every text-level best-product error.
    per_transition_limit = Fraction(1, 2 * 10**precision)
    derived_bound = min(Fraction(1), longest_weighted_path * per_transition_limit)
    if derived_bound >= 1:
        minimum = _minimum_fsg_precision(longest_weighted_path)
        raise ValueError(
            "FSG product tolerance reaches 1 and gives no accuracy guarantee; "
            f"decimal_precision must be at least {minimum} for this graph"
        )
    exact_arc_error = max(
        (
            abs(Fraction(Decimal(transition.probability)) - probability)
            for transition, probability in zip(transitions, probabilities, strict=True)
        ),
        default=Fraction(0),
    )
    required_bound = longest_weighted_path * exact_arc_error
    if required_bound > derived_bound:
        raise ValueError(
            "FSG product tolerance exceeds the bound implied by decimal_precision and "
            "longest weighted path; refusing export"
        )
    with localcontext(Context(prec=max(precision + 8, 32), rounding=ROUND_HALF_EVEN)):
        return Decimal(derived_bound.numerator) / Decimal(derived_bound.denominator)


def to_fsg_text(
    alignment: AlignGraph,
    name: str,
    *,
    decimal_precision: int = DEFAULT_PRECISION,
    work_budget: int = DEFAULT_WORK_BUDGET,
) -> tuple[str, ArcManifest]:
    """Return generic finite-state-grammar text and its provenance manifest."""
    if not name or any(character.isspace() for character in name):
        raise ValueError("grammar name must be nonempty and contain no whitespace")
    transitions, manifest, states, probabilities, _budget = _prepare(
        alignment, "finite-state grammar text", decimal_precision, work_budget
    )
    tolerance = _fsg_tolerance(
        transitions, probabilities, decimal_precision, manifest.longest_weighted_path
    )
    manifest = replace(manifest, product_tolerance=format(tolerance, "f"))
    lines = [
        f"FSG_BEGIN {name}",
        f"NUM_STATES {len(states)}",
        f"START_STATE {manifest.start_state}",
        f"FINAL_STATE {manifest.final_state}",
    ]
    for transition in transitions:
        line = f"TRANSITION {transition.source} {transition.target} {transition.probability}"
        if transition.word is not None:
            line += f" {transition.word}"
        lines.append(line)
    lines.extend(["FSG_END", ""])
    return "\n".join(lines), manifest


def _att_tolerance_limit(precision: int, longest_path: int) -> Decimal:
    # At p significant digits, rounding ln(p_arc) and then exp(-cost) contributes
    # less than 10**(2-p) absolute probability error per arc: within each decimal
    # decade the log ulp grows by ten as exp(-cost) falls by at least e**-10.
    # A product of values in (0, 1] is 1-Lipschitz in each factor, so summing that
    # per-arc bound over the longest path is a conservative path-product bound.
    return min(Decimal(1), longest_path * Decimal(1).scaleb(2 - precision))


def _att_values_and_tolerance(
    probabilities: tuple[Fraction, ...],
    longest_path: int,
    precision: int,
    budget: _Budget,
) -> tuple[tuple[Decimal, ...], Decimal]:
    context = Context(prec=precision, rounding=ROUND_HALF_EVEN)
    costs = tuple(_fraction_cost(probability, context, budget) for probability in probabilities)
    with localcontext(context):
        recovered = tuple((-cost).exp() for cost in costs)
        arc_error = max(
            (
                abs(Decimal(exact.numerator) / Decimal(exact.denominator) - recovered_probability)
                for exact, recovered_probability in zip(probabilities, recovered, strict=True)
            ),
            default=Decimal(0),
        )
        unit = Decimal(1).scaleb(1 - precision)
        tolerance = min(Decimal(1), longest_path * (arc_error + 2 * unit))
    return costs, tolerance


def _minimum_att_precision(
    probabilities: tuple[Fraction, ...],
    longest_path: int,
    budget: _Budget,
) -> int:
    precision = 1
    while True:
        _costs, tolerance = _att_values_and_tolerance(
            probabilities, longest_path, precision, budget
        )
        if tolerance < 1:
            return precision
        precision += 1


def to_att_text(
    alignment: AlignGraph,
    *,
    decimal_precision: int = DEFAULT_PRECISION,
    work_budget: int = DEFAULT_WORK_BUDGET,
) -> tuple[str, str, ArcManifest]:
    """Return AT&T acceptor text, a symbol table, and the shared manifest."""
    transitions, manifest, _states, probabilities, budget = _prepare(
        alignment, "AT&T finite-state acceptor text", decimal_precision, work_budget
    )
    context = Context(prec=decimal_precision, rounding=ROUND_HALF_EVEN)
    longest = [0] * len(_states)
    for transition in transitions:
        longest[transition.target] = max(longest[transition.target], longest[transition.source] + 1)
    longest_path = max(longest)
    costs, tolerance = _att_values_and_tolerance(
        probabilities, longest_path, decimal_precision, budget
    )
    if tolerance >= 1:
        minimum = _minimum_att_precision(probabilities, longest_path, budget)
        raise ValueError(
            "AT&T product tolerance reaches 1 and gives no accuracy guarantee; "
            f"decimal_precision must be at least {minimum} for this graph"
        )
    with localcontext(context):
        tolerance_limit = _att_tolerance_limit(decimal_precision, max(longest))
        if tolerance > tolerance_limit:
            raise ValueError(
                "AT&T product tolerance exceeds the bound implied by decimal_precision "
                "and longest path; refusing export"
            )
        manifest = replace(
            manifest,
            product_tolerance=format(tolerance, "f"),
            budget_used=budget.record(),
        )
        lines = [
            f"{transition.source}\t{transition.target}\t{transition.word or '<eps>'}\t"
            f"{format(cost, 'f')}"
            for transition, cost in zip(transitions, costs, strict=True)
        ]
    assert manifest.start_state == 0
    assert lines and int(lines[0].split("\t", 1)[0]) == manifest.start_state
    lines.extend([str(manifest.final_state), ""])
    symbols = [
        "<eps>\t0",
        *(f"{word}\t{index}" for index, word in enumerate(manifest.symbols, 1)),
        "",
    ]
    return "\n".join(lines), "\n".join(symbols), manifest


def _write_prepared(path: str | Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def write_fsg(path: str | Path, alignment: AlignGraph, name: str, **kwargs) -> ArcManifest:
    """Preflight completely, then write an FSG file."""
    text, manifest = to_fsg_text(alignment, name, **kwargs)
    _write_prepared(path, text)
    return manifest


def write_att(
    fst_path: str | Path,
    symbols_path: str | Path,
    alignment: AlignGraph,
    **kwargs,
) -> ArcManifest:
    """Preflight completely, then write AT&T graph and symbol-table files."""
    fst, symbols, manifest = to_att_text(alignment, **kwargs)
    _write_prepared(fst_path, fst)
    _write_prepared(symbols_path, symbols)
    return manifest
