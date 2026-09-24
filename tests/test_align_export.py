"""Behavioral falsifiers for alignment text exports."""

from __future__ import annotations

import importlib.metadata
import random
import re
from collections import defaultdict
from dataclasses import replace
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from fractions import Fraction
from time import perf_counter

import pytest

import irn.align_export as export_module
from irn.align_export import ExportBudgetError, to_att_text, to_fsg_text, write_att, write_fsg
from irn.align_graph import build_align_graph
from irn.fold_resolve import CoverScore
from irn.lattice import ChoiceGraph, ReadingEdge, ReadingRank, SemanticRank, resolve_choices
from irn.type_priors import ReadingPrior
from irn.verbalize import SpokenAlternative, VerbalizedUnit


def prior(p: str) -> ReadingPrior:
    return ReadingPrior("number", "A", Decimal(p), 100, True, "measured")


def choices(text, specs=()):
    """Build a real carrier; specs are (start, end, forms, prior)."""
    base = resolve_choices([], source_text=text)
    edges = []
    units = []
    for index, (start, end, forms, measurement) in enumerate(specs):
        edge_id = f"c{index}"
        geometry = CoverScore(end - start, 1, 0)
        semantic = SemanticRank("unsupported", None, None)
        edges.append(
            ReadingEdge(
                edge_id,
                start,
                end,
                None,
                "reading",
                geometry,
                measurement,
                ReadingRank(end - start, 1, 0, semantic),
            )
        )
        units.append(
            VerbalizedUnit(
                edge_id,
                tuple(
                    SpokenAlternative(form, f"source:{alternative}")
                    for alternative, form in enumerate(forms)
                ),
                None,
                None,
                True,
            )
        )
    return ChoiceGraph(replace(base, edges=tuple(edges) + base.edges), tuple(units))


def graph_exact_oracle(alignment):
    """Enumerate token sequence -> exact best product."""
    plan = alignment.count_plan()
    maxima = defaultdict(lambda: Decimal(0))
    for item in alignment.items.values():
        if item.weight.scored:
            maxima[item.span] = max(maxima[item.span], item.weight.p)

    def walk(index, tokens, product):
        item = alignment.items[plan.labels[index]]
        if item.weight.scored:
            product *= Fraction(item.weight.p) / Fraction(maxima[item.span])
        tokens = (*tokens, *item.tokens)
        if not plan.children[index]:
            yield tokens, product
        for child in plan.children[index]:
            yield from walk(child, tokens, product)

    exact = {}
    for tokens, product in walk(plan.roots[0], (), Fraction(1)):
        exact[tokens] = max(exact.get(tokens, Fraction(0)), product)
    return exact


def _round_oracle(exact, precision):
    context = Context(prec=precision, rounding=ROUND_HALF_EVEN)
    with localcontext(context):
        return {
            tokens: Decimal(product.numerator) / Decimal(product.denominator)
            for tokens, product in exact.items()
        }


def graph_oracle(alignment, precision):
    """Enumerate token sequence -> exact best product, then round each once."""
    return _round_oracle(graph_exact_oracle(alignment), precision)


def _enumerate_decimal(start, final, transitions, precision):
    result = {}

    def walk(state, tokens, product):
        if state == final:
            result[tokens] = max(result.get(tokens, Decimal(0)), product)
            return
        for target, word, probability in transitions[state]:
            with localcontext(Context(prec=precision)):
                next_product = product * probability
            walk(target, tokens if word is None else (*tokens, word), next_product)

    walk(start, (), Decimal(1))
    return result


def _enumerate_exact_fractions(start, final, transitions):
    exact = {}

    def walk(state, tokens, product):
        if state == final:
            exact[tokens] = max(exact.get(tokens, Fraction(0)), product)
            return
        for target, word, probability in transitions[state]:
            walk(
                target,
                tokens if word is None else (*tokens, word),
                product * probability,
            )

    walk(start, (), Fraction(1))
    return exact


def _enumerate_fractions(start, final, transitions, precision):
    return _round_oracle(_enumerate_exact_fractions(start, final, transitions), precision)


def parse_fsg(text, manifest):
    start = final = None
    transitions = defaultdict(list)
    records = []
    with localcontext(Context(prec=manifest.decimal_precision)):
        for line in text.splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "START_STATE":
                start = int(fields[1])
            elif fields[0] == "FINAL_STATE":
                final = int(fields[1])
            elif fields[0] == "TRANSITION":
                source, target = map(int, fields[1:3])
                record = (source, target, fields[4] if len(fields) == 5 else None)
                records.append(record)
                transitions[source].append((target, record[2], Fraction(Decimal(fields[3]))))
    return _enumerate_fractions(start, final, transitions, manifest.decimal_precision), records


def parse_fsg_exact(text, manifest):
    start = final = None
    transitions = defaultdict(list)
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "START_STATE":
            start = int(fields[1])
        elif fields[0] == "FINAL_STATE":
            final = int(fields[1])
        elif fields[0] == "TRANSITION":
            source, target = map(int, fields[1:3])
            word = fields[4] if len(fields) == 5 else None
            transitions[source].append((target, word, Fraction(Decimal(fields[3]))))
    return _enumerate_exact_fractions(start, final, transitions)


def manifest_exact_oracle(manifest):
    transitions = defaultdict(list)
    for transition in manifest.transitions:
        transitions[transition.source].append(
            (
                transition.target,
                transition.word,
                Fraction(transition.exact_numerator, transition.exact_denominator),
            )
        )
    return _enumerate_exact_fractions(manifest.start_state, manifest.final_state, transitions)


def parse_att(text, manifest):
    transitions = defaultdict(list)
    final = None
    records = []
    with localcontext(Context(prec=manifest.decimal_precision)):
        for line in text.splitlines():
            fields = line.split("\t")
            if len(fields) == 1:
                final = int(fields[0])
                continue
            assert len(fields) == 4
            source, target = map(int, fields[:2])
            word = None if fields[2] == "<eps>" else fields[2]
            probability = (-Decimal(fields[3])).exp()
            transitions[source].append((target, word, probability))
            records.append((source, target, word))
    return (
        _enumerate_decimal(manifest.start_state, final, transitions, manifest.decimal_precision),
        records,
    )


def assert_att_close(actual, expected, manifest):
    assert actual.keys() == expected.keys()
    tolerance = Decimal(manifest.product_tolerance)
    with localcontext(Context(prec=manifest.decimal_precision)):
        assert all(abs(actual[tokens] - expected[tokens]) <= tolerance for tokens in expected)


def assert_fsg_contract(alignment, text, manifest):
    expected_exact = graph_exact_oracle(alignment)
    actual_exact = parse_fsg_exact(text, manifest)
    tolerance = Fraction(Decimal(manifest.product_tolerance))
    assert actual_exact.keys() == expected_exact.keys()
    assert all(
        abs(actual_exact[tokens] - expected_exact[tokens]) <= tolerance for tokens in expected_exact
    )
    assert manifest_exact_oracle(manifest) == expected_exact


def _att_longest_path(text, final_state):
    longest = [0] * (final_state + 1)
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) == 4:
            source, target = map(int, fields[:2])
            longest[target] = max(longest[target], longest[source] + 1)
    return max(longest)


ORACLE_CARRIERS = [
    choices("a,b"),
    choices("3 - 4", [(2, 3, ["minus", ""], prior("1"))]),
    choices(
        "ab",
        [
            (0, 1, ["same first", "same second"], prior("1")),
            (1, 2, ["next"], prior("2")),
            (0, 2, ["whole"], prior("3")),
        ],
    ),
    choices(
        "xy", [(0, 2, ["many tokens", "many choices"], prior("1")), (0, 2, ["winner"], prior("3"))]
    ),
    choices("!!", [(0, 1, ["bang", ""], prior("1")), (1, 2, ["bang"], prior("2"))]),
]


def _expect(statement, call, *args, **kwargs):
    """Run an export whose success on this graph is the stated expectation.

    A refusal fails the test with the statement, so the kill is carried by what the
    test says should happen rather than by a production crash inside it.
    """
    try:
        return call(*args, **kwargs)
    except (ValueError, TypeError) as error:
        pytest.fail(f"{statement}, but it raised {error!r}")


@pytest.mark.parametrize("carrier", ORACLE_CARRIERS)
def test_exports_equal_graph_language_and_best_products(carrier):
    """Kill pushing, repeated factors, sum merges, lost junctions, and dropped words."""
    alignment = build_align_graph(carrier)
    precision = 18
    expected = graph_oracle(alignment, precision)
    fsg, fsg_manifest = _expect(
        "every oracle graph exports as FSG",
        to_fsg_text,
        alignment,
        "oracle",
        decimal_precision=precision,
    )
    att, _symbols, att_manifest = _expect(
        "every oracle graph exports as AT&T", to_att_text, alignment, decimal_precision=precision
    )
    assert_fsg_contract(alignment, fsg, fsg_manifest)
    assert_att_close(parse_att(att, att_manifest)[0], expected, att_manifest)


def test_nonterminating_probability_text_and_log_tolerance():
    alignment = build_align_graph(
        choices(
            "x",
            [
                (0, 1, ["one"], prior("1")),
                (0, 1, ["two"], prior("2")),
                (0, 1, ["three"], prior("3")),
            ],
        )
    )
    precision = 18
    expected = graph_oracle(alignment, precision)
    fsg, fsg_manifest = _expect(
        "thirds export as FSG within tolerance",
        to_fsg_text,
        alignment,
        "thirds",
        decimal_precision=precision,
    )
    att, _symbols, att_manifest = _expect(
        "thirds export as AT&T within tolerance",
        to_att_text,
        alignment,
        decimal_precision=precision,
    )
    actual_fsg = parse_fsg(fsg, fsg_manifest)[0]
    actual_att = parse_att(att, att_manifest)[0]
    assert actual_fsg == expected
    assert actual_fsg[("one",)] == Decimal("0.333333333333333333")
    assert actual_fsg[("two",)] == Decimal("0.666666666666666667")
    assert actual_att[("one",)] != expected[("one",)]
    assert_att_close(actual_att, expected, att_manifest)
    assert Decimal(fsg_manifest.product_tolerance) > 0
    assert Decimal(att_manifest.product_tolerance) > 0


@pytest.mark.parametrize("precision", [2, 50])
def test_three_factor_word_then_null_chain_rounds_exact_product_once(precision):
    """Cover 2/3 followed by two empty 1/3 readings across a word boundary."""
    alignment = build_align_graph(
        choices(
            "abc",
            [
                (0, 1, ["x"], prior("2")),
                (0, 1, ["y"], prior("3")),
                (1, 2, [""], prior("1")),
                (1, 2, ["q"], prior("3")),
                (2, 3, [""], prior("1")),
                (2, 3, ["r"], prior("3")),
            ],
        )
    )
    expected = graph_oracle(alignment, precision)
    text, manifest = to_fsg_text(alignment, "three_factors", decimal_precision=precision)
    actual = parse_fsg(text, manifest)[0]
    assert actual == expected
    assert actual[("x",)] == (
        Decimal("0.074")
        if precision == 2
        else Decimal("0.074074074074074074074074074074074074074074074074074")
    )


@pytest.mark.parametrize("precision", [2, 3, 18, 50])
def test_two_visible_two_thirds_obeys_fsg_route_tolerance(precision):
    alignment = build_align_graph(
        choices(
            "aa",
            [
                (0, 1, ["w0"], prior("2")),
                (0, 1, ["v0"], prior("3")),
                (1, 2, ["w1"], prior("2")),
                (1, 2, ["v1"], prior("3")),
            ],
        )
    )
    text, manifest = _expect(
        "two visible two-thirds factors export within the route tolerance",
        to_fsg_text,
        alignment,
        "two_thirds",
        decimal_precision=precision,
    )
    actual = parse_fsg(text, manifest)[0]
    expected = graph_oracle(alignment, precision)
    assert manifest.longest_weighted_path == 2
    assert abs(actual[("w0", "w1")] - expected[("w0", "w1")]) <= Decimal(manifest.product_tolerance)
    assert_fsg_contract(alignment, text, manifest)


def test_random_whole_routes_obey_fsg_text_tolerance_and_exact_manifest():
    rng = random.Random(77411)
    primes = [3, 7, 11, 13, 17, 19]
    for precision in range(2, 19):
        specs = []
        factors = [(2, 3), (2, 3)] if precision == 2 else []
        factors.extend(
            (numerator, denominator)
            for denominator in (rng.choice(primes) for _ in range(rng.randrange(3, 6)))
            for numerator in [rng.randrange(1, denominator)]
        )
        for index, (numerator, denominator) in enumerate(factors):
            form = (
                "" if index % 3 == 0 else f"word{index} tail" if index % 3 == 1 else f"word{index}"
            )
            specs.extend(
                [
                    (
                        index,
                        index + 1,
                        [form],
                        prior(str(numerator)),
                    ),
                    (
                        index,
                        index + 1,
                        [f"maximum{index}"],
                        prior(str(denominator)),
                    ),
                ]
            )
        alignment = build_align_graph(choices("a" * len(factors), specs))
        text, manifest = to_fsg_text(alignment, f"random{precision}", decimal_precision=precision)
        assert_fsg_contract(alignment, text, manifest)


def test_att_large_rational_logs_keep_recorded_precision():
    alignment = build_align_graph(
        choices(
            "x",
            [
                (0, 1, ["small"], prior("0.1234567890123")),
                (0, 1, ["large"], prior("0.9876543120991")),
            ],
        )
    )
    text, _symbols, _manifest = to_att_text(alignment, decimal_precision=4)
    costs = [line.rsplit("\t", 1)[1] for line in text.splitlines() if "\t" in line]
    assert "2.079" in costs


@pytest.mark.parametrize(
    ("precision", "difference"),
    [
        (2, 114_999_997_000),
        (8, 112_345_674_999_999_999),
        (18, 112_345_678_901_234_567_499_999_999),
    ],
)
def test_att_near_tie_logs_are_independently_correctly_rounded(precision, difference):
    denominator = 10**564
    factor = Fraction(denominator - difference, denominator)
    needed_digits = 564 + precision

    def oracle(extra_factor):
        work = Context(prec=extra_factor * needed_digits, rounding=ROUND_HALF_EVEN)
        with localcontext(work):
            exact_approximation = Decimal(factor.denominator).ln() - Decimal(factor.numerator).ln()
        with localcontext(Context(prec=precision, rounding=ROUND_HALF_EVEN)):
            return +exact_approximation

    expected = oracle(4)
    assert oracle(8) == expected
    budget = export_module._Budget(1 << 20)
    assert (
        export_module._fraction_cost(
            factor, Context(prec=precision, rounding=ROUND_HALF_EVEN), budget
        )
        == expected
    )


def test_att_near_tie_exports_correctly_rounded_cost():
    denominator = 10**564
    difference = 114_999_997_000
    numerator = denominator - difference
    probability = f"0.{numerator:0564d}"
    alignment = build_align_graph(
        choices(
            "x",
            [
                (0, 1, ["near"], prior(probability)),
                (0, 1, ["one"], prior("1")),
            ],
        )
    )
    text, _symbols, _manifest = to_att_text(alignment, decimal_precision=2)
    costs = [Decimal(line.rsplit("\t", 1)[1]) for line in text.splitlines() if "\t" in line]
    assert Decimal("1.1E-553") in costs
    assert Decimal("1.2E-553") not in costs


def test_att_log_iteration_refuses_when_its_reserved_work_exhausts_budget():
    context = Context(prec=2, rounding=ROUND_HALF_EVEN)
    with pytest.raises(ExportBudgetError, match="log_steps"):
        export_module._fraction_cost(Fraction(2, 3), context, export_module._Budget(9))


@pytest.mark.parametrize(
    ("size", "precision", "prior_limit"),
    [
        (12, 8, Decimal("0.00000567")),
        (12, 16, Decimal("0.0000000000000594")),
        (24, 8, Decimal("0.00001071")),
        (24, 16, Decimal("0.0000000000001071")),
    ],
)
def test_att_near_one_visible_words_match_pre_exact_regression_tolerance(
    size, precision, prior_limit
):
    alignment = build_align_graph(
        choices(
            "!" * size,
            [
                spec
                for index in range(size)
                for spec in (
                    (index, index + 1, [f"lower{index}"], prior("0.9999123456789")),
                    (index, index + 1, [f"upper{index}"], prior("0.9999999999999")),
                )
            ],
        )
    )
    _text, _symbols, manifest = to_att_text(alignment, decimal_precision=precision)
    assert Decimal(manifest.product_tolerance) <= prior_limit


@pytest.mark.parametrize("format_name", ["fsg", "att"])
def test_vacuous_product_tolerance_refuses_and_names_working_precision(format_name):
    size = 250
    alignment = build_align_graph(
        choices(
            "!" * size,
            [
                spec
                for index in range(size)
                for spec in (
                    (index, index + 1, [f"lower{index}"], prior("2")),
                    (index, index + 1, [f"upper{index}"], prior("3")),
                )
            ],
        )
    )

    def export(precision):
        if format_name == "fsg":
            return to_fsg_text(alignment, "long", decimal_precision=precision, work_budget=1 << 30)
        return to_att_text(alignment, decimal_precision=precision, work_budget=1 << 30)

    with pytest.raises(ValueError, match="tolerance reaches 1") as refused:
        export(2)
    match = re.search(r"at least (\d+)", str(refused.value))
    assert match is not None
    minimum = int(match.group(1))
    result = export(minimum)
    manifest = result[1] if format_name == "fsg" else result[2]
    assert Decimal(manifest.product_tolerance) < 1


def test_att_precision_contract_bounds_tolerance_and_written_digits():
    alignment = build_align_graph(
        choices(
            "x",
            [
                (0, 1, ["one"], prior("1")),
                (0, 1, ["two"], prior("2")),
                (0, 1, ["three"], prior("3")),
            ],
        )
    )
    precision = 18
    text, _symbols, manifest = _expect(
        "thirds export as AT&T within the precision contract",
        to_att_text,
        alignment,
        decimal_precision=precision,
    )
    weights = [line.rsplit("\t", 1)[1] for line in text.splitlines() if "\t" in line]

    # Zero is exact; every inexact logarithm must carry the context's full
    # significant-digit precision in the actual text given to the reader.
    assert all(
        Decimal(weight).is_zero() or len(Decimal(weight).as_tuple().digits) == precision
        for weight in weights
    )
    longest_path = _att_longest_path(text, manifest.final_state)
    independent_limit = min(
        Decimal(1), longest_path * Decimal(1).scaleb(2 - manifest.decimal_precision)
    )
    assert Decimal(manifest.product_tolerance) <= independent_limit


def test_generated_exports_equal_graph_oracle():
    rng = random.Random(90210)
    forms = ["", "!", "a", "a b", "a c", "word"]
    for _ in range(30):
        text = rng.choice(["ab", "a-b", "a b", "!!", "3 - 4"])
        specs = []
        for _reading in range(3):
            start = rng.randrange(len(text))
            end = rng.randrange(start + 1, len(text) + 1)
            specs.append((start, end, rng.sample(forms, 3), prior(rng.choice(["1", "2", "3"]))))
        alignment = build_align_graph(choices(text, specs))
        precision = 16
        expected = graph_oracle(alignment, precision)
        fsg, fsg_manifest = to_fsg_text(alignment, "generated", decimal_precision=precision)
        att, _symbols, att_manifest = to_att_text(alignment, decimal_precision=precision)
        assert_fsg_contract(alignment, fsg, fsg_manifest)
        assert_att_close(parse_att(att, att_manifest)[0], expected, att_manifest)


def test_fsg_has_no_composable_nulls_or_duplicate_labels():
    alignment = build_align_graph(
        choices("!!", [(0, 1, ["", "one"], prior("1")), (1, 2, ["two"], prior("2"))])
    )
    text, manifest = to_fsg_text(alignment, "constraints")
    _language, records = parse_fsg(text, manifest)
    assert len(records) == len(set(records))
    null_targets = {target for _source, target, word in records if word is None}
    null_sources = {source for source, _target, word in records if word is None}
    assert null_targets.isdisjoint(null_sources)


def _write(format_name, tmp_path, alignment, **kwargs):
    if format_name == "fsg":
        targets = (tmp_path / "target.fsg",)

        def action():
            return write_fsg(targets[0], alignment, "refusal", **kwargs)

    else:
        targets = (tmp_path / "target.att", tmp_path / "target.sym")

        def action():
            return write_att(*targets, alignment, **kwargs)

    return action, targets


@pytest.mark.parametrize("format_name", ["fsg", "att"])
@pytest.mark.parametrize("bad_probability", [Decimal("0"), Decimal("-1"), Decimal("2")])
def test_invalid_probability_refuses_every_writer(
    monkeypatch, tmp_path, format_name, bad_probability
):
    alignment = build_align_graph(choices("x"))
    monkeypatch.setattr(
        export_module,
        "_factors",
        lambda graph, budget: {item_id: Fraction(bad_probability) for item_id in graph.items},
    )
    action, targets = _write(format_name, tmp_path, alignment)
    with pytest.raises(ValueError, match="probability"):
        action()
    assert not any(target.exists() for target in targets)


@pytest.mark.parametrize("format_name", ["fsg", "att"])
@pytest.mark.parametrize(
    ("case", "alignment", "kwargs", "error", "message"),
    [
        (
            "underflow",
            build_align_graph(
                choices("x", [(0, 1, ["tiny"], prior("1e-50")), (0, 1, ["large"], prior("1"))])
            ),
            {},
            ValueError,
            "underflows float32",
        ),
        ("empty", build_align_graph(choices("!")), {}, ValueError, "only accepted"),
        (
            "budget",
            build_align_graph(choices("hello")),
            {"work_budget": 1},
            ExportBudgetError,
            "work units",
        ),
        (
            "fraction-bound",
            build_align_graph(
                choices(
                    "x",
                    [
                        (0, 1, ["tiny"], prior("1e-2000")),
                        (0, 1, ["ordinary"], prior("1")),
                    ],
                )
            ),
            {},
            ExportBudgetError,
            "4096-bit exact Fraction bound",
        ),
    ],
)
def test_structural_refusals_leave_no_writer_files(
    tmp_path, format_name, case, alignment, kwargs, error, message
):
    target_dir = tmp_path / case
    target_dir.mkdir()
    action, targets = _write(format_name, target_dir, alignment, **kwargs)
    with pytest.raises(error, match=message):
        action()
    assert not any(target.exists() for target in targets)


def test_cold_count_plan_is_charged_before_preparation(monkeypatch):
    alignment = build_align_graph(choices("hello"))
    assert "_count_plan" not in alignment.__dict__
    events = []
    original_count_plan = type(alignment).count_plan
    original_charge = export_module._Budget.charge

    def count_plan_spy(self):
        events.append("count_plan_prepare")
        return original_count_plan(self)

    def charge_spy(self, field, units=1):
        events.append(("charge", field, units))
        return original_charge(self, field, units)

    monkeypatch.setattr(type(alignment), "count_plan", count_plan_spy)
    monkeypatch.setattr(export_module._Budget, "charge", charge_spy)
    reservation = len(alignment.items) + len(alignment.graph.relations)
    with pytest.raises(ExportBudgetError, match="topological_steps"):
        to_fsg_text(alignment, "budget", work_budget=reservation)
    assert events[0] == ("charge", "topological_steps", reservation)
    assert events[1] == "count_plan_prepare"

    cold = build_align_graph(choices("hello"))
    assert "_count_plan" not in cold.__dict__
    events.clear()
    with pytest.raises(ExportBudgetError, match="topological_steps"):
        to_fsg_text(cold, "budget", work_budget=reservation - 1)
    assert events == [("charge", "topological_steps", reservation)]


def test_huge_decimal_factor_refuses_before_fraction_construction_is_expensive(monkeypatch):
    alignment = build_align_graph(
        choices(
            "x",
            [
                (0, 1, ["tiny"], prior("1e-1000000")),
                (0, 1, ["ordinary"], prior("1")),
            ],
        )
    )
    original_fraction = export_module.Fraction

    def fraction_spy(numerator=0, denominator=None):
        if isinstance(numerator, Decimal) and numerator.adjusted() < -4096:
            raise AssertionError("oversized Decimal reached Fraction construction")
        if denominator is None:
            return original_fraction(numerator)
        return original_fraction(numerator, denominator)

    monkeypatch.setattr(export_module, "Fraction", fraction_spy)
    started = perf_counter()
    with pytest.raises(ExportBudgetError, match="4096-bit exact Fraction bound"):
        to_fsg_text(alignment, "huge")
    elapsed = perf_counter() - started
    assert elapsed < 0.05, f"preflight took {elapsed:.3f}s; bound is 0.05s"


def test_accepts_empty_is_exact():
    alongside = build_align_graph(choices("!", [(0, 1, ["bang"], prior("1"))]))
    ordinary = build_align_graph(choices("x", [(0, 1, ["word"], prior("1"))]))
    assert to_fsg_text(alongside, "alongside")[1].accepts_empty is True
    assert to_fsg_text(ordinary, "ordinary")[1].accepts_empty is False


def test_manifest_contract_and_graph_item_round_trip():
    alignment = build_align_graph(
        choices("x!", [(0, 1, ["ex", "X"], prior("1")), (0, 1, ["why"], prior("3"))])
    )
    _text, manifest = to_fsg_text(alignment, "manifest")
    payload = export_module.json.loads(manifest.to_json())
    assert payload["semantics"] == export_module.SEMANTICS
    assert "decimal_precision significant digits" in payload["weight_representation"]
    assert "differ" in payload["best_product_guarantee"]
    assert payload["longest_weighted_path"] >= 1
    assert "No claim is made" in payload["numeric_disclaimer"]
    assert payload["decimal_precision"] == 50
    assert payload["budget_used"]["total"] > 0
    assert payload["budget_used"]["topological_steps"] > 0
    assert payload["budget_used"]["factor_steps"] > 0
    assert payload["budget_used"]["provenance_steps"] > 0
    assert payload["budget_used"]["source_states_scanned"] > 0
    assert len(payload["transitions"]) == payload["budget_used"]["transitions_written"]
    for transition in payload["transitions"]:
        assert transition["exact_numerator"] >= 1
        assert transition["exact_denominator"] >= transition["exact_numerator"]
        assert transition["origins"]
        assert all(origin["items"] for origin in transition["origins"])
        assert all(
            item_id in alignment.items
            for origin in transition["origins"]
            for item_id in origin["items"]
        )
        if transition["scored"]:
            assert transition["word"] is None
        for reading in transition["readings"]:
            assert ("p" in reading) is reading["scored"]
            assert ("n" in reading) is reading["scored"]


def test_att_is_four_column_acceptor_with_explicit_start():
    alignment = build_align_graph(
        choices("x", [(0, 1, ["one"], prior("1")), (0, 1, ["two"], prior("3"))])
    )
    text, _symbols, manifest = to_att_text(alignment)
    arc_lines = [line for line in text.splitlines() if "\t" in line]
    assert manifest.format == "AT&T finite-state acceptor text"
    assert manifest.start_state == 0
    assert int(arc_lines[0].split("\t", 1)[0]) == manifest.start_state
    assert all(len(line.split("\t")) == 4 for line in arc_lines)
    assert len(text.splitlines()[-1].split("\t")) == 1
    assert "negated Decimal natural logs" in manifest.weight_representation
    assert "product_tolerance" in manifest.best_product_guarantee


def _separator_carrier(size):
    return choices(
        "!" * size,
        [(index, index + 1, ["", f"word{index}"], prior(str(index + 1))) for index in range(size)],
    )


def test_null_closure_scales_polynomially():
    measurements = []
    for size in (16, 24):
        alignment = build_align_graph(_separator_carrier(size))
        started = perf_counter()
        _text, manifest = to_fsg_text(alignment, f"scale{size}")
        elapsed = perf_counter() - started
        assert elapsed < 2.0, f"n={size} export took {elapsed:.3f}s; bound is 2.0s"
        assert manifest.budget_used["total"] < 80 * size**3
        measurements.append((size, manifest.budget_used["total"], elapsed))
    assert measurements[1][1] <= 4 * measurements[0][1]
    print("scaling", measurements)


def _fsg_transition_count(text: str) -> int:
    return sum(line.startswith("TRANSITION ") for line in text.splitlines())


def test_real_fsg_reader(tmp_path):
    """Ask the installed reader to validate weighted syntax, language, and arc count."""
    pocketsphinx = pytest.importorskip("pocketsphinx")
    version = importlib.metadata.version("pocketsphinx")
    print(f"pocketsphinx {version}")
    carriers = [
        choices(
            "x",
            [
                (0, 1, ["one", "first"], prior("1")),
                (0, 1, ["two"], prior("2")),
                (0, 1, ["three"], prior("3")),
            ],
        ),
        choices("3 - 4", [(2, 3, ["minus", ""], prior("1"))]),
        choices("ab", [(0, 1, ["same first", "same second"], prior("2"))]),
    ]
    saw_subunit = False
    for index, carrier in enumerate(carriers):
        alignment = build_align_graph(carrier)
        source = tmp_path / f"source-{index}.fsg"
        round_trip = tmp_path / f"round-trip-{index}.fsg"
        text, manifest = to_fsg_text(alignment, f"reader{index}")
        saw_subunit = saw_subunit or any(
            Decimal(transition.probability) < 1 for transition in manifest.transitions
        )
        source.write_text(text, encoding="utf-8")
        model = pocketsphinx.FsgModel.readfile(str(source), pocketsphinx.LogMath(), 1.0)
        expected = graph_oracle(alignment, manifest.decimal_precision)
        assert model is not None
        for tokens in expected:
            assert model.accept(" ".join(tokens))
        for foreign in ["foreign", "one foreign", "three one"]:
            assert not model.accept(foreign)
        model.writefile(str(round_trip))
        assert _fsg_transition_count(round_trip.read_text(encoding="utf-8")) == len(
            manifest.transitions
        )
    assert saw_subunit
