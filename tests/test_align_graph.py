"""Alignment falsifiers; mutation targets are listed beside each check."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
from tiergraph import (
    AttributeValuation,
    ChildCombination,
    FoldDeclaration,
    FoldTransition,
    dumps,
    loads,
)
from tiergraph.pathplan import PathPlan
from tiergraph.semiring import COUNTING

import irn.align_graph as module
from irn.align_graph import arc_weight, build_align_graph
from irn.fold_resolve import CoverScore, resolve, resolve_cover
from irn.lattice import (
    ChoiceGraph,
    ReadingEdge,
    ReadingRank,
    SemanticRank,
    compose_choices,
    resolve_choices,
    resolve_lattice,
    route_geometry,
)
from irn.spoken_priors import normalize_spoken, spoken_tokens
from irn.type_priors import BlendedPrior, CorpusPrior, IcuBackfillTable, PriorTable, ReadingPrior
from irn.verbalize import SpokenAlternative, VerbalizedUnit


def prior(p=".2", *, tier="measured", supported=True, n=100, generated_p=None, group="a"):
    return ReadingPrior(
        group,
        "A",
        Decimal(p) if p is not None else None,
        n,
        supported,
        tier,
        generated_p=generated_p,
    )


def backfill_fixture():
    blend = BlendedPrior(
        PriorTable({"N": {"date": 3}}, {}),
        IcuBackfillTable(
            {"date": {"A": 1}, "fraction": {"A": 3}}, {"date": 10, "fraction": 10}, {}
        ),
    )
    detections = [
        {"text": "x", "start": 0, "end": 1, "type": t, "value": None, "captures": ()}
        for t in ["date", "fraction"]
    ]
    alone = resolve_choices(detections[:1], feature_sources=[blend], source_text="x")
    together = resolve_choices(detections, feature_sources=[blend], source_text="x")
    return alone.edges[0].prior, together.edges[0].prior


def _expect(statement, call, *args):
    """Run a call whose success on valid input is the stated expectation.

    A refusal here fails the test with the statement, so a kill is carried by what
    the test says should happen rather than by a production crash inside it.
    """
    try:
        return call(*args)
    except (ValueError, TypeError) as error:
        pytest.fail(f"{statement}, but it raised {error!r}")


def choices(text, specs=()):
    """Specs are (start, end, forms, prior); unit skips come from the real carrier."""
    base = resolve_choices([], source_text=text)
    edges, units = [], []
    for index, (start, end, forms, measurement) in enumerate(specs):
        id = f"c{index}"
        geometry = CoverScore(end - start, 1, 0)
        semantic = SemanticRank("unsupported", None, None)
        edges.append(
            ReadingEdge(
                id,
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
                id,
                tuple(
                    SpokenAlternative(form, f"source:{i}", Decimal(".7"))
                    for i, form in enumerate(forms)
                ),
                None,
                None,
                True,
            )
        )
    return ChoiceGraph(replace(base, edges=tuple(edges) + base.edges), tuple(units))


def paths(alignment):
    plan = alignment.log_plan()

    def walk(index, path):
        path = (*path, index)
        if not plan.children[index]:
            yield path
        else:
            for child in plan.children[index]:
                yield from walk(child, path)

    return list(walk(plan.roots[0], ()))


def path_key(alignment, path):
    labels = alignment.log_plan().labels
    return (
        tuple(labels[i] for i in path if alignment.items[labels[i]].role == "reading"),
        tuple(token for i in path for token in alignment.items[labels[i]].tokens),
    )


def carrier_oracle(carrier):
    """Enumerate actual code-point tilings, render buffered skips with reading boundaries."""
    text = carrier.lattice.source_text
    units = {unit.edge_id: unit for unit in carrier.units}

    def walk(position, ids, reading_ids, rendered, skipped):
        if position == len(text):
            route_geometry(carrier.lattice, ids)
            yield reading_ids, spoken_tokens(rendered + skipped)
            return
        for edge in carrier.lattice.edges:
            if edge.start != position:
                continue
            if edge.kind == "passthrough":
                yield from walk(
                    edge.end,
                    (*ids, edge.id),
                    reading_ids,
                    rendered,
                    skipped + text[edge.start : edge.end],
                )
            else:
                forms = {spoken_tokens(form.text) for form in units[edge.id].alternatives}
                for tokens in forms:
                    yield from walk(
                        edge.end,
                        (*ids, edge.id),
                        (*reading_ids, edge.id),
                        rendered + skipped + " " + " ".join(tokens) + " ",
                        "",
                    )

    return Counter(walk(0, (), (), "", ""))


def test_routes_match_brute_force_enumeration():
    """Kill missing pre, dropped pairs/junctions, cross-reading sharing, duplicate links."""
    fixtures = [
        choices(""),
        choices("ΟΣ'Α"),
        choices("-5 5th\u00a0x"),
        choices("3 - 4", [(2, 3, ["minus", ""], None)]),
        choices(
            "abcdef",
            [
                (1, 3, ["X", "x!", "x y", ""], None),
                (2, 4, ["x", "x z"], None),
                (3, 5, ["q"], None),
                (0, 6, ["all"], None),
            ],
        ),
    ]
    rng = random.Random(718)
    for _ in range(35):
        text = rng.choice(["abcd", "a--b", "a b", "ΟΣ'Α", "-5th"])
        specs = []
        for _ in range(4):
            a = rng.randrange(len(text))
            b = rng.randrange(a + 1, len(text) + 1)
            specs.append((a, b, rng.sample(["x", "x y", "x z", "", "!", "Z"], 3), None))
        fixtures.append(choices(text, specs))
    for carrier in fixtures:
        alignment = _expect("every fixture carrier builds", build_align_graph, carrier)
        expected = carrier_oracle(carrier)
        assert Counter(path_key(alignment, path) for path in paths(alignment)) == expected
        assert alignment.count_plan().evaluate().value == sum(expected.values())


def test_posteriors_match_enumeration():
    """Kill doubled prior, token/junction factors, and incorrect value-order gathering."""
    rng = random.Random(191)
    for _ in range(30):
        text = rng.choice(["abc", "a--b", "a b", "-5th"])
        spans = []
        for _span in range(2):
            start = rng.randrange(len(text))
            spans.append((start, rng.randrange(start + 1, len(text) + 1)))
        specs = [
            (*spans[0], rng.sample(["a b", "a c", "", "a"], 3), prior(".2")),
            (*spans[0], ["d"], prior(".8")),
            (*spans[1], rng.sample(["e f", "e", ""], 2), prior(".3")),
            (*spans[1], ["g"], prior(".9")),
        ]
        alignment = build_align_graph(choices(text, specs))
        plan = alignment.log_plan()
        acoustics = tuple(
            rng.uniform(-4, 0) if alignment.items[label].tokens else 0 for label in plan.labels
        )
        values = tuple(v + a for v, a in zip(plan.values, acoustics, strict=True))
        expected_priors = {}
        for index, (start, end, _forms, measurement) in enumerate(specs):
            maximum = max(m.p for s, e, _f, m in specs if (s, e) == (start, end))
            expected_priors[f"c{index}"] = math.log(float(measurement.p / maximum))
        oracle_values = tuple(
            expected_priors.get(label, 0.0) + a
            for label, a in zip(plan.labels, acoustics, strict=True)
        )
        all_paths = paths(alignment)
        products = [math.exp(math.fsum(oracle_values[i] for i in path)) for path in all_paths]
        total = math.fsum(products)
        expected = [
            math.fsum(p for path, p in zip(all_paths, products, strict=True) if i in path) / total
            for i in range(len(values))
        ]
        result = plan.marginals(values).posteriors(readout="normalize")
        assert result.values == pytest.approx(expected, rel=1e-9, abs=1e-12)
        assert plan.marginals(values).total == pytest.approx(math.log(total), rel=1e-9)


_WEIGHT_RULES = [
    (prior(), True, "measured"),
    (prior(None, supported=False), False, "attested-zero"),
    (prior("0", supported=False), False, "attested-zero"),
    (prior(None, n=None, supported=False), False, "sparse-shape"),
    (backfill_fixture()[1], False, "icu-backfill"),
    (prior(None, tier="unsupported", supported=False), False, "unsupported"),
    (prior(".2", supported=False), False, "unsupported"),
    (None, False, "no-corpus-source"),
]


def _weight_rule_carrier(measurement):
    return choices("x", [(0, 1, ["one two", ""], measurement), (0, 1, ["other"], prior(".8"))])


@pytest.mark.parametrize("measurement,scored,reason", _WEIGHT_RULES)
def test_arc_weight_rule_table_directly(measurement, scored, reason):
    """Assert each weight rule from arc_weight itself, before any graph is built."""
    readings = [
        edge for edge in _weight_rule_carrier(measurement).lattice.edges if edge.kind == "reading"
    ]
    weight = arc_weight(readings[0], readings)
    assert (weight.scored, weight.reason) == (scored, reason)
    assert weight.log_weight == (float(Decimal(".25").ln()) if scored else 0.0)


def test_arc_weight_survives_an_underflowing_ratio_directly():
    """A ratio below float range still weighs exactly, computed in Decimal."""
    readings = [
        edge
        for edge in choices(
            "x", [(0, 1, ["a"], prior("1e-400")), (0, 1, ["b"], prior(".8"))]
        ).lattice.edges
        if edge.kind == "reading"
    ]
    weight = _expect(
        "a representable ratio weighs without raising", arc_weight, readings[0], readings
    )
    assert weight.log_weight == float((Decimal("1e-400") / Decimal(".8")).ln())


@pytest.mark.parametrize("measurement,scored,reason", _WEIGHT_RULES)
def test_weight_rule_table(measurement, scored, reason):
    """Kill backfill p/generated_p, zero/sparse/unsupported scores and spoken shares."""
    alignment = _expect(
        "every weight rule builds a graph", build_align_graph, _weight_rule_carrier(measurement)
    )
    weight = alignment.items["c0"].weight
    assert (weight.scored, weight.reason) == (scored, reason)
    assert weight.log_weight == (float(Decimal(".25").ln()) if scored else 0.0)
    for item in alignment.items.values():
        if item.role != "reading":
            assert not item.weight.scored and item.weight.log_weight == 0.0
            assert item.weight.reason == {
                "token": "passthrough",
                "word": "spoken-form",
                "separator": "separator",
            }.get(item.role, "structure")


def test_overlapping_groups():
    """Kill group-local maxima: corpus cardinal 60 / decimal 20 / other 20."""
    table = PriorTable({"N": {"cardinal": 60, "decimal": 20, "other": 20}}, {})
    ps = [
        table.reading_prior({"text": "3", "type": type_})
        for type_ in ("number:cardinal", "number:decimal")
    ]
    assert [p.p for p in ps] == [Decimal(".6"), Decimal(".8")]
    alignment = build_align_graph(
        choices("3", [(0, 1, ["three"], ps[0]), (0, 1, ["three"], ps[1])])
    )
    assert [math.exp(alignment.items[f"c{i}"].weight.log_weight) for i in range(2)] == [0.75, 1]


def test_best_scored_ties_passthrough():
    """Kill absolute log p; best measured reading has exactly the unit factor."""
    alignment = build_align_graph(choices("x", [(0, 1, ["spoken"], prior(".2"))]))
    assert alignment.items["c0"].weight.scored
    assert alignment.items["c0"].weight.log_weight == 0.0
    plan = alignment.log_plan()
    posterior = plan.marginals().posteriors(readout="normalize")
    assert posterior.values[plan.labels.index("c0")] == 0.5
    assert posterior.values[plan.labels.index("tok0_1")] == 0.5


def test_factor_is_on_reading_item():
    """Kill moving an exclusive reading factor to form exits (numerically equivalent)."""
    alignment = build_align_graph(
        choices("x", [(0, 1, ["a b", "a c", ""], prior(".2")), (0, 1, ["d"], prior(".8"))])
    )
    assert alignment.items["c0"].weight.log_weight < 0
    for item in alignment.items.values():
        if item.role != "reading":
            assert item.weight.log_weight == 0
    for graph_item in loads(dumps(alignment.graph)).tiers[0].items:
        attrs = {a.name: a.lexical for a in graph_item.attributes}
        item = alignment.items[graph_item.durable_id]
        assert float(attrs[module._LOG_WEIGHT]) == item.weight.log_weight
        assert attrs[module._SCORED] == ("true" if item.weight.scored else "false")
        assert attrs[module._REASON] == item.weight.reason
        assert module._PRIOR_KIND in attrs
        assert (module._P in attrs) == item.weight.scored
        if item.weight.scored:
            assert Decimal(attrs[module._P]) == item.weight.p
            assert int(attrs[module._N]) == item.weight.n


def test_carrier_log_survives_underflowing_p():
    """Kill math.log(float(ratio)); use a much larger scored span-mate."""
    alignment = build_align_graph(
        choices("x", [(0, 1, ["a"], prior("1e-400")), (0, 1, ["b"], prior(".8"))])
    )
    assert alignment.items["c0"].weight.log_weight == float(
        (Decimal("1e-400") / Decimal(".8")).ln()
    )
    assert math.isfinite(alignment.log_plan().marginals().total)


def test_backfill_mate_does_not_change_measured_weights():
    """Kill including generated span-mates in measured maximum."""
    alone, together = backfill_fixture()
    assert alone.p == 1 and together.p == Decimal(".25")
    assert alone.generated_p == together.generated_p == Decimal(".1")
    carrier = choices(
        "x",
        [
            (0, 1, ["a"], prior(".2")),
            (0, 1, ["b"], prior(".4")),
            (0, 1, ["c"], prior(".9", tier="icu-backfill")),
        ],
    )
    a = build_align_graph(carrier)
    b = build_align_graph(
        choices(
            "x",
            [
                (0, 1, ["a"], prior(".2")),
                (0, 1, ["b"], prior(".4")),
                (0, 1, ["c"], prior(".99", tier="icu-backfill")),
                (0, 1, ["d"], prior(".01", tier="icu-backfill")),
            ],
        )
    )
    assert a.items["c0"].weight == b.items["c0"].weight
    assert a.items["c0"].weight.log_weight == float(Decimal(".5").ln())


def test_item_cap_refuses_whole(monkeypatch):
    """Kill cap truncation and constructing Graph before refusal."""
    called = []
    real = module.Graph

    def graph(*args, **kwargs):
        called.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "Graph", graph)
    carrier = choices("abc", [(0, 3, ["one two three"], None)])
    with pytest.raises(ValueError, match="graph-item"):
        build_align_graph(carrier, item_cap=3)
    assert not called
    for bad in [0, True, -1, 1.5]:
        with pytest.raises(ValueError, match="positive integer"):
            build_align_graph(carrier, item_cap=bad)


def test_missing_source_and_dead_end_refuse_before_graph(monkeypatch):
    """Kill source fabrication and allowing a childless non-sink reading."""
    called = []
    real = module.Graph

    def graph(*args, **kwargs):
        called.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "Graph", graph)
    carrier = choices("x", [(0, 1, [], None)])
    with pytest.raises(ValueError, match="non-sink"):
        build_align_graph(carrier)
    with pytest.raises(ValueError, match="source_text"):
        build_align_graph(replace(carrier, lattice=replace(carrier.lattice, source_text=None)))
    assert not called


def test_plans_accept_graph_and_only_final_sink():
    """Kill sink links and AND encoding; count double valuation must be refused."""
    alignment = _expect(
        "a carrier with a skip builds",
        build_align_graph,
        choices("a--b", [(2, 3, ["dash", ""], None)]),
    )
    log_plan = _expect("the log plan prepares over the graph", alignment.log_plan)
    count_plan = _expect("the count plan prepares over the graph", alignment.count_plan)
    for plan in [log_plan, count_plan]:
        assert PathPlan.prepare(plan.declaration).children == plan.children
        assert plan is (
            alignment.log_plan() if plan is alignment.log_plan() else alignment.count_plan()
        )
        assert len(plan.roots) == 1 and plan.labels[plan.roots[0]] == "B0"
        assert [plan.labels[i] for i, c in enumerate(plan.children) if not c] == ["B4"]
        assert plan.declaration.transitions[0].combination is ChildCombination.OR
    assert alignment.kept_positions == frozenset([0, 2, 3, 4])
    with pytest.raises(ValueError, match="double"):
        FoldDeclaration(
            "bad",
            alignment.graph,
            AttributeValuation("bad", module._LOG_WEIGHT, (module._TIER,)),
            COUNTING,
            lambda v, _label: int(v),
            (FoldTransition(module._NEXT, ChildCombination.OR),),
            roots=(alignment.root,),
        )


def test_all_negative_infinity_has_zero_mass():
    """Kill stale values or fabrication of a uniform zero-mass distribution."""
    alignment = build_align_graph(choices("x", [(0, 1, ["a"], prior())]))
    plan = alignment.log_plan()
    result = plan.marginals([-math.inf] * len(plan.items)).posteriors(readout="normalize")
    assert result.zero_mass and result.values is None
    assert not plan.marginals().posteriors(readout="normalize").zero_mass


def test_lexicon_filter_trims_and_records():
    """Kill omission of co-reachability trim; retained-route oracle still agrees."""
    carrier = choices("x", [(0, 1, ["unknown", "known"], None)])
    alignment = build_align_graph(carrier, pronounceable=lambda t: t != "unknown")
    assert Counter(path_key(alignment, p) for p in paths(alignment)) == Counter(
        [((), ("x",)), (("c0",), ("known",))]
    )
    assert alignment.unalignable[0][1].text == "unknown"
    with pytest.raises(ValueError, match="no complete"):
        build_align_graph(carrier, pronounceable=lambda _t: False)


def test_normalization_contract_generated():
    """Kill join-contract drift, whitespace/punctuation drift and whole-string lowering."""

    def oracle(text):
        tokens = []
        token = []
        for character in text:
            if character.isspace() or character in "'-!":
                if token:
                    tokens.append("".join(token).lower())
                    token = []
            else:
                token.append(character)
        if token:
            tokens.append("".join(token).lower())
        return tuple(tokens)

    rng = random.Random(40)
    alphabet = "AZΟΣ'Α\u00a0\t\n\x1cİ-!$+中文"
    for _ in range(500):
        text = "".join(rng.choices(alphabet, k=rng.randrange(50)))
        expected = oracle(text)
        assert spoken_tokens(text) == expected
        assert normalize_spoken(text) == " ".join(expected)
    assert spoken_tokens("ΟΣ'Α") == ("ος", "α")


def test_one_best_has_no_alignment_import():
    """Kill an import linking the exact resolver to the alignment module."""
    root = Path(__file__).resolve().parents[1]
    for file in ["fold_resolve.py", "lattice.py"]:
        tree = ast.parse((root / "irn" / file).read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not any("align" in name for name in imports)
        assert not any(
            alias.name.startswith("align")
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        )


def test_one_best_untouched():
    """Kill resolver behavior drift against a baseline-generated byte golden."""
    from icukit.detectors import detect
    from icukit.recognize import (
        FlexibleCurrencyDetector,
        FlexibleDateDetector,
        FlexibleFractionDetector,
        FlexibleNumberDetector,
        LetterNameDetector,
        SingleLetterWordDetector,
    )

    detectors = [
        FlexibleDateDetector("en_US"),
        FlexibleNumberDetector("en_US"),
        LetterNameDetector("en_US"),
        SingleLetterWordDetector("en_US"),
        FlexibleCurrencyDetector("en_US", "USD"),
        FlexibleFractionDetector("en_US"),
    ]
    data = Path(__file__).parent / "data"
    provenance = json.loads((data / "align_one_best.provenance.json").read_text())
    assert provenance == {
        "baseline_commit": "7c6ef7b5e77adf9016655ee54c9fadce0ee62710",
        "command": (
            "cd $IRN_BASELINE_CHECKOUT && PYTHONPATH=. python -B "
            "$IRN_ALIGN_WORKTREE/tests/generate_align_one_best.py "
            "$IRN_ALIGN_WORKTREE/tests/data/align_one_best.json"
        ),
    }
    rows = json.loads((data / "align_one_best.json").read_text())
    for row in rows:
        text = row["text"]
        ds = detect(text, detectors)
        outputs = [
            repr(resolve(ds, source_text=text)),
            repr(resolve_lattice(ds, source_text=text)),
            repr(resolve_cover(ds, source_text=text)),
        ]
        assert [hashlib.sha256(value.encode()).hexdigest() for value in outputs] == row["outputs"]


def test_exact_ranking_below_double_resolution():
    """Kill rounding the resolver's Decimal prior axis to float."""

    class Source(CorpusPrior):
        def reading_prior(self, detection):
            return prior(
                ".500000000000000000000000000002"
                if detection["type"] == "number:z"
                else ".500000000000000000000000000001"
            )

        def features(self, detection, context):
            return ()

    ds = [
        {"start": 0, "end": 1, "text": "3", "type": t, "value": None, "captures": ()}
        for t in ["number:a", "number:z"]
    ]
    with localcontext() as context:
        context.prec = 60
        assert (
            resolve_cover(ds, feature_sources=[Source()], source_text="3").best[0]["type"]
            == "number:z"
        )
        lattice = resolve_choices(ds, feature_sources=[Source()], source_text="3")
        weights = [
            arc_weight(e, lattice.edges).log_weight for e in lattice.edges if e.kind == "reading"
        ]
        assert weights[0] != weights[1]
        # Two lower competitors differ below double resolution of the log ratio.
        mate = replace(lattice.edges[0], id="mate", prior=prior(".9"))
        weights = [
            arc_weight(e, (*lattice.edges, mate)).log_weight
            for e in lattice.edges
            if e.kind == "reading"
        ]
        assert weights[0] == weights[1]


def test_upstream_caps_still_refuse(monkeypatch):
    """Kill silent prefixes in either upstream completeness bound."""
    import irn.lattice as lattice_module

    ds = [
        {"start": 0, "end": 1, "text": "3", "type": t, "value": None, "captures": ()}
        for t in ["number:a", "number:z"]
    ]
    with pytest.raises(ValueError, match="reading bound"):
        resolve_choices(ds, source_text="3", reading_cap=1)
    monkeypatch.setattr(lattice_module, "_COMPOSED_SPOKEN_CAP", 1)
    with pytest.raises(ValueError, match="spoken forms bound"):
        compose_choices(resolve_choices([], source_text="abc"))


def test_partial_date_alternatives_are_never_cut(monkeypatch):
    """Kill month-first slicing in the early-return branch for a year-only date."""
    from icukit.detectors import all_detectors

    import irn.verbalize as verbalize

    real_leaf = verbalize._number_leaf
    extra = tuple(SpokenAlternative(f"synthetic year {i}", "test:year") for i in range(12))

    def widened(value, kind, locale):
        forms = real_leaf(value, kind, locale)
        return (*forms, *extra) if kind == "year" else forms

    monkeypatch.setattr(verbalize, "_number_leaf", widened)
    detection = next(
        d for d in all_detectors("en_US", ("y",)).detect("2024") if d["type"] == "date:y"
    )
    unit = compose_choices(resolve_choices([detection], source_text="2024")).units[0]
    assert {a.text for a in extra} <= {a.text for a in unit.alternatives}
