"""Unselected reading choices and their composed spoken graph."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest
from icukit.detectors import Capture, NumberValue

import irn.fold_resolve as fold_resolve
import irn.lattice as lattice_module
from irn import ChoiceGraph, ChoiceLattice, compose_choices, resolve_choices, resolve_lattice
from irn.fold_resolve import CoverScore
from irn.lattice import LatticeNode, ReadingEdge, _content_identity, route_geometry
from irn.type_priors import CorpusPrior, ReadingPrior
from irn.verbalize import verbalize_edge


def _det(text, start, end, type_, *, value=None, captures=(), **extra):
    return {
        "text": text[start:end],
        "start": start,
        "end": end,
        "type": type_,
        "value": value,
        "captures": tuple(captures),
        **extra,
    }


def _fraction(denominator: str):
    return _det(
        "1/4",
        0,
        3,
        "number:fraction",
        value=NumberValue(Decimal("0.5")),
        captures=(
            Capture("numerator", 0, 1, "1", value=Decimal(1)),
            Capture("denominator", 2, 3, denominator, value=Decimal(denominator)),
        ),
    )


def _readings(lattice):
    return [edge for edge in lattice.edges if edge.kind == "reading"]


def test_choices_carry_every_reading_distinct_under_the_carrier_identity():
    text = "xxxx"
    detections = [
        _det(text, 0, 4, "date:Md", captures=(Capture("x", 0, 1, "x"),)),
        _det(text, 0, 4, "number:fraction"),
    ]
    assert [edge.detection["type"] for edge in _readings(resolve_choices(detections))] == [
        "date:Md",
        "number:fraction",
    ]


def test_choices_record_geometry_and_prior_on_every_reading_edge():
    choices = resolve_choices([_det("12", 0, 2, "number:cardinal")], source_text="12")
    (edge,) = _readings(choices)
    assert edge.geometry == CoverScore(2, 1, 0)
    assert edge.prior is not None
    assert edge.rank.semantic.tier == edge.prior.tier
    assert edge.rank.semantic.p == edge.prior.p


def test_choices_never_enumerate_covers(monkeypatch):
    monkeypatch.setattr(fold_resolve, "FoldDeclaration", lambda *args, **kwargs: pytest.fail())
    assert _readings(resolve_choices([_det("1", 0, 1, "number:cardinal")]))


def test_choices_stay_defined_where_enumeration_is_intractable(monkeypatch):
    monkeypatch.setattr(fold_resolve, "FoldDeclaration", lambda *args, **kwargs: pytest.fail())
    detections = [_det("x" * 40, i * 2, i * 2 + 1, f"x:{i}") for i in range(20)]
    assert len(_readings(resolve_choices(detections))) == 20


def test_choices_refuse_rather_than_truncating_at_the_reading_cap(monkeypatch):
    monkeypatch.setattr(lattice_module, "_snapshot", lambda value: pytest.fail())
    detections = [_det("ab", 0, 1, "a"), _det("ab", 1, 2, "b")]
    with pytest.raises(ValueError, match="reading bound.*prefix"):
        resolve_choices(detections, reading_cap=1)
    assert not hasattr(ChoiceLattice(0, (), ()), "truncated")


def test_choice_edges_equal_the_ranked_lattice_edges_where_the_identities_agree():
    detections = [
        _det("12", 0, 2, "number:cardinal"),
        _det("12", 0, 2, "date:y"),
    ]
    assert (
        resolve_choices(detections, source_text="12").edges
        == resolve_lattice(detections, source_text="12", output_cap=99).edges
    )


def test_choices_record_dropped_detections_rather_than_discarding_them():
    detections = [
        _det("xxxx", 0, 0, "zero"),
        _det("xxxx", 2, 1, "reverse"),
        _det("xxxx", -1, 1, "negative"),
        _det("xxxx", 1, 3, "valid"),
    ]
    choices = resolve_choices(detections)
    assert len(choices.dropped) == 3
    assert [edge.id for edge in _readings(choices)] == ["c3"]


@pytest.mark.parametrize("reading_cap", [0, -1, True, 1.0])
def test_choices_reject_a_non_positive_reading_cap(reading_cap):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_choices([], reading_cap=reading_cap)


def test_choices_reject_source_text_shorter_than_the_detection_extent():
    with pytest.raises(ValueError, match="shorter than detection extent"):
        resolve_choices([_det("xx", 0, 2, "x")], source_text="x")


def test_choices_snapshot_detections_against_caller_mutation():
    nested = {"parts": ["kept"]}
    bad = _det("xx", 0, 0, "bad", nested=nested)
    good = _det("xx", 0, 2, "good", nested=nested)
    choices = resolve_choices([bad, good])
    nested["parts"].append("changed")
    assert choices.dropped[0]["nested"]["parts"] == ("kept",)
    assert _readings(choices)[0].detection["nested"]["parts"] == ("kept",)


def test_choices_of_no_detections_are_the_bare_position_lattice():
    choices = resolve_choices([], source_text="abc")
    assert choices.text_length == 3
    assert len(choices.nodes) == 4
    assert all(edge.kind == "passthrough" for edge in choices.edges)


def test_choice_carrier_size_is_linear_in_text_and_readings():
    text = "x" * 40_000
    choices = resolve_choices([_det(text, 0, 1, "a"), _det(text, 2, 3, "b")], source_text=text)
    assert len(choices.edges) == 40_002
    assert len(choices.nodes) == 40_001


def test_route_geometry_refuses_anything_that_is_not_a_route():
    choices = resolve_choices([_det("xxxx", 0, 4, "a"), _det("xxxx", 0, 4, "b")])
    with pytest.raises(ValueError, match="overlap"):
        route_geometry(choices, ("c0", "c1"))
    with pytest.raises(ValueError, match="partial"):
        route_geometry(choices, ())
    with pytest.raises(ValueError, match="not in this lattice"):
        route_geometry(choices, ("missing",))
    with pytest.raises(ValueError, match="gap"):
        route_geometry(choices, ("skip1", "skip2", "skip3"))
    backward = ReadingEdge(
        "back", 2, 1, None, "passthrough", CoverScore(0, 0, 0), None, choices.edges[0].rank
    )
    hand_built = ChoiceLattice(4, tuple(LatticeNode(i) for i in range(5)), (backward,))
    with pytest.raises(ValueError, match="strictly forward"):
        route_geometry(hand_built, ("back",))
    assert route_geometry(choices, ("skip3", "skip2", "skip1", "skip0")) == CoverScore(0, 0, 0)
    with_reading = resolve_choices([_det("xxxx", 1, 3, "x")], source_text="xxxx")
    route = ("skip3", "c0", "skip0")
    assert route_geometry(with_reading, route) == route_geometry(
        with_reading, tuple(reversed(route))
    )
    assert route_geometry(with_reading, route) == CoverScore(2, 1, 0)

    ranked = resolve_lattice(
        [_det("xxxx", 0, 4, "a"), _det("xxxx", 1, 3, "b")],
        source_text="xxxx",
        output_cap=99,
    )
    carrier = ChoiceLattice(ranked.text_length, ranked.nodes, ranked.edges, source_text="xxxx")
    for path in ranked.paths:
        assert route_geometry(carrier, tuple(reversed(path.edge_ids))) == path.geometry


def test_readings_with_different_spoken_forms_are_never_collapsed():
    first, second = _fraction("2"), _fraction("4")
    forward = _readings(resolve_choices([first, second], source_text="1/4"))
    reverse = _readings(resolve_choices([second, first], source_text="1/4"))
    assert len(forward) == len(reverse) == 2
    forms = [
        {item.text for item in verbalize_edge(edge, source_text="1/4").alternatives}
        for edge in forward
    ]
    assert forms[0].isdisjoint(forms[1])
    assert {frozenset(forms_) for forms_ in forms} == {
        frozenset(item.text for item in verbalize_edge(edge, source_text="1/4").alternatives)
        for edge in reverse
    }


def test_total_content_identity_is_type_safe_stable_and_order_independent():
    identities = {
        _content_identity(_det("x", 0, 1, "x", payload=value))
        for value in (1, True, Decimal("1"), 1.0, "1")
    }
    assert len(identities) == 5
    left = _det("x", 0, 1, "x", payload={1: "a", "1": "b"})
    right = dict(reversed(tuple(left.items())))
    assert _content_identity(left) == _content_identity(right)
    assert hash(_content_identity(left))

    class SameRepresentation:
        def __repr__(self):
            return "same"

    first, second = SameRepresentation(), SameRepresentation()
    forward = _det("x", 0, 1, "x", payload={first: "a", second: "b"})
    reverse = _det("x", 0, 1, "x", payload={second: "b", first: "a"})
    assert forward["payload"] == reverse["payload"]
    assert _content_identity(forward) == _content_identity(reverse)


def test_the_composed_graph_carries_every_readings_spoken_forms():
    choices = resolve_choices(
        [_fraction("2"), _det("1/4", 0, 1, "number:cardinal", value=NumberValue("1"))],
        source_text="1/4",
    )
    graph = compose_choices(choices)
    assert isinstance(graph, ChoiceGraph)
    assert [unit.edge_id for unit in graph.units] == [edge.id for edge in choices.edges]
    assert [unit.alternatives for unit in graph.units] == [
        verbalize_edge(edge, source_text="1/4").alternatives for edge in choices.edges
    ]


def test_the_composed_graph_carries_no_selection_facts():
    graph = compose_choices(resolve_choices([], source_text="x"))
    for name in ("best_path", "truncated", "ambiguous", "rank"):
        assert not hasattr(graph, name)


def test_the_carrier_builds_edges_by_index_not_by_content_key():
    edges = _readings(resolve_choices([_fraction("2"), _fraction("4")]))
    assert [edge.id for edge in edges] == ["c0", "c1"]
    assert [edge.detection["captures"][1].value for edge in edges] == [Decimal(2), Decimal(4)]


def test_content_key_collisions_keep_each_readings_own_prior():
    class CapturePrior(CorpusPrior):
        def features(self, detection, context):
            del detection, context
            return ()

        def reading_prior(self, detection):
            denominator = detection["captures"][1].value
            return ReadingPrior(
                group="fraction",
                shape="N/N",
                p=Decimal(1) / denominator,
                n=1,
                supported=True,
                tier="measured",
                provenance="test:capture",
            )

    edges = _readings(
        resolve_choices([_fraction("2"), _fraction("4")], feature_sources=(CapturePrior(),))
    )
    assert [edge.prior.p for edge in edges] == [Decimal("0.5"), Decimal("0.25")]


def test_the_composer_refuses_a_carrier_with_no_source_text(monkeypatch):
    monkeypatch.setattr("irn.verbalize.verbalize_edge", lambda *args, **kwargs: pytest.fail())
    with pytest.raises(ValueError, match="source_text"):
        compose_choices(resolve_choices([]))


def test_the_composed_graph_refuses_rather_than_truncating_at_the_spoken_bound(monkeypatch):
    choices = resolve_choices([], source_text="x")
    alternative = object()
    unit = type("Unit", (), {"alternatives": (alternative,) * ((1 << 16) + 1)})()
    monkeypatch.setattr("irn.verbalize.verbalize_edge", lambda *args, **kwargs: unit)
    with pytest.raises(ValueError, match="spoken forms.*not readings.*prefix"):
        compose_choices(choices)
    assert not hasattr(ChoiceGraph(choices, ()), "truncated")

    two_edges = resolve_choices([], source_text="xy")
    exact = type("Unit", (), {"alternatives": (alternative,) * (1 << 15)})()
    monkeypatch.setattr("irn.verbalize.verbalize_edge", lambda *args, **kwargs: exact)
    assert len(compose_choices(two_edges).units) == 2

    calls = iter((1 << 15, (1 << 15) + 1))

    def spread(*args, **kwargs):
        del args, kwargs
        return type("Unit", (), {"alternatives": (alternative,) * next(calls)})()

    monkeypatch.setattr("irn.verbalize.verbalize_edge", spread)
    with pytest.raises(ValueError, match="spoken forms.*not readings.*prefix"):
        compose_choices(two_edges)


def test_choice_types_are_frozen():
    choices = resolve_choices([], source_text="x")
    graph = compose_choices(choices)
    with pytest.raises(FrozenInstanceError):
        choices.text_length = 2
    with pytest.raises(FrozenInstanceError):
        graph.units = ()
