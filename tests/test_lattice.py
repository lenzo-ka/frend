"""Public, distilled reading-lattice behavior."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from inspect import getdoc

import pytest
from icukit.detectors import Capture, NumberValue
from icukit.recognize import (
    FlexibleNumberDetector,
    LetterNameDetector,
    SingleLetterWordDetector,
)

import frend
from frend import ReadingLattice, resolve_lattice
from frend.fold_resolve import CoverScore, _content_key, resolve
from frend.type_priors import BlendedPrior, IcuBackfillTable, PriorTable
from frend.verbalize import verbalize_edge


def _det(text: str, start: int, end: int, type_: str, captures=()) -> dict:
    return {
        "text": text[start:end],
        "start": start,
        "end": end,
        "type": type_,
        "value": None,
        "captures": tuple(captures),
    }


def _isolated_letter_detections(text: str) -> list[dict]:
    detectors = (
        LetterNameDetector("en_US"),
        SingleLetterWordDetector("en_US"),
        FlexibleNumberDetector("en_US"),
    )
    return [detection for detector in detectors for detection in detector.detect(text)]


def test_single_letter_word_wins_same_geometry_competition():
    resolution = resolve(_isolated_letter_detections("A"))

    assert resolution.best[0]["type"] == "word:single-letter"
    assert resolution.semantic_ambiguous is False
    assert [reading.detection["type"] for reading in resolution.spans[0].readings] == [
        "word:single-letter",
        "letter:name",
    ]


def test_non_roman_letter_carries_its_measured_prior():
    resolution = resolve(_isolated_letter_detections("B"))

    assert [reading.detection["type"] for reading in resolution.spans[0].readings] == [
        "letter:name"
    ]
    prior = resolution.spans[0].readings[0].prior
    assert prior is not None
    assert prior.supported is True
    assert prior.group == "letter"
    assert prior.shape == "<Lu>"


def test_roman_letter_competition_is_decided_by_the_corpus_prior():
    """An isolated Roman letter is resolved by evidence, not by capture count.

    This replaces a test that characterized the opposite. The letter readings
    once carried no capture while the Roman cardinal carried one, so geometry
    filtered them out of the span before any prior was read and the cardinal
    won unopposed. icukit now gives the letter readings a capture, so all
    readings of the span reach the prior and the measured base rate decides.
    A regression to the old behavior shows up here as the cardinal winning
    again, or as the competitors vanishing from the span.
    """
    detections = _isolated_letter_detections("V")
    resolution = resolve(detections)
    lattice = resolve_lattice(detections, source_text="V")
    priors = {
        edge.detection["type"]: edge.prior
        for edge in lattice.edges
        if edge.kind == "reading" and edge.detection is not None
    }

    assert priors["letter:name"] is not None
    assert priors["number:cardinal:roman"] is not None
    assert priors["letter:name"].p > priors["number:cardinal:roman"].p

    # Both readings survive the span's geometry filter, which is what the
    # capture buys, and the winner is the one the corpus attests more often.
    span_readings = [reading.detection["type"] for reading in resolution.spans[0].readings]
    assert span_readings == ["letter:name", "number:cardinal:roman"]
    assert resolution.best[0]["type"] == "letter:name"

    # The competitors tie on capture count; nothing is decided by structure.
    captures = {
        detection["type"]: len(detection["captures"])
        for detection in detections
        if detection["type"] in {"letter:name", "number:cardinal:roman"}
    }
    assert captures["letter:name"] == captures["number:cardinal:roman"]


def test_nodes_reading_edges_and_resolution_consistency():
    text = "on 12 now"
    number = _det(text, 3, 5, "number:cardinal")

    lattice = resolve_lattice([number], source_text=text)
    resolution = resolve([number])

    assert isinstance(lattice, ReadingLattice)
    assert tuple(node.position for node in lattice.nodes) == tuple(range(len(text) + 1))
    (edge,) = [edge for edge in lattice.edges if edge.kind == "reading"]
    assert edge.detection == number
    assert edge.detection is not number
    assert edge.geometry == CoverScore(coverage=2, span_count=1, capture_count=0)
    assert edge.rank.coverage == 2
    assert edge.rank.semantic.p == edge.prior.p
    assert lattice.best_path.readings == resolution.best == (number,)
    assert lattice.best_path.geometry == CoverScore(2, 1, 0)


def test_reading_edge_prior_is_the_existing_per_span_prior():
    text = "3/24"
    date = _det(text, 0, 4, "date:Md")
    fraction = _det(text, 0, 4, "number:fraction")

    lattice = resolve_lattice([date, fraction], source_text=text, output_cap=2)
    resolution = resolve([date, fraction], n=2)
    span_priors = {
        reading.detection["type"]: reading.prior for reading in resolution.spans[0].readings
    }

    for edge in lattice.edges:
        if edge.detection is not None:
            assert edge.prior == span_priors[edge.detection["type"]]


def test_passthrough_edges_and_path_reconstruct_uncovered_text():
    text = "ab12z"
    number = _det(text, 2, 4, "number:cardinal")
    lattice = resolve_lattice([number], source_text=text)

    skips = [edge for edge in lattice.edges if edge.kind == "passthrough"]
    assert [(edge.start, edge.end) for edge in skips] == [
        (position, position + 1) for position in range(len(text))
    ]
    assert lattice.best_path.edge_ids == ("skip0", "skip1", "c0", "skip4")
    assert all(edge.detection is None for edge in skips)
    assert all(edge.geometry == CoverScore(0, 0, 0) for edge in skips)


def test_ranked_projection_is_capped_and_reports_truncation():
    text = "xxxx"
    detections = [_det(text, 0, 4, "date:Md"), _det(text, 0, 4, "number:fraction")]

    one = resolve_lattice(detections, source_text=text)
    two = resolve_lattice(detections, source_text=text, output_cap=2)
    all_three = resolve_lattice(detections, source_text=text, output_cap=3)

    assert len(one.paths) == 1
    assert one.truncated is True
    assert len(two.paths) == 2
    assert two.truncated is True
    assert len(all_three.paths) == 3
    assert all_three.truncated is False
    assert [path.rank for path in two.paths] == [0, 1]
    assert one.edges == two.edges == all_three.edges
    contract = getdoc(resolve_lattice)
    assert contract is not None
    assert "cap bounds only the paths exposed" in contract
    assert "complete top-geometry equivalence class" in contract
    assert "can grow exponentially" in contract


def test_detection_snapshot_is_independent_of_nested_input_mutation():
    text = "x7"
    capture = {"name": "integer", "parts": ["7"]}
    detection = _det(text, 1, 2, "number:cardinal", captures=[capture])
    lattice = resolve_lattice([detection], source_text=text)

    detection["type"] = "MUTATED"
    detection["captures"][0]["name"] = "MUTATED"
    capture["parts"].append("MUTATED")

    edge_detection = next(edge.detection for edge in lattice.edges if edge.detection is not None)
    path_detection = lattice.best_path.readings[0]
    assert edge_detection["type"] == "number:cardinal"
    assert edge_detection["captures"][0]["name"] == "integer"
    assert edge_detection["captures"][0]["parts"] == ("7",)
    assert path_detection is edge_detection


def test_capture_value_is_recursively_frozen_through_dataclass_fields():
    text = "x7"
    value = {
        "nested": [
            Decimal("1.25"),
            {"x": ["kept"]},
        ]
    }
    capture = Capture("integer", 1, 2, "7", value=value)
    detection = _det(text, 1, 2, "number:cardinal", captures=[capture])
    lattice = resolve_lattice([detection], source_text=text)

    # Mutating the caller-owned payload cannot alter the lattice snapshot.
    value["nested"][0] = Decimal("888")
    value["nested"][1]["x"].append("CALLER-MUTATED")

    stored = lattice.best_path.readings[0]["captures"][0]
    assert isinstance(stored, Capture)
    assert stored.value["nested"][0] == Decimal("1.25")
    assert stored.value["nested"][1]["x"][0] == "kept"

    with pytest.raises(TypeError):
        stored.value["nested"][0] = Decimal("999")
    with pytest.raises((AttributeError, TypeError)):
        stored.value["nested"][1]["x"].append("MUTATED")

    assert stored.value["nested"][0] == Decimal("1.25")
    assert stored.value["nested"][1]["x"] == ("kept",)


def test_lattice_is_deterministic_and_dataclasses_are_frozen():
    text = "x7"
    detections = [_det(text, 1, 2, "number:cardinal")]
    first = resolve_lattice(detections, source_text=text)
    second = resolve_lattice(detections, source_text=text)

    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.text_length = 99
    with pytest.raises(FrozenInstanceError):
        first.edges[0].start = 99
    with pytest.raises(FrozenInstanceError):
        first.paths[0].rank = 99


def test_non_english_passthrough_smoke():
    text = "前12後"
    number = _det(text, 1, 3, "number:cardinal")
    lattice = resolve_lattice([number], source_text=text)

    assert lattice.text_length == 4
    assert lattice.best_path.edge_ids == ("skip0", "c0", "skip3")
    assert lattice.best_path.readings == (number,)


@pytest.mark.parametrize(
    ("detections", "expected"),
    [
        (
            [
                _det("xxx", 0, 1, "A"),
                _det("xxx", 1, 3, "B"),
                _det("xxx", 0, 2, "C"),
                _det("xxx", 2, 3, "D"),
            ],
            (True, False, True),
        ),
        ([_det("xxx", 0, 3, "A"), _det("xxx", 0, 3, "B")], (False, True, True)),
        ([_det("xxx", 0, 3, "A")], (False, False, False)),
    ],
)
def test_lattice_ambiguity_matches_resolution(detections, expected):
    lattice = resolve_lattice(detections, source_text="xxx")
    resolution = resolve(detections)

    actual = (
        lattice.structural_ambiguous,
        lattice.semantic_ambiguous,
        lattice.ambiguous,
    )
    compatible = (
        resolution.structural_ambiguous,
        resolution.semantic_ambiguous,
        resolution.ambiguous,
    )
    assert actual == compatible == expected
    assert lattice.ambiguous == (lattice.structural_ambiguous or lattice.semantic_ambiguous)


def test_resolve_lattice_contract_names_the_choice_carrier():
    assert "future phase" not in getdoc(resolve_lattice)
    assert "resolve_choices" in getdoc(resolve_lattice)
    assert "future phase" not in getdoc(ReadingLattice)


def test_the_public_surface_names_the_carrier_and_the_composed_graph():
    names = {"ChoiceLattice", "ChoiceGraph", "resolve_choices", "compose_choices", "route_geometry"}
    exported = names & set(frend.__all__)
    bound = {name for name in names if hasattr(frend, name)}
    assert exported == bound == names


def test_path_geometry_is_the_sum_of_its_edge_geometries_over_a_tiling_traversal():
    def assert_tiling(edges, text_length):
        assert all(edge.end > edge.start for edge in edges)
        covered = [position for edge in edges for position in range(edge.start, edge.end)]
        assert len(covered) == len(set(covered))
        assert sorted(covered) == list(range(text_length))

    for detections, text in [
        ([], ""),
        ([_det("1/2", 0, 3, "number:fraction", (Capture("n", 0, 1, "1"),))], "1/2"),
    ]:
        lattice = resolve_lattice(detections, source_text=text, output_cap=99)
        by_id = {edge.id: edge for edge in lattice.edges}
        for path in lattice.paths:
            edges = [by_id[edge_id] for edge_id in path.edge_ids]
            assert len({edge.id for edge in edges}) == len(edges)
            assert_tiling(edges, lattice.text_length)
            assert path.geometry == CoverScore(
                sum(edge.geometry.coverage for edge in edges),
                sum(edge.geometry.span_count for edge in edges),
                sum(edge.geometry.capture_count for edge in edges),
            )

    lattice = resolve_lattice([_det("abc", 0, 2, "reading")], source_text="abc")
    by_id = {edge.id: edge for edge in lattice.edges}
    with pytest.raises(AssertionError):
        assert_tiling([by_id["c0"], by_id["skip1"], by_id["skip2"]], 3)
    with pytest.raises(AssertionError):
        assert_tiling([by_id["c0"]], 3)


def test_ranked_covers_span_more_than_the_top_geometry_level():
    text = "1/3/2026"
    detections = [
        _det(text, 0, 8, "date:flexible", (Capture("date", 0, 8, text),)),
        _det(text, 0, 3, "number:fraction"),
        _det(text, 4, 8, "number:cardinal"),
    ]
    lattice = resolve_lattice(detections, source_text="1/3/2026", output_cap=1000)
    assert (
        len(
            {
                (-path.geometry.coverage, path.geometry.span_count, -path.geometry.capture_count)
                for path in lattice.paths
            }
        )
        > 1
    )
    assert any(path.geometry.coverage == 7 for path in lattice.paths)


def test_icu_backfill_edge_prior_and_tier_are_conditional_on_span_mates():
    blend = BlendedPrior(
        PriorTable({"N": {"date": 3}}, {"source": "test"}),
        IcuBackfillTable(
            {"date": {"A": 1}, "fraction": {"A": 3}},
            {"date": 10, "fraction": 10},
            {"status": "test"},
        ),
    )
    lone = _det("x", 0, 1, "date")
    mate = _det("x", 0, 1, "fraction")
    alone = next(
        edge
        for edge in resolve_lattice([lone], source_text="x", feature_sources=(blend,)).edges
        if edge.kind == "reading"
    )
    together = {
        edge.detection["type"]: edge
        for edge in resolve_lattice(
            [lone, mate], source_text="x", output_cap=9, feature_sources=(blend,)
        ).edges
        if edge.kind == "reading"
    }
    assert alone.prior.p == Decimal(1)
    assert together["date"].prior.p == Decimal("0.25")
    assert all(edge.prior.tier == "icu-backfill" for edge in together.values())

    zeroed = BlendedPrior(
        blend.measured,
        blend.backfill,
        class_prior={"date": Decimal(0), "fraction": Decimal(1)},
        class_prior_source="test",
    )
    alone_zero = next(
        edge
        for edge in resolve_lattice([lone], source_text="x", feature_sources=(zeroed,)).edges
        if edge.kind == "reading"
    )
    together_zero = {
        edge.detection["type"]: edge
        for edge in resolve_lattice(
            [lone, mate], source_text="x", output_cap=9, feature_sources=(zeroed,)
        ).edges
        if edge.kind == "reading"
    }["date"]
    assert (alone_zero.prior.tier, alone_zero.prior.supported, alone_zero.prior.p) == (
        "unsupported",
        False,
        None,
    )
    assert (alone_zero.rank.semantic.tier, alone_zero.rank.semantic.p) == ("unsupported", None)
    assert (together_zero.prior.tier, together_zero.prior.supported, together_zero.prior.p) == (
        "icu-backfill",
        True,
        Decimal(0),
    )
    assert (together_zero.rank.semantic.tier, together_zero.rank.semantic.p) == (
        "icu-backfill",
        Decimal(0),
    )


def test_the_resolver_content_key_does_not_separate_differently_spoken_readings():
    def fraction(denominator):
        return _det(
            "1/4",
            0,
            3,
            "number:fraction",
            (
                Capture("numerator", 0, 1, "1", value=Decimal(1)),
                Capture("denominator", 2, 3, denominator, value=Decimal(denominator)),
            ),
        ) | {"value": NumberValue(Decimal("0.5"))}

    first, second = fraction("2"), fraction("4")
    assert _content_key(first) == _content_key(second)
    first_edge = next(
        edge for edge in resolve_lattice([first], source_text="1/4").edges if edge.kind == "reading"
    )
    second_edge = next(
        edge
        for edge in resolve_lattice([second], source_text="1/4").edges
        if edge.kind == "reading"
    )
    assert {
        item.text for item in verbalize_edge(first_edge, source_text="1/4").alternatives
    }.isdisjoint(item.text for item in verbalize_edge(second_edge, source_text="1/4").alternatives)
