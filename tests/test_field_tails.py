"""Detection captures retain their surface order in tiergraph field tails."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest
from icukit.recognize import FlexibleDateDetector, FlexibleNumberDetector
from tiergraph import (
    AttributeDeclaration,
    AttributeDomain,
    AttributeValue,
    Graph,
    Item,
    ItemRef,
    NamespaceDeclaration,
    PolyadicRelationDeclaration,
    PolyadicRelationInstance,
    QualifiedName,
    RelationEndpointKind,
    RelationSideDeclaration,
    SimpleRelationDeclaration,
    Tier,
    TierDeclaration,
    XsdType,
    dumps,
)

import frend.field_tails as field_tails
from frend.field_tails import build_field_tails, ordered_fields

NS = "https://ogion.org/frend/field-tails"


def _name(local: str) -> QualifiedName:
    return QualifiedName(NS, local)


def _value(local: str, value_type: XsdType, value: object) -> AttributeValue:
    return AttributeValue(_name(local), value_type, str(value))


def _direct_graph(detections) -> Graph:
    """Reproduce the pre-migration graph as an exact parity oracle."""
    candidate_items = []
    field_items = []
    relations = []
    for detection_index, detection in enumerate(detections):
        candidate_ref = ItemRef(_name("candidates"), detection_index)
        detection_type = str(detection["type"])
        candidate_id = f"{detection_type}-{detection_index}"
        candidate_items.append(
            Item(
                candidate_id,
                (
                    _value("type", XsdType.STRING, detection_type),
                    _value("start", XsdType.INTEGER, detection["start"]),
                    _value("end", XsdType.INTEGER, detection["end"]),
                ),
            )
        )
        targets = []
        for capture_index, capture in enumerate(detection.get("captures", ())):
            targets.append(ItemRef(_name("fields"), len(field_items)))
            field_items.append(
                Item(
                    f"{candidate_id}-field-{capture_index}",
                    (
                        _value("name", XsdType.STRING, capture.name),
                        _value("value", XsdType.STRING, capture.value),
                        _value("form", XsdType.STRING, capture.form),
                        _value("start", XsdType.INTEGER, capture.start),
                        _value("end", XsdType.INTEGER, capture.end),
                    ),
                )
            )
        relations.append(
            PolyadicRelationInstance(
                _name("field-tail"),
                (candidate_ref,),
                tuple(targets),
                durable_id=f"{candidate_id}-field-tail",
            )
        )
    candidates = _name("candidates")
    fields = _name("fields")
    return Graph(
        namespaces=(NamespaceDeclaration("frend", NS),),
        tiers=(
            Tier(TierDeclaration(candidates, "Detection candidates"), tuple(candidate_items)),
            Tier(TierDeclaration(fields, "Detection capture fields"), tuple(field_items)),
        ),
        relation_declarations=(
            SimpleRelationDeclaration(
                _name("candidate-membership"), candidates, _name("candidate")
            ),
            SimpleRelationDeclaration(_name("field-membership"), fields, _name("field")),
            PolyadicRelationDeclaration(
                _name("field-tail"),
                sources=RelationSideDeclaration(
                    (RelationEndpointKind.ITEM,), tiers=(candidates,), maximum=1
                ),
                targets=RelationSideDeclaration(
                    (RelationEndpointKind.ITEM,), tiers=(fields,), allow_empty=True
                ),
                unique_sources=True,
                distinct_targets=True,
            ),
        ),
        attribute_declarations=tuple(
            AttributeDeclaration(_name(local), AttributeDomain.ITEM, value_type)
            for local, value_type in (
                ("type", XsdType.STRING),
                ("name", XsdType.STRING),
                ("value", XsdType.STRING),
                ("form", XsdType.STRING),
                ("start", XsdType.INTEGER),
                ("end", XsdType.INTEGER),
            )
        ),
        polyadic_relations=tuple(relations),
    )


def _date(locale: str, text: str):
    return FlexibleDateDetector(locale).detect(text)[0]


@pytest.mark.parametrize(
    "detections",
    [
        [],
        [dict(_date("en_US", "1/3/2026"), captures=())],
        [_date("en_US", "1/3/2026")],
        [FlexibleNumberDetector("en_US").detect("-1,234.50")[0]],
        [
            _date("ja_JP", "2026/1/3"),
            FlexibleNumberDetector("en_US").detect("1,234.50")[0],
            _date("en_US", "1/3/2026"),
        ],
    ],
    ids=("no-detections", "empty-captures", "date", "number", "multiple"),
)
def test_builder_graph_has_exact_direct_construction_parity(detections):
    graph = build_field_tails(detections).graph
    expected = _direct_graph(detections)

    assert graph == expected
    assert dumps(graph) == dumps(expected)


def test_builder_preserves_stringifiable_integer_span_inputs():
    detection = {
        "type": "decimal-spans",
        "start": Decimal("1"),
        "end": Decimal("4"),
        "captures": (
            SimpleNamespace(
                name="field",
                value="value",
                form="surface",
                start=Decimal("2"),
                end=Decimal("3"),
            ),
        ),
    }

    graph = build_field_tails([detection]).graph
    expected = _direct_graph([detection])

    assert graph == expected
    assert dumps(graph) == dumps(expected)


def test_build_field_tails_uses_builder_without_direct_graph_construction(monkeypatch):
    calls = 0
    actual_document = field_tails.document

    def tracking_document(*args, **kwargs):
        nonlocal calls
        calls += 1
        return actual_document(*args, **kwargs)

    def reject_direct_graph_construction(*args, **kwargs):
        raise AssertionError("build_field_tails constructed Graph directly")

    monkeypatch.setattr(field_tails, "document", tracking_document)
    monkeypatch.setattr(field_tails, "Graph", reject_direct_graph_construction)
    build_field_tails([_date("en_US", "1/3/2026")])

    assert calls == 1


def test_month_first_date_reads_in_surface_order():
    detection = _date("en_US", "1/3/2026")
    tails = build_field_tails([detection])

    assert tuple(field.name for field in ordered_fields(tails, tails.candidates[0])) == (
        "M",
        "d",
        "y",
    )


def test_date_field_order_is_non_commutative():
    month_first = _date("en_US", "1/3/2026")
    year_first = _date("ja_JP", "2026/1/3")
    tails = build_field_tails([month_first, year_first])

    orders = [
        tuple(field.name for field in ordered_fields(tails, candidate))
        for candidate in tails.candidates
    ]
    assert orders == [("M", "d", "y"), ("y", "M", "d")]
    assert orders[0] != orders[1]


def test_number_fields_read_in_surface_order():
    detection = FlexibleNumberDetector("en_US").detect("1,234.50")[0]
    tails = build_field_tails([detection])

    assert tuple(field.name for field in ordered_fields(tails, tails.candidates[0])) == (
        "integer",
        "decimal-separator",
        "fraction",
    )


def test_field_values_round_trip_as_strings_through_traversal():
    detection = FlexibleNumberDetector("en_US").detect("-1,234.50")[0]
    tails = build_field_tails([detection])

    actual = tuple(field.value for field in ordered_fields(tails, tails.candidates[0]))
    expected = tuple(str(capture.value) for capture in detection["captures"])
    assert actual == expected


def test_ordered_fields_reads_the_graph_snapshot_not_the_live_input():
    # Mutating the input detection after the tail is built must not change the
    # read: ordered_fields resolves through the graph, not the source captures.
    detection = dict(_date("en_US", "1/3/2026"))
    tails = build_field_tails([detection])
    detection["captures"] = ()
    assert tuple(field.name for field in ordered_fields(tails, tails.candidates[0])) == (
        "M",
        "d",
        "y",
    )


def test_empty_captures_build_and_read_as_no_fields():
    detection = dict(_date("en_US", "1/3/2026"))
    detection["captures"] = ()
    tails = build_field_tails([detection])
    assert ordered_fields(tails, tails.candidates[0]) == ()


def test_multiple_detections_have_independent_ordered_tails():
    detections = [
        _date("ja_JP", "2026/1/3"),
        FlexibleNumberDetector("en_US").detect("1,234.50")[0],
        _date("en_US", "1/3/2026"),
    ]
    tails = build_field_tails(detections)

    assert [
        tuple(field.name for field in ordered_fields(tails, candidate))
        for candidate in tails.candidates
    ] == [
        ("y", "M", "d"),
        ("integer", "decimal-separator", "fraction"),
        ("M", "d", "y"),
    ]
