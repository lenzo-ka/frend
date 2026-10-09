"""Ordered capture-field tails for detected normalization candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tiergraph import (
    Graph,
    ItemRef,
    OrderedPolyadicTraversal,
    PolyadicRelationDeclaration,
    PolyadicRelationInstance,
    PolyadicSide,
    QualifiedName,
    RelationEndpointKind,
    RelationSideDeclaration,
    XsdType,
)
from tiergraph.build import document, item

__all__ = ["Field", "FieldTails", "build_field_tails", "ordered_fields"]

NS = "https://ogion.org/frend/field-tails"
_CANDIDATES = QualifiedName(NS, "candidates")
_CANDIDATE = QualifiedName(NS, "candidate")
_FIELDS = QualifiedName(NS, "fields")
_FIELD = QualifiedName(NS, "field")
_CANDIDATE_MEMBERSHIP = QualifiedName(NS, "candidate-membership")
_FIELD_MEMBERSHIP = QualifiedName(NS, "field-membership")
_FIELD_TAIL = QualifiedName(NS, "field-tail")

_TYPE = QualifiedName(NS, "type")
_NAME = QualifiedName(NS, "name")
_VALUE = QualifiedName(NS, "value")
_FORM = QualifiedName(NS, "form")
_START = QualifiedName(NS, "start")
_END = QualifiedName(NS, "end")


@dataclass(frozen=True, slots=True)
class Field:
    """One capture's name and string-valued semantic value."""

    name: str
    value: str


@dataclass(frozen=True, slots=True)
class FieldTails:
    """One graph of candidate-to-capture tails and its candidate references."""

    graph: Graph
    candidates: tuple[ItemRef, ...]


def build_field_tails(detections: Sequence[Mapping[str, Any]]) -> FieldTails:
    """Build ordered polyadic capture tails for ``detections`` in one graph."""
    graph_document = document(NS, prefix="frend")
    graph_document.attributes(
        {
            _TYPE: XsdType.STRING,
            _NAME: XsdType.STRING,
            _VALUE: XsdType.STRING,
            _FORM: XsdType.STRING,
            _START: XsdType.INTEGER,
            _END: XsdType.INTEGER,
        }
    )
    candidate_items = []
    field_items = []
    tails: list[tuple[int, tuple[int, ...], str]] = []

    for detection_index, detection in enumerate(detections):
        detection_type = str(detection["type"])
        candidate_id = f"{detection_type}-{detection_index}"
        candidate_items.append(
            item(
                candidate_id,
                attrs={
                    _TYPE: detection_type,
                    _START: str(detection["start"]),
                    _END: str(detection["end"]),
                },
            )
        )

        targets: list[int] = []
        for capture_index, capture in enumerate(detection.get("captures", ())):
            targets.append(len(field_items))
            field_items.append(
                item(
                    f"{candidate_id}-field-{capture_index}",
                    attrs={
                        _NAME: str(capture.name),
                        _VALUE: str(capture.value),
                        _FORM: str(capture.form),
                        _START: str(capture.start),
                        _END: str(capture.end),
                    },
                )
            )
        tails.append((detection_index, tuple(targets), f"{candidate_id}-field-tail"))

    candidates = graph_document.tier(
        _CANDIDATES,
        candidate_items,
        item_type=_CANDIDATE,
        membership=_CANDIDATE_MEMBERSHIP,
        long_name="Detection candidates",
    )
    fields = graph_document.tier(
        _FIELDS,
        field_items,
        item_type=_FIELD,
        membership=_FIELD_MEMBERSHIP,
        long_name="Detection capture fields",
    )
    graph_document.declare(
        PolyadicRelationDeclaration(
            _FIELD_TAIL,
            sources=RelationSideDeclaration(
                (RelationEndpointKind.ITEM,), tiers=(_CANDIDATES,), maximum=1
            ),
            targets=RelationSideDeclaration(
                (RelationEndpointKind.ITEM,), tiers=(_FIELDS,), allow_empty=True
            ),
            unique_sources=True,
            distinct_targets=True,
        )
    )
    for source, targets, durable_id in tails:
        graph_document.relate(
            PolyadicRelationInstance(
                _FIELD_TAIL,
                (candidates.ref(source),),
                tuple(fields.ref(target) for target in targets),
                durable_id=durable_id,
            )
        )

    return FieldTails(
        graph_document.build(),
        tuple(candidates.ref(index) for index in range(len(candidate_items))),
    )


def ordered_fields(tails: FieldTails, candidate: ItemRef) -> tuple[Field, ...]:
    """Read one candidate's fields through the ordered polyadic traversal."""
    sequence = OrderedPolyadicTraversal(
        tails.graph,
        _FIELD_TAIL,
        PolyadicSide.SOURCES,
        PolyadicSide.TARGETS,
    ).direct(candidate)
    fields: list[Field] = []
    for node in sequence.nodes:
        reference = node.reference
        if not isinstance(reference, ItemRef):
            raise TypeError("field-tail traversal returned a non-item endpoint")
        tier = next(
            candidate
            for candidate in tails.graph.tiers
            if candidate.declaration.name == reference.tier
        )
        item = tier.items[reference.index]
        attributes = {attribute.name: attribute.lexical for attribute in item.attributes}
        fields.append(Field(attributes[_NAME], attributes[_VALUE]))
    return tuple(fields)
