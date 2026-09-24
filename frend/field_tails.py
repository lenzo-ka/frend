"""Ordered capture-field tails for detected normalization candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tiergraph import (
    AttributeDeclaration,
    AttributeDomain,
    AttributeValue,
    Graph,
    Item,
    ItemRef,
    NamespaceDeclaration,
    OrderedPolyadicTraversal,
    PolyadicRelationDeclaration,
    PolyadicRelationInstance,
    PolyadicSide,
    QualifiedName,
    RelationEndpointKind,
    RelationSideDeclaration,
    SimpleRelationDeclaration,
    Tier,
    TierDeclaration,
    XsdType,
)

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


def _attribute(name: QualifiedName, value_type: XsdType, value: object) -> AttributeValue:
    return AttributeValue(name, value_type, str(value))


def build_field_tails(detections: Sequence[Mapping[str, Any]]) -> FieldTails:
    """Build ordered polyadic capture tails for ``detections`` in one graph."""
    candidate_items: list[Item] = []
    field_items: list[Item] = []
    candidate_refs: list[ItemRef] = []
    relations: list[PolyadicRelationInstance] = []

    for detection_index, detection in enumerate(detections):
        candidate_ref = ItemRef(_CANDIDATES, detection_index)
        candidate_refs.append(candidate_ref)
        detection_type = str(detection["type"])
        candidate_id = f"{detection_type}-{detection_index}"
        candidate_items.append(
            Item(
                candidate_id,
                (
                    _attribute(_TYPE, XsdType.STRING, detection_type),
                    _attribute(_START, XsdType.INTEGER, detection["start"]),
                    _attribute(_END, XsdType.INTEGER, detection["end"]),
                ),
            )
        )

        targets: list[ItemRef] = []
        for capture_index, capture in enumerate(detection.get("captures", ())):
            field_ref = ItemRef(_FIELDS, len(field_items))
            targets.append(field_ref)
            field_items.append(
                Item(
                    f"{candidate_id}-field-{capture_index}",
                    (
                        _attribute(_NAME, XsdType.STRING, capture.name),
                        _attribute(_VALUE, XsdType.STRING, capture.value),
                        _attribute(_FORM, XsdType.STRING, capture.form),
                        _attribute(_START, XsdType.INTEGER, capture.start),
                        _attribute(_END, XsdType.INTEGER, capture.end),
                    ),
                )
            )
        relations.append(
            PolyadicRelationInstance(
                _FIELD_TAIL,
                (candidate_ref,),
                tuple(targets),
                durable_id=f"{candidate_id}-field-tail",
            )
        )

    graph = Graph(
        namespaces=(NamespaceDeclaration("frend", NS),),
        tiers=(
            Tier(TierDeclaration(_CANDIDATES, "Detection candidates"), tuple(candidate_items)),
            Tier(TierDeclaration(_FIELDS, "Detection capture fields"), tuple(field_items)),
        ),
        relation_declarations=(
            SimpleRelationDeclaration(_CANDIDATE_MEMBERSHIP, _CANDIDATES, _CANDIDATE),
            SimpleRelationDeclaration(_FIELD_MEMBERSHIP, _FIELDS, _FIELD),
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
            ),
        ),
        attribute_declarations=(
            AttributeDeclaration(_TYPE, AttributeDomain.ITEM, XsdType.STRING),
            AttributeDeclaration(_NAME, AttributeDomain.ITEM, XsdType.STRING),
            AttributeDeclaration(_VALUE, AttributeDomain.ITEM, XsdType.STRING),
            AttributeDeclaration(_FORM, AttributeDomain.ITEM, XsdType.STRING),
            AttributeDeclaration(_START, AttributeDomain.ITEM, XsdType.INTEGER),
            AttributeDeclaration(_END, AttributeDomain.ITEM, XsdType.INTEGER),
        ),
        polyadic_relations=tuple(relations),
    )
    return FieldTails(graph, tuple(candidate_refs))


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
