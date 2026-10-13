"""Consumer paths for positions and ranked readings in a frend resolution."""

from __future__ import annotations

from collections.abc import Sequence

from tiergraph import Graph, ItemRef, QualifiedName, parse_path_index_segment
from tiergraph.path import (
    AlternativeRef,
    CanonicalPath,
    ItemBinding,
    PathBinding,
    PathOffender,
    PathRefusal,
    PathRefusalCode,
)

from frend.fold_resolve import _POS, NS, Detection, _dedupe, build_lattice, resolve

__all__ = ["FrendReadingProfile"]

_READINGS = QualifiedName(NS, "readings")


class FrendReadingProfile:
    """Address one resolution snapshot's lattice offsets and ranked readings.

    The vocabulary is ``/offset/N`` for the structural position-tier item at
    offset ``N`` and ``/reading/K`` for the K-th descending ranked cover.
    """

    def __init__(
        self,
        detections: Sequence[Detection],
        *,
        locale: str = "en_US",
        n: int = 8,
    ) -> None:
        """Build one lattice snapshot and its ranked resolution readings."""
        unique = _dedupe(detections)
        self.graph, _roots, _id_to_index = build_lattice(unique)
        self.covers = resolve(unique, locale=locale, n=n).covers

    def _require_snapshot(self, graph: Graph, path: CanonicalPath | None = None) -> None:
        if graph is not self.graph:
            raise PathRefusal(
                PathRefusalCode.PROFILE_REFUSED,
                PathOffender(
                    text="" if path is None else str(path),
                    path=path,
                    profile_reason="different_resolution_snapshot",
                ),
            )

    def bind(self, path: CanonicalPath, graph: Graph) -> PathBinding:
        """Bind an offset item or ranked-reading alternative."""
        self._require_snapshot(graph, path)
        segments = path.segments
        if len(segments) != 2 or segments[0] not in {"offset", "reading"}:
            raise PathRefusal(
                PathRefusalCode.UNKNOWN_FORM,
                PathOffender(text=str(path), path=path),
            )
        index = parse_path_index_segment(segments[1], path=path, segment_index=1)
        if segments[0] == "offset":
            return ItemBinding(ItemRef(_POS, index))
        return AlternativeRef(ItemRef(_POS, 0), _READINGS, index)

    def spell(self, binding: PathBinding, graph: Graph) -> CanonicalPath:
        """Spell a supported binding in the frend reading vocabulary."""
        if graph is not self.graph:
            raise PathRefusal(
                PathRefusalCode.UNSPELLABLE,
                PathOffender(text="", profile_reason="different_resolution_snapshot"),
            )
        if isinstance(binding, ItemBinding):
            reference = binding.reference
            if not isinstance(reference, ItemRef) or reference.tier != _POS:
                raise PathRefusal(
                    PathRefusalCode.UNSPELLABLE,
                    PathOffender(text="", profile_reason="unsupported_item"),
                )
            self.graph.resolve_item(reference)
            return CanonicalPath(("offset", str(reference.index)))
        if isinstance(binding, AlternativeRef):
            if binding.relation != _READINGS:
                reason = "unsupported_relation"
            elif binding.owner != ItemRef(_POS, 0):
                reason = "unsupported_owner"
            elif type(binding.index) is not int or not 0 <= binding.index < len(self.covers):
                reason = "unsupported_index"
            else:
                return CanonicalPath(("reading", str(binding.index)))
            raise PathRefusal(
                PathRefusalCode.UNSPELLABLE,
                PathOffender(text="", profile_reason=reason),
            )
        raise PathRefusal(
            PathRefusalCode.UNSPELLABLE,
            PathOffender(text="", profile_reason="unsupported_binding"),
        )

    def alternatives(
        self, owner: ItemRef, relation: QualifiedName, graph: Graph
    ) -> tuple[object, ...]:
        """Return covers in this resolution snapshot's descending rank order."""
        self._require_snapshot(graph)
        if relation != _READINGS:
            raise PathRefusal(
                PathRefusalCode.PROFILE_REFUSED,
                PathOffender(text="", relation=relation, profile_reason="unsupported_relation"),
            )
        if owner != ItemRef(_POS, 0):
            raise PathRefusal(
                PathRefusalCode.PROFILE_REFUSED,
                PathOffender(text="", tier=owner.tier, profile_reason="unsupported_owner"),
            )
        return self.covers
