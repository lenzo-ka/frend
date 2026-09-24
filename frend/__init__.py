"""frend -- the front end.

frend composes a fold/hypergraph substrate (tiergraph), internationalization
recognition (icukit), and later phonetics (ipakit) into a text-normalization pipeline:
recognize formatted values in running text, resolve overlapping readings into a best
non-overlapping cover, and verbalize the result. It uses tiergraph as a substrate and
keeps icukit recognition-only; resolution and verbalization live here.
"""

from __future__ import annotations

__version__ = "0.1.0"

from frend.fold_resolve import Resolution, resolve
from frend.lattice import (
    ChoiceGraph,
    ChoiceLattice,
    LatticeNode,
    PriorSummary,
    ReadingEdge,
    ReadingLattice,
    ReadingPath,
    ReadingRank,
    SemanticRank,
    compose_choices,
    resolve_choices,
    resolve_lattice,
    route_geometry,
)
from frend.verbalize import (
    SpokenAlternative,
    VerbalizedLattice,
    VerbalizedPath,
    VerbalizedUnit,
    register_curated_alternative,
    verbalize_edge,
    verbalize_lattice,
)

__all__ = [
    "ChoiceGraph",
    "ChoiceLattice",
    "LatticeNode",
    "PriorSummary",
    "ReadingEdge",
    "ReadingLattice",
    "ReadingPath",
    "ReadingRank",
    "Resolution",
    "SemanticRank",
    "SpokenAlternative",
    "VerbalizedLattice",
    "VerbalizedPath",
    "VerbalizedUnit",
    "__version__",
    "compose_choices",
    "resolve",
    "resolve_lattice",
    "resolve_choices",
    "route_geometry",
    "register_curated_alternative",
    "verbalize_edge",
    "verbalize_lattice",
]
