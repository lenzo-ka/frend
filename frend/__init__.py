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
from frend.input_folds import InputFold, apply_input_fold
from frend.input_limits import (
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_UNIT_CHARS,
    MAX_NON_TEXT_SHARE,
    InputValidationError,
    validate_input,
    validate_unit_length,
)
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
from frend.normalize import NormalizedText, NormalizedUnit, normalize
from frend.runtime import freeze_after_setup
from frend.symbols import DEFAULT_SYMBOL_RUN_THRESHOLD
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
    "DEFAULT_MAX_INPUT_CHARS",
    "DEFAULT_MAX_UNIT_CHARS",
    "DEFAULT_SYMBOL_RUN_THRESHOLD",
    "InputValidationError",
    "InputFold",
    "LatticeNode",
    "MAX_NON_TEXT_SHARE",
    "NormalizedText",
    "NormalizedUnit",
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
    "apply_input_fold",
    "freeze_after_setup",
    "normalize",
    "resolve",
    "resolve_lattice",
    "resolve_choices",
    "route_geometry",
    "register_curated_alternative",
    "verbalize_edge",
    "verbalize_lattice",
    "validate_input",
    "validate_unit_length",
]
