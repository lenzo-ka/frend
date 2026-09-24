"""Import fallback for the composed projects when they are not installed.

frend composes tiergraph and icukit. When either is already importable this does
nothing. Otherwise, as a development convenience, fall back to a sibling checkout
laid out beside this one (``../tiergraph/src``, ``../icukit``) if present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SIBLINGS = {
    "tiergraph": _REPO.parent / "tiergraph" / "src",
    "icukit": _REPO.parent / "icukit",
}

for _module, _path in _SIBLINGS.items():
    if importlib.util.find_spec(_module) is None and _path.is_dir():
        sys.path.insert(0, str(_path))
