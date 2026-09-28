"""Import fallback for the composed projects when they are not installed.

frend composes tiergraph and icukit, and reads its context trees with cartlet. When
each is already importable this does nothing. Otherwise, as a development convenience,
fall back to a sibling checkout laid out beside this one (``../tiergraph/src``,
``../icukit``, ``../cartlet``) if present.

``no_context_trees`` reads with no context trees (frend's own order, the range
connector still offered), for a test whose subject is that order.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SIBLINGS = {
    "tiergraph": _REPO.parent / "tiergraph" / "src",
    "icukit": _REPO.parent / "icukit",
    "cartlet": _REPO.parent / "cartlet",
}

for _module, _path in _SIBLINGS.items():
    if importlib.util.find_spec(_module) is None and _path.is_dir():
        sys.path.insert(0, str(_path))


import pytest  # noqa: E402


@pytest.fixture
def no_context_trees(monkeypatch):
    """No locale has context trees while the test runs."""
    from frend import context

    monkeypatch.setattr(context, "_context_model", lambda locale: None)
