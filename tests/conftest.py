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
from dataclasses import replace
from decimal import Decimal
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


def load_tool_module(name: str, path: Path | None = None, *, register: bool = False):
    """Load one repository script without repeating importlib scaffolding in tests."""
    path = path or _REPO / "tools" / f"{name}.py"
    tools = str(path.parent)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tool_loader():
    """Fixture form of :func:`load_tool_module` for tests needing an injected loader."""
    return load_tool_module


def alignment_choices(text: str, specs=()):
    """Build the small retained-alignment carrier shared by graph tests."""
    from frend.fold_resolve import CoverScore
    from frend.lattice import ChoiceGraph, ReadingEdge, ReadingRank, SemanticRank, resolve_choices
    from frend.verbalize import SpokenAlternative, VerbalizedUnit

    base = resolve_choices([], source_text=text)
    edges, units = [], []
    for index, (start, end, forms, measurement) in enumerate(specs):
        item_id = f"c{index}"
        geometry = CoverScore(end - start, 1, 0)
        semantic = SemanticRank("unsupported", None, None)
        edges.append(
            ReadingEdge(
                item_id,
                start,
                end,
                None,
                "reading",
                geometry,
                measurement,
                ReadingRank(end - start, 1, 0, semantic),
            )
        )
        units.append(
            VerbalizedUnit(
                item_id,
                tuple(
                    SpokenAlternative(form, f"source:{form_index}", Decimal(".7"))
                    for form_index, form in enumerate(forms)
                ),
                None,
                None,
                True,
            )
        )
    return ChoiceGraph(replace(base, edges=tuple(edges) + base.edges), tuple(units))


def alignment_paths(alignment, *, plan="log"):
    """Enumerate root-to-sink item-index paths from an alignment plan."""
    selected = alignment.log_plan() if plan == "log" else alignment.count_plan()

    def walk(index, path):
        path = (*path, index)
        if not selected.children[index]:
            yield path
        else:
            for child in selected.children[index]:
                yield from walk(child, path)

    return [path for root in selected.roots for path in walk(root, ())]


def assert_dependency_floor(text: str, dependency_name: str) -> None:
    """Assert that a dependency floor and its explanatory comment stay aligned."""
    import re
    import tomllib

    metadata = tomllib.loads(text)
    dependency = next(
        value for value in metadata["project"]["dependencies"] if value.startswith(dependency_name)
    )
    dependency_floor = re.fullmatch(rf"{re.escape(dependency_name)}>=(\d+(?:\.\d+)*)", dependency)
    preamble = text.partition("[project.urls]")[0]
    comments = [
        line
        for line in preamble.splitlines()
        if line.startswith("#") and dependency_name in line.casefold()
    ]
    assert dependency_floor is not None
    assert len(comments) == 1
    comment_floor = re.fullmatch(
        rf"# {re.escape(dependency_name)} (\d+(?:\.\d+)*) is the floor: .+", comments[0]
    )
    assert comment_floor is not None
    assert comment_floor.group(1) == dependency_floor.group(1)


def assert_count_map(
    value: object,
    *,
    names=None,
    positive: bool = True,
    nonempty: bool = False,
) -> None:
    """Assert the aggregate count-map contract shared by shipped-data schemas."""
    assert type(value) is dict
    if nonempty:
        assert value
    for name, count in value.items():
        assert type(name) is str and (names is None or name in names)
        assert type(count) is int and count >= int(positive)
