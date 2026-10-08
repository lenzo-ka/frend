"""Assertions for human-readable explanations in project metadata."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO / "pyproject.toml"


def test_cartlet_floor_comment_matches_dependency_specifier():
    text = _PYPROJECT.read_text(encoding="utf-8")
    metadata = tomllib.loads(text)
    cartlet = next(
        dependency
        for dependency in metadata["project"]["dependencies"]
        if dependency.startswith("cartlet")
    )

    dependency_floor = re.fullmatch(r"cartlet>=(\d+(?:\.\d+)*)", cartlet)
    project_preamble = text.partition("[project.urls]")[0]
    cartlet_comments = [
        line
        for line in project_preamble.splitlines()
        if line.startswith("#") and "cartlet" in line.casefold()
    ]

    assert dependency_floor is not None
    assert len(cartlet_comments) == 1
    comment_floor = re.fullmatch(r"# cartlet (\d+(?:\.\d+)*) is the floor: .+", cartlet_comments[0])
    assert comment_floor is not None
    assert comment_floor.group(1) == dependency_floor.group(1)
