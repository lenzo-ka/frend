"""Assertions for human-readable explanations in project metadata."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO / "pyproject.toml"
_CHANGELOG = _REPO / "CHANGELOG.md"


def test_unreleased_changelog_names_public_features_since_pr_57():
    text = _CHANGELOG.read_text(encoding="utf-8")
    unreleased = text.partition("## Unreleased")[2].partition("\n## ")[0]

    public_features = (
        "`normalize`",
        "external `google-tn` profile",
        "behavior schemas",
        "input limits",
        "telephone numbers",
        "ISBNs and grouped IDs",
        "fractional durations",
        "symbol runs",
        "electronic spans",
        "spell-out dictionary",
    )
    missing = [feature for feature in public_features if feature not in unreleased]

    assert not missing, f"Unreleased changelog is missing: {', '.join(missing)}"
    assert "every limit and the fold are configurable" not in unreleased
    assert (
        "The document and\n"
        "  graph-building unit limits and the fold are configurable through `normalize`"
    ) in unreleased
    assert "Standalone `ElectronicDetector` exposes separate URL, email, and input caps" in (
        unreleased
    )


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


def test_tiergraph_floor_comment_matches_dependency_specifier():
    text = _PYPROJECT.read_text(encoding="utf-8")
    metadata = tomllib.loads(text)
    tiergraph = next(
        dependency
        for dependency in metadata["project"]["dependencies"]
        if dependency.startswith("tiergraph")
    )

    dependency_floor = re.fullmatch(r"tiergraph>=(\d+(?:\.\d+)*)", tiergraph)
    project_preamble = text.partition("[project.urls]")[0]
    tiergraph_comments = [
        line
        for line in project_preamble.splitlines()
        if line.startswith("#") and "tiergraph" in line.casefold()
    ]

    assert dependency_floor is not None
    assert len(tiergraph_comments) == 1
    comment_floor = re.fullmatch(
        r"# tiergraph (\d+(?:\.\d+)*) is the floor: .+", tiergraph_comments[0]
    )
    assert comment_floor is not None
    assert comment_floor.group(1) == dependency_floor.group(1)
