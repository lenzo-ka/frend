"""Assertions for human-readable explanations in project metadata."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_README = _REPO / "README.md"
_PACKAGE_INIT = _REPO / "frend" / "__init__.py"
_PYPROJECT = _REPO / "pyproject.toml"
_CHANGELOG = _REPO / "CHANGELOG.md"
_FREND = _REPO / "frend"
_SERVICE_LOCALES_TEST = _REPO / "tests" / "test_service_locales.py"


def _service_profile_bullets(readme: str) -> tuple[str, ...]:
    section = readme.partition("### Eleven-locale service profile")[2].partition("\n### ")[0]
    return tuple(re.findall(r"^- (.+)$", section, re.MULTILINE))


def test_readme_lists_checked_eleven_locale_service_profile():
    readme = _README.read_text(encoding="utf-8")
    documented = _service_profile_bullets(readme)

    service_test = ast.parse(_SERVICE_LOCALES_TEST.read_text(encoding="utf-8"))
    assignment = next(
        statement
        for statement in service_test.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "LOCALES" for target in statement.targets
        )
    )
    checked = ast.literal_eval(assignment.value)

    assert documented == tuple(f"`{locale}`" for locale in checked)


def test_readme_service_profile_parser_includes_unformatted_bullets():
    readme = _README.read_text(encoding="utf-8")
    with_unformatted_locale = readme.replace("- `en_US`", "- `en_US`\n- en_GB", 1)

    assert "en_GB" in _service_profile_bullets(with_unformatted_locale)


def test_description_names_only_current_pipeline_dependencies():
    metadata = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    readme = _README.read_text(encoding="utf-8")

    dependency_names = {
        re.split(r"[<>=!~]", dependency, maxsplit=1)[0] for dependency in metadata["dependencies"]
    }
    pipeline_components = set(re.findall(r"^- \*\*(\w+)\*\*", readme, re.MULTILINE))
    later_components = set(re.findall(r"^- \*\*(\w+)\*\* \(later\)", readme, re.MULTILINE))
    named_components = {
        component
        for component in pipeline_components
        if re.search(rf"\b{re.escape(component)}\b", metadata["description"])
    }

    assert named_components == (pipeline_components & dependency_names) - later_components


def test_project_metadata_names_phonetization_as_the_callers_step():
    readme = _README.read_text(encoding="utf-8")
    package_docstring = ast.get_docstring(
        ast.parse(_PACKAGE_INIT.read_text(encoding="utf-8"), filename=str(_PACKAGE_INIT))
    )

    assert "A caller can then connect those words to its own `phonetize (ipakit)` step." in readme
    assert "ipakit** (caller-owned)" in readme
    assert "outside frend; a caller may pass the returned words to ipakit" in readme
    assert package_docstring is not None
    assert "Callers may pass the resulting words" in package_docstring
    assert "later phonetics (ipakit) into" not in package_docstring


def test_frend_modules_do_not_import_ipakit():
    imported_by = []
    for path in sorted(_FREND.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = [
            name
            for node in ast.walk(tree)
            for name in (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
        ]
        if any(name == "ipakit" or name.startswith("ipakit.") for name in imports):
            imported_by.append(path.relative_to(_REPO).as_posix())

    assert imported_by == []


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
