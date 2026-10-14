"""Assertions for human-readable explanations in project metadata."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

from conftest import assert_dependency_floor

from frend.generated_rule_locales import GENERATED_RULE_CAPABILITIES

_REPO = Path(__file__).resolve().parents[1]
_README = _REPO / "README.md"
_PACKAGE_INIT = _REPO / "frend" / "__init__.py"
_PYPROJECT = _REPO / "pyproject.toml"
_CHANGELOG = _REPO / "CHANGELOG.md"
_FREND = _REPO / "frend"
_SERVICE_LOCALES_TEST = _REPO / "tests" / "test_service_locales.py"
_API_REFERENCE = _REPO / "docs" / "api.md"


def _service_profile_rows(readme: str) -> tuple[tuple[str, str], ...]:
    section = readme.partition("### Checked locale service profile")[2].partition("\n### ")[0]
    return tuple(re.findall(r"^\| `([^`]+)` \| (.+) \|$", section, re.MULTILINE))


def test_readme_lists_checked_locale_service_profile():
    readme = _README.read_text(encoding="utf-8")
    documented = _service_profile_rows(readme)

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

    assert tuple(locale for locale, _ in documented) == checked
    families = ("date", "time", "fraction", "money", "electronic", "telephone")
    for locale, coverage in documented:
        if locale == "en_US":
            assert "English measured/default tables" in coverage
            assert "measured telephone route" in coverage
        else:
            assert {family for family in families if family in coverage} == set(
                GENERATED_RULE_CAPABILITIES[locale]
            )


def test_readme_service_profile_parser_includes_an_added_matrix_row():
    readme = _README.read_text(encoding="utf-8")
    with_extra_locale = readme.replace("| `en_US` |", "| `en_GB` | best effort |\n| `en_US` |", 1)

    assert ("en_GB", "best effort") in _service_profile_rows(with_extra_locale)


def test_description_names_only_current_pipeline_dependencies():
    metadata = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    readme = _README.read_text(encoding="utf-8")

    dependency_names = {
        re.split(r"\s*@\s*|[<>=!~]", dependency, maxsplit=1)[0]
        for dependency in metadata["dependencies"]
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


def test_readme_distinguishes_source_boundary_whitespace_from_spoken_spacing():
    readme = " ".join(_README.read_text(encoding="utf-8").split())

    assert "Leading and trailing source whitespace is discarded in both return forms" in readme
    assert "inter-sentence source whitespace contributes one space" in readme
    assert (
        "Reader emissions can add spaces of their own, including at the edges of the returned "
        "text and next to inter-sentence spacing" in readme
    )
    assert "offset units cover discarded boundary-whitespace spans explicitly" in readme
    assert "leading and trailing whitespace is trimmed" not in readme


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


def test_010_changelog_names_public_features_since_pr_57():
    text = _CHANGELOG.read_text(encoding="utf-8")
    release = text.partition("## 0.1.0 (2026-10-14)")[2].partition("\n## ")[0]

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
    missing = [feature for feature in public_features if feature not in release]

    assert not missing, f"0.1.0 changelog is missing: {', '.join(missing)}"
    assert "every limit and the fold are configurable" not in release
    assert (
        "The document and\n"
        "  graph-building unit limits and the fold are configurable through `normalize`"
    ) in release
    assert "Standalone `ElectronicDetector` exposes separate URL, email, and input caps" in (
        release
    )


def test_pep639_license_metadata_and_notice_files_are_packaged():
    configuration = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    project = configuration["project"]

    assert configuration["build-system"]["requires"][0] == "setuptools>=77.0"
    assert project["license"] == (
        "BSD-2-Clause AND CC-BY-SA-4.0 AND Unicode-3.0 AND Apache-2.0 AND "
        "LicenseRef-Festival AND LicenseRef-IANA-public"
    )
    assert project["license-files"] == [
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/*.txt",
    ]
    assert (_REPO / "LICENSE").is_file()
    assert (_REPO / "THIRD_PARTY_NOTICES.md").is_file()
    assert {path.name for path in (_REPO / "LICENSES").glob("*.txt")} == {
        "Apache-2.0.txt",
        "CC-BY-SA-4.0.txt",
        "LicenseRef-Festival.txt",
        "LicenseRef-IANA-public.txt",
        "Unicode-3.0.txt",
    }


def test_public_api_reference_covers_exactly_the_package_exports():
    package = ast.parse(_PACKAGE_INIT.read_text(encoding="utf-8"))
    assignment = next(
        statement
        for statement in package.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in statement.targets
        )
    )
    exports = ast.literal_eval(assignment.value)
    documented = re.findall(
        r"^- `([^`]+)` — ", _API_REFERENCE.read_text(encoding="utf-8"), re.MULTILINE
    )

    assert documented == exports


def test_exported_aliases_and_constants_have_attribute_docstrings():
    definitions = {
        "__version__": _PACKAGE_INIT,
        "InputFold": _FREND / "input_folds.py",
        "SentenceBase": _FREND / "normalize.py",
        "DEFAULT_MAX_INPUT_CHARS": _FREND / "input_limits.py",
        "DEFAULT_MAX_UNIT_CHARS": _FREND / "input_limits.py",
        "MAX_NON_TEXT_SHARE": _FREND / "input_limits.py",
        "DEFAULT_SYMBOL_RUN_THRESHOLD": _FREND / "symbols.py",
        "DEFAULT_TIERGRAPH_WORK_BUDGET": _FREND / "work_budgets.py",
    }

    for name, path in definitions.items():
        statements = ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body
        index = next(
            position
            for position, statement in enumerate(statements)
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
            and (
                any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in statement.targets
                )
                if isinstance(statement, ast.Assign)
                else isinstance(statement.target, ast.Name) and statement.target.id == name
            )
        )
        docstring = statements[index + 1]
        assert isinstance(docstring, ast.Expr), name
        assert isinstance(docstring.value, ast.Constant), name
        assert isinstance(docstring.value.value, str) and docstring.value.value, name


def test_cartlet_floor_comment_matches_dependency_specifier():
    text = _PYPROJECT.read_text(encoding="utf-8")
    assert_dependency_floor(text, "cartlet")


def test_tiergraph_floor_comment_matches_dependency_specifier():
    text = _PYPROJECT.read_text(encoding="utf-8")
    assert_dependency_floor(text, "tiergraph")


def test_tiergraph_floor_includes_shared_path_and_plan_apis():
    metadata = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]

    assert "tiergraph>=0.9.0" in metadata["dependencies"]
