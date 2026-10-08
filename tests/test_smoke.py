"""Smoke tests for the installed package and its first-use documentation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from frend import __version__

_ROOT = Path(__file__).resolve().parents[1]


def test_version_is_a_nonempty_string():
    assert isinstance(__version__, str)
    assert __version__


def test_readme_usage_example_runs_without_external_profile_tables(tmp_path):
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    usage = readme.split("### Usage\n", 1)[1]
    example = usage.split("```python\n", 1)[1].split("```", 1)[0]
    env = os.environ.copy()
    env.pop("FREND_GOOGLE_TN_PROFILE_PATH", None)
    env.pop("FREND_GOOGLE_TN_BRITISHISMS_PATH", None)
    env["XDG_CACHE_HOME"] = str(tmp_path / "empty-cache")

    subprocess.run(
        [sys.executable, "-c", example],
        cwd=_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_readme_profile_example_configures_both_tables():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    usage = readme.split("### Usage\n", 1)[1]
    profile_example = usage.split("```sh\n", 1)[1].split("```", 1)[0]

    assert "FREND_GOOGLE_TN_PROFILE_PATH=acronym_surfaces.json" in profile_example
    assert "FREND_GOOGLE_TN_BRITISHISMS_PATH=britishisms.json" in profile_example
