"""Smoke tests for the installed package and its first-use documentation."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from frend import __version__

_ROOT = Path(__file__).resolve().parents[1]


def test_version_is_a_nonempty_string():
    assert isinstance(__version__, str)
    assert __version__


def test_every_readme_python_example_runs_without_external_profile_tables(tmp_path):
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    examples = re.findall(r"^```python\n(.*?)^```$", readme, re.MULTILINE | re.DOTALL)
    assert len(examples) == 2
    env = os.environ.copy()
    env.pop("FREND_GOOGLE_TN_PROFILE_PATH", None)
    env.pop("FREND_GOOGLE_TN_BRITISHISMS_PATH", None)
    env["XDG_CACHE_HOME"] = str(tmp_path / "empty-cache")

    outputs = [
        subprocess.run(
            [sys.executable, "-c", example],
            cwd=_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for example in examples
    ]
    assert outputs[0] == (
        "' meet   me   at   five thirty .  it   costs   twelve dollars and fifty cents .'\n"
    )


def test_readme_profile_example_configures_both_tables():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Set `FREND_GOOGLE_TN_PROFILE_PATH`" in readme
    assert "`acronym_surfaces.json`" in readme
    assert "`FREND_GOOGLE_TN_BRITISHISMS_PATH`" in readme
    assert "`britishisms.json`" in readme
    assert "```sh" not in readme
