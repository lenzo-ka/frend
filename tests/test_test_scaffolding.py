"""Regression checks for the shared test and composition-benchmark scaffolding."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import compose_benchmark

_REPO = Path(os.environ.get("FREND_SCAFFOLD_ROOT", Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("name", ("date", "time", "fraction"))
def test_composition_benchmark_scripts_support_direct_invocation(name, tmp_path):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", str(_REPO / "tools" / f"bench_{name}_compose.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("usage:")


def test_dynamic_loader_and_composition_harness_are_single_sourced(tmp_path, monkeypatch):
    loader_call = "spec_from" + "_file_location"
    loader_files = sorted(
        path.relative_to(_REPO).as_posix()
        for path in (_REPO / "tests").glob("*.py")
        if loader_call in path.read_text(encoding="utf-8")
    )
    assert loader_files == ["tests/conftest.py"]

    for name in ("date", "time", "fraction"):
        source = (_REPO / "tools" / f"bench_{name}_compose.py").read_text(encoding="utf-8")
        assert "run_composition_benchmark(" in source
        assert "def percentile(" not in source

    def completed(command, **kwargs):
        locale = command[command.index("--child-locale") + 1]
        value = 10 if locale == "a" else 20
        return SimpleNamespace(stdout=f'{{"locale": "{locale}", "p50_ns": {value}}}\n')

    monkeypatch.setattr(compose_benchmark.subprocess, "run", completed)
    output = tmp_path / "receipt.json"
    report = compose_benchmark.run_composition_benchmark(
        script=Path("worker.py"),
        repository=_REPO,
        locales=("a", "b"),
        output=output,
        iterations=3,
        repeat=2,
        limits={"p50_ns": 15},
    )

    assert [row["run"] for row in report["rows"]] == [1, 1, 2, 2]
    assert report["summary"] == {"a": {"p50_ns": 10}, "b": {"p50_ns": 20}}
    assert report["failures"] == [{"locale": "b", "metric": "p50_ns", "value": 20, "limit": 15}]
    assert output.read_text(encoding="utf-8").endswith("\n")
