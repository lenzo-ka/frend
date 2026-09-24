"""Behavioral checks for the alignment mutation runner itself."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_runner():
    path = Path(__file__).parent / "run_align_mutations.py"
    spec = importlib.util.spec_from_file_location("run_align_mutations", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prefailing_target_and_noop_mutation_is_invalid(tmp_path):
    runner = _load_runner()
    (tmp_path / "sample.py").write_text("VALUE = 1\n")
    (tmp_path / "test_sample.py").write_text(
        "from sample import VALUE\n\ndef test_target():\n    assert VALUE == 2\n"
    )
    mutation = runner.Mutation(
        "No-op whitespace mutation",
        "test_sample.py::test_target",
        "sample.py",
        "VALUE = 1",
        "VALUE  = 1",
    )

    rows = runner.run_mutations([mutation], root=tmp_path, timeout=10)

    assert rows[0]["status"] == "invalid"
    assert not rows[0]["killed"]
    assert not rows[0]["behavioral"]
    assert rows[0]["baseline"]["failures"] == 1
    assert (tmp_path / "sample.py").read_text() == "VALUE = 1\n"


@pytest.mark.parametrize(
    ("source", "before", "after", "exception_type"),
    [
        (
            "VALUE = 1\n\ndef helper():\n    return VALUE\n",
            "return VALUE",
            "return MISSING_VALUE",
            "NameError",
        ),
        (
            "def helper():\n    return 1\n",
            "def helper():",
            "def renamed_helper():",
            "AttributeError",
        ),
        (
            "VALUE = 1\n\ndef helper():\n    return VALUE\n",
            "return VALUE",
            "result = VALUE\n    VALUE = 2\n    return result",
            "UnboundLocalError",
        ),
    ],
)
def test_symbol_absence_is_not_a_behavioral_kill(tmp_path, source, before, after, exception_type):
    runner = _load_runner()
    (tmp_path / "sample.py").write_text(source)
    (tmp_path / "test_sample.py").write_text(
        "import sample\n\ndef test_target():\n    assert sample.helper() == 1\n"
    )
    mutation = runner.Mutation(
        "Remove target symbol",
        "test_sample.py::test_target",
        "sample.py",
        before,
        after,
    )

    row = runner.run_mutations([mutation], root=tmp_path, timeout=10)[0]

    details = row["mutation_result"]["failure_details"]
    assert len(details) == 1
    assert details[0]["testcase"] == "test_target"
    assert details[0]["outcome"] == "failure"
    assert details[0]["exception_type"] == exception_type
    assert details[0]["message"]
    assert row["status"] == "invalid"
    assert not row["killed"]
    assert not row["behavioral"]
    assert exception_type in row["reason"]


def test_refused_mutation_leaves_existing_evidence_untouched(tmp_path, monkeypatch):
    runner = _load_runner()
    source = tmp_path / "sample.py"
    evidence = tmp_path / "evidence.json"
    source.write_text("VALUE = 1\n")
    evidence.write_text('[{"existing": true}]\n')
    valid_mutation = runner.Mutation(
        "Valid target",
        "test_sample.py::test_target",
        "sample.py",
        "VALUE = 1",
        "VALUE = 2",
    )
    refused_mutation = runner.Mutation(
        "Missing target",
        "test_sample.py::test_target",
        "sample.py",
        "VALUE = 2",
        "VALUE = 3",
    )
    monkeypatch.setattr(
        runner,
        "_pytest_result",
        lambda *args: pytest.fail("pytest ran before every mutation was validated"),
    )

    with pytest.raises(ValueError, match="mutation target is not unique"):
        runner.run_mutations(
            [valid_mutation, refused_mutation],
            root=tmp_path,
            output_path=evidence,
            timeout=10,
        )

    assert evidence.read_text() == '[{"existing": true}]\n'
    assert source.read_text() == "VALUE = 1\n"


def test_checked_in_mutation_evidence_matches_current_tree():
    runner = _load_runner()
    root = Path(__file__).parents[1]
    evidence = json.loads((root / "tests/data/align_mutations.json").read_text())
    mutations = {mutation.name: mutation for mutation in runner.MUTATIONS}
    runner_hash = hashlib.sha256((root / "tests/run_align_mutations.py").read_bytes()).hexdigest()

    assert "irn/align_export.py" in {mutation.path for mutation in runner.MUTATIONS}
    assert "tests/test_align_export.py" in {
        mutation.test.split("::", 1)[0] for mutation in runner.MUTATIONS
    }
    assert len(evidence) == len(runner.MUTATIONS)
    assert [row["mutation"] for row in evidence] == [mutation.name for mutation in runner.MUTATIONS]
    for row in evidence:
        mutation = mutations[row["mutation"]]
        test_path = mutation.test.split("::", 1)[0]
        assert (
            row["source_sha256"] == hashlib.sha256((root / mutation.path).read_bytes()).hexdigest()
        )
        assert row["test_sha256"] == hashlib.sha256((root / test_path).read_bytes()).hexdigest()
        assert row["before_sha256"] == hashlib.sha256(mutation.before.encode()).hexdigest()
        assert row["after_sha256"] == hashlib.sha256(mutation.after.encode()).hexdigest()
        assert row["runner_sha256"] == runner_hash
        assert row["status"] == "killed"
        assert row["killed"]
        assert row["behavioral"]
