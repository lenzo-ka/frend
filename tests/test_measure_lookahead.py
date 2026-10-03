from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from measure_lookahead import (  # noqa: E402
    _prefix_readings,
    _text_and_spans,
    _tree_names,
    commit_lookahead,
    validate_output_path,
)


def _fixture_lookaheads(words: tuple[str, ...]) -> list[int]:
    sentence = [("PLAIN", word, "<self>") for word in words]
    text, spans, ends = _text_and_spans(sentence)
    histories = [[] for _ in sentence]
    for prefix_index, end in enumerate(ends):
        readings = _prefix_readings(text[:end], spans[: prefix_index + 1], ("default",))["default"]
        for token_index, reading in enumerate(readings):
            histories[token_index].append(reading.signature)
    return [commit_lookahead(history) for history in histories]


def test_later_token_changes_reading_and_unambiguous_token_commits_immediately():
    # "5" starts as a cardinal but becomes the left edge of "5 - 10" only when
    # the right end arrives.  An ordinary word's reading is already final.
    assert _fixture_lookaheads(("5", "-", "10"))[0] == 2
    assert _fixture_lookaheads(("plain", "arrives"))[0] == 0


def test_commit_requires_equality_at_every_longer_prefix():
    assert commit_lookahead(["final", "other", "final"]) == 2


def test_output_must_be_external_and_below_streaming_root(tmp_path: Path):
    repo = tmp_path / "repo"
    output_root = tmp_path / "processed" / "streaming"
    repo.mkdir()
    output_root.mkdir(parents=True)
    accepted = output_root / "measurement.json"
    assert validate_output_path(accepted, output_root=output_root, repo=repo) == accepted
    with pytest.raises(ValueError, match="repository"):
        validate_output_path(repo / "result.json", output_root=repo, repo=repo)
    with pytest.raises(ValueError, match="under"):
        validate_output_path(tmp_path / "elsewhere.json", output_root=output_root, repo=repo)


def test_static_tree_extent_uses_decision_features_not_complete_schema():
    from frend.context import context_model

    model = context_model("en_US")
    assert model is not None
    leaf_problem = next(
        problem for problem, info in model.index["trees"].items() if info["nodes"] == 1
    )
    assert _tree_names(leaf_problem) == ()
