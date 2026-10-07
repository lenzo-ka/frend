from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from measure_lookahead import (  # noqa: E402
    _context_requirement,
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


def test_processed_root_without_option_or_environment_refuses(tmp_path, monkeypatch, capsys):
    import measure_lookahead

    monkeypatch.delenv("FREND_PROCESSED", raising=False)
    with pytest.raises(SystemExit):
        measure_lookahead.main(["--output", str(tmp_path / "measurement.json")])
    assert "--processed-root is required when FREND_PROCESSED is not set" in capsys.readouterr().err


def test_static_tree_extent_uses_decision_features_not_complete_schema():
    from frend.context import context_model, features

    model = context_model("en_US")
    assert model is not None
    leaf_problem = next(
        problem for problem, info in model.index["trees"].items() if info["nodes"] == 1
    )
    assert _tree_names(leaf_problem, {}) == ()

    problem = (
        "cldr-symbol:dash\tcldr-symbol:hyphen\tcldr-symbol:hyphen-minus\t"
        "cldr-symbol:minus\tlexical:en_US\tsurface:silence"
    )
    values = features(
        "x a b",
        0,
        1,
        first="cldr-symbol:dash",
        first_weight=None,
        locale="en_US",
        frequent=model.frequent,
        curated=model.curated,
    )
    names = _tree_names(problem, values)
    tree = model.tree(problem)
    assert tree is not None
    all_decision_names = {
        tree.names[int(decision[0])] for decision in tree._predictor.model["decisions"]
    }
    assert "w_wb+3" in names
    assert set(names) < all_decision_names


def test_right_character_offset_skips_only_boundary_whitespace():
    from frend.context import class_windows

    text = "x a b"
    spans = [(0, 1), (2, 3), (4, 5)]
    assert class_windows(text, 0, 1)["w_gc+3"] == "Ll"
    assert _context_requirement(text, 0, 1, ("w_gc+3",), spans) == (2, False, 0, 3)
