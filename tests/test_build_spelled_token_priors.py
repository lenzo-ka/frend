from __future__ import annotations

from pathlib import Path

from conftest import load_tool_module


def _builder():
    return load_tool_module(
        "build_spelled_token_priors",
        Path(__file__).resolve().parents[1] / "tools" / "build_spelled_token_priors.py",
    )


def test_relevance_is_literal():
    literal = _builder().literal_outcome
    assert literal("LETTERS", "pdf", "p d f", "en_US") == "spell"
    assert literal("PLAIN", "pdf", "<self>", "en_US") == "say"
    assert literal("LETTERS", "pdf", "portable document format", "en_US") is None
    assert literal("PLAIN", "pdf", "document", "en_US") is None
    assert literal("LETTERS", "NASA", "n a s a", "en_US") is None


def test_say_only_tokens_need_no_stored_entry(tmp_path):
    shard = tmp_path / "output-00000-of-00002"
    shard.write_text("PLAIN\tword\t<self>\n<eos>\t<eos>\n", encoding="utf-8")
    document = _builder().build_document(tmp_path)
    assert document["tokens"] == {}
    assert document["provenance"]["training_shards"] == [shard.name]


def test_only_rule_exceptions_are_stored(tmp_path):
    shard = tmp_path / "output-00000-of-00002"
    shard.write_text(
        "LETTERS\tfMRI\tf m r i\nPLAIN\tby\t<self>\nPLAIN\tword\t<self>\n",
        encoding="utf-8",
    )
    document = _builder().build_document(tmp_path)
    assert set(document["tokens"]) == {"fMRI", "by"}
    assert document["provenance"]["full_measured_tokens"] == 3
    assert document["provenance"]["exceptions"] == 2
    assert "pronunciation_sources" not in document["provenance"]
