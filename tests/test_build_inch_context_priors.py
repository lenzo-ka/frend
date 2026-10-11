"""Builder coverage for the learned bare-inch context counts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _builder():
    path = _REPO / "tools" / "build_inch_context_priors.py"
    spec = importlib.util.spec_from_file_location("build_inch_context_priors", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_builder_counts_following_classes_and_sentence_ends(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shard = corpus / "output-00000-of-00001"
    shard.write_text(
        "MEASURE\t5 in\tfive inches\n"
        "PLAIN\tlong\t<self>\n"
        "<eos>\t<eos>\n"
        "MEASURE\t2 in\ttwo inches\n"
        "<eos>\t<eos>\n"
        "CARDINAL\t3\tthree\n"
        "PLAIN\tin\t<self>\n"
        "PLAIN\tthe\t<self>\n"
        "<eos>\t<eos>\n"
        "ORDINAL\t4th\tfourth\n"
        "PLAIN\tin\t<self>\n"
        "<eos>\t<eos>\n",
        encoding="utf-8",
    )

    document = _builder().build_document(corpus)

    assert sum(document["counts"]["measure"].values()) == 2
    assert sum(document["counts"]["preposition"].values()) == 2
    assert document["counts"]["measure"]["<EOS>"] == 1
    assert document["counts"]["preposition"]["<EOS>"] == 1
    assert document["provenance"]["shards"] == [shard.name]
    assert document["provenance"]["builder"] == "tools/build_inch_context_priors.py"
