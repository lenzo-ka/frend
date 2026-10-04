"""``tools/build_range_priors.py``: the range table (E and J), counted from the training
shards only, reproducibly."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
_TABLE = _REPO / "frend" / "data" / "en" / "range_priors.json"


def _builder():
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(
        "build_range_priors", _TOOLS / "build_range_priors.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_range_priors"] = module
    spec.loader.exec_module(module)
    return module


_ROWS = [
    # A range said "to" (credited), and one said as nothing between plain cardinals.
    [("PLAIN", "pages", "<self>"), ("CARDINAL", "5", "five"), ("PLAIN", "-", "to"),
     ("CARDINAL", "10", "ten")],
    [("CARDINAL", "57", "fifty seven"), ("PUNCT", "-", "sil"), ("CARDINAL", "58", "fifty eight")],
    # R6: a year-shaped cardinal, then a dash said as nothing: punctuation.
    [("CARDINAL", "1994", "one thousand nine hundred ninety four"), ("PUNCT", "-", "sil"),
     ("CARDINAL", "95", "ninety five")],
    # R6: money ends, the dash silent; and a money range said "to" (counted).
    [("MONEY", "$15,000", "fifteen thousand dollars"), ("VERBATIM", "-", "sil"),
     ("MONEY", "$25,000", "twenty five thousand dollars")],
    [("MONEY", "$15,000", "fifteen thousand dollars"), ("PLAIN", "-", "to"),
     ("MONEY", "$25,000", "twenty five thousand dollars")],
    # R1: the right end starts with a word, so no rule can emit on it.
    [("DATE", "Oct. 29, 1951", "october twenty ninth nineteen fifty one"),
     ("PUNCT", "-", "sil"), ("DATE", "April 28, 1953", "april twenty eighth nineteen fifty three")],
    # Single tokens of a range's written shape: a phone's and a clock's.
    [("TELEPHONE", "555-1212", "five five five sil one two one two")],
    [("TIME", "10:30", "ten thirty")],
    [("TIME", "2:08.34", "two minutes eight seconds and thirty four milliseconds")],
]  # fmt: skip


def _corpus(root: Path) -> Path:
    root.mkdir()
    lines = []
    for sentence in _ROWS:
        lines += ["\t".join(row) for row in sentence] + ["<eos>\t<eos>"]
    (root / "output-00000-of-00002").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_money_rows_are_counted(tmp_path):
    """The superset predicate reaches a money-left triple: "$15,000 - $25,000" said "to"
    and the correctly read silent form are both valid range evidence."""
    document = _builder().build_document(_corpus(tmp_path / "corpus"))
    other = document["kinds"]["dash"]["sub_keys"]["other"]["source_matched"]
    assert other == {
        "range:money+silent+money": 1,
        "range:money+to+money": 1,
    }


def test_relevance_predicates_in_the_builder(tmp_path):
    """E and J count only what the rules can emit on and R6 keeps: the punctuation dashes
    and the unemittable date are counted apart, in the provenance."""
    document = _builder().build_document(_corpus(tmp_path / "corpus"))
    relevance = document["provenance"]["relevance"]
    assert relevance["wider_not_counted"] == {"not_emittable": 1, "punctuation_dash": 0}
    assert relevance["joint_outcomes"] == {"credited": 5, "not_emittable": 1}
    readings = document["readings"]
    assert readings["dash:1+2"] == {"range": 1, "single_token": {}}
    assert readings["dash:3+4"] == {"range": 0, "single_token": {"TELEPHONE": 1}}
    assert readings["ratio:clock"] == {"range": 0, "single_token": {"TIME": 1}}
    sources = document["kinds"]["dash"]["source_matched"]
    assert sources == {
        "range:cardinal+silent+cardinal": 2,
        "range:cardinal+to+cardinal": 1,
        "range:money+silent+money": 1,
        "range:money+to+money": 1,
    }


def test_fractional_duration_preference_is_derived_from_counts(tmp_path):
    document = _builder().build_document(_corpus(tmp_path / "corpus"))
    assert document["numeric_duration"] == {
        "shape": "whole-token M:SS.hh",
        "classes": {"TIME": 1},
        "outcomes": {"second": 1},
        "total": 1,
        "preferred_unit": "second",
        "preferred_count": 1,
        "other_count": 0,
    }


def test_a_silent_dash_with_correctly_read_ends_is_kept(tmp_path):
    """R6 refined: silence between correctly read ends is a valid range reading."""
    document = _builder().build_document(_corpus(tmp_path / "corpus"))
    four_two = document["kinds"]["dash"]["sub_keys"]["4+2"]["source_matched"]
    assert four_two == {"range:cardinal+silent+cardinal": 1}


def test_check_round_trips_a_fixture_build(tmp_path):
    builder = _builder()
    corpus = _corpus(tmp_path / "corpus")
    out = tmp_path / "range_priors.json"
    rendered = builder.render(builder.build_document(corpus))
    out.write_text(rendered, encoding="utf-8")
    assert out.read_text(encoding="utf-8") == builder.render(builder.build_document(corpus))
    out.write_text(out.read_text(encoding="utf-8").replace('"range": 1', '"range": 2', 1))
    assert out.read_text(encoding="utf-8") != builder.render(builder.build_document(corpus))


def test_shipped_table_names_its_shards():
    """E counts the 90 training shards, J every tenth of them; none is held out."""
    provenance = json.loads(_TABLE.read_text(encoding="utf-8"))["provenance"]
    assert provenance["emit_shards"] == [f"output-{i:05d}-of-00100" for i in range(90)]
    assert provenance["shards"] == [f"output-{i:05d}-of-00100" for i in range(0, 90, 10)]
    assert provenance["corpus"] == "google-tn:en_with_types"


def test_check_matches_shipped_table(tmp_path):
    """``--check`` re-derives the shipped table from the corpus, byte for byte. The corpus
    is not in CI: this runs where it is present."""
    builder = _builder()
    from build_spoken_priors import _default_corpus_dir

    corpus = _default_corpus_dir()
    if not (corpus / "output-00000-of-00100").is_file():
        pytest.skip(f"the Google TN corpus is not at {corpus}")
    assert (
        builder.main(
            [
                "--locale",
                "en_US",
                "--source-id",
                "google/tn-en_with_types",
                "--pool",
                "training",
                "--receipt",
                str(tmp_path / "receipt.json"),
                "--check",
                "--jobs",
                "8",
            ]
        )
        == 0
    )
