"""``tools/build_abbreviation_priors.py``: how the corpus says each abbreviation, by fold and
written case, and its shipped table."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
_TABLE = _REPO / "frend" / "data" / "en" / "abbreviation_priors.json"


def _builder():
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(
        "build_abbreviation_priors", _TOOLS / "build_abbreviation_priors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ROWS = [
    ("PLAIN", "st", "saint"),
    ("PLAIN", "st", "street"),
    ("PLAIN", "ST", "s t"),
    ("PLAIN", "St.", "saint"),
    ("PLAIN", "No", "<self>"),
    ("PLAIN", "No", "number"),
    ("PLAIN", "no", "<self>"),
    ("PLAIN", "Mr", "mister"),
    ("PLAIN", "mR", "mister"),
    ("LETTERS", "e.g.", "e g"),
    ("PLAIN", "e.g.", "for example"),
    ("PLAIN", "vs", "verses"),
    ("PLAIN", "in", "<self>"),
    ("PLAIN", "p", "p"),
    ("PLAIN", "sat", "<self>"),
]


def _corpus(root: Path) -> Path:
    root.mkdir()
    body = "".join(f"{c}\t{w}\t{s}\n" for c, w, s in _ROWS) + "<eos>\t<eos>\n"
    (root / "output-00000-of-00002").write_text(body, encoding="utf-8")
    # Every tenth shard is sampled, so the second is never read.
    (root / "output-00001-of-00002").write_text("PLAIN\tsun\t<self>\n", encoding="utf-8")
    return root


def test_rows_are_counted_by_fold_case_and_what_the_corpus_says(tmp_path):
    """Each row is keyed by its ICU fold less one trailing period, under its written case,
    by what the corpus says: spelled, as written, a lexicon expansion, or other. A token in
    none of the three cases ("mR"), and a surface that is no key ("in", "p"), are not
    counted."""
    document = _builder().build_document(_corpus(tmp_path / "corpus"))
    assert document["keys"] == {
        "st": {"lower": {"saint": 1, "street": 1}, "title": {"saint": 1}, "upper": {"spelled": 1}},
        "no": {"title": {"as-written": 1, "number": 1}, "lower": {"as-written": 1}},
        "mr": {"title": {"mister": 1}},
        "e.g": {"lower": {"spelled": 1, "for example": 1}},
        "vs": {"lower": {"other": 1}},
        "sat": {"lower": {"as-written": 1}},
    }
    provenance = document["provenance"]
    assert provenance["corpus"] == "google-tn:corpus"
    assert provenance["shards"] == ["output-00000-of-00002"]


def test_check_round_trips_a_fixture_build(tmp_path):
    builder = _builder()
    corpus = _corpus(tmp_path / "corpus")
    out = tmp_path / "abbreviation_priors.json"
    rendered = builder._render(builder.build_document(corpus))
    out.write_text(rendered, encoding="utf-8")
    assert out.read_text(encoding="utf-8") == builder._render(builder.build_document(corpus))
    out.write_text(out.read_text(encoding="utf-8").replace('"saint": 1', '"saint": 2', 1))
    assert out.read_text(encoding="utf-8") != builder._render(builder.build_document(corpus))


def test_shipped_table_counts_every_tenth_training_shard():
    """The shipped table counts the spoken priors' sample: every tenth training shard
    (00, 10, ..., 80), none held out."""
    provenance = json.loads(_TABLE.read_text(encoding="utf-8"))["provenance"]
    assert provenance["shards"] == [f"output-{index:05d}-of-00100" for index in range(0, 90, 10)]
    assert provenance["locale"] == "en"
    assert provenance["corpus"] == "google-tn:en_with_types"
    keys = json.loads(_TABLE.read_text(encoding="utf-8"))["keys"]
    # The corpus never writes these keys in title case (dualplan-ranges M3).
    for key in ("st", "dr", "mr", "mrs", "ltd", "jr", "sr", "mt"):
        assert "title" not in keys[key], key
    assert "title" in keys["no"] and "title" in keys["vol"]


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
            ]
        )
        == 0
    )
