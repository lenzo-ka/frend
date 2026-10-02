"""Measured bare four-digit number choices."""

from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

from frend.context import TextContext
from frend.number_priors import NumberPriorTable, load_number_priors
from frend.verbalize import SpokenAlternative, _bare_number_ranked

_REPO = Path(__file__).resolve().parents[1]


def _builder():
    path = _REPO / "tools" / "build_number_priors.py"
    spec = importlib.util.spec_from_file_location("build_number_priors", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_table_uses_a_century_row_and_a_measured_fallback():
    table = NumberPriorTable(
        {
            "all": {"date": 6, "cardinal": 3, "digit": 1},
            "centuries": {"19": {"date": 8, "cardinal": 2}},
        }
    )
    assert table.lookup("1979", "date").share == Decimal("0.8")
    assert table.lookup("7199", "cardinal").share == Decimal("0.3")
    assert table.lookup("979", "date") is None


def test_google_tn_profile_uses_its_selected_aggregate_but_default_does_not(monkeypatch):
    table = NumberPriorTable(
        {
            "all": {"date": 9, "cardinal": 1},
            "centuries": {"20": {"date": 9, "cardinal": 1}},
            "google_tn_profile": {
                "selected_key": "decade+structure+neighbors",
                "keys": {"201|ordinary|other|other": {"date": 1, "cardinal": 9}},
            },
        }
    )
    monkeypatch.setattr("frend.verbalize.load_number_priors", lambda _locale: table)
    alternatives = (
        SpokenAlternative("twenty twelve", "icu-rbnf:%spellout-numbering-year"),
        SpokenAlternative("two thousand twelve", "icu-rbnf:%spellout-numbering"),
    )
    detection = {"text": "2012"}
    context = TextContext("in 2012 today", 3)

    default = _bare_number_ranked(alternatives, detection, "en_US", context=context, start=0, end=4)
    profiled = _bare_number_ranked(
        alternatives,
        detection,
        "en_US",
        context=context,
        start=0,
        end=4,
        profile="google-tn",
    )

    assert default[0].text == "twenty twelve"
    assert profiled[0].text == "two thousand twelve"


def test_fixture_builder_counts_only_bare_four_digit_choices(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "output-00000-of-00001").write_text(
        "DATE\t1979\tnineteen seventy nine\n"
        "CARDINAL\t1999\tone thousand nine hundred ninety nine\n"
        "DIGIT\t1900\tone nine o o\n"
        "DATE\tMay 1979\tmay nineteen seventy nine\n"
        "TELEPHONE\t1971\tone nine seven one\n",
        encoding="utf-8",
    )
    document = _builder().build_document(corpus)
    assert document["all"] == {"date": 1, "cardinal": 1, "digit": 1}
    assert document["centuries"] == {"19": {"date": 1, "cardinal": 1, "digit": 1}}


def test_builder_selects_a_finer_key_on_development(tmp_path, monkeypatch):
    training = tmp_path / "output-00000-of-00100"
    development = tmp_path / "output-00080-of-00100"
    training.write_text(
        "CARDINAL\t2012\ttwo thousand twelve\n<eos>\t<eos>\n"
        "DATE\t2099\ttwenty ninety nine\n<eos>\t<eos>\n"
        "DATE\t2098\ttwenty ninety eight\n<eos>\t<eos>\n",
        encoding="utf-8",
    )
    development.write_text(
        "CARDINAL\t2013\ttwo thousand thirteen\n<eos>\t<eos>\n",
        encoding="utf-8",
    )
    builder = _builder()
    monkeypatch.setattr(builder, "_shards", lambda _corpus, _inputs=None: [training, development])

    profile = builder.build_document(tmp_path)["google_tn_profile"]

    assert profile["selection"]["candidates"]["century"]["correct"] == 0
    assert profile["selection"]["candidates"]["decade"]["correct"] == 1
    assert profile["selected_key"] == "decade"
    assert profile["keys"]["201"] == {"cardinal": 2}


def test_shipped_table_names_only_training_shards():
    table = load_number_priors("en_US")
    assert table is not None
    assert table.provenance["shards"] == [f"output-{index:05d}-of-00100" for index in range(90)]
    json.dumps(dict(table.provenance))
