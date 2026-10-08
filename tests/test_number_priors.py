"""Measured bare four-digit number choices."""

from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest

from frend.number_priors import NumberPrior, NumberPriorTable, load_number_priors

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


@pytest.mark.parametrize(
    ("section", "value"),
    [
        ("centuries", True),
        ("all", True),
        ("centuries", -1),
        ("all", 1.5),
    ],
)
def test_table_refuses_invalid_counts(section, value):
    document = {
        "all": {"date": 6, "cardinal": 3, "digit": 1},
        "centuries": {"19": {"date": 8, "cardinal": 2}},
    }
    if section == "centuries":
        document[section]["19"]["date"] = value
    else:
        document[section]["date"] = value

    with pytest.raises(ValueError, match="counts must be nonnegative integers"):
        NumberPriorTable(document)


@pytest.mark.parametrize("section", ["centuries", "all"])
def test_table_refuses_unknown_choice_keys(section):
    document = {
        "all": {"date": 6, "cardinal": 3, "digit": 1},
        "centuries": {"19": {"date": 8, "cardinal": 2}},
    }
    if section == "centuries":
        document[section]["19"]["year"] = 10
    else:
        document[section]["year"] = 10

    with pytest.raises(ValueError, match="unknown choice keys"):
        NumberPriorTable(document)


def test_table_retains_exact_counts_and_shares_for_valid_rows():
    table = NumberPriorTable(
        {
            "all": {"date": 6, "cardinal": 3, "digit": 1},
            "centuries": {"19": {"date": 8, "cardinal": 2}},
        }
    )

    assert table.lookup("1979", "date") == NumberPrior(8, Decimal("0.8"), 10)
    assert table.lookup("7199", "digit") == NumberPrior(1, Decimal("0.1"), 10)


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


def test_shipped_table_names_only_training_shards():
    table = load_number_priors("en_US")
    assert table is not None
    assert table.provenance["shards"] == [f"output-{index:05d}-of-00100" for index in range(90)]
    json.dumps(dict(table.provenance))
