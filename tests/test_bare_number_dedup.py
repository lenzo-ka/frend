"""PR1 regressions for bare-number text-class deduplication."""

from __future__ import annotations

import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import evaluate_google_tn  # noqa: E402

from frend.number_priors import load_number_priors  # noqa: E402
from frend.verbalize import (  # noqa: E402
    SpokenAlternative,
    _bare_number_classes,
    _bare_number_ranked,
    _rank_final,
    _ranked,
    _spoken_digits,
    _year_leaf,
)
from frend.written_forms import DigitsValue  # noqa: E402


@pytest.mark.parametrize(
    ("written", "target", "before", "after"),
    [
        pytest.param("5000", "five thousand", "the", "meters", id="R1"),
        pytest.param("3000", "three thousand", "the", "x", id="R2"),
        pytest.param("9001", "nine thousand one", "", "", id="X1"),
    ],
)
def test_repaired_bare_number_rows(written, target, before, after):
    assert evaluate_google_tn._score_text(written, target, before, after)[0]


@pytest.mark.parametrize(
    ("written", "target", "before", "after"),
    [
        pytest.param("2012", "twenty twelve", "In", ", he", id="C1"),
        pytest.param("1908", "nineteen o eight", "in", ".", id="C2"),
        pytest.param("1000", "one thousand", "the", "people", id="C3"),
        pytest.param("1990", "nineteen ninety", "the band in MID", ".", id="C5"),
        pytest.param("2012", "twenty twelve", "MID", ",", id="C6"),
    ],
)
def test_characterization_scorer_rows_stay_first(written, target, before, after):
    assert evaluate_google_tn._score_text(written, target, before, after)[0]


def _pre_bare_alternatives(written: str) -> tuple[SpokenAlternative, ...]:
    value = Decimal(written)
    alternatives = _ranked(
        [*_year_leaf(value, "en_US"), *_spoken_digits(DigitsValue(written), "en_US")]
    )
    return _rank_final(alternatives, "date", None, "en_US")


def _old_first(written: str) -> str:
    table = load_number_priors("en_US")
    assert table is not None
    weighted = []
    for item in _pre_bare_alternatives(written):
        if "numbering-year" in item.provenance:
            choice = "date"
        elif item.provenance.startswith("icu-rbnf:%spellout-cardinal"):
            choice = "digit"
        else:
            choice = "cardinal"
        prior = table.lookup(written, choice)
        weighted.append(
            item if prior is None else SpokenAlternative(item.text, item.provenance, prior.share)
        )
    return _ranked(weighted)[0].text


def _new_first(written: str) -> str:
    detection = {"text": written}
    return _bare_number_ranked(_pre_bare_alternatives(written), detection, "en_US")[0].text


def test_r13_exact_70_value_sweep():
    words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }
    expected = {}
    for thousands in range(3, 10):
        lead = words[thousands]
        expected[f"{thousands}000"] = (f"{lead} o o o", f"{lead} thousand")
        for ones in range(1, 10):
            expected[f"{thousands}00{ones}"] = (
                f"{lead} thousand and {words[ones]}",
                f"{lead} thousand {words[ones]}",
            )

    changed = {}
    for number in range(1000, 10000):
        written = str(number)
        before, after = _old_first(written), _new_first(written)
        if before != after:
            changed[written] = (before, after)

    assert len(expected) == 70
    assert changed == expected


def test_date_class_is_only_year_text_and_lexical_o_variant():
    classes = _bare_number_classes("1908", Decimal(1908), "en_US")
    assert classes["nineteen oh-eight"] == frozenset({"date"})
    assert classes["nineteen o eight"] == frozenset({"date"})
    assert classes["one thousand nine hundred eight"] == frozenset({"cardinal"})

    shared = _bare_number_classes("5000", Decimal(5000), "en_US")
    assert shared["five thousand"] == frozenset({"date", "cardinal"})


def test_g8_retains_index_and_range_trees_from_main():
    result = subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "07e3899",
            "--",
            "frend/data/en/context/index.json",
            "frend/data/en/context/trees/",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_r16_retained_range_tree_row():
    before = "Count Alessandro Contini Bonacossi ( 18 March"
    after = "October 1955 ) was an Italian politician , art collector , dealer and philatelist ."
    assert evaluate_google_tn._score_text(
        "1878-22", "eighteen seventy-eight to twenty-two", before, after
    )[0]
