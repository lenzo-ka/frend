"""A zero is said as the corpus says it in that kind of reading: "o" in a date, mostly
"o" after a decimal point, "zero" in front of one."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from icukit.detectors import detect
from icukit.recognize import FlexibleNumberDetector, FlexibleTimeDetector

from frend import resolve_lattice
from frend.verbalize import _zero_key, verbalize_lattice

_DETECTORS = (FlexibleNumberDetector("en_US"), FlexibleTimeDetector("en_US"))
_TABLE = Path(__file__).resolve().parents[1] / "frend" / "data" / "zero_priors.json"


def _first(text: str) -> str:
    lattice = resolve_lattice(list(detect(text, _DETECTORS)), source_text=text)
    (unit,) = [
        unit
        for unit in verbalize_lattice(lattice).best_path.units
        if unit.best.provenance != "surface:passthrough"
    ]
    return unit.best.text


@pytest.mark.parametrize(
    ("text", "expected"),
    [("3.05", "three point o five"), ("0.5", "zero point five"), ("10:05", "ten o five")],
)
def test_zero_is_said_as_measured(text, expected):
    assert _first(text) == expected


def test_readings_group_by_everything_but_the_zero_word():
    assert _zero_key("nineteen oh-five") == _zero_key("nineteen o five")
    assert _zero_key("three point zero five") != _zero_key("three point five")


def test_the_table_says_o_for_dates_and_never_oh():
    kinds = json.loads(_TABLE.read_text(encoding="utf-8"))["kinds"]
    assert kinds["date"]["o"] > 1000 * kinds["date"].get("zero", 0)
    assert all(words.get("oh", 0) == 0 for words in kinds.values())
