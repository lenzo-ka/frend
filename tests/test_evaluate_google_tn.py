"""The evaluator's running-text rejoining: by written form only, never overlapping."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _evaluator():
    sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(
        "evaluate_google_tn", _TOOLS / "evaluate_google_tn.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_number_separator_number_triple_is_rejoined_as_written():
    sentence = [
        ("PLAIN", "pages", "<self>"),
        ("CARDINAL", "5", "five"),
        ("PLAIN", "-", "to"),
        ("CARDINAL", "10", "ten"),
    ]
    assert _evaluator()._running_text([sentence]) == [("-", "to", "5-10", "five to ten")]


def test_silence_and_non_numbers_and_overlaps():
    evaluate = _evaluator()
    silent = [("CARDINAL", "3", "three"), ("PUNCT", ":", "sil"), ("CARDINAL", "2", "two")]
    assert evaluate._running_text([silent]) == [(":", "(silence)", "3:2", "three  two")]
    words = [("PLAIN", "well", "<self>"), ("PUNCT", "-", "sil"), ("PLAIN", "known", "<self>")]
    assert evaluate._running_text([words]) == []
    chain = [
        ("CARDINAL", "1", "one"),
        ("PLAIN", "-", "to"),
        ("CARDINAL", "2", "two"),
        ("PLAIN", "-", "to"),
        ("CARDINAL", "3", "three"),
    ]
    assert [item[2] for item in evaluate._running_text([chain])] == ["1-2"]
