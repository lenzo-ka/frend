from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import frend
from frend import InputValidationError, NormalizedText


def _evaluator():
    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "normalize_evaluator", tools / "evaluate_google_tn.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_returns_a_string_by_default():
    result = frend.normalize("I paid $12.")
    assert isinstance(result, str)
    assert "twelve" in result


def test_offsets_tile_output_and_slice_the_original_source():
    source = "I paid $12.\n\nIt was 5% off."
    result = frend.normalize(source, offsets=True)
    assert isinstance(result, NormalizedText)
    at = 0
    for unit in result.units:
        assert unit.output_span[0] == at
        assert unit.output_span[1] >= unit.output_span[0]
        assert result.text[slice(*unit.output_span)]
        at = unit.output_span[1]
        start, end = unit.source_span
        assert 0 <= start < end <= len(source)
        assert source[start:end]
    assert at == len(result.text)
    separators = [unit for unit in result.units if unit.provenance.endswith("whitespace")]
    assert len(separators) == 1
    assert source[slice(*separators[0].source_span)] == "\n\n"
    assert result.text[slice(*separators[0].output_span)] == " "
    assert "\n" not in result.text


def test_first_choice_agrees_with_the_evaluator_join_on_a_fixture():
    evaluator = _evaluator()
    source = "I paid $12 on Jan 2."
    from icukit.detectors import detect

    from frend import resolve_lattice, verbalize_lattice

    detections = list(detect(source, evaluator._detectors("en_US")))
    path = verbalize_lattice(resolve_lattice(detections, source_text=source)).best_path
    expected = evaluator._joined(
        (unit.best.text, unit.best.provenance == "surface:passthrough") for unit in path.units
    )
    assert frend.normalize(source) == expected


def test_input_limit_refusals_pass_through():
    with pytest.raises(InputValidationError, match="max_input_chars=3"):
        frend.normalize("four", max_input_chars=3)
    with pytest.raises(InputValidationError, match="NUL"):
        frend.normalize("bad\x00text")


def test_sentence_over_unit_bound_names_the_missing_icukit_forced_break():
    with pytest.raises(
        InputValidationError, match=r"sentence has 4.*max_unit_chars=3.*forced sentence break"
    ):
        frend.normalize("four", max_unit_chars=3)


def test_document_input_is_validated_once(monkeypatch):
    import importlib

    normalize_module = importlib.import_module("frend.normalize")
    original = normalize_module.validate_input
    calls = []

    def recording_validate(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(normalize_module, "validate_input", recording_validate)
    frend.normalize("It cost $12.\n\nNow it costs $13.")
    assert len(calls) == 1


def test_text_mode_skips_offset_record_construction(monkeypatch):
    import importlib

    normalize_module = importlib.import_module("frend.normalize")

    monkeypatch.setattr(
        normalize_module,
        "NormalizedUnit",
        lambda *_args, **_kwargs: pytest.fail("text mode constructed offset bookkeeping"),
    )
    assert isinstance(frend.normalize("12"), str)
