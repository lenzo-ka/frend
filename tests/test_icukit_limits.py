"""frend keeps icukit resource failures visible and bounded captures non-fatal."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from frend import normalize, resolve_lattice, verbalize_lattice

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


class SentinelRegexError(RuntimeError):
    pass


class _FailingDetector:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    def detect(self, _text: str):
        self.calls += 1
        raise self.error


def _tool(name: str):
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_propagates_sentence_break_failure_without_retry(monkeypatch):
    normalization = importlib.import_module("frend.normalize")

    error = SentinelRegexError("sentence limit")
    calls = 0

    def fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(normalization, "break_sentence_spans", fail)

    with pytest.raises(SentinelRegexError) as caught:
        normalize("one")
    assert caught.value is error
    assert calls == 1


def test_normalize_propagates_detection_failure_without_retry(monkeypatch):
    normalization = importlib.import_module("frend.normalize")

    error = SentinelRegexError("detection limit")
    detector = _FailingDetector(error)
    monkeypatch.setattr(normalization, "_reading_detectors", lambda *_args: detector)

    with pytest.raises(SentinelRegexError) as caught:
        normalize("one")
    assert caught.value is error
    assert detector.calls == 1


def test_normalize_propagates_sentence_date_detection_failure_without_retry(monkeypatch):
    normalization = importlib.import_module("frend.normalize")

    error = SentinelRegexError("date detection limit")
    detector = _FailingDetector(error)
    monkeypatch.setattr(normalization, "_date_detector_gang", lambda *_args: detector)

    with pytest.raises(SentinelRegexError) as caught:
        normalize("one")
    assert caught.value is error
    assert detector.calls == 1


def test_resolve_lattice_propagates_detection_failure_without_retry(monkeypatch):
    import icukit.detectors

    error = SentinelRegexError("detection limit")
    calls = 0

    def fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(icukit.detectors, "detect", fail)

    with pytest.raises(SentinelRegexError) as caught:
        resolve_lattice(source_text="one")
    assert caught.value is error
    assert calls == 1


def test_none_numeric_duration_capture_falls_back_instead_of_raising():
    from icukit import FlexibleNumericDurationDetector

    written = "1:02"
    detections = FlexibleNumericDurationDetector("en_US").detect(written)
    detection = detections[0]
    captures = tuple(
        replace(capture, value=None) if capture.name in {"h", "m"} else capture
        for capture in detection["captures"]
    )

    lattice = resolve_lattice(
        [{**detection, "captures": captures}],
        source_text=written,
        max_input_chars=None,
        max_unit_chars=None,
    )
    result = verbalize_lattice(lattice, max_input_chars=None, max_unit_chars=None)

    assert result.best_path.units[0].verbalized is False
    assert result.best_path.units[0].best.provenance == "surface:unsupported"


def test_absent_overlong_fraction_candidate_remains_exact_surface_passthrough():
    written = "1/" + "9" * (sys.int_info.default_max_str_digits + 1)
    lattice = resolve_lattice(
        [],
        source_text=written,
        max_input_chars=None,
        max_unit_chars=None,
    )
    result = verbalize_lattice(lattice, max_input_chars=None, max_unit_chars=None)

    assert "".join(unit.best.text for unit in result.best_path.units) == written
    assert all(unit.verbalized is True for unit in result.best_path.units)
    assert all(unit.best.provenance == "surface:passthrough" for unit in result.best_path.units)


@pytest.mark.parametrize(
    ("tool_name", "operation", "invoke"),
    [
        (
            "build_spoken_priors",
            "detect number alternatives",
            lambda tool, detector: tool._alternatives("1", "number", {"number": (detector,)}),
        ),
        (
            "build_context_trees",
            "detect context-tree token",
            lambda tool, _detector: tool._lattice("1"),
        ),
        (
            "build_range_priors",
            "detect numeric duration",
            lambda tool, _detector: tool._duration_outcome("1:02", "one minute two seconds"),
        ),
    ],
)
def test_builders_name_icukit_detection_failures(monkeypatch, tool_name, operation, invoke):
    import icukit
    import icukit.detectors

    tool = _tool(tool_name)
    error = SentinelRegexError("stack limit")
    detector = _FailingDetector(error)
    if tool_name == "build_context_trees":
        tool._LATTICES.clear()
        monkeypatch.setattr(tool, "_detectors", lambda: ())
        monkeypatch.setattr(
            icukit.detectors,
            "detect",
            lambda *_args, **_kwargs: detector.detect("fixture"),
        )
    elif tool_name == "build_range_priors":
        monkeypatch.setattr(icukit, "FlexibleNumericDurationDetector", lambda _locale: detector)

    with pytest.raises(RuntimeError, match=operation) as caught:
        invoke(tool, detector)
    assert caught.value.__cause__ is error
    assert detector.calls == 1


def test_evaluator_records_detection_exception_type(monkeypatch):
    import frend

    evaluate = _tool("evaluate_google_tn")
    error = SentinelRegexError("time limit")
    monkeypatch.setattr(
        frend,
        "resolve_lattice",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    assert evaluate._score_text_status("1", "one") == (False, False, "SentinelRegexError")

    records = []
    monkeypatch.setattr(
        evaluate,
        "_map",
        lambda *_args: [("CARDINAL", False, False, "SentinelRegexError")],
    )
    report = evaluate._per_token(
        [[("CARDINAL", "1", "one")]],
        1,
        record_sink=records,
    )
    assert report["exception_counts"] == {"SentinelRegexError": 1}
    assert records[0]["exception_counts"] == {"SentinelRegexError": 1}


def test_triage_records_detection_exception_type_in_accuracy_only_output(monkeypatch):
    import icukit.detectors

    triage = _tool("triage_misses")
    error = SentinelRegexError("time limit")
    monkeypatch.setattr(triage, "_detectors", lambda _locale: ())
    monkeypatch.setattr(
        icukit.detectors,
        "detect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    sentence = (("CARDINAL", "1", "one"),)

    result = triage._score_sentence(("shard", 0, sentence, None, None, False))

    assert result["exception_counts"] == {"SentinelRegexError": 1}
