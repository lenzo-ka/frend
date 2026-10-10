from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect

import pytest

import frend
from frend import (
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_UNIT_CHARS,
    InputValidationError,
    compose_choices,
    resolve,
    resolve_choices,
    resolve_lattice,
    validate_input,
    verbalize_edge,
    verbalize_lattice,
)
from frend.context import TextContext
from frend.electronic import ElectronicDetector
from frend.fold_resolve import resolve_cover


@pytest.mark.parametrize("source_text", [None, b"123", 123, ["123"]])
def test_normalize_refuses_non_string_input_before_folding(monkeypatch, source_text):
    normalize_module = importlib.import_module("frend.normalize")

    def unexpected_fold(*args, **kwargs):
        pytest.fail("input folding ran before plain-text validation")

    monkeypatch.setattr(normalize_module, "apply_input_fold", unexpected_fold)

    with pytest.raises(InputValidationError, match="source_text must be a str"):
        frend.normalize(source_text)


@pytest.mark.parametrize("source_text", ["", " ", "\t\n"])
def test_normalize_retains_empty_and_whitespace_only_results(source_text):
    assert frend.normalize(source_text) == ""


def test_binary_and_core_file_like_input_is_refused_before_resolution():
    with pytest.raises(InputValidationError, match="including invalid UTF-8"):
        resolve_lattice([], source_text=b"\x7fELF\x02\x01\x00\xff")  # type: ignore[arg-type]
    with pytest.raises(InputValidationError, match="must be a str decoded from valid UTF-8"):
        resolve_lattice([], source_text=b"plain")  # type: ignore[arg-type]


def test_nul_is_refused_at_every_public_text_bearing_entry_point():
    lattice = resolve_lattice([], source_text="x")
    choices = resolve_choices([], source_text="x")
    bad_lattice = dataclasses.replace(lattice, source_text="x\x00")
    bad_choices = dataclasses.replace(choices, source_text="x\x00")

    calls = [
        lambda: resolve([], source_text="x\x00"),
        lambda: resolve_cover([], source_text="x\x00"),
        lambda: resolve_choices([], source_text="x\x00"),
        lambda: resolve_lattice([], source_text="x\x00"),
        lambda: compose_choices(bad_choices),
        lambda: verbalize_edge(lattice.edges[0], source_text="x\x00"),
        lambda: verbalize_lattice(bad_lattice),
        lambda: verbalize_edge(lattice.edges[0], source_text="x", context=TextContext("x\x00")),
        lambda: verbalize_lattice(lattice, context=TextContext("x\x00")),
    ]
    for call in calls:
        with pytest.raises(InputValidationError, match="NUL"):
            call()


def test_public_lattice_apis_expose_no_validation_bypass():
    assert "_input_validated" not in inspect.signature(resolve_lattice).parameters
    assert "_input_validated" not in inspect.signature(verbalize_lattice).parameters


def test_default_document_budget_refuses_without_building_a_huge_lattice():
    text = "x" * (DEFAULT_MAX_INPUT_CHARS + 1)
    with pytest.raises(InputValidationError, match=r"4194305.*max_input_chars=4194304"):
        resolve_lattice([], source_text=text)


@pytest.mark.parametrize("char", ["\x01", "\u0378"])
def test_high_non_text_share_is_refused(char):
    with pytest.raises(InputValidationError, match="non-text share"):
        validate_input("ordinary" + char)


def test_lone_surrogate_is_refused_independently_of_share():
    with pytest.raises(InputValidationError, match="lone surrogate"):
        validate_input("a" * 10_000 + "\ud800")


def test_non_text_share_threshold_is_inclusive_at_exactly_one_percent():
    assert validate_input("a" * 99 + "\x01") == "a" * 99 + "\x01"
    with pytest.raises(InputValidationError, match=r"2/199.*1.01%"):
        validate_input("a" * 197 + "\x01\x02")


def test_multilingual_format_private_use_and_whitespace_text_is_accepted():
    text = "漢字 ไทย e\u0301 👩\u200d💻️ \u200eRTL\u200f \ue000\t\n\r\f\v\x85"
    assert validate_input(text) == text


def test_budget_can_be_lowered_or_disabled_but_plain_text_guard_remains():
    with pytest.raises(InputValidationError, match="max_input_chars=3"):
        validate_input("four", max_input_chars=3)
    assert validate_input("four", max_input_chars=None) == "four"
    with pytest.raises(InputValidationError, match="NUL"):
        validate_input("four\x00", max_input_chars=None)


def test_under_budget_and_budget_disabled_output_match_main_byte_for_byte():
    text = "Ordinary prose."
    default = verbalize_lattice(resolve_lattice([], source_text=text))
    disabled = verbalize_lattice(
        resolve_lattice([], source_text=text, max_input_chars=None),
        max_input_chars=None,
    )
    expected = "e134b486e7b08b53c37948f26b0b8a8cba1cc656cc76f11964f9392795185f32"
    assert repr(default) == repr(disabled)
    assert hashlib.sha256(repr(default).encode()).hexdigest() == expected


def test_recognized_reading_output_matches_main_byte_for_byte():
    text = "Email jane.doe@example.org today."
    detections = ElectronicDetector().detect(text)
    result = verbalize_lattice(resolve_lattice(detections, source_text=text))
    expected = "49612cab37913d15a9cc50baf1da02de1a2fd3bd5280bcdb4cd447e2dc0fd0a4"
    assert hashlib.sha256(repr(result).encode()).hexdigest() == expected


@pytest.mark.parametrize("resolver", [resolve_choices, resolve_lattice])
def test_omitted_source_uses_detection_extent_for_unit_bound(resolver):
    detection = {
        "start": 0,
        "end": DEFAULT_MAX_UNIT_CHARS + 1,
        "type": "unsupported:test",
        "text": "x",
    }
    with pytest.raises(InputValidationError, match=r"8193.*max_unit_chars=8192.*sentence-break"):
        resolver([detection])
    with pytest.raises(InputValidationError, match=r"8193.*max_input_chars=1.*sentence-break"):
        resolver([detection], max_input_chars=1, max_unit_chars=None)


@pytest.mark.parametrize("resolver", [resolve_choices, resolve_lattice])
def test_resolution_unit_bound_is_separate_and_configurable(resolver):
    text = "x" * (DEFAULT_MAX_UNIT_CHARS + 1)
    with pytest.raises(InputValidationError, match="sentence-break"):
        resolver([], source_text=text)
    assert resolver([], source_text=text, max_unit_chars=None).text_length == len(text)


def test_verbalize_paths_enforce_the_unit_bound():
    text = "x" * (DEFAULT_MAX_UNIT_CHARS + 1)
    lattice = resolve_lattice([], source_text=text, max_unit_chars=None)
    choices = resolve_choices([], source_text=text, max_unit_chars=None)
    with pytest.raises(InputValidationError, match="sentence-break"):
        verbalize_lattice(lattice)
    with pytest.raises(InputValidationError, match="sentence-break"):
        verbalize_edge(lattice.edges[0], source_text=text)
    with pytest.raises(InputValidationError, match="sentence-break"):
        compose_choices(choices)
