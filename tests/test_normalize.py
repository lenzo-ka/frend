from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from tiergraph import WorkBudget, WorkMeter

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


@pytest.mark.parametrize(
    ("locale", "digit", "day", "year"),
    [
        ("en_US", "one", "fifth", "twenty-four"),
        ("de_DE", "eins", "fünf", "zweitausendvierundzwanzigste"),
    ],
)
def test_keycap_digit_is_spoken_without_hiding_a_following_date(locale, digit, day, year):
    assert frend.normalize("1️⃣", locale=locale).strip() == digit

    sentence = frend.normalize("Call 1️⃣ on March 5, 2024.", locale=locale)
    assert {"call", "on", "march"} <= set(sentence.casefold().split())
    assert digit in sentence.split()
    assert day in sentence
    assert year in sentence


@pytest.mark.parametrize("locale", ["en_US", "de_DE", "fr_FR", "ja_JP"])
@pytest.mark.parametrize("base", [*"0123456789", "#", "*"])
@pytest.mark.parametrize("variation_selector", ["", "\ufe0f"])
def test_every_keycap_reads_exactly_as_its_base(locale, base, variation_selector):
    keycap = f"{base}{variation_selector}\u20e3"

    assert frend.normalize(keycap, locale=locale) == frend.normalize(base, locale=locale)


def test_normalize_shares_one_work_meter_across_sentences(monkeypatch):
    normalize_module = importlib.import_module("frend.normalize")

    meters = []

    def sentence(text, **kwargs):
        meters.append(kwargs["work_budget"])
        return None, text

    monkeypatch.setattr(normalize_module, "_sentence", sentence)
    meter = WorkMeter(WorkBudget(steps=100))
    assert frend.normalize("First. Second.", work_budget=meter) == "First. Second."
    assert meters == [meter, meter]


def test_missing_optional_detector_family_degrades_only_that_family():
    plain = frend.normalize("12", locale="ar_EG")
    result = frend.normalize("12", locale="ar_EG", offsets=True)
    retained = frend.normalize("١ رطل و٢ أونصة", locale="ar_EG", offsets=True)

    assert plain
    assert isinstance(result, NormalizedText)
    assert result.text == plain
    assert result.missing_detector_families == ("measure:foot-and-inch",)
    assert isinstance(retained, NormalizedText)
    assert [unit.reader for unit in retained.units] == ["measure:pound-and-ounce"]


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


def _assert_alignment_tiles(source: str, result: NormalizedText):
    output_at = 0
    source_at = 0
    for unit in result.units:
        assert unit.output_span[0] == output_at
        assert unit.source_span[0] == source_at
        output_at = unit.output_span[1]
        source_at = unit.source_span[1]
    assert output_at == len(result.text)
    assert source_at == len(source)


@pytest.mark.parametrize(
    ("source", "without_boundary_whitespace"),
    [
        ("\nPlain text.", "Plain text."),
        ("  a.  b.  ", "a.  b."),
        ("a.\n\nb", "a. b"),
        (" \n ", ""),
    ],
)
def test_boundary_whitespace_is_trimmed_or_collapsed_without_losing_source_coverage(
    source, without_boundary_whitespace
):
    assert frend.normalize(source) == frend.normalize(without_boundary_whitespace)
    result = frend.normalize(source, offsets=True)
    assert isinstance(result, NormalizedText)
    _assert_alignment_tiles(source, result)
    for unit in result.units:
        source_text = source[slice(*unit.source_span)]
        if unit.provenance.endswith("whitespace"):
            assert source_text.isspace()
            output_text = result.text[slice(*unit.output_span)]
            assert output_text == (" " if "inter-sentence" in unit.provenance else "")


def test_trailing_text_without_terminal_punctuation_is_preserved():
    source = "a. trailing text without terminal punctuation"
    result = frend.normalize(source, offsets=True)
    assert isinstance(result, NormalizedText)
    _assert_alignment_tiles(source, result)
    assert result.units[-1].source_span[1] == len(source)
    assert result.text.endswith("punctuation")


def test_non_ascii_offsets_are_code_point_exact_and_tile_both_texts():
    source = "\U0001f600 \U00010400 e\u0301 \u6f22\u5b57."
    result = frend.normalize(source, offsets=True)
    assert isinstance(result, NormalizedText)
    _assert_alignment_tiles(source, result)
    source_slices = [source[slice(*unit.source_span)] for unit in result.units]
    assert source_slices == list(source)
    assert "".join(source_slices) == source


def test_typographic_fold_is_length_preserving_declared_and_slices_raw_text():
    source = "’94\u2010’95, “quoted”, 5\u00a0km, and 26\u221227."
    folded = frend.apply_input_fold(source)
    assert folded == "'94-'95, \"quoted\", 5 km, and 26-27."
    assert len(folded) == len(source)

    result = frend.normalize(source, offsets=True)
    assert isinstance(result, NormalizedText)
    assert result.fold == "typographic"
    assert result.text == frend.normalize(folded, fold=None)
    _assert_alignment_tiles(source, result)
    for unit in result.units:
        assert source[slice(*unit.source_span)]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201b", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u201f", '"'),
        ("\u00a0", " "),
        ("\u2007", " "),
        ("\u2009", " "),
        ("\u202f", " "),
        ("\u2010", "-"),
        ("\u2011", "-"),
        ("\u2212", "-"),
        ("\u2013", "\u2013"),
    ],
)
def test_typographic_fold_has_the_declared_one_code_point_mapping(source, expected):
    assert frend.apply_input_fold(source) == expected
    assert len(source) == len(expected)


@pytest.mark.parametrize(
    ("typographic", "ascii_twin"),
    [
        ("’94", "'94"),
        ("’94–’95", "'94-'95"),
        ("“quoted”", '"quoted"'),
        ("5\u00a0km", "5 km"),
        ("26–27", "26-27"),
        ("June 26–27", "June 26-27"),
    ],
)
def test_typographic_examples_read_as_their_folded_twins(typographic, ascii_twin):
    assert frend.normalize(typographic) == frend.normalize(ascii_twin, fold=None)


def test_project_authored_typographic_sentences_have_exact_outputs():
    path = Path(__file__).parent / "data" / "typographic_fold_synthetic.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["authorship"] == "project-authored"

    failed_without_fold = []
    for item in document["items"]:
        with_fold = frend.normalize(item["written"])
        without_fold = frend.normalize(item["written"], fold=None)
        assert with_fold == item["expected_spoken"]
        assert without_fold == item["expected_without_fold"]
        assert (without_fold != item["expected_spoken"]) is item["requires_fold"]
        if without_fold != item["expected_spoken"]:
            failed_without_fold.append(item["name"])
    assert failed_without_fold == [
        "curly-quotation",
        "minus-range",
        "nonbreaking-hyphen-range",
    ]


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

    electronic_module = importlib.import_module("frend.electronic")
    lattice_module = importlib.import_module("frend.lattice")
    normalize_module = importlib.import_module("frend.normalize")
    verbalize_module = importlib.import_module("frend.verbalize")
    original = normalize_module.validate_input
    calls = []

    def recording_validate(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    for module in (normalize_module, electronic_module, lattice_module, verbalize_module):
        monkeypatch.setattr(module, "validate_input", recording_validate)
    text = "a" * 100 + ". " + "b" * 10 + "\x01" + "b" * 10 + "."
    assert frend.normalize(text)
    assert len(calls) == 1


def test_sentence_range_protection_advances_through_acronym_spans_once(monkeypatch):
    import importlib

    normalize_module = importlib.import_module("frend.normalize")
    sentence_count = 100
    sentence_length = 10
    text = "x" * (sentence_count * sentence_length)
    spans = [
        {
            "text": text[start : start + sentence_length],
            "start": start,
            "end": start + sentence_length,
        }
        for start in range(0, len(text), sentence_length)
    ]
    protected = tuple(
        (start - 2, start + 2) for start in range(sentence_length, len(text), sentence_length)
    )

    class CountingSpans:
        def __init__(self, values):
            self.values = values
            self.accesses = 0

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            self.accesses += 1
            return self.values[index]

    counting = CountingSpans(protected)

    monkeypatch.setattr(normalize_module, "break_sentence_spans", lambda *_args: spans)
    monkeypatch.setattr(normalize_module, "_dotted_acronym_suffix_spans", lambda _text: counting)

    assert normalize_module._sentence_ranges(text, "en_US") == [(0, len(text))]
    assert counting.accesses <= 5 * sentence_count


def test_text_mode_skips_offset_record_construction(monkeypatch):
    import importlib

    normalize_module = importlib.import_module("frend.normalize")

    monkeypatch.setattr(
        normalize_module,
        "NormalizedUnit",
        lambda *_args, **_kwargs: pytest.fail("text mode constructed offset bookkeeping"),
    )
    assert isinstance(frend.normalize("12"), str)
