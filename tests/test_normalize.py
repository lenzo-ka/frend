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
    ("canonical", "aliases", "localized_reading"),
    [
        ("en_US", ("en-US", "EN_us"), " one hundred twenty-three "),
        ("es_MX", ("es-mx", "ES_mx"), " ciento veintitrés "),
        ("es_ES", ("es-es", "ES_es"), " ciento veintitrés "),
        ("fr_FR", ("fr-fr", "FR_fr"), " cent vingt-trois "),
        ("de_DE", ("de-de", "DE_de"), " einhundertdreiundzwanzig "),
        ("pt_BR", ("pt-br", "PT_br"), " cento e vinte e três "),
        ("it_IT", ("it-it", "IT_it"), " centoventitré "),
        ("zh_CN", ("zh-cn", "ZH_cn"), " 一百二十三 "),
        ("ko_KR", ("ko-kr", "KO_kr"), " 백이십삼 "),
        ("ja_JP", ("ja-jp", "JA_jp"), " 百二十三 "),
        ("pt_PT", ("pt-pt", "PT_pt"), " cento e vinte e três "),
    ],
)
def test_normalize_locale_aliases_are_byte_identical(canonical, aliases, localized_reading):
    expected = frend.normalize("123", locale=canonical)
    assert expected == localized_reading

    for alias in aliases:
        assert frend.normalize("123", locale=alias) == expected, alias


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("Chapter I", "one"),
        ("Chapter V", "five"),
        ("Chapter IV", "four"),
        ("Chapter IX", "nine"),
        ("Act IV", "four"),
        ("Volume IV", "four"),
        ("Super Bowl LVIII", "fifty-eight"),
    ],
)
def test_roman_numerals_after_title_cues_read_as_numbers(written, spoken):
    assert frend.normalize(written).split()[-1] == spoken


@pytest.mark.parametrize("locale", ["en_US", "fr_FR", "ja_JP"])
@pytest.mark.parametrize("abbreviation", ["U.S.A.", "Dr."])
@pytest.mark.parametrize("punctuation", [",", "!", "?", ":", ";", "…", ")", "]", "}"])
def test_period_ending_abbreviation_preserves_following_punctuation(
    locale, abbreviation, punctuation
):
    expected_reading = frend.normalize(abbreviation, locale=locale).rstrip()
    result = frend.normalize(f"{abbreviation}{punctuation}", locale=locale).rstrip()

    assert result.endswith(punctuation)
    assert result.removesuffix(punctuation).rstrip() == expected_reading


@pytest.mark.parametrize("acronym", ["F.B.I.s", "P.C.s"])
def test_sentence_period_preserves_a_dotted_acronyms_plural_reading(acronym):
    expected_reading = frend.normalize(acronym).rstrip()

    for punctuation in [",", "!", "?", "."]:
        assert frend.normalize(f"{acronym}{punctuation}").rstrip() == (
            f"{expected_reading} {punctuation}"
        )


def test_genuine_dotted_lexicon_abbreviation_retains_precedence():
    assert frend.normalize("Ph.D.") == " Doctor of Philosophy "


@pytest.mark.parametrize("written", ["COVID-19", "F-16", "UTF-8", "Q4-2024", "ISO-8859-1"])
def test_hyphenated_alphanumeric_identifier_is_not_negative_or_a_range(written):
    spoken = frend.normalize(written)

    assert "minus" not in spoken
    assert " to " not in spoken


def test_leading_negative_and_numeric_range_keep_their_readings():
    assert frend.normalize("-19") == " minus nineteen "
    assert frend.normalize("4-2024") == " four twenty twenty-four "
    assert frend.normalize("1-2-3") == " one  minus two  minus three "


@pytest.mark.parametrize("separator", ["-", "–"])
def test_period_bearing_month_abbreviations_normalize_like_periodless_twins(separator):
    periodless = frend.normalize(f"Dec 31{separator}Jan 2")

    assert frend.normalize(f"Dec. 31{separator}Jan. 2") == periodless
    assert periodless == " December thirty-first to January second "
    assert frend.normalize(f"Sept. 3{separator}5") == " September third to fifth "
    assert frend.normalize(f"Sept.{separator}Oct. 2024") == (
        " September to October twenty twenty-four "
    )


def test_period_bearing_month_range_support_does_not_change_other_periods():
    assert frend.normalize("Dec.") == " December "
    assert frend.normalize("The deadline is Dec. 31.") == (
        " the  deadline  is   December thirty-first ."
    )


@pytest.mark.parametrize("apostrophe", ["'", "’"])
@pytest.mark.parametrize("punctuation", [",", ".", "!", "?", ":", ";"])
def test_plural_possessive_apostrophe_is_silent_and_keeps_following_punctuation(
    apostrophe, punctuation
):
    result = frend.normalize(f"teachers{apostrophe}{punctuation}").rstrip()

    assert result == f"teachers'{punctuation}"


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


def test_source_boundary_whitespace_is_discarded_without_trimming_reader_emissions():
    without_source_padding = frend.normalize("12")
    with_source_padding = frend.normalize("  12  ")

    assert without_source_padding == " twelve "
    assert with_source_padding == without_source_padding

    result = frend.normalize("  12  ", offsets=True)
    assert isinstance(result, NormalizedText)
    boundary_units = [
        unit for unit in result.units if unit.provenance == "surface:boundary-whitespace"
    ]
    assert [(unit.source_span, unit.output_span) for unit in boundary_units] == [
        ((0, 2), (0, 0)),
        ((4, 6), (8, 8)),
    ]


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
    ("locale", "one_kilogram"),
    [
        ("en_US", " one kilogram "),
        ("ja_JP", " 一 キログラム "),
        ("zh_CN", " 一千克 "),
    ],
)
def test_fullwidth_decimal_digits_fold_to_ascii_numbers_and_measures(locale, one_kilogram):
    ascii_digits = "0123456789"
    fullwidth_digits = "０１２３４５６７８９"

    assert frend.apply_input_fold(fullwidth_digits) == ascii_digits
    assert frend.apply_input_fold(fullwidth_digits, fold=None) == fullwidth_digits
    assert frend.normalize(fullwidth_digits, locale=locale, fold=None) == fullwidth_digits
    assert frend.normalize(fullwidth_digits, locale=locale) == frend.normalize(
        ascii_digits, locale=locale, fold=None
    )
    assert frend.normalize(fullwidth_digits + "kg", locale=locale) == frend.normalize(
        ascii_digits + "kg", locale=locale, fold=None
    )

    measure = frend.normalize("１kg", locale=locale, offsets=True)
    assert isinstance(measure, NormalizedText)
    assert measure.text == one_kilogram
    assert [unit.reader for unit in measure.units] == ["measure:kilogram"]
    assert [unit.source_span for unit in measure.units] == [(0, 3)]

    for ascii_digit, fullwidth_digit in zip(ascii_digits, fullwidth_digits, strict=True):
        result = frend.normalize(fullwidth_digit, locale=locale, offsets=True)
        assert isinstance(result, NormalizedText)
        assert result.text == frend.normalize(ascii_digit, locale=locale, fold=None)
        assert [unit.source_span for unit in result.units] == [(0, 1)]
        assert fullwidth_digit[slice(*result.units[0].source_span)] == fullwidth_digit


@pytest.mark.parametrize("locale", ["en_US", "ja_JP", "zh_CN"])
@pytest.mark.parametrize(
    ("fullwidth", "ascii_twin"),
    [
        ("３．５km", "3.5km"),
        ("５０％", "50%"),
        ("１，２３４", "1,234"),
        ("３．５ｋｍ", "3.5km"),
    ],
)
def test_fullwidth_number_punctuation_and_units_read_as_ascii_twins(locale, fullwidth, ascii_twin):
    assert frend.normalize(fullwidth, locale=locale) == frend.normalize(
        ascii_twin, locale=locale, fold=None
    )


def test_fullwidth_number_punctuation_is_contextual_and_length_preserving():
    source = "＋５ ５－６ ５％ ３．５ １，２３４ ３．５ｋｍ"
    expected = "+5 5-6 5% 3.5 1,234 3.5km"
    assert frend.apply_input_fold(source) == expected
    assert len(source) == len(expected)

    result = frend.normalize(source, offsets=True)
    assert isinstance(result, NormalizedText)
    _assert_alignment_tiles(source, result)


def test_fullwidth_cjk_punctuation_and_non_unit_letters_stay_fullwidth():
    source = "語，語。 語．語 １Ａ"
    assert frend.apply_input_fold(source) == source.replace("１", "1")


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


@pytest.mark.parametrize(
    ("source", "quote"),
    [('"1"', '"'), ("'1'", "'"), ("“1”", '"'), ("‘1’", "'")],
)
@pytest.mark.parametrize("locale", ["en_US", "fr_FR", "ja_JP"])
def test_paired_quotes_around_integer_remain_quotes(source, quote, locale):
    assert frend.normalize(source, locale=locale) == (
        quote + frend.normalize("1", locale=locale) + quote
    )


@pytest.mark.parametrize(
    ("locale", "foot", "inch", "mixed"),
    [
        ("en_US", " one foot ", " one inch ", " five feet, two inches "),
        ("fr_FR", " un\u00a0pied ", " un\u00a0pouce ", " cinq pieds et deux\u00a0pouces "),
        ("ja_JP", " 一 フィート ", " 一 インチ ", " 五 フィート 二 インチ "),
    ],
)
def test_prime_measures_remain_unambiguous(locale, foot, inch, mixed):
    assert frend.normalize("1′", locale=locale) == foot
    assert frend.normalize("1″", locale=locale) == inch
    assert frend.normalize("5′2″", locale=locale) == mixed


@pytest.mark.parametrize(
    ("source", "spoken"),
    [
        ("1 ft", " one foot "),
        ("2 lbs", " two pounds "),
        ("60 mph", " sixty miles per hour "),
        ("100 sq ft", " one hundred square feet "),
        ("60 km/h", " sixty kilometers per hour "),
        ("2 mi²", " two square miles "),
        ("1 lb", " one pound "),
        ("1 mph", " one mile per hour "),
        ("1 sq ft", " one square foot "),
        ("1 mi²", " one square mile "),
    ],
)
def test_public_normalize_reads_common_english_unit_abbreviations(source, spoken):
    assert frend.normalize(source) == spoken


@pytest.mark.parametrize(
    ("source", "reader", "spoken"),
    [
        ("1 sq yd", "measure:square-yard", " one square yard "),
        ("2 sq yd", "measure:square-yard", " two square yards "),
        ("1 sq in", "measure:square-inch", " one square inch "),
        ("2 sq in", "measure:square-inch", " two square inches "),
        ("1 cu ft", "measure:cubic-foot", " one cubic foot "),
        ("2 cu ft", "measure:cubic-foot", " two cubic feet "),
    ],
)
def test_common_area_and_volume_units_use_one_measure_span(source, reader, spoken):
    normalized = frend.normalize(source, locale="en_US", offsets=True)

    assert isinstance(normalized, NormalizedText)
    assert normalized.text == spoken
    assert [(unit.reader, unit.source_span) for unit in normalized.units] == [
        (reader, (0, len(source)))
    ]


@pytest.mark.parametrize(
    ("locale", "degree", "celsius", "fahrenheit"),
    [
        ("en_US", " one degree ", " one degree Celsius ", " one degree Fahrenheit "),
        ("es_MX", " uno grado ", " uno grado Celsius ", " uno grado Fahrenheit "),
        ("es_ES", " uno grado ", " uno grado Celsius ", " uno grado Fahrenheit "),
        (
            "fr_FR",
            " un\N{NO-BREAK SPACE}degré ",
            " un\N{NO-BREAK SPACE}degré Celsius ",
            " un\N{NO-BREAK SPACE}degré Fahrenheit ",
        ),
        ("de_DE", " eins Grad ", " eins Grad Celsius ", " eins Grad Fahrenheit "),
        ("pt_BR", " um grau ", " um grau Celsius ", " um grau Fahrenheit "),
        ("it_IT", " uno grado ", " uno grado Celsius ", " uno grado Fahrenheit "),
        ("zh_CN", " 一度 ", " 一摄氏度 ", " 一华氏度 "),
        ("ko_KR", " 일도 ", " 섭씨 일도 ", " 화씨 일도 "),
        ("ja_JP", " 一 度 ", " 摂氏 一 度 ", " 華氏 一 度 "),
        ("pt_PT", " um grau ", " um grau Celsius ", " um grau Fahrenheit "),
    ],
)
def test_bare_degree_sign_does_not_invent_a_temperature_scale(locale, degree, celsius, fahrenheit):
    bare = frend.normalize("1°", locale=locale, offsets=True)
    assert isinstance(bare, NormalizedText)
    assert bare.text == degree
    assert {unit.reader for unit in bare.units} == {"measure:degree"}
    assert not {"measure:celsius", "measure:fahrenheit"} & {unit.reader for unit in bare.units}

    assert frend.normalize("1°C", locale=locale) == celsius
    assert frend.normalize("1°F", locale=locale) == fahrenheit


def test_bare_degree_range_does_not_invent_a_temperature_scale():
    normalized = frend.normalize("1°–2°", locale="en_US", offsets=True)
    assert isinstance(normalized, NormalizedText)
    assert normalized.text == " one degree to two degrees "
    assert {unit.reader for unit in normalized.units} == {"measure:range"}


@pytest.mark.parametrize(
    ("source", "spoken"),
    [
        ("45°N", " forty-five degrees north "),
        ("122°W", " one hundred twenty-two degrees west "),
        ("45° 30′ N", " forty-five degrees thirty minutes north "),
        ("45°", " forty-five degrees "),
    ],
)
def test_compass_coordinates_read_as_degrees_and_direction(source, spoken):
    assert frend.normalize(source, locale="en_US") == spoken


@pytest.mark.parametrize("locale", ["de_DE", "fr_FR", "ja_JP"])
def test_compass_coordinate_is_not_invented_without_a_localized_reading(locale):
    assert frend.normalize("45°N", locale=locale).rstrip().endswith("°N")


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
