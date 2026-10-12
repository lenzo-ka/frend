"""English scientific notation is one semantic reading, not operator fragments."""

from __future__ import annotations

import sys

import pytest

from frend import normalize
from frend.scientific import ScientificNotationDetector
from frend.spoken_priors import normalize_spoken


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("6.02e23", "six point o two times ten to the twenty third"),
        ("6.02E+23", "six point o two times ten to the twenty third"),
        ("6.02e-23", "six point o two times ten to the minus twenty third"),
        ("1.5×10^6", "one point five times ten to the sixth"),
    ],
)
def test_scientific_notation_is_one_english_reading_span(written, spoken):
    result = normalize(written, locale="en_US", offsets=True)

    assert normalize_spoken(result.text) == spoken
    assert [(unit.source_span, unit.reader) for unit in result.units] == [
        ((0, len(written)), "number:scientific")
    ]
    tokens = normalize_spoken(result.text).split()
    assert "e" not in tokens
    assert not any(operator in result.text for operator in ("E", "×", "^"))


@pytest.mark.parametrize("marker", ["e", "E"])
@pytest.mark.parametrize(("sign", "spoken_sign"), [("+", ""), ("-", "minus "), ("−", "minus ")])
def test_e_notation_speaks_a_negative_exponent_sign(marker, sign, spoken_sign):
    result = normalize(f"3{marker}{sign}4", locale="en_US", offsets=True)

    assert normalize_spoken(result.text) == f"three times ten to the {spoken_sign}fourth"
    assert [(unit.source_span, unit.reader) for unit in result.units] == [
        ((0, 4), "number:scientific")
    ]


@pytest.mark.parametrize("operator", ["×", "x", "·", "*"])
def test_times_ten_accepts_each_multiplication_operator(operator):
    written = f"2 {operator} 10^-3"
    result = normalize(written, locale="en_US", offsets=True)

    assert normalize_spoken(result.text) == "two times ten to the minus third"
    assert [(unit.source_span, unit.reader) for unit in result.units] == [
        ((0, len(written)), "number:scientific")
    ]


@pytest.mark.parametrize(
    ("exponent", "spoken"),
    [
        ("³", "third"),
        ("⁺³", "third"),
        ("+³", "third"),
        ("⁻³", "minus third"),
        ("-³", "minus third"),
        ("−³", "minus third"),
    ],
)
def test_times_ten_accepts_superscript_exponents(exponent, spoken):
    written = f"2×10{exponent}"
    result = normalize(written, locale="en_US", offsets=True)

    assert normalize_spoken(result.text) == f"two times ten to the {spoken}"
    assert [(unit.source_span, unit.reader) for unit in result.units] == [
        ((0, len(written)), "number:scientific")
    ]


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("3E-4 m", "three times ten to the minus fourth meters"),
        ("2 x 10^−3 kg", "two times ten to the minus third kilograms"),
        ("1e0 m", "one times ten to the zeroth meter"),
    ],
)
def test_scientific_notation_composes_with_icu_measure_units(written, spoken):
    result = normalize(written, locale="en_US", offsets=True)

    assert normalize_spoken(result.text) == spoken
    assert [(unit.source_span, unit.reader) for unit in result.units] == [
        ((0, len(written)), "number:scientific")
    ]


@pytest.mark.parametrize(
    ("written", "spoken"),
    [("e23", " e twenty-three "), ("2×3", " two by three "), ("10^6", " ten ^ six ")],
)
def test_scientific_detector_leaves_other_expressions_unchanged(written, spoken):
    assert normalize(written, locale="en_US") == spoken


@pytest.mark.parametrize("written", ["e2e4", "the 2e version", "0x1e3", "A1e"])
def test_scientific_lookalikes_are_not_read_as_scientific_numbers(written):
    result = normalize(written, locale="en_US", offsets=True)

    assert all(unit.reader != "number:scientific" for unit in result.units)


def test_scientific_notation_is_us_english_only():
    assert ScientificNotationDetector("en_GB").detect("6.02e23") == []


@pytest.mark.parametrize("minus", ["-", "−"])
def test_scientific_notation_preserves_negative_zero_mantissa(minus):
    result = normalize(f"{minus}0e3", locale="en_US", offsets=True)

    assert normalize_spoken(result.text) == "minus zero times ten to the third"
    assert [(unit.source_span, unit.reader) for unit in result.units] == [
        ((0, 4), "number:scientific")
    ]


@pytest.mark.parametrize(
    "written",
    [
        "1e{digits}",
        "1 x 10^{digits}",
        "1×10{super_digits}",
    ],
)
def test_scientific_detector_skips_unrepresentable_exponents_atomically(written):
    limit = sys.int_info.default_max_str_digits
    text = written.format(digits="9" * (limit + 1), super_digits="⁹" * (limit + 1))

    detections = ScientificNotationDetector().detect(f"2e3 {text} 4e5")

    assert [(item["text"], item["start"], item["end"]) for item in detections] == [
        ("2e3", 0, 3),
        ("4e5", len(text) + 5, len(text) + 8),
    ]
    assert [item["value"].exponent for item in detections] == [3, 5]


def test_scientific_detector_uses_current_integer_conversion_limit():
    original_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        text = "1e" + "9" * 641

        assert ScientificNotationDetector().detect(text) == []
    finally:
        sys.set_int_max_str_digits(original_limit)


def test_scientific_detector_honors_disabled_integer_conversion_limit():
    original_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        text = "1e" + "9" * (sys.int_info.default_max_str_digits + 1)

        detections = ScientificNotationDetector().detect(text)
    finally:
        sys.set_int_max_str_digits(original_limit)

    assert [item["text"] for item in detections] == [text]
