"""Declared, length-preserving folds applied before text recognition."""

from __future__ import annotations

from typing import Literal

__all__ = ["InputFold", "apply_input_fold"]

InputFold = Literal["typographic"]

# Every replacement is exactly one code point. U+2013 EN DASH is deliberately absent:
# icukit 0.8.0's generated en_US date-interval readers accept ICU's en dash but do not
# accept a hyphen-minus in the same patterns. Frend's own range reader already handles
# both surfaces where it has evidence, without making them equivalent everywhere.
_TYPOGRAPHIC = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201f": '"',
        "\u00a0": " ",
        "\u2007": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2212": "-",
        "\uff10": "0",
        "\uff11": "1",
        "\uff12": "2",
        "\uff13": "3",
        "\uff14": "4",
        "\uff15": "5",
        "\uff16": "6",
        "\uff17": "7",
        "\uff18": "8",
        "\uff19": "9",
    }
)

_ASCII_DIGITS = frozenset("0123456789")
_FULLWIDTH_DIGITS = frozenset("０１２３４５６７８９")
_DECIMAL_DIGITS = _ASCII_DIGITS | _FULLWIDTH_DIGITS
_BETWEEN_DIGITS = {"，": ",", "．": "."}
_NEXT_TO_DIGIT = {"％": "%", "＋": "+", "－": "-"}

# Fold only unit symbols the installed measure readers already recognize, and only as
# a fullwidth Latin run immediately following a number. This deliberately does not
# make fullwidth Latin text generally equivalent to ASCII (for example, ``1Ａ``).
_MEASURE_UNIT_SYMBOLS = frozenset(
    {
        "cm",
        "ft",
        "g",
        "GB",
        "GHz",
        "GW",
        "h",
        "ha",
        "hp",
        "Hz",
        "in",
        "kB",
        "kg",
        "kHz",
        "km",
        "kW",
        "L",
        "lb",
        "m",
        "MB",
        "mbar",
        "mg",
        "MHz",
        "mi",
        "min",
        "mL",
        "mm",
        "mph",
        "nm",
        "oz",
        "TB",
        "V",
        "W",
        "yd",
    }
)


def _is_fullwidth_latin(char: str) -> bool:
    return "Ａ" <= char <= "Ｚ" or "ａ" <= char <= "ｚ"


def _ascii_fullwidth_latin(text: str) -> str:
    return "".join(chr(ord(char) - 0xFEE0) for char in text)


def _contextual_number_fold(text: str, folded: list[str]) -> None:
    """Fold punctuation and unit letters only where they are part of a number."""
    for index, char in enumerate(text):
        before_digit = index > 0 and text[index - 1] in _DECIMAL_DIGITS
        after_digit = index + 1 < len(text) and text[index + 1] in _DECIMAL_DIGITS
        if char in _BETWEEN_DIGITS and before_digit and after_digit:
            folded[index] = _BETWEEN_DIGITS[char]
        elif char in _NEXT_TO_DIGIT and (before_digit or after_digit):
            folded[index] = _NEXT_TO_DIGIT[char]

    index = 1
    while index < len(text):
        if text[index - 1] not in _DECIMAL_DIGITS or not _is_fullwidth_latin(text[index]):
            index += 1
            continue
        end = index + 1
        while end < len(text) and _is_fullwidth_latin(text[end]):
            end += 1
        unit = _ascii_fullwidth_latin(text[index:end])
        if unit in _MEASURE_UNIT_SYMBOLS:
            folded[index:end] = unit
        index = end


def apply_input_fold(text: str, fold: InputFold | None = "typographic") -> str:
    """Return ``text`` after the declared input ``fold``.

    ``None`` disables folding. The typographic fold is code-point-length preserving,
    so every offset in the folded recognition text is also an offset in the raw text.
    """
    if fold is None:
        return text
    if fold != "typographic":
        raise ValueError(f"unknown input fold: {fold!r}")
    folded_chars = list(text.translate(_TYPOGRAPHIC))
    _contextual_number_fold(text, folded_chars)
    folded = "".join(folded_chars)
    assert len(folded) == len(text)
    return folded
