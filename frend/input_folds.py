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
    }
)


def apply_input_fold(text: str, fold: InputFold | None = "typographic") -> str:
    """Return ``text`` after the declared input ``fold``.

    ``None`` disables folding. The typographic fold is code-point-length preserving,
    so every offset in the folded recognition text is also an offset in the raw text.
    """
    if fold is None:
        return text
    if fold != "typographic":
        raise ValueError(f"unknown input fold: {fold!r}")
    folded = text.translate(_TYPOGRAPHIC)
    assert len(folded) == len(text)
    return folded
