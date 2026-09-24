"""Reflective shape signature of a surface string.

A *shape* abstracts a surface into the coarse token pattern the corpus prior is
keyed on: a single uppercase letter has its own shape, other maximal letter runs
collapse to ``A``, maximal decimal-digit runs collapse to ``N``, currency symbols
to ``¤``, and structural punctuation plus space is preserved verbatim. Every
classification is drawn from the Unicode
general category, never a hard-coded character range, so the signature is script
neutral: ``shape("٣/٤") == shape("3/24") == "N/N"`` because Arabic-Indic digits
carry category ``Nd`` exactly as ASCII digits do (garment, not brittle).

The surface is NFC-normalized before classification, so a combining sequence and
its precomposed form share one shape: ``shape("é") == shape("é")``.
Without this the builder and the runtime could key a shape differently for the
same visible text depending only on its normalization form.
"""

from __future__ import annotations

import unicodedata

__all__ = ["SINGLE_UPPERCASE_SHAPE", "is_single_uppercase", "shape"]

# Structural punctuation and space kept verbatim, so "N/N", "N.N", "A N" and the
# like remain distinct shapes. Anything else outside Nd/L*/Sc also passes through
# verbatim; it is simply never collapsed into a run.
_PRESERVED = frozenset("/.,:-' ")

_DIGIT = "N"
_LETTER = "A"
_CURRENCY = "¤"
SINGLE_UPPERCASE_SHAPE = "<Lu>"


def is_single_uppercase(surface: str) -> bool:
    """Return whether NFC ``surface`` is one Unicode uppercase-letter character."""
    normalized = unicodedata.normalize("NFC", surface)
    return len(normalized) == 1 and unicodedata.category(normalized) == "Lu"


def _token(ch: str) -> str:
    """The run token a character belongs to, or the character itself verbatim."""
    category = unicodedata.category(ch)
    if category == "Nd":
        return _DIGIT
    if category[0] == "L":
        return _LETTER
    if category == "Sc":
        return _CURRENCY
    return ch


def shape(surface: str) -> str:
    """Return the reflective shape signature of ``surface``.

    Consecutive characters that share a run token (``N``, ``A``, or ``¤``)
    collapse to a single token; preserved punctuation and any other character
    appear once per occurrence and interrupt a run.
    """
    normalized = unicodedata.normalize("NFC", surface)
    if is_single_uppercase(normalized):
        return SINGLE_UPPERCASE_SHAPE
    out: list[str] = []
    run: str | None = None
    for ch in normalized:
        token = _token(ch)
        if token in (_DIGIT, _LETTER, _CURRENCY):
            if token != run:
                out.append(token)
                run = token
        else:
            out.append(token)
            run = None
    return "".join(out)
