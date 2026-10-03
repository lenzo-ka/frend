"""Plain-text validation and the document-sized public input budget."""

from __future__ import annotations

import unicodedata

__all__ = [
    "DEFAULT_MAX_INPUT_CHARS",
    "DEFAULT_MAX_UNIT_CHARS",
    "MAX_NON_TEXT_SHARE",
    "InputValidationError",
    "validate_input",
    "validate_unit_length",
]


# Sentence limits belong to the breaker. This is deliberately document-sized: callers
# can submit a modest book or batch while a mistaken core file is refused before frend
# allocates a node and passthrough edge for every code point.
DEFAULT_MAX_INPUT_CHARS = 4 * 1024 * 1024

# Resolution constructs at least one node and passthrough edge per code point. The
# isolated Gutenberg timing buckets recorded in the length-bounds lane keep this at a
# practical sentence-sized unit while leaving the document-scale ingress guard above
# available to recognizers and callers that batch their own sentences.
DEFAULT_MAX_UNIT_CHARS = 8 * 1024

# The five Gutenberg books used to choose the sentence bound and a 10,000-sentence
# sample of Google TN runtime-evaluation shards 90--94 all measured zero here.
MAX_NON_TEXT_SHARE = 0.01


class InputValidationError(ValueError):
    """The supplied value is not bounded plain Unicode text."""


def _bytes_error() -> InputValidationError:
    return InputValidationError(
        "source_text must be a str decoded from valid UTF-8, not bytes; byte input "
        "(including invalid UTF-8) is refused before it is scanned"
    )


def _validate_limit(name: str, value: int | None) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or None, got {value!r}")


def validate_unit_length(
    text_length: int,
    *,
    max_unit_chars: int | None = DEFAULT_MAX_UNIT_CHARS,
    max_input_chars: int | None = None,
) -> int:
    """Refuse a derived length against document and per-unit graph-work bounds."""
    _validate_limit("max_unit_chars", max_unit_chars)
    _validate_limit("max_input_chars", max_input_chars)
    if not isinstance(text_length, int) or isinstance(text_length, bool) or text_length < 0:
        raise ValueError(f"text_length must be a non-negative integer, got {text_length!r}")
    if max_input_chars is not None and text_length > max_input_chars:
        raise InputValidationError(
            f"derived text length is {text_length} code points, exceeding the "
            f"max_input_chars={max_input_chars} document bound; sentence-break the input "
            "first and resolve each sentence separately"
        )
    if max_unit_chars is not None and text_length > max_unit_chars:
        raise InputValidationError(
            f"resolution unit has {text_length} code points, exceeding the "
            f"max_unit_chars={max_unit_chars} work bound; sentence-break the input first "
            "and resolve each sentence separately"
        )
    return text_length


def validate_input(
    source_text: str | bytes | bytearray | memoryview | None,
    *,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
) -> str | None:
    """Return bounded plain text, or refuse binary and suspicious input.

    ``max_input_chars`` counts Unicode code points. ``None`` disables only the size
    budget for a caller that already imposes one; binary, NUL-bearing, and predominantly
    non-text input is still refused. The control characters in Unicode White_Space are
    ordinary text controls. Other controls and unassigned code points together may
    occupy at most :data:`MAX_NON_TEXT_SHARE` of a non-empty input. A lone surrogate is
    always refused independently of that share.
    """
    _validate_limit("max_input_chars", max_input_chars)
    if source_text is None:
        return None
    if isinstance(source_text, (bytes, bytearray, memoryview)):
        raise _bytes_error()
    if not isinstance(source_text, str):
        raise InputValidationError(
            f"source_text must be a str containing plain text, got {type(source_text).__name__}"
        )
    if max_input_chars is not None and len(source_text) > max_input_chars:
        raise InputValidationError(
            f"source_text has {len(source_text)} code points, exceeding the "
            f"max_input_chars={max_input_chars} budget; split the document into "
            "sentences or bounded chunks before calling frend"
        )
    if "\x00" in source_text:
        raise InputValidationError("source_text contains a NUL character; input must be text")
    if not source_text:
        return source_text

    for char in source_text:
        if unicodedata.category(char) == "Cs":
            raise InputValidationError(
                "source_text contains a lone surrogate code point; input must be decoded "
                "as strict Unicode text"
            )

    suspicious = 0
    for char in source_text:
        category = unicodedata.category(char)
        # These are exactly the Cc members of Unicode White_Space. Other White_Space
        # characters have separator categories and do not enter this branch.
        if (category == "Cc" and char not in "\t\n\v\f\r\x85") or category == "Cn":
            suspicious += 1
    share = suspicious / len(source_text)
    if share > MAX_NON_TEXT_SHARE:
        raise InputValidationError(
            f"source_text contains {suspicious}/{len(source_text)} "
            f"({share:.2%}) non-whitespace control or unassigned code points; the maximum "
            f"non-text share is {MAX_NON_TEXT_SHARE:.2%}"
        )
    return source_text
