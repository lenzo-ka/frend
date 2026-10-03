"""Plain-text validation and the document-sized public input budget."""

from __future__ import annotations

import unicodedata

__all__ = [
    "DEFAULT_MAX_INPUT_CHARS",
    "MAX_NON_TEXT_SHARE",
    "InputValidationError",
    "validate_input",
]


# Sentence limits belong to the breaker. This is deliberately document-sized: callers
# can submit a modest book or batch while a mistaken core file is refused before frend
# allocates a node and passthrough edge for every code point.
DEFAULT_MAX_INPUT_CHARS = 4 * 1024 * 1024

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


def validate_input(
    source_text: str | bytes | bytearray | memoryview | None,
    *,
    max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
) -> str | None:
    """Return bounded plain text, or refuse binary and suspicious input.

    ``max_input_chars`` counts Unicode code points. ``None`` disables only the size
    budget for a caller that already imposes one; binary, NUL-bearing, and predominantly
    non-text input is still refused. Tabs and line endings are ordinary text controls.
    Other control characters, unassigned code points, and lone surrogates together may
    occupy at most :data:`MAX_NON_TEXT_SHARE` of a non-empty input.
    """
    if max_input_chars is not None and (
        not isinstance(max_input_chars, int)
        or isinstance(max_input_chars, bool)
        or max_input_chars < 0
    ):
        raise ValueError(
            f"max_input_chars must be a non-negative integer or None, got {max_input_chars!r}"
        )
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

    suspicious = 0
    for char in source_text:
        category = unicodedata.category(char)
        if (category == "Cc" and char not in "\t\n\r") or category in {"Cn", "Cs"}:
            suspicious += 1
    share = suspicious / len(source_text)
    if share > MAX_NON_TEXT_SHARE:
        raise InputValidationError(
            f"source_text contains {suspicious}/{len(source_text)} "
            f"({share:.2%}) control, unassigned, or surrogate code points; the maximum "
            f"non-text share is {MAX_NON_TEXT_SHARE:.2%}"
        )
    return source_text
