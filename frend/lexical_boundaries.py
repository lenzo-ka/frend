"""Shared lexical-boundary and written-separator primitives."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

SEPARATOR_CLASSES = ("range", "ratio", "dimension")


def letter_bearing_hyphenated_alnum_before(text: str, end: int) -> bool:
    """Whether the joined hyphenated alphanumeric run before ``end`` has a letter."""
    cursor = end
    while True:
        group_end = cursor
        while cursor > 0 and text[cursor - 1].isalnum():
            cursor -= 1
        if cursor == group_end:
            return False
        if any(char.isalpha() for char in text[cursor:group_end]):
            return True
        if cursor == 0 or text[cursor - 1] != "-":
            return False
        cursor -= 1


def separator_classes(
    connectors: Mapping[str, object],
    written: Mapping[str, object],
    cldr_range: str | None,
) -> Mapping[str, frozenset[str]]:
    """Derive enabled separator sets from loader-specific lexical values."""
    result: dict[str, frozenset[str]] = {}
    for separator_class in SEPARATOR_CLASSES:
        raw = written.get(separator_class, ())
        separators = set(raw) if isinstance(raw, (list, tuple, set, frozenset)) else set()
        if separator_class == "range" and cldr_range:
            separators.add(cldr_range)
        if separators and connectors.get(separator_class):
            result[separator_class] = frozenset(separators)
    return MappingProxyType(result)
