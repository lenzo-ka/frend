"""Checksum-valid ISBNs and conservatively measured grouped digit identifiers."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from icukit.detectors import Capture

from frend.locale_data import LOCALE_CACHE, canonical_locale, measured_table

__all__ = [
    "GroupedDigitsDetector",
    "GroupedDigitsValue",
    "grouped_id_shape",
    "is_valid_isbn",
]

_CANDIDATE = re.compile(r"[0-9]+(?:[- ][0-9]+)+(?:[- ][Xx])?")
_SHAPE = re.compile(r"N[1-9][0-9]*(?:[- ]N[1-9][0-9]*)+(?:[- ]X)?\Z")


def grouped_id_shape(text: str) -> str:
    """A width-preserving ASCII shape (``0-306-40615-2`` -> ``N1-N3-N5-N1``)."""
    out: list[str] = []
    at = 0
    while at < len(text):
        if text[at] in "0123456789":
            end = at + 1
            while end < len(text) and text[end] in "0123456789":
                end += 1
            out.append(f"N{end - at}")
            at = end
        else:
            out.append(text[at].upper())
            at += 1
    return "".join(out)


def _groups(text: str) -> tuple[str, ...] | None:
    """Return ASCII digit groups, with an optional final ISBN-10 ``X`` group."""
    if _CANDIDATE.fullmatch(text) is None:
        return None
    return tuple(part.upper() for part in re.findall(r"[0-9]+|[Xx]", text))


def _candidate_matches(text: str) -> Iterator[re.Match[str]]:
    """Yield semantically possible prefixes so token boundaries remain load-bearing."""
    for token in _CANDIDATE.finditer(text):
        digits = 0
        yielded_full = False
        for end in range(token.start() + 1, token.end() + 1):
            char = text[end - 1]
            if char in "0123456789":
                digits += 1
            if digits > 13:
                break
            if digits not in {9, 10, 13} or (char in "Xx" and digits != 9):
                continue
            match = _CANDIDATE.fullmatch(text, token.start(), end)
            if match is not None:
                yield match
                yielded_full = end == token.end()
        if not yielded_full:
            yield token


def is_valid_isbn(text: str) -> bool:
    """Validate the ISBN-10 modulus-11 or ISBN-13 modulus-10 check digit.

    ISBN-13 uses the International ISBN Agency's alternating 1/3 weights. ISBN-10
    uses its standard descending 10..1 weights and permits X=10 only as the check
    digit. Separators do not enter either calculation.

    https://www.isbn-international.org/content/what-isbn/10
    https://www.isbn-international.org/index.php/content/isbn-users-manual/29
    """
    groups = _groups(text)
    if groups is None:
        return False
    compact = "".join(groups)
    if len(compact) == 10 and compact[:9].isdigit() and compact[-1] in "0123456789X":
        values = [int(char) for char in compact[:9]] + [
            10 if compact[-1] == "X" else int(compact[-1])
        ]
        return sum((10 - index) * value for index, value in enumerate(values)) % 11 == 0
    if len(compact) == 13 and compact.isdigit() and compact[:3] in {"978", "979"}:
        weighted = sum(
            (1 if index % 2 == 0 else 3) * int(char) for index, char in enumerate(compact)
        )
        return weighted % 10 == 0
    return False


def _unicode_numeric_continuation(char: str) -> bool:
    """Treat non-ASCII digits and dash punctuation as parts of numeric tokens."""
    return (char not in "0123456789" and char.isdigit()) or (
        char != "-" and unicodedata.category(char) == "Pd"
    )


def _valid_left_boundary(text: str, start: int) -> bool:
    """Exclude token continuations and signed or currency-prefixed candidates."""
    if start == 0:
        return True
    previous = text[start - 1]
    if previous.isalnum() or previous == "_" or _unicode_numeric_continuation(previous):
        return False
    if previous == "." and start >= 2 and text[start - 2].isdigit():
        return False

    at = start - 1
    while at >= 0 and (text[at].isspace() or text[at] in "+-"):
        if text[at] in "+-":
            return False
        at -= 1
    return at < 0 or unicodedata.category(text[at]) != "Sc"


def _is_percent(char: str) -> bool:
    return "PERCENT SIGN" in unicodedata.name(char, "")


def _valid_right_boundary(text: str, end: int) -> bool:
    """Exclude token continuations and signed, currency, or percent suffixes."""
    if end == len(text):
        return True
    following = text[end]
    if following.isalnum() or following == "_" or _unicode_numeric_continuation(following):
        return False
    if following in ".-" and end + 1 < len(text) and text[end + 1].isdigit():
        return False

    after_space = end
    while after_space < len(text) and text[after_space].isspace():
        after_space += 1
    if (
        after_space > end
        and after_space < len(text)
        and (text[after_space].isdigit() or text[after_space] in "Xx")
    ):
        return False

    at = end
    while at < len(text) and (text[at].isspace() or text[at] in "+-"):
        if text[at] in "+-":
            return False
        at += 1
    return at == len(text) or (unicodedata.category(text[at]) != "Sc" and not _is_percent(text[at]))


def _generic_id_shape(groups: tuple[str, ...]) -> bool:
    """The development-selected generic family: three groups and nine ASCII digits."""
    return (
        len(groups) == 3 and all(group.isdigit() for group in groups) and sum(map(len, groups)) == 9
    )


@dataclass(frozen=True)
class GroupedDigitsValue:
    """Written groups to say character by character, retaining group boundaries."""

    groups: tuple[str, ...]
    isbn: bool


def _count_map(value: object, *, positive: bool) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(name, str)
            and bool(name)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and (count > 0 if positive else count >= 0)
            for name, count in value.items()
        )
    )


@lru_cache(maxsize=LOCALE_CACHE)
def _prior(locale: str) -> frozenset[str]:
    document = measured_table("grouped_id_priors", locale)
    if document is None:
        return frozenset()
    if document.get("schema_version") != 1:
        raise ValueError("invalid grouped-ID prior table: wrong schema version")
    rows = document.get("shapes")
    if not isinstance(rows, dict):
        raise ValueError("invalid grouped-ID prior table: shapes must be an object")
    loaded = set()
    for shape, row in rows.items():
        classes = row.get("classes") if isinstance(row, dict) else None
        readings = row.get("readings") if isinstance(row, dict) else None
        if (
            not isinstance(shape, str)
            or _SHAPE.fullmatch(shape) is None
            or not isinstance(row, dict)
            or set(row) != {"classes", "readings"}
            or not _count_map(classes, positive=True)
            or not _count_map(readings, positive=False)
            or set(readings) != {"grouped", "other"}
        ):
            raise ValueError(f"invalid grouped-ID prior row for {shape!r}")
        grouped = readings["grouped"]
        if grouped < 3 or 2 * grouped <= sum(readings.values()):
            raise ValueError(f"unselected grouped-ID prior row for {shape!r}")
        if sum(classes.values()) != sum(readings.values()):
            raise ValueError(f"inconsistent grouped-ID prior row for {shape!r}")
        loaded.add(shape)
    return frozenset(loaded)


class GroupedDigitsDetector:
    """Detect validated ISBNs or measured three-group, nine-digit identifiers."""

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = canonical_locale(locale)

    def detect(self, text: str) -> list[dict]:
        shapes = _prior(self.locale)
        found = []
        for match in _candidate_matches(text):
            if not _valid_left_boundary(text, match.start()) or not _valid_right_boundary(
                text, match.end()
            ):
                continue
            surface = match.group()
            groups = _groups(surface)
            if groups is None:
                continue
            isbn = is_valid_isbn(surface)
            measured = grouped_id_shape(surface) in shapes
            if not isbn and (not _generic_id_shape(groups) or not measured):
                continue
            found.append(self._detection(match, groups, isbn))
        return found

    @staticmethod
    def _detection(match, groups: tuple[str, ...], isbn: bool) -> dict:
        surface = match.group()
        capture = Capture("grouped-id", match.start(), match.end(), surface, groups, "numeric")
        return {
            "text": surface,
            "start": match.start(),
            "end": match.end(),
            "type": "identifier:grouped-digits",
            "value": GroupedDigitsValue(groups, isbn),
            "captures": (capture,),
        }
