"""Corpus-measured readings for conservatively phone-shaped digit groups."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from icukit.detectors import Capture

from frend.locale_data import LOCALE_CACHE, canonical_locale, measured_table

__all__ = [
    "TelephoneDetector",
    "TelephoneValue",
    "telephone_shape",
]

_CANDIDATE = re.compile(r"(?<!\w)[(\d][\d() .-]*\d(?!\w)")
_SPACED_DASH = re.compile(r"\s-|\-\s")


def telephone_shape(text: str) -> str:
    """A digit-width-preserving surface shape (``212-555-1212`` -> ``N3-N3-N4``)."""
    out: list[str] = []
    at = 0
    while at < len(text):
        if unicodedata.category(text[at]) == "Nd":
            end = at + 1
            while end < len(text) and unicodedata.category(text[end]) == "Nd":
                end += 1
            out.append(f"N{end - at}")
            at = end
        else:
            out.append(text[at])
            at += 1
    return "".join(out)


def telephone_reading_key(groups: tuple[str, ...]) -> str:
    """Coarse digit-content strata that keep zero and repeated-run evidence apart."""
    return ":".join(
        "z" if "0" in group else "r" if len(group) > 1 and len(set(group)) == 1 else "n"
        for group in groups
    )


def _phone_groups(text: str) -> tuple[str, ...] | None:
    """Return unambiguous NANP groups, excluding range-like spaced dashes."""
    if _SPACED_DASH.search(text):
        return None
    groups = tuple(re.findall(r"\d+", text))
    widths = tuple(map(len, groups))
    if widths not in ((3, 3, 4), (1, 3, 3, 4)):
        return None
    if widths == (1, 3, 3, 4) and groups[0] != "1":
        return None
    if "(" in text or ")" in text:
        if widths != (3, 3, 4) or not text.startswith(f"({groups[0]})"):
            return None
        if text.count("(") != 1 or text.count(")") != 1:
            return None
    return groups


@dataclass(frozen=True)
class TelephoneValue:
    """Digit groups and their measured whole-reading signatures."""

    groups: tuple[str, ...]
    readings: tuple[tuple[tuple[str, ...], int], ...]


@lru_cache(maxsize=LOCALE_CACHE)
def _rows(locale: str) -> dict[str, tuple[tuple[tuple[str, ...], int], ...]]:
    document = measured_table("telephone_priors", locale)
    if document is None:
        return {}
    if document.get("schema_version") != 1:
        raise ValueError("invalid telephone prior table: wrong schema version")
    rows = document.get("shapes")
    if not isinstance(rows, dict):
        raise ValueError("invalid telephone prior table: shapes must be an object")
    loaded = {}
    for shape, row in rows.items():
        patterns = row.get("patterns") if isinstance(row, dict) else None
        if not isinstance(shape, str) or not isinstance(patterns, dict):
            raise ValueError("invalid telephone prior table row")
        for pattern, pattern_row in patterns.items():
            readings = pattern_row.get("readings") if isinstance(pattern_row, dict) else None
            if not isinstance(pattern, str) or not isinstance(readings, list):
                raise ValueError("invalid telephone prior pattern row")
            parsed = []
            for reading in readings:
                modes = reading.get("modes") if isinstance(reading, dict) else None
                count = reading.get("count") if isinstance(reading, dict) else None
                if (
                    not isinstance(modes, list)
                    or not modes
                    or not all(
                        mode in {"digits", "digits-o", "digits-zero", "cardinal"} for mode in modes
                    )
                    or not isinstance(count, int)
                    or isinstance(count, bool)
                    or count <= 0
                ):
                    raise ValueError(f"invalid telephone prior reading for {shape!r}")
                parsed.append((tuple(modes), count))
            loaded[f"{shape}|{pattern}"] = tuple(parsed)
    return loaded


class TelephoneDetector:
    """Detect only measured NANP shapes; readings remain corpus-derived."""

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = canonical_locale(locale)

    def detect(self, text: str) -> list[dict]:
        table = _rows(self.locale)
        if not table:
            return []
        found = []
        for match in _CANDIDATE.finditer(text):
            surface = match.group()
            groups = _phone_groups(surface)
            readings = None
            if groups is not None:
                shape = telephone_shape(surface)
                pattern = telephone_reading_key(groups)
                readings = table.get(f"{shape}|{pattern}")
                if readings is None and "r" in pattern:
                    readings = table.get(f"{shape}|{pattern.replace('r', 'n')}")
            if groups is None or readings is None:
                continue
            found.append(
                {
                    "text": surface,
                    "start": match.start(),
                    "end": match.end(),
                    "type": "telephone:number",
                    "value": TelephoneValue(groups, readings),
                    "captures": (
                        Capture(
                            "telephone",
                            match.start(),
                            match.end(),
                            surface,
                            groups,
                            "numeric",
                        ),
                    ),
                }
            )
        return found
