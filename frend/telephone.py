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

_CANDIDATE = re.compile(r"[(0-9][0-9() .-]*[0-9](?!\w)")
_SPACED_DASH = re.compile(r"\s-|\-\s")
_LEFT_TOKEN = re.compile(r"(\S+)\s*$")


def telephone_shape(text: str) -> str:
    """A digit-width-preserving surface shape (``212-555-1212`` -> ``N3-N3-N4``)."""
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
    """Return unambiguous ASCII NANP-width groups, excluding spaced dashes."""
    if _SPACED_DASH.search(text) or re.fullmatch(r"[0-9() .-]+", text) is None:
        return None
    groups = tuple(re.findall(r"[0-9]+", text))
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


def _valid_nanp(groups: tuple[str, ...]) -> bool:
    """Whether groups satisfy the published NANP NXX-NXX-XXXX structure."""
    area, exchange = groups[-3:-1]
    # NANPA publishes NXX with N=2-9 for both the NPA and central-office code,
    # and lists N11 as unavailable/special-use abbreviated dialing resources:
    # https://www.nanpa.com/about
    # https://www.nanpa.com/reports/co-code-reports/cocodes_assign
    return all(group[0] in "23456789" and group[1:] != "11" for group in (area, exchange))


def _valid_left_boundary(text: str, start: int) -> bool:
    """Exclude candidates directly following letters, numbers, or symbols."""
    if start == 0:
        return True
    previous = text[start - 1]
    return not (previous.isalnum() or unicodedata.category(previous).startswith("S"))


def _left_context(text: str, start: int) -> str | None:
    """The case-folded non-space token immediately before a candidate."""
    match = _LEFT_TOKEN.search(text[:start])
    return match.group(1).casefold() if match is not None else None


@dataclass(frozen=True)
class TelephoneValue:
    """Digit groups and their measured whole-reading signatures."""

    groups: tuple[str, ...]
    readings: tuple[tuple[tuple[str, ...], int], ...]


@lru_cache(maxsize=LOCALE_CACHE)
def _prior(
    locale: str,
) -> tuple[dict[str, tuple[tuple[tuple[str, ...], int], ...]], frozenset[str]]:
    document = measured_table("telephone_priors", locale)
    if document is None:
        return {}, frozenset()
    if document.get("schema_version") != 2:
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
    context_rows = document.get("left_contexts")
    if not isinstance(context_rows, dict):
        raise ValueError("invalid telephone prior table: left_contexts must be an object")
    vetoes = set()
    for context, row in context_rows.items():
        classes = row.get("classes") if isinstance(row, dict) else None
        prediction = row.get("prediction") if isinstance(row, dict) else None
        if (
            not isinstance(context, str)
            or not context
            or context != context.casefold()
            or not isinstance(classes, dict)
            or not classes
            or not all(
                isinstance(name, str)
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
                for name, count in classes.items()
            )
            or not isinstance(prediction, str)
            or prediction in {"PLAIN", "PUNCT", "TELEPHONE"}
            or prediction not in classes
            or sum(classes.values()) < 3
            or 2 * classes[prediction] <= sum(classes.values())
        ):
            raise ValueError(f"invalid telephone left context {context!r}")
        vetoes.add(context)
    return loaded, frozenset(vetoes)


class TelephoneDetector:
    """Detect trained NANP shapes using only ASCII digits and ``- .()`` separators.

    Wider digit scripts and Unicode separator support require separately measured priors.
    """

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = canonical_locale(locale)

    def detect(self, text: str) -> list[dict]:
        table, left_context_vetoes = _prior(self.locale)
        if not table:
            return []
        found = []
        for match in _CANDIDATE.finditer(text):
            if not _valid_left_boundary(text, match.start()):
                continue
            surface = match.group()
            groups = _phone_groups(surface)
            readings = None
            if groups is not None and _valid_nanp(groups):
                shape = telephone_shape(surface)
                pattern = telephone_reading_key(groups)
                readings = table.get(f"{shape}|{pattern}")
                if readings is None and "r" in pattern:
                    readings = table.get(f"{shape}|{pattern.replace('r', 'n')}")
            if groups is None or readings is None:
                continue
            if _left_context(text, match.start()) in left_context_vetoes:
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
