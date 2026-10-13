"""Corpus-measured readings for conservatively phone-shaped digit groups."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from icukit import add_likely_subtags, parse_locale
from icukit.detectors import Capture
from phonenumbers import (
    Leniency,
    NumberParseException,
    PhoneMetadata,
    PhoneNumberMatcher,
    is_valid_number,
    national_significant_number,
    parse,
)

from frend.locale_data import LOCALE_CACHE, canonical_locale, measured_table
from frend.shape import ascii_width_shape

__all__ = [
    "TelephoneDetector",
    "TelephoneValue",
    "telephone_shape",
]

_SPACED_DASH = re.compile(r"\s-|\-\s")
_LEFT_TOKEN = re.compile(r"(\S+)\s*$")
_LEFT_CONTEXT_CLASSES = frozenset(
    {
        "digit",
        "letter",
        "lower",
        "mark",
        "mixed-letter",
        "other",
        "punct-pc",
        "punct-pd",
        "punct-pe",
        "punct-pf",
        "punct-pi",
        "punct-po",
        "punct-ps",
        "separator",
        "symbol-sc",
        "symbol-sk",
        "symbol-sm",
        "symbol-so",
        "title",
        "upper",
    }
)
_LEFT_CONTEXT_LENGTHS = frozenset({"1", "2", "3", "4", "5-8", "9+"})


def telephone_shape(text: str) -> str:
    """A digit-width-preserving surface shape (``212-555-1212`` -> ``N3-N3-N4``)."""
    return ascii_width_shape(text, lambda character: character)


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


def _international_phone_groups(text: str, country_code: str) -> tuple[str, ...] | None:
    """Return the national digit runs written after an explicit ``+CC`` prefix."""
    prefix = f"+{country_code}"
    if not text.startswith(prefix) or len(text) == len(prefix):
        return None
    if re.fullmatch(r"\+[0-9() .-]+", text) is None:
        return None
    groups = tuple(re.findall(r"[0-9]+", text[len(prefix) :]))
    return groups or None


@lru_cache(maxsize=LOCALE_CACHE)
def _phone_region(locale: str) -> str | None:
    """Return the locale's CLDR likely region for phone-number metadata lookup."""
    likely = add_likely_subtags(locale)
    region = parse_locale(likely)["region"]
    return region if isinstance(region, str) else None


@lru_cache(maxsize=LOCALE_CACHE)
def _general_phone_pattern(region: str) -> re.Pattern[str] | None:
    """Return libphonenumber's general national-number pattern for ``region``."""
    metadata = PhoneMetadata.metadata_for_region(region)
    if metadata is None or metadata.general_desc is None:
        return None
    pattern = metadata.general_desc.national_number_pattern
    return re.compile(pattern) if pattern else None


def _valid_left_boundary(text: str, start: int) -> bool:
    """Exclude candidates directly following letters, numbers, or symbols."""
    if start == 0:
        return True
    previous = text[start - 1]
    return not (previous.isalnum() or unicodedata.category(previous).startswith("S"))


def _valid_right_boundary(text: str, end: int) -> bool:
    """Exclude candidates directly followed by a word character."""
    return end == len(text) or not (text[end].isalnum() or text[end] == "_")


def _length_bucket(length: int) -> str:
    if length <= 4:
        return str(length)
    return "5-8" if length <= 8 else "9+"


def _left_context_shape(token: str) -> str | None:
    """A surface-free character-class shape for one non-space token."""
    if not token or any(char.isspace() for char in token):
        return None
    runs: list[tuple[str, str]] = []
    at = 0
    while at < len(token):
        category = unicodedata.category(token[at])
        if token[at].isalpha():
            family = "letter"
        elif token[at].isdecimal():
            family = "digit"
        elif category.startswith("P"):
            family = f"punct-{category.casefold()}"
        elif category.startswith("S"):
            family = f"symbol-{category.casefold()}"
        elif category.startswith("M"):
            family = "mark"
        elif category.startswith("Z"):
            family = "separator"
        else:
            family = "other"
        end = at + 1
        while end < len(token):
            next_category = unicodedata.category(token[end])
            if family == "letter":
                same = token[end].isalpha()
            elif family == "digit":
                same = token[end].isdecimal()
            elif family.startswith("punct-"):
                same = next_category.casefold() == family.removeprefix("punct-")
            elif family.startswith("symbol-"):
                same = next_category.casefold() == family.removeprefix("symbol-")
            elif family == "mark":
                same = next_category.startswith("M")
            elif family == "separator":
                same = next_category.startswith("Z")
            else:
                same = not (
                    token[end].isalpha() or token[end].isdecimal() or next_category[0] in "PMSZ"
                )
            if not same:
                break
            end += 1
        run = token[at:end]
        if family == "letter":
            if run.isupper():
                family = "upper"
            elif run.islower():
                family = "lower"
            elif run.istitle():
                family = "title"
            elif any(char.isupper() or char.islower() for char in run):
                family = "mixed-letter"
        runs.append((family, _length_bucket(len(run))))
        at = end
    return "+".join(f"{family}:{length}" for family, length in runs)


def _is_left_context_shape(value: str) -> bool:
    """Whether ``value`` uses only the fixed, surface-free shape alphabet."""
    if not value:
        return False
    runs = value.split("+")
    return all(
        run.count(":") == 1
        and run.split(":", 1)[0] in _LEFT_CONTEXT_CLASSES
        and run.split(":", 1)[1] in _LEFT_CONTEXT_LENGTHS
        for run in runs
    )


def _left_context(text: str, start: int) -> str | None:
    """The character-class shape of the non-space token before a candidate."""
    match = _LEFT_TOKEN.search(text[:start])
    return _left_context_shape(match.group(1)) if match is not None else None


@dataclass(frozen=True)
class TelephoneValue:
    """Digit groups, reading signatures, and an optional explicit country code."""

    groups: tuple[str, ...]
    readings: tuple[tuple[tuple[str, ...], int], ...]
    country_code: str | None = None


@lru_cache(maxsize=LOCALE_CACHE)
def _prior(
    locale: str,
) -> tuple[dict[str, tuple[tuple[tuple[str, ...], int], ...]], frozenset[str]]:
    document = measured_table("telephone_priors", locale)
    if document is None:
        return {}, frozenset()
    if document.get("schema_version") != 3:
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
            or not _is_left_context_shape(context)
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
    """Detect trained NANP shapes and valid explicit international numbers.

    Wider digit scripts and Unicode separator support require separately measured priors.
    """

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = canonical_locale(locale)

    def detect(self, text: str) -> list[dict]:
        table, left_context_vetoes = _prior(self.locale)
        if not table:
            return []
        region = _phone_region(self.locale)
        general_pattern = _general_phone_pattern(region) if region is not None else None
        if region is None or general_pattern is None:
            return []
        found = []
        matches = PhoneNumberMatcher(text, region, leniency=Leniency.POSSIBLE)
        for match in matches:
            if not _valid_left_boundary(text, match.start) or not _valid_right_boundary(
                text, match.end
            ):
                continue
            surface = match.raw_string
            number = match.number
            country_code = None
            if surface.startswith("+"):
                try:
                    number = parse(surface, None)
                except NumberParseException:
                    continue
                country_code = str(number.country_code)
            groups = (
                _international_phone_groups(surface, country_code)
                if country_code is not None
                else _phone_groups(surface)
            )
            readings = None
            national = national_significant_number(number)
            if groups is not None:
                if (
                    country_code is not None
                    and "".join(groups) == national
                    and is_valid_number(number)
                ):
                    readings = ((tuple("digits" for _group in groups), 1),)
                elif country_code is None and general_pattern.fullmatch(national) is not None:
                    shape = telephone_shape(surface)
                    pattern = telephone_reading_key(groups)
                    readings = table.get(f"{shape}|{pattern}")
                    if readings is None and "r" in pattern:
                        readings = table.get(f"{shape}|{pattern.replace('r', 'n')}")
            if groups is None or readings is None:
                continue
            if _left_context(text, match.start) in left_context_vetoes:
                continue
            found.append(
                {
                    "text": surface,
                    "start": match.start,
                    "end": match.end,
                    "type": "telephone:number",
                    "value": TelephoneValue(groups, readings, country_code),
                    "captures": (
                        Capture(
                            "telephone",
                            match.start,
                            match.end,
                            surface,
                            ((f"+{country_code}", *groups) if country_code is not None else groups),
                            "numeric",
                        ),
                    ),
                }
            )
        return found
