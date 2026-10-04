"""Resolve ICU's ambiguous fractional numeric-duration parses."""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache

from icukit import FlexibleNumericDurationDetector as _IcuNumericDurationDetector

from frend.locale_data import LOCALE_CACHE, canonical_locale, measured_table

__all__ = ["NumericDurationDetector", "fractional_duration_shape"]


_FRACTIONAL_DURATION = re.compile(r"[0-9]+:[0-5][0-9]\.[0-9]{2}")


def fractional_duration_shape(text: str, locale: str = "en_US") -> bool:
    """Whether ``text`` is the exact English whole-token ``M:SS.hh`` shape."""
    canonical = canonical_locale(locale)
    return canonical.split("_", 1)[0] == "en" and _FRACTIONAL_DURATION.fullmatch(text) is not None


def _unit(detection: Mapping) -> str | None:
    value = detection.get("value")
    unit = getattr(value, "unit", None)
    return None if unit is None else str(unit)


def _has_fraction(detection: Mapping) -> bool:
    return any(
        getattr(capture, "name", None) == "fraction" for capture in detection.get("captures", ())
    )


@lru_cache(maxsize=LOCALE_CACHE)
def _preferred_unit(locale: str) -> str | None:
    """The exact shape's unit when its measured support beats all opposition."""
    canonical = canonical_locale(locale)
    if canonical.split("_", 1)[0] != "en":
        return None
    document = measured_table("range_priors", canonical)
    record = None if document is None else document.get("numeric_duration")
    if not isinstance(record, Mapping):
        return None
    preferred = record.get("preferred_unit")
    support = record.get("preferred_count")
    opposition = record.get("other_count")
    if (
        not isinstance(preferred, str)
        or not isinstance(support, int)
        or not isinstance(opposition, int)
        or support <= opposition
    ):
        return None
    return preferred


class NumericDurationDetector:
    """ICU numeric durations with fractional ``M:SS.hh`` resolved as elapsed time.

    ICU intentionally offers both hours/minutes and minutes/seconds for a colon.  For
    an exact English whole-token ``M:SS.hh``, retain the unit preferred by the aggregate
    training counts in ``range_priors.json``.  Every other locale, shape, and embedded
    occurrence remains exactly ICU's result.
    """

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = canonical_locale(locale)
        self._icu = _IcuNumericDurationDetector(self.locale)

    def detect(self, text: str) -> list[dict]:
        found = list(self._icu.detect(text))
        preferred = _preferred_unit(self.locale)
        if preferred is None or not fractional_duration_shape(text, self.locale):
            return found
        preferred_readings = [
            item
            for item in found
            if int(item["start"]) == 0
            and int(item["end"]) == len(text)
            and _unit(item) == preferred
            and _has_fraction(item)
        ]
        return preferred_readings or found
