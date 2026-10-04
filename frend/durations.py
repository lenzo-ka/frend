"""Resolve ICU's ambiguous fractional numeric-duration parses."""

from __future__ import annotations

from collections.abc import Mapping

from icukit import FlexibleNumericDurationDetector as _IcuNumericDurationDetector

from frend.locale_data import canonical_locale

__all__ = ["NumericDurationDetector"]


def _unit(detection: Mapping) -> str | None:
    value = detection.get("value")
    unit = getattr(value, "unit", None)
    return None if unit is None else str(unit)


def _has_fraction(detection: Mapping) -> bool:
    return any(
        getattr(capture, "name", None) == "fraction" for capture in detection.get("captures", ())
    )


class NumericDurationDetector:
    """ICU numeric durations with fractional ``M:SS.hh`` resolved as elapsed time.

    ICU intentionally offers both hours/minutes and minutes/seconds for a colon.
    On training shards 00--89, the fractional shape is the corpus's race-time form;
    retain its seconds parse and leave un-fractioned clock/duration ambiguity intact.
    """

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = canonical_locale(locale)
        self._icu = _IcuNumericDurationDetector(self.locale)

    def detect(self, text: str) -> list[dict]:
        found = list(self._icu.detect(text))
        elapsed_spans = {
            (int(item["start"]), int(item["end"]))
            for item in found
            if _unit(item) == "second" and _has_fraction(item)
        }
        return [
            item
            for item in found
            if (int(item["start"]), int(item["end"])) not in elapsed_spans
            or _unit(item) == "second"
        ]
