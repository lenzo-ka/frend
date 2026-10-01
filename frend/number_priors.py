"""Measured choices for a bare four-digit token.

The English table counts the corpus's DATE, CARDINAL, and DIGIT classes by century.
It is evidence about an otherwise bare token only: dates with a month, era, or other
structure remain icukit's reading.  Runtime derives shares from raw counts and falls
back to the measured all-century row when a bucket is absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from types import MappingProxyType

from frend.locale_data import LOCALE_CACHE, canonical_locale, measured_table

__all__ = ["NumberPrior", "NumberPriorTable", "load_number_priors"]


@dataclass(frozen=True)
class NumberPrior:
    """One class's measured count and share in a century bucket."""

    count: int
    share: Decimal
    total: int


class NumberPriorTable:
    """Raw DATE/CARDINAL/DIGIT counts for bare four-digit tokens."""

    def __init__(self, document: Mapping[str, object]) -> None:
        raw = document.get("centuries")
        if not isinstance(raw, Mapping):
            raise ValueError("a number-prior table needs century counts")
        rows: dict[str, Mapping[str, int]] = {}
        for key, value in raw.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"centuries.{key} must be a mapping")
            counts = {str(name): count for name, count in value.items()}
            if any(not isinstance(count, int) or count < 0 for count in counts.values()):
                raise ValueError(f"centuries.{key} counts must be nonnegative integers")
            rows[str(key)] = MappingProxyType(counts)
        total = document.get("all")
        if not isinstance(total, Mapping):
            raise ValueError("a number-prior table needs all-century counts")
        all_counts = {str(name): count for name, count in total.items()}
        if any(not isinstance(count, int) or count < 0 for count in all_counts.values()):
            raise ValueError("all-century counts must be nonnegative integers")
        self._rows = MappingProxyType(rows)
        self._all = MappingProxyType(all_counts)
        self.provenance = MappingProxyType(dict(document.get("provenance", {})))

    def lookup(self, written: str, choice: str) -> NumberPrior | None:
        """The measured ``choice`` share for a four-digit ASCII token."""
        if len(written) != 4 or not written.isascii() or not written.isdigit():
            return None
        counts = self._rows.get(written[:2], self._all)
        total = sum(counts.values())
        if not total:
            return None
        count = counts.get(choice, 0)
        return NumberPrior(count, Decimal(count) / Decimal(total), total)


def load_number_priors(locale: str = "en_US") -> NumberPriorTable | None:
    """The locale's measured bare-number choices, or ``None`` without a table."""
    return _number_priors(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _number_priors(locale: str) -> NumberPriorTable | None:
    document = measured_table("number_priors", locale)
    return None if document is None else NumberPriorTable(document)
