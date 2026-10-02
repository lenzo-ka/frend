"""Measured choices for a bare four-digit token.

The English table counts the corpus's DATE, CARDINAL, and DIGIT classes by century.
It is evidence about an otherwise bare token only: dates with a month, era, or other
structure remain icukit's reading.  Runtime derives shares from raw counts and falls
back to the measured all-century row when a bucket is absent.

The optional ``google-tn`` conformance profile adds a development-selected aggregate
key. It remains dormant unless its caller explicitly selects that profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from types import MappingProxyType

from frend.locale_data import LOCALE_CACHE, canonical_locale, measured_table

__all__ = ["NumberPrior", "NumberPriorTable", "load_number_priors"]

_PROFILE_MODEL = "decade+structure+neighbors"


def _structure(written: str) -> str:
    if written.startswith("0"):
        return "leading-zero"
    if written.endswith("000"):
        return "round-thousand"
    if written.endswith("00"):
        return "round-hundred"
    return "ordinary"


@lru_cache(maxsize=65536)
def _neighbor_class(surface: str, edge: str, locale: str = "en_US") -> str:
    """The existing context feature's coarse class for one neighboring token."""
    if not surface:
        return edge
    from frend.context import _word_class

    return _word_class(surface, canonical_locale(locale), frozenset())


def _profile_keys(written: str, before: str, after: str, locale: str = "en_US") -> dict[str, str]:
    """Candidate aggregate keys for one token and its neighboring surfaces."""
    decade = written[:3]
    structure = _structure(written)
    left = _neighbor_class(before, "<BOS>", locale)
    right = _neighbor_class(after, "<EOS>", locale)
    return {
        "century": written[:2],
        "decade": decade,
        "structure": structure,
        "decade+structure": f"{decade}|{structure}",
        "decade+left": f"{decade}|{left}",
        "decade+right": f"{decade}|{right}",
        "decade+neighbors": f"{decade}|{left}|{right}",
        "decade+structure+left": f"{decade}|{structure}|{left}",
        "decade+structure+right": f"{decade}|{structure}|{right}",
        _PROFILE_MODEL: f"{decade}|{structure}|{left}|{right}",
    }


def _profile_key(model: str, written: str, before: str, after: str, locale: str = "en_US") -> str:
    """One candidate aggregate key selected by ``model``."""
    try:
        return _profile_keys(written, before, after, locale)[model]
    except KeyError as exc:
        raise ValueError(f"unknown google-tn number-prior model {model!r}") from exc


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
        profile = document.get("google_tn_profile")
        self._profile_model: str | None = None
        profile_rows: dict[str, Mapping[str, int]] = {}
        if profile is not None:
            if not isinstance(profile, Mapping):
                raise ValueError("google_tn_profile must be a mapping")
            model = profile.get("selected_key")
            raw_profile_rows = profile.get("keys")
            if (
                not isinstance(model, str)
                or model not in _profile_keys("2012", "", "")
                or not isinstance(raw_profile_rows, Mapping)
            ):
                raise ValueError("google_tn_profile needs its selected key and counts")
            for key, value in raw_profile_rows.items():
                if not isinstance(value, Mapping):
                    raise ValueError(f"google_tn_profile.keys.{key} must be a mapping")
                counts = {str(name): count for name, count in value.items()}
                if any(not isinstance(count, int) or count < 0 for count in counts.values()):
                    raise ValueError(
                        f"google_tn_profile.keys.{key} counts must be nonnegative integers"
                    )
                profile_rows[str(key)] = MappingProxyType(counts)
            self._profile_model = model
        self._profile_rows = MappingProxyType(profile_rows)
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

    def lookup_profile(
        self,
        written: str,
        choice: str,
        before: str = "",
        after: str = "",
        locale: str = "en_US",
    ) -> NumberPrior | None:
        """The selected ``google-tn`` aggregate, falling back to the century prior."""
        if self._profile_model is None:
            return self.lookup(written, choice)
        if len(written) != 4 or not written.isascii() or not written.isdigit():
            return None
        key = _profile_key(self._profile_model, written, before, after, locale)
        counts = self._profile_rows.get(key)
        if counts is None:
            return self.lookup(written, choice)
        total = sum(counts.values())
        if not total:
            return self.lookup(written, choice)
        count = counts.get(choice, 0)
        return NumberPrior(count, Decimal(count) / Decimal(total), total)


def load_number_priors(locale: str = "en_US") -> NumberPriorTable | None:
    """The locale's measured bare-number choices, or ``None`` without a table."""
    return _number_priors(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _number_priors(locale: str) -> NumberPriorTable | None:
    document = measured_table("number_priors", locale)
    return None if document is None else NumberPriorTable(document)
