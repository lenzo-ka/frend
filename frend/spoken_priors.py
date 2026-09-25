"""Read-only access to measured spoken-alternative source counts."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "SourceMeasurement",
    "SpokenPriorTable",
    "load_spoken_prior_table",
    "measurement_sub_key",
    "source_prior",
    "spoken_tokens",
    "normalize_spoken",
]

_DATA = Path(__file__).parent / "data" / "spoken_priors.json"
_UNMATCHED_REASONS = {"unrecognized", "unverbalized", "no_alternative_matched"}
# How many matched rows the kind-level share is worth when a sub-key share is blended
# toward it: at this many rows a sub-key's own evidence carries half the weight.
SUB_KEY_PRIOR_STRENGTH = Decimal(5)


@dataclass(frozen=True)
class SourceMeasurement:
    """A source's matched-row count and its share of matched rows for one kind."""

    count: int
    share: Decimal


class SpokenPriorTable:
    """Immutable view over measured spoken-alternative matches and conditioning."""

    def __init__(self, kinds: Mapping[str, Mapping[str, object]], provenance: Mapping[str, object]):
        sub_key_rules = provenance.get("sub_key_rules")
        if not isinstance(sub_key_rules, Mapping) or set(sub_key_rules) != set(kinds):
            raise ValueError("provenance.sub_key_rules must declare every kind")
        for kind, rule in sub_key_rules.items():
            if rule is not None and (not isinstance(rule, str) or not rule.strip()):
                raise ValueError(f"provenance.sub_key_rules.{kind} must be null or text")
        validated: dict[str, tuple[int, dict[str, int], dict[str, tuple[int, dict[str, int]]]]] = {}
        for kind, record in kinds.items():
            total = _count(record.get("total"), f"{kind}.total")
            matched = _count(record.get("matched"), f"{kind}.matched")
            unmatched = _count(record.get("unmatched"), f"{kind}.unmatched")
            if matched + unmatched != total:
                raise ValueError(f"{kind}: matched + unmatched must equal total")
            sources = _count_mapping(record.get("source_matched"), f"{kind}.source_matched")
            if sum(sources.values()) != matched:
                raise ValueError(f"{kind}: source counts must sum to matched")
            reasons = _count_mapping(
                record.get("unmatched_by_reason"), f"{kind}.unmatched_by_reason"
            )
            if set(reasons) != _UNMATCHED_REASONS:
                raise ValueError(f"{kind}: unmatched reasons do not match the schema")
            if sum(reasons.values()) != unmatched:
                raise ValueError(f"{kind}: reason counts must sum to unmatched")
            if matched == 0 and sources:
                raise ValueError(f"{kind}: a source cannot be present when matched is zero")
            sub_keys = record.get("sub_keys")
            if not isinstance(sub_keys, Mapping):
                raise ValueError(f"{kind}.sub_keys must be a mapping")
            validated_sub_keys: dict[str, tuple[int, dict[str, int]]] = {}
            for sub_key, sub_record in sub_keys.items():
                if not isinstance(sub_record, Mapping):
                    raise ValueError(f"{kind}.sub_keys.{sub_key} must be a mapping")
                sub_matched = _count(
                    sub_record.get("matched"), f"{kind}.sub_keys.{sub_key}.matched"
                )
                sub_sources = _count_mapping(
                    sub_record.get("source_matched"),
                    f"{kind}.sub_keys.{sub_key}.source_matched",
                )
                if sum(sub_sources.values()) != sub_matched:
                    raise ValueError(
                        f"{kind}.sub_keys.{sub_key}: source counts must sum to matched"
                    )
                if sub_matched == 0 and sub_sources:
                    raise ValueError(
                        f"{kind}.sub_keys.{sub_key}: a source cannot be present "
                        "when matched is zero"
                    )
                validated_sub_keys[str(sub_key)] = sub_matched, sub_sources
            if (
                validated_sub_keys
                and sum(item[0] for item in validated_sub_keys.values()) != matched
            ):
                raise ValueError(f"{kind}: sub-key matched counts must sum to kind matched")
            if sub_key_rules[kind] is None and validated_sub_keys:
                raise ValueError(f"{kind}: sub-key rows require a declared rule")
            if sub_key_rules[kind] is not None and not validated_sub_keys:
                raise ValueError(f"{kind}: a declared sub-key rule requires measured rows")
            validated[kind] = matched, sources, validated_sub_keys
        self._kinds = MappingProxyType(
            {
                kind: MappingProxyType(sources)
                for kind, (_matched, sources, _sub_keys) in validated.items()
            }
        )
        self._matched = MappingProxyType(
            {kind: matched for kind, (matched, _sources, _sub_keys) in validated.items()}
        )
        self._sub_keys = MappingProxyType(
            {
                kind: MappingProxyType(
                    {
                        sub_key: (matched, MappingProxyType(sources))
                        for sub_key, (matched, sources) in sub_keys.items()
                    }
                )
                for kind, (_matched, _sources, sub_keys) in validated.items()
            }
        )
        self.provenance = MappingProxyType(dict(provenance))

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(self._kinds)

    def lookup(
        self, kind: str, source: str, sub_key: str | None = None
    ) -> SourceMeasurement | None:
        """Return a source measurement at kind level, or blended at a present sub-key.

        Without a shipped sub-key row, the share is the kind row's raw count over its
        matched rows. With one, every source in the comparison gets the same blend
        of the sub-key's evidence and the kind-level share, weighted by how much
        evidence the sub-key has: ``(sub-key count + A * kind share) / (sub-key
        matched + A)``, with ``A`` = ``SUB_KEY_PRIOR_STRENGTH``. One observation
        therefore barely moves a sub-key away from the kind, and hundreds dominate
        it. A source attested at neither level is unmeasured. ``count`` is the
        source's matched count in the narrowest scope used: the sub-key's own
        evidence, which may be zero for a source measured only at kind level.
        """
        matched = self._matched.get(kind)
        sources = self._kinds.get(kind)
        kind_share = (
            Decimal(sources[source]) / Decimal(matched)
            if sources is not None and source in sources and matched
            else None
        )
        sub_key_record = self._sub_keys.get(kind, {}).get(sub_key) if sub_key is not None else None
        if sub_key_record is None:
            if kind_share is None:
                return None
            return SourceMeasurement(sources[source], kind_share)
        sub_matched, sub_sources = sub_key_record
        sub_count = sub_sources.get(source, 0)
        if sub_count == 0 and kind_share is None:
            return None
        strength = SUB_KEY_PRIOR_STRENGTH
        blended = (Decimal(sub_count) + strength * (kind_share or Decimal(0))) / (
            Decimal(sub_matched) + strength
        )
        return SourceMeasurement(sub_count, blended)


def measurement_sub_key(kind: str | None, detection: object) -> str | None:
    """Derive the declared measurement sub-key shared by building and ranking.

    Fraction measurements use the captured denominator's decimal value, and
    measure measurements the reading's ICU unit identifier (``per-square-kilometer``,
    ``kilometer``; ``percent`` for a percent), so a rate can learn the corpus's plural
    after "per" without every unit taking it. No other kind declares a sub-key, so
    those kinds retain kind-level shares.
    """
    if kind == "measure":
        if str(detection.get("type", "")) == "number:percent":  # type: ignore[union-attr]
            return "percent"
        unit = getattr(detection.get("value"), "unit", None)  # type: ignore[union-attr]
        return None if unit is None else str(unit)
    if kind != "fraction":
        return None
    for capture in detection.get("captures", ()):  # type: ignore[union-attr]
        if getattr(capture, "name", None) == "denominator":
            try:
                value = Decimal(str(capture.value))
            except (ArithmeticError, ValueError):
                return None
            if value != value.to_integral_value():
                return None
            return str(int(value))
    return None


def _count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _count_mapping(value: object, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return {str(key): _count(count, f"{field}.{key}") for key, count in value.items()}


@lru_cache(maxsize=1)
def load_spoken_prior_table() -> SpokenPriorTable:
    document = json.loads(_DATA.read_text(encoding="utf-8"))
    return SpokenPriorTable(document["kinds"], document["provenance"])


def source_prior(kind: str, source: str, sub_key: str | None = None) -> SourceMeasurement | None:
    """Return a source's share of matched rows in the selected scope, if attested.

    Matched rows are the denominator because an unmatched row credits no source;
    the shares therefore form a distribution over rows that credit a source. With
    no shipped sub-key row the kind-level share is used throughout; with one, every
    source's share is the sub-key's evidence blended toward the kind-level share by
    how many rows the sub-key has (``SpokenPriorTable.lookup``), so a single
    observation cannot decide a ranking alone. The returned measurement also
    carries the sub-key's own matched count.
    """
    return load_spoken_prior_table().lookup(kind, source, sub_key)


def spoken_tokens(text: str) -> tuple[str, ...]:
    """Split at Unicode punctuation and split-whitespace, then lower each token.

    Case mapping is local to each token, including contextual Greek final sigma.
    This permits path-independent emissions in the alignment graph.
    """
    separated = "".join(
        " " if unicodedata.category(character).startswith("P") else character for character in text
    )
    return tuple(token.lower() for token in separated.split())


def normalize_spoken(text: str) -> str:
    """Return the canonical space-joined spoken tokens."""
    return " ".join(spoken_tokens(text))
