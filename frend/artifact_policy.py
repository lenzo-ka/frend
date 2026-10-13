"""Versioned selection policies shared by measured-artifact builders and runtime."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

GROUPED_ID_POLICY = MappingProxyType(
    {
        "id": "grouped-id-reading-majority",
        "version": 1,
        "minimum_support": 3,
        "label": "grouped",
        "strict_majority": True,
    }
)
TELEPHONE_SHAPE_POLICY = MappingProxyType(
    {
        "id": "telephone-shape-majority",
        "version": 1,
        "minimum_support": 3,
        "label": "TELEPHONE",
        "strict_majority": True,
    }
)
TELEPHONE_READING_POLICY = MappingProxyType(
    {
        "id": "telephone-reading-support",
        "version": 1,
        "minimum_support": 3,
    }
)
TELEPHONE_CONTEXT_POLICY = MappingProxyType(
    {
        "id": "telephone-left-context-veto",
        "version": 1,
        "minimum_observations": 3,
        "strict_majority": True,
        "excluded_predictions": ("PLAIN", "PUNCT", "TELEPHONE"),
    }
)
ELECTRONIC_SPAN_POLICY = MappingProxyType(
    {
        "id": "electronic-span-feature-majority",
        "version": 1,
        "minimum_support": 3,
        "label": "ELECTRONIC",
        "strict_majority": True,
    }
)


def strict_majority(
    counts: Mapping[str, int],
    label: str,
    *,
    minimum_support: int = 0,
    minimum_observations: int = 0,
) -> bool:
    """Whether ``label`` meets support and strictly exceeds all other observations."""
    selected = counts.get(label, 0)
    total = sum(counts.values())
    return (
        selected >= minimum_support
        and total >= minimum_observations
        and selected > total - selected
    )


def policy_document(policy: Mapping[str, object]) -> dict[str, object]:
    """Return a JSON-ready copy of an immutable declared policy."""
    return {
        key: list(value) if isinstance(value, tuple) else value for key, value in policy.items()
    }


def require_policy(observed: object, expected: Mapping[str, object], artifact: str) -> None:
    """Reject an artifact whose policy identity, version, or parameters drifted."""
    if observed != policy_document(expected):
        raise ValueError(f"invalid {artifact} policy")
