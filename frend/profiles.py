"""Named, caller-selected conformance profiles and their user-local data paths."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "GOOGLE_TN",
    "GROUP_ROLES",
    "GroupOrders",
    "TTS_SANITY",
    "google_tn_britishisms_path",
    "google_tn_profile_path",
    "validate_profile",
    "validate_groups",
]

GOOGLE_TN = "google-tn"
TTS_SANITY = "tts-sanity"
GROUP_ROLES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {TTS_SANITY: ("described", "named", "silent")}
)
type GroupOrders = Mapping[str, Sequence[str]]
_PROFILE_PATH_ENV = "FREND_GOOGLE_TN_PROFILE_PATH"
_BRITISHISMS_PATH_ENV = "FREND_GOOGLE_TN_BRITISHISMS_PATH"


def validate_profile(profile: str | None) -> str | None:
    """Return a supported profile name; the default ``None`` is general-purpose frend."""
    if profile not in (None, GOOGLE_TN):
        raise ValueError(f"unknown conformance profile {profile!r}; known profiles: {GOOGLE_TN!r}")
    return profile


def validate_groups(
    groups: GroupOrders | None,
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    """Validate and freeze caller-selected orders for known behavior groups."""
    if groups is None:
        return None
    if not isinstance(groups, Mapping):
        raise ValueError(f"groups must be a mapping or None, got {type(groups).__name__}")
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for group, raw_order in groups.items():
        if group not in GROUP_ROLES:
            known = ", ".join(repr(name) for name in GROUP_ROLES)
            raise ValueError(f"unknown verbalization group {group!r}; known groups: {known}")
        if isinstance(raw_order, (str, bytes)) or not isinstance(raw_order, Sequence):
            order = ()
        else:
            order = tuple(raw_order)
        roles = GROUP_ROLES[group]
        if len(order) != len(roles) or set(order) != set(roles):
            listed = ", ".join(repr(role) for role in roles)
            raise ValueError(f"group {group!r} order must list each of {listed} exactly once")
        normalized.append((group, order))
    return tuple(normalized)


def google_tn_profile_path() -> Path:
    """The external Google-TN profile table selected by environment or user cache.

    ``FREND_GOOGLE_TN_PROFILE_PATH`` names the file exactly. Otherwise the XDG cache
    root (or ``~/.cache``) holds it; this data is never a package resource.
    """
    configured = os.environ.get(_PROFILE_PATH_ENV)
    if configured:
        return Path(configured).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return root / "frend" / "profiles" / GOOGLE_TN / "acronym_surfaces.json"


def google_tn_britishisms_path() -> Path:
    """The external Google-TN spelling table selected by environment or user cache."""
    configured = os.environ.get(_BRITISHISMS_PATH_ENV)
    if configured:
        return Path(configured).expanduser()
    return google_tn_profile_path().with_name("britishisms.json")
