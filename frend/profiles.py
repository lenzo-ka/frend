"""Named, caller-selected conformance profiles and their user-local data paths."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "CHAR_DETAIL",
    "GOOGLE_TN",
    "GROUP_OPTIONS",
    "GROUP_ROLES",
    "GroupOrders",
    "GroupSetting",
    "OPT_IN_GROUPS",
    "TTS_SANITY",
    "google_tn_britishisms_path",
    "google_tn_profile_path",
    "validate_profile",
    "validate_groups",
]

GOOGLE_TN = "google-tn"
TTS_SANITY = "tts-sanity"
CHAR_DETAIL = "char-detail"
GROUP_ROLES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        TTS_SANITY: ("described", "named", "silent"),
        CHAR_DETAIL: ("named", "reading", "spelled"),
    }
)
GROUP_OPTIONS: Mapping[str, Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {CHAR_DETAIL: MappingProxyType({"digits": ("each", "number")})}
)
OPT_IN_GROUPS: frozenset[str] = frozenset({CHAR_DETAIL})


@dataclass(frozen=True)
class GroupSetting:
    """One validated group order and its filled option values."""

    order: tuple[str, ...]
    options: Mapping[str, str]


type GroupOrders = Mapping[str, Sequence[str] | Mapping[str, object] | GroupSetting]
_PROFILE_PATH_ENV = "FREND_GOOGLE_TN_PROFILE_PATH"
_BRITISHISMS_PATH_ENV = "FREND_GOOGLE_TN_BRITISHISMS_PATH"


def validate_profile(profile: str | None) -> str | None:
    """Return a supported profile name; the default ``None`` is general-purpose frend."""
    if profile not in (None, GOOGLE_TN):
        raise ValueError(f"unknown conformance profile {profile!r}; known profiles: {GOOGLE_TN!r}")
    return profile


def validate_groups(
    groups: GroupOrders | None,
) -> tuple[tuple[str, GroupSetting], ...] | None:
    """Validate and freeze caller-selected orders for known behavior groups."""
    if groups is None:
        return None
    if not isinstance(groups, Mapping):
        raise ValueError(f"groups must be a mapping or None, got {type(groups).__name__}")
    normalized: list[tuple[str, GroupSetting]] = []
    for group, raw_setting in groups.items():
        if group not in GROUP_ROLES:
            known = ", ".join(repr(name) for name in GROUP_ROLES)
            raise ValueError(f"unknown verbalization group {group!r}; known groups: {known}")
        if isinstance(raw_setting, GroupSetting):
            normalized.append((group, raw_setting))
            continue
        option_registry = GROUP_OPTIONS.get(group, {})
        options = {name: choices[0] for name, choices in option_registry.items()}
        if isinstance(raw_setting, Mapping):
            if "order" not in raw_setting:
                raise ValueError(
                    f"group {group!r} must be an order sequence or an object with 'order'; "
                    f"got {type(raw_setting).__name__} without 'order'"
                )
            raw_order = raw_setting["order"]
            for option in raw_setting.keys() - {"order"}:
                if option not in option_registry:
                    known = ", ".join(repr(name) for name in option_registry) or "none"
                    raise ValueError(
                        f"group {group!r} has unknown option {option!r}; known options: {known}"
                    )
                raw_value = raw_setting[option]
                choices = option_registry[option]
                if raw_value not in choices:
                    listed = ", ".join(repr(choice) for choice in choices)
                    raise ValueError(
                        f"group {group!r} {option} must be one of {listed}; got {raw_value!r}"
                    )
                options[option] = raw_value
        else:
            raw_order = raw_setting
        if isinstance(raw_order, (str, bytes)) or not isinstance(raw_order, Sequence):
            order = ()
        else:
            order = tuple(raw_order)
        roles = GROUP_ROLES[group]
        if len(order) != len(roles) or set(order) != set(roles):
            listed = ", ".join(repr(role) for role in roles)
            raise ValueError(f"group {group!r} order must list each of {listed} exactly once")
        normalized.append((group, GroupSetting(order, MappingProxyType(options))))
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
