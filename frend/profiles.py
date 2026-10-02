"""Named, caller-selected conformance profiles and their user-local data paths."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "GOOGLE_TN",
    "google_tn_britishisms_path",
    "google_tn_profile_path",
    "validate_profile",
]

GOOGLE_TN = "google-tn"
_PROFILE_PATH_ENV = "FREND_GOOGLE_TN_PROFILE_PATH"
_BRITISHISMS_PATH_ENV = "FREND_GOOGLE_TN_BRITISHISMS_PATH"


def validate_profile(profile: str | None) -> str | None:
    """Return a supported profile name; the default ``None`` is general-purpose frend."""
    if profile not in (None, GOOGLE_TN):
        raise ValueError(f"unknown conformance profile {profile!r}; known profiles: {GOOGLE_TN!r}")
    return profile


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
