"""Race-safe identity keys for external profile files."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

type ProfileKey = tuple[str, int, int, str]
type Freshness = tuple[int, int, int, int, int]

_KEYS: dict[str, tuple[Freshness, ProfileKey]] = {}


def _freshness(stat) -> Freshness:
    return (stat.st_dev, stat.st_ino, stat.st_ctime_ns, stat.st_mtime_ns, stat.st_size)


def verified_profile_key(
    path: Path,
    *,
    missing: Callable[[Path], FileNotFoundError],
    changed: Callable[[Path], ValueError],
) -> ProfileKey:
    """Resolve, hash, re-stat, and cache one external file by full filesystem identity."""
    resolved = path.expanduser().resolve()
    try:
        before = resolved.stat()
    except FileNotFoundError:
        raise missing(resolved) from None
    freshness = _freshness(before)
    cached = _KEYS.get(str(resolved))
    if cached is not None and cached[0] == freshness:
        return cached[1]
    try:
        content = resolved.read_bytes()
        after = resolved.stat()
    except FileNotFoundError:
        raise missing(resolved) from None
    if freshness != _freshness(after):
        raise changed(resolved)
    key = (str(resolved), before.st_mtime_ns, before.st_size, hashlib.sha256(content).hexdigest())
    _KEYS[str(resolved)] = (freshness, key)
    return key
