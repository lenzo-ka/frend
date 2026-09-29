"""Verify corpus store identity, split membership, and bytes before row iteration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_CATALOG = Path(__file__).with_name("corpora.json")
_CORPUS_ROOT = Path("/Volumes/k02/corpora/unpacked")
_LICENSE_CLASSES = {
    "google/tn-en_with_types": "shippable-share-alike",
    "google/tn-ru_with_types": "shippable-share-alike",
}


@dataclass(frozen=True)
class CorpusSplit:
    training: frozenset[str]
    runtime_eval: frozenset[str]
    acceptance: frozenset[str]
    report: frozenset[str]

    def names(self, pools: Sequence[str]) -> frozenset[str]:
        try:
            return frozenset().union(*(getattr(self, pool.replace("-", "_")) for pool in pools))
        except AttributeError as exc:
            raise ValueError(f"unknown corpus pool in {tuple(pools)!r}") from exc


def _pool(first: int, last: int) -> frozenset[str]:
    return frozenset(f"output-{index:05d}-of-00100" for index in range(first, last + 1))


GOOGLE_TN_SPLIT = CorpusSplit(_pool(0, 89), _pool(90, 94), _pool(95, 95), _pool(99, 99))
CORPUS_SPLITS: Mapping[str, CorpusSplit] = {
    "google/tn-en_with_types": GOOGLE_TN_SPLIT,
    "google/tn-ru_with_types": GOOGLE_TN_SPLIT,
}


@dataclass(frozen=True)
class VerifiedInput:
    source_id: str
    relative_path: str
    sha256: str
    license_class: str


def _catalog(path: Path = _CATALOG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(source_id: str) -> dict:
    aliases = {"google/tn-en_with_types": "google-tn"}
    for key, entry in _catalog().items():
        if entry.get("store_id") == source_id or aliases.get(source_id) == key:
            return entry
    raise ValueError(f"unknown corpus store id {source_id!r}")


def store_root(source_id: str) -> Path:
    entry = _entry(source_id)
    provider, store_name = source_id.split("/", 1)
    return (_CORPUS_ROOT / provider / store_name / str(entry["dest"])).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verified_inputs(
    source_id: str,
    paths: Sequence[Path],
    *,
    pools: tuple[str, ...],
) -> tuple[VerifiedInput, ...]:
    """Verify the complete requested set before returning anything a row reader can open."""
    if not paths:
        raise ValueError("at least one corpus input is required")
    entry = _entry(source_id)
    split = CORPUS_SPLITS.get(source_id)
    if split is None:
        raise ValueError(f"source {source_id!r} has no declared split")
    allowed = split.names(pools)
    pins = entry.get("shards", {})
    root = store_root(source_id)
    pending: list[tuple[str, str]] = []
    seen: set[str] = set()
    for supplied in paths:
        path = Path(supplied)
        if path.is_symlink():
            raise ValueError(f"source shard must not be a symlink: {path}")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"{path} is outside corpus store root {root}") from exc
        if "/" in relative or relative not in allowed:
            raise ValueError(f"{relative!r} does not belong to requested pools {pools!r}")
        if relative in seen:
            raise ValueError(f"duplicate corpus input {relative!r}")
        seen.add(relative)
        expected = pins.get(relative)
        if not isinstance(expected, str):
            raise ValueError(f"catalog pins no digest for {source_id}/{relative}")
        actual = _sha256(resolved)
        if actual != expected:
            raise ValueError(f"{relative}: sha256 {actual}, catalog pins {expected}")
        pending.append((relative, actual))
    license_class = _LICENSE_CLASSES[source_id]
    return tuple(VerifiedInput(source_id, name, digest, license_class) for name, digest in pending)
