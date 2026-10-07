"""Verify corpus store identity, split membership, and bytes before row iteration."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from frend.locale_data import canonical_locale

_CATALOG = Path(__file__).with_name("corpora.json")
_LICENSE_CLASSES = {
    "google/tn-en_with_types": "shippable-share-alike",
    "google/tn-ru_with_types": "shippable-share-alike",
}
_SOURCE_LOCALES = {
    "google/tn-en_with_types": "en_US",
    "google/tn-ru_with_types": "ru_RU",
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
    path: Path | None = field(default=None, compare=False, repr=False)

    def receipt_fields(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "license_class": self.license_class,
        }


def _catalog(path: Path = _CATALOG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(source_id: str) -> dict:
    aliases = {"google/tn-en_with_types": "google-tn"}
    for key, entry in _catalog().items():
        if entry.get("store_id") == source_id or aliases.get(source_id) == key:
            return entry
    raise ValueError(f"unknown corpus store id {source_id!r}")


def store_root(source_id: str) -> Path:
    corpus_root = os.environ.get("FREND_CORPORA")
    if corpus_root is None:
        raise ValueError("--corpus-dir is required when FREND_CORPORA is not set")
    entry = _entry(source_id)
    provider, store_name = source_id.split("/", 1)
    return (Path(corpus_root) / provider / store_name / str(entry["dest"])).resolve()


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
    locale: str,
    pools: tuple[str, ...],
    root: Path | None = None,
) -> tuple[VerifiedInput, ...]:
    """Verify the complete requested set before returning anything a row reader can open."""
    if not paths:
        raise ValueError("at least one corpus input is required")
    locale = canonical_locale(locale)
    expected_locale = _SOURCE_LOCALES.get(source_id)
    if expected_locale is None:
        raise ValueError(f"source {source_id!r} has no declared locale")
    if locale != expected_locale:
        raise ValueError(f"source {source_id!r} is declared for {expected_locale}, not {locale}")
    entry = _entry(source_id)
    split = CORPUS_SPLITS.get(source_id)
    if split is None:
        raise ValueError(f"source {source_id!r} has no declared split")
    allowed = split.names(pools)
    pins = entry.get("shards", {})
    root = (store_root(source_id) if root is None else Path(root)).resolve(strict=True)
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
        pending.append((relative, actual, resolved))
    license_class = _LICENSE_CLASSES[source_id]
    return tuple(
        VerifiedInput(source_id, name, digest, license_class, path)
        for name, digest, path in pending
    )


@contextmanager
def open_verified(item: VerifiedInput) -> TextIO:
    """Open a verified shard, rechecking its identity and bytes at the opening boundary."""
    if not isinstance(item, VerifiedInput):
        raise TypeError("row readers require a verified input")
    if item.path is None:
        raise ValueError("verified input has no bound path")
    path = Path(item.path)
    if path.is_symlink():
        raise ValueError(f"verified input must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved.name != item.relative_path or "/" in item.relative_path:
        raise ValueError("verified input path no longer matches its receipt identity")
    actual = _sha256(resolved)
    if actual != item.sha256:
        raise ValueError(f"{item.relative_path}: sha256 changed after verification")
    with resolved.open(encoding="utf-8") as handle:
        yield handle


def verification_receipt(
    inputs: Sequence[VerifiedInput], *, locale: str, pools: Sequence[str]
) -> dict[str, object]:
    """Canonical receipt for the exact verified inputs a producer or evaluator opens."""
    if not inputs:
        raise ValueError("cannot receipt an empty verified input set")
    source_ids = {item.source_id for item in inputs}
    if len(source_ids) != 1:
        raise ValueError("one verification receipt cannot mix corpus source ids")
    payload: dict[str, object] = {
        "schema_version": 1,
        "source_id": next(iter(source_ids)),
        "locale": canonical_locale(locale),
        "pools": sorted(set(pools)),
        "inputs": [item.receipt_fields() for item in inputs],
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def write_verification_receipt(
    path: Path, inputs: Sequence[VerifiedInput], *, locale: str, pools: Sequence[str]
) -> dict[str, object]:
    receipt = verification_receipt(inputs, locale=locale, pools=pools)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
