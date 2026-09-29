"""Canonical derivation receipts with transitive strictest-license inheritance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

try:
    from corpus_inputs import VerifiedInput
except ModuleNotFoundError:  # imported as tools.corpus_receipts
    from tools.corpus_inputs import VerifiedInput

from frend.data_sources import KNOWN_INTERNAL_DIGESTS, _register_receipt

_LICENSE_ORDER = {
    "shippable": 0,
    "shippable-share-alike": 1,
    "derived-shippable": 2,
    "internal-only": 3,
}


@dataclass(frozen=True)
class DerivationReceipt:
    inputs: tuple[VerifiedInput, ...]
    ancestors: tuple[str, ...]
    producer_commit: str
    command: tuple[str, ...]
    artifacts: Mapping[str, str]
    license_class: str
    fingerprint: str


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _leaf_class(item: VerifiedInput) -> str:
    classes = [item.license_class]
    if item.source_id.lower().startswith("ldc/") or item.sha256 in KNOWN_INTERNAL_DIGESTS:
        classes.append("internal-only")
    try:
        return max(classes, key=_LICENSE_ORDER.__getitem__)
    except KeyError as exc:
        raise ValueError(f"unknown license class {exc.args[0]!r}") from exc


def _payload(receipt: DerivationReceipt) -> dict[str, object]:
    return {
        "inputs": [item.receipt_fields() for item in receipt.inputs],
        "ancestors": sorted(set(receipt.ancestors)),
        "producer_commit": receipt.producer_commit,
        "command": list(receipt.command),
        "artifacts": dict(sorted(receipt.artifacts.items())),
        "license_class": receipt.license_class,
    }


def _validate_parent(receipt: DerivationReceipt) -> None:
    fingerprint = hashlib.sha256(_canonical(_payload(receipt))).hexdigest()
    if fingerprint != receipt.fingerprint:
        raise ValueError("parent receipt fingerprint is not canonical for its contents")
    effective = max(
        (_leaf_class(item) for item in receipt.inputs),
        default="shippable",
        key=_LICENSE_ORDER.__getitem__,
    )
    if receipt.license_class != effective:
        raise ValueError(
            f"parent receipt license class {receipt.license_class!r} does not match "
            f"flattened leaves {effective!r}"
        )


def derive_receipt(
    *,
    inputs: Sequence[VerifiedInput | DerivationReceipt],
    producer_commit: str,
    command: tuple[str, ...],
    artifacts: Mapping[str, str],
    shipping: bool,
) -> DerivationReceipt:
    leaves: list[VerifiedInput] = []
    ancestors: list[str] = []
    classes: list[str] = []
    for item in inputs:
        if isinstance(item, DerivationReceipt):
            _validate_parent(item)
            leaves.extend(item.inputs)
            ancestors.extend((item.fingerprint, *item.ancestors))
            classes.extend(_leaf_class(leaf) for leaf in item.inputs)
        else:
            leaves.append(item)
            classes.append(_leaf_class(item))
    effective = max(classes or ["shippable"], key=_LICENSE_ORDER.__getitem__)
    if shipping and effective == "internal-only":
        raise ValueError("internal-only corpus ancestry cannot produce a shipping receipt")
    payload = {
        "inputs": [item.receipt_fields() for item in leaves],
        "ancestors": sorted(set(ancestors)),
        "producer_commit": producer_commit,
        "command": list(command),
        "artifacts": dict(sorted(artifacts.items())),
        "license_class": effective,
    }
    fingerprint = hashlib.sha256(_canonical(payload)).hexdigest()
    receipt = DerivationReceipt(
        tuple(leaves),
        tuple(payload["ancestors"]),
        producer_commit,
        command,
        dict(sorted(artifacts.items())),
        effective,
        fingerprint,
    )
    _register_receipt(fingerprint, _payload(receipt))
    return receipt
