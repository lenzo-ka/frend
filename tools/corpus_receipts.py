"""Canonical derivation receipts with transitive strictest-license inheritance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

try:
    from corpus_inputs import VerifiedInput
except ModuleNotFoundError:  # imported as tools.corpus_receipts
    from tools.corpus_inputs import VerifiedInput

from frend.data_sources import KNOWN_INTERNAL_DIGESTS

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
        if isinstance(item, VerifiedInput):
            leaves.append(item)
            classes.append(item.license_class)
            if item.source_id.lower().startswith("ldc/") or item.sha256 in KNOWN_INTERNAL_DIGESTS:
                classes.append("internal-only")
        else:
            leaves.extend(item.inputs)
            ancestors.extend((item.fingerprint, *item.ancestors))
            classes.append(item.license_class)
    effective = max(classes or ["shippable"], key=_LICENSE_ORDER.__getitem__)
    if shipping and effective == "internal-only":
        raise ValueError("internal-only corpus ancestry cannot produce a shipping receipt")
    payload = {
        "inputs": [asdict(item) for item in leaves],
        "ancestors": sorted(set(ancestors)),
        "producer_commit": producer_commit,
        "command": list(command),
        "artifacts": dict(sorted(artifacts.items())),
        "license_class": effective,
    }
    fingerprint = hashlib.sha256(_canonical(payload)).hexdigest()
    return DerivationReceipt(
        tuple(leaves),
        tuple(payload["ancestors"]),
        producer_commit,
        command,
        dict(sorted(artifacts.items())),
        effective,
        fingerprint,
    )
