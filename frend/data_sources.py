"""The sources frend may ship data from, each with its license and license class.

Every table under ``frend/data/`` names what it was made from by a source id (the
corpora conventions, item 1: kal's store id, such as ``google/tn-en_with_types``, and
``icu/<version>/<family>`` for material ICU generates). That id must be one of
:data:`SHIPPABLE_SOURCES`, and its class one frend ships (``tests/test_locale_data.py``
checks every file). An id not listed, one under ``ldc/``, or a table naming no source
is refused: frend ships nothing derived from an LDC corpus (kal, 2026-09-28), whatever
the corpus's own terms allow.

Tables built before store ids name the corpus by an older label (``corpus_label``,
``google-tn:en_with_types``); :data:`SOURCE_LABELS` maps each such label to its id, so
the tables keep their bytes and their builders' ``--check``. A file that is a source
vendored unchanged (IANA's top-level domains) carries no provenance of its own; its
source lists it under ``vendored``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType

__all__ = [
    "INTERNAL_ONLY",
    "LICENSE_CLASSES",
    "SHIPPABLE_CLASSES",
    "SHIPPABLE_SOURCES",
    "SOURCE_LABELS",
    "receipt_refusals",
    "register_receipt",
    "source_id",
    "source_class",
    "shipping_refusals",
    "validate_receipt",
]

# Receipts created in this process are indexed by canonical fingerprint so a package
# boundary can resolve an ``ancestors`` reference instead of trusting an opaque hash.
_RECEIPT_INDEX: dict[str, Mapping[str, object]] = {}


def _register_receipt(fingerprint: str, receipt: Mapping[str, object]) -> None:
    """Register a validated canonical receipt for ancestry checks."""
    _RECEIPT_INDEX[fingerprint] = receipt


INTERNAL_ONLY = "internal-only"
# The shared license classes (conventions item 3, with kal's R2 ruling).
LICENSE_CLASSES = ("shippable", "shippable-share-alike", "derived-shippable", INTERNAL_ONLY)
SHIPPABLE_CLASSES = frozenset({"shippable", "shippable-share-alike", "derived-shippable"})

# id -> SPDX/license identifier, notice, allowed use, license class, and files under
# frend/data/ that are the source itself. An id ending in "/" is a namespace.
SHIPPABLE_SOURCES: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "google/tn-en_with_types": {
            "license": "CC-BY-SA-4.0",
            "notice": "Google text normalization corpus; attribution and share-alike apply.",
            "allowed_use": "normalization evidence and ranking",
            "class": "shippable-share-alike",
            "vendored": (),
        },
        "google/tn-ru_with_types": {
            "license": "CC-BY-SA-4.0",
            "notice": "Google text normalization corpus; attribution and share-alike apply.",
            "allowed_use": "normalization evidence and ranking",
            "class": "shippable-share-alike",
            "vendored": (),
        },
        "cldr/": {
            "license": "Unicode-3.0",
            "notice": "Unicode CLDR copyright and permission notice applies.",
            "allowed_use": "locale data generation and independent public evidence",
            "class": "shippable",
            "vendored": (),
        },
        "icu/": {
            "license": "Unicode-3.0",
            "notice": "Unicode ICU copyright and permission notice applies.",
            "allowed_use": "runtime generation and independent public evidence",
            "class": "shippable",
            "vendored": (),
        },
        "unicode/": {
            "license": "Unicode-3.0",
            "notice": "Unicode data files copyright and permission notice applies.",
            "allowed_use": "character data and independent public evidence",
            "class": "shippable",
            "vendored": (),
        },
        "libphonenumber": {
            "license": "Apache-2.0",
            "notice": "libphonenumber copyright and Apache 2.0 notice applies.",
            "allowed_use": "normalization fixtures and rules",
            "class": "shippable",
            "vendored": (),
        },
        "nemo": {
            "license": "Apache-2.0",
            "notice": "NVIDIA NeMo-text-processing copyright and Apache 2.0 notice applies.",
            "allowed_use": "compatibility evidence only; never ranking evidence",
            "class": "shippable",
            "vendored": (),
        },
        "wikidata": {
            "license": "CC0-1.0",
            "notice": "Wikidata structured data is dedicated to the public domain under CC0.",
            "allowed_use": "public labels and independent public evidence",
            "class": "shippable",
            "vendored": (),
        },
        "iana/tlds-alpha-by-domain": {
            "license": "LicenseRef-IANA-public",
            "notice": "IANA Root Zone Database attribution applies.",
            "allowed_use": "vendored public suffix source data",
            "class": "shippable",
            "vendored": ("root/tlds-alpha-by-domain.txt",),
        },
        # kal's own break-exception lists, curated by tools/import_break_exceptions.py.
        "lenzo/break_exceptions": {
            "license": "LicenseRef-kal-own-work",
            "notice": "Original frend project data.",
            "allowed_use": "normalization rules",
            "class": "shippable",
            "vendored": (),
        },
        # Forms frend writes by hand, each with its reason (<locale>/lexical.json).
        "frend/curated": {
            "license": "BSD-2-Clause",
            "notice": "frend copyright and BSD 2-Clause notice applies.",
            "allowed_use": "cited normalization rules and lexical forms",
            "class": "shippable",
            "vendored": (),
        },
        # Festival's hand-curated word lists (github.com/festvox/festival): free to use
        # and distribute with its notice kept and changes marked, which
        # en/context/festival_classes.json carries.
        "festvox/festival": {
            "license": "LicenseRef-Festival",
            "notice": "The Festival notice is retained in each derived data artifact.",
            "allowed_use": "derived normalization classes with notice and marked changes",
            "class": "shippable",
            "vendored": (),
        },
    }
)

# An older label a table names its source by -> the source id; ``{field}`` is filled
# from the same provenance record.
SOURCE_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "google-tn:en_with_types": "google/tn-en_with_types",
        "google-tn-en_with_types": "google/tn-en_with_types",
        "icu-reflective-generation": "icu/{icu_version}/reflective-generation",
        "break-exceptions-en-curated": "lenzo/break_exceptions",
    }
)

_ICU_ID = re.compile(r"icu/\d+(?:\.\d+)*/[a-z0-9][a-z0-9-]*")
_VERSIONED_SOURCE_ID = re.compile(r"(?:cldr|unicode)/\d+(?:\.\d+)*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NEMO_ALLOWED_USES = frozenset({"compatibility evidence", "compatibility evidence only"})
KNOWN_INTERNAL_DIGESTS = frozenset(
    {"5d1b84c18b57c62e2c369ce93b3fdc607a2487de17acb74ec72503e242300034"}
)
_LDC = re.compile(r"(?<![a-z0-9])ldc(?:\d{2}[a-z]\d|(?=$|[^a-z0-9]))", re.IGNORECASE)


def source_id(label: str, provenance: Mapping[str, object] | None = None) -> str:
    """The source id ``label`` stands for: itself, or the id :data:`SOURCE_LABELS` maps
    it to, filled from ``provenance``. A label naming a field it lacks is itself."""
    template = SOURCE_LABELS.get(label)
    if template is None:
        return label
    try:
        return template.format_map(dict(provenance or {}))
    except KeyError:
        return label


def source_class(source: str) -> str | None:
    """The license class frend ships ``source`` under, or ``None`` when it is not a
    declared source. Every ``ldc/`` id is internal-only, declared or not."""
    if source.lower().startswith("ldc/"):
        return INTERNAL_ONLY
    entry = None if source.endswith("/") else SHIPPABLE_SOURCES.get(source)
    if entry is None:
        if _ICU_ID.fullmatch(source):
            entry = SHIPPABLE_SOURCES["icu/"]
        elif _VERSIONED_SOURCE_ID.fullmatch(source):
            entry = SHIPPABLE_SOURCES[f"{source.partition('/')[0]}/"]
    return None if entry is None else str(entry["class"])


def _nemo_use_refusals(document: Mapping[object, object]) -> tuple[str, ...]:
    if document.get("source") != "nemo":
        return ()
    use = document.get("use")
    normalized = use.strip().casefold() if isinstance(use, str) else None
    if normalized not in _NEMO_ALLOWED_USES:
        return ("nemo may be used only as compatibility evidence, never for ranking",)
    return ()


def _receipt_shape_refusals(receipt: object) -> tuple[str, ...]:
    if not isinstance(receipt, Mapping):
        return ("receipt is not a mapping",)

    refusals: list[str] = []
    source = receipt.get("source")
    if not isinstance(source, str) or source_class(source) not in SHIPPABLE_CLASSES:
        refusals.append(f"source {source!r} is not registered for shipping")

    for field in ("revision", "fetched_at"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            refusals.append(f"receipt field {field!r} is missing or empty")
    if not any(
        isinstance(receipt.get(field), str) and receipt[field].strip()
        for field in ("locator", "repository")
    ):
        refusals.append("receipt requires a nonempty locator or repository")

    for field in ("sha256", "license_sha256"):
        value = receipt.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            refusals.append(f"receipt field {field!r} is not a lowercase SHA-256")

    fetched_at = receipt.get("fetched_at")
    if isinstance(fetched_at, str) and fetched_at.strip():
        try:
            fetched = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        except ValueError:
            fetched = None
        if fetched is None or fetched.tzinfo is None:
            refusals.append("receipt field 'fetched_at' is not an ISO 8601 timestamp with timezone")

    command = receipt.get("transform_command")
    valid_command = isinstance(command, str) and bool(command.strip())
    if isinstance(command, (list, tuple)):
        valid_command = bool(command) and all(isinstance(part, str) and part for part in command)
    if not valid_command:
        refusals.append("receipt field 'transform_command' is missing or empty")

    refusals.extend(_nemo_use_refusals(receipt))

    return tuple(dict.fromkeys(refusals))


def receipt_refusals(receipt: object) -> tuple[str, ...]:
    """Return reasons an acquisition receipt cannot enter the registered catalog."""
    refusals = list(_receipt_shape_refusals(receipt))

    refusals.extend(shipping_refusals(receipt))
    return tuple(dict.fromkeys(refusals))


def validate_receipt(receipt: object) -> None:
    """Raise ``ValueError`` unless ``receipt`` satisfies acquisition intake rules."""
    refusals = receipt_refusals(receipt)
    if refusals:
        raise ValueError("; ".join(refusals))


def register_receipt(receipt: Mapping[str, object]) -> str:
    """Validate and register an acquisition receipt, returning its canonical hash."""
    validate_receipt(receipt)
    canonical = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    fingerprint = hashlib.sha256(canonical).hexdigest()
    _register_receipt(fingerprint, MappingProxyType(json.loads(canonical)))
    return fingerprint


def _validated_receipt_catalog(
    receipt_index: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, Mapping[str, object]]:
    """Return caller-supplied acquisition receipts whose contents match their keys."""
    catalog: dict[str, Mapping[str, object]] = {}
    for fingerprint, receipt in (receipt_index or {}).items():
        if receipt_refusals(receipt):
            continue
        try:
            canonical = json.dumps(
                receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        except (TypeError, ValueError):
            continue
        if (
            isinstance(fingerprint, str)
            and _SHA256.fullmatch(fingerprint)
            and hashlib.sha256(canonical).hexdigest() == fingerprint
        ):
            catalog[fingerprint] = receipt
    return catalog


def shipping_refusals(
    document: object,
    path: str = "",
    *,
    receipt_index: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[str, ...]:
    """Return package-boundary refusals, including identity hidden behind labels."""
    refusals: list[str] = []
    normalized = path.replace("\\", "/").casefold()
    if "/processed/frend/ldc/" in f"/{normalized.strip('/')}/":
        refusals.append(f"{path}: path is under the internal LDC store")

    catalog = {**_RECEIPT_INDEX, **_validated_receipt_catalog(receipt_index)}
    resolved: set[str] = set()
    walked_containers: set[int] = set()

    pending = [document]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            folded = value.casefold()
            if _LDC.search(value):
                refusals.append("document mentions an LDC corpus")
            if folded in KNOWN_INTERNAL_DIGESTS:
                refusals.append("document contains a cataloged internal digest")
            if folded == INTERNAL_ONLY:
                refusals.append("document carries internal-only ancestry")
            if value in catalog and value not in resolved:
                resolved.add(value)
                pending.append(catalog[value])
        elif isinstance(value, Mapping):
            identity = id(value)
            if identity in walked_containers:
                continue
            walked_containers.add(identity)
            refusals.extend(_nemo_use_refusals(value))
            for key, item in reversed(list(value.items())):
                if key == "ancestors":
                    ancestors = item if isinstance(item, (list, tuple, set, frozenset)) else (item,)
                    for ancestor in ancestors:
                        if not isinstance(ancestor, str) or ancestor not in catalog:
                            refusals.append(
                                f"ancestry does not close at a registered receipt: {ancestor!r}"
                            )
                pending.append(item)
                pending.append(key)
        elif isinstance(value, (list, tuple, set, frozenset)):
            identity = id(value)
            if identity in walked_containers:
                continue
            walked_containers.add(identity)
            pending.extend(reversed(list(value)))
    return tuple(dict.fromkeys(refusals))
