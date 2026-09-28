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

import re
from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "INTERNAL_ONLY",
    "LICENSE_CLASSES",
    "SHIPPABLE_CLASSES",
    "SHIPPABLE_SOURCES",
    "SOURCE_LABELS",
    "source_id",
    "source_class",
]

INTERNAL_ONLY = "internal-only"
# The shared license classes (conventions item 3, with kal's R2 ruling).
LICENSE_CLASSES = ("shippable", "shippable-share-alike", "derived-shippable", INTERNAL_ONLY)
SHIPPABLE_CLASSES = frozenset({"shippable", "shippable-share-alike", "derived-shippable"})

# id -> {"license": SPDX id or LicenseRef, "class": license class, "vendored": files
# under frend/data/ that are the source itself}. An id ending in "/" is a namespace:
# ``icu/`` admits ``icu/<version>/<family>``.
SHIPPABLE_SOURCES: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "google/tn-en_with_types": {
            "license": "CC-BY-SA-4.0",
            "class": "shippable-share-alike",
            "vendored": (),
        },
        "icu/": {"license": "Unicode-3.0", "class": "shippable", "vendored": ()},
        "iana/tlds-alpha-by-domain": {
            "license": "LicenseRef-IANA-public",
            "class": "shippable",
            "vendored": ("root/tlds-alpha-by-domain.txt",),
        },
        # kal's own break-exception lists (~/dev/lenzo/break_exceptions), curated by
        # tools/import_break_exceptions.py.
        "lenzo/break_exceptions": {
            "license": "LicenseRef-kal-own-work",
            "class": "shippable",
            "vendored": (),
        },
        # Forms frend writes by hand, each with its reason (<locale>/lexical.json).
        "frend/curated": {"license": "BSD-2-Clause", "class": "shippable", "vendored": ()},
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
    if entry is None and _ICU_ID.fullmatch(source):
        entry = SHIPPABLE_SOURCES["icu/"]
    return None if entry is None else str(entry["class"])
