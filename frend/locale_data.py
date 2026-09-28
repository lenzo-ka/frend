"""Where frend's data lives by locale, and the order a locale looks it up in.

Tables sit under ``frend/data/<locale>/``. A locale looks along its chain, found by
truncating subtags: ``en_US`` -> ``en`` -> ``root``. A measured table (counted from one
language's corpus) is looked up along the chain but never in ``root``, so a locale with
no measured table gets ``None``, never another language's counts. A root-only table
(the ICU shape backfill, the IANA list of top-level domains) is read from ``root``.

CLDR ``parentLocales`` exceptions are not handled: truncation is exact for en and ru.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from types import MappingProxyType

__all__ = [
    "ROOT",
    "lexical_forms",
    "locale_chain",
    "measured_table",
    "root_table",
]

ROOT = "root"


def locale_chain(locale: str) -> tuple[str, ...]:
    """The locales ``locale`` looks in, most specific first, ending at ``root``.

    ``"en_US"`` -> ``("en_US", "en", "root")``; a hyphen separates subtags as an
    underscore does.
    """
    if not locale or locale == ROOT:
        return (ROOT,)
    tags = locale.replace("-", "_").split("_")
    return (*("_".join(tags[:count]) for count in range(len(tags), 0, -1)), ROOT)


def _data() -> Traversable:
    return files("frend").joinpath("data")


def _file_name(name: str) -> str:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"not a table name: {name!r}")
    return f"{name}.json"


def _measured_resource(file_name: str, locale: str) -> Traversable | None:
    for tag in locale_chain(locale):
        if tag == ROOT:
            break
        resource = _data().joinpath(tag, file_name)
        if resource.is_file():
            return resource
    return None


def measured_table(name: str, locale: str) -> dict | None:
    """The measured table ``name`` for ``locale``, or ``None`` when its chain has none.

    ``root`` is never consulted: a measured table is one language's counts.
    """
    resource = _measured_resource(_file_name(name), locale)
    if resource is None:
        return None
    return json.loads(resource.read_text(encoding="utf-8"))


def root_table(name: str) -> dict:
    """The cross-locale table ``name``, read from ``root``."""
    resource = _data().joinpath(ROOT, _file_name(name))
    return json.loads(resource.read_text(encoding="utf-8"))


def lexical_forms(locale: str) -> Mapping[str, object]:
    """The locale's lexical forms (``lexical.json``'s ``forms``); empty when it has none."""
    document = measured_table("lexical", locale)
    if document is None:
        return MappingProxyType({})
    return MappingProxyType(dict(document.get("forms", {})))
