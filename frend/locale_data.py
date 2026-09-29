"""Where frend's data lives by locale, and the order a locale looks it up in.

Tables sit under ``frend/data/<locale>/``. A locale looks along its chain, found by
truncating subtags: ``en_US`` -> ``en`` -> ``root``. A measured table (counted from one
language's corpus) is looked up along the chain but never in ``root``, so a locale with
no measured table gets ``None``, never another language's counts. A root-only table
(the ICU shape backfill, the IANA list of top-level domains) is read from ``root``.

A tag is validated and canonicalized (:func:`canonical_locale`) before it names a
directory: ``"-"`` and ``"_"`` both separate subtags, ICU's base name sets the case
(``EN-us`` -> ``en_US``), ``root`` in any case is ``root``, and anything that is not a
well-formed tag (a ``/``, ``..``, an empty subtag) is refused with ``ValueError``.

CLDR ``parentLocales`` exceptions are not handled: truncation is exact for en and ru.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from types import MappingProxyType

import icu

__all__ = [
    "ROOT",
    "canonical_locale",
    "lexical_forms",
    "locale_chain",
    "measured_directory",
    "measured_table",
    "root_table",
]

ROOT = "root"
# Locale-keyed caches hold at most this many locales (absent ones included).
LOCALE_CACHE = 16


# A well-formed tag's shape (BCP 47's language, script, region and variant subtags; no
# extensions), with "-" or "_" between subtags. Anything else never reaches a path.
_WELL_FORMED = re.compile(
    r"(?:[A-Za-z]{2,3}|[A-Za-z]{5,8})"  # language
    r"(?:[-_][A-Za-z]{4})?"  # script
    r"(?:[-_](?:[A-Za-z]{2}|[0-9]{3}))?"  # region
    r"(?:[-_](?:[A-Za-z0-9]{5,8}|[0-9][A-Za-z0-9]{3}))*",  # variants
)
_CANONICAL = re.compile(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*")


def canonical_locale(locale: str) -> str:
    """``locale`` as the one spelling its data directory and caches use.

    ``"EN-US"``, ``"en-us"`` and ``"en_US"`` are all ``"en_US"`` (ICU's base name:
    language lower, script title, region upper); ``root`` in any case, and ``und``
    (CLDR's code for root, with any script or region), is ``"root"``.
    Raises ``ValueError`` for anything that is not a well-formed tag.
    """
    if not isinstance(locale, str):
        raise ValueError(f"not a locale tag: {locale!r}")
    if locale.lower() == ROOT:
        return ROOT
    if not _WELL_FORMED.fullmatch(locale):
        raise ValueError(f"not a locale tag: {locale!r}")
    if re.split(r"[-_]", locale)[0].lower() == "und":
        # CLDR's root locale is "und" (ICU's base name for it is empty).
        return ROOT
    parsed = icu.Locale(locale.replace("-", "_"))
    variants = [variant for variant in parsed.getVariant().split("_") if variant]
    if len(set(variants)) != len(variants):
        # BCP 47 allows no repeated variant ("en_US_POSIX_POSIX").
        raise ValueError(f"not a locale tag: {locale!r}")
    canonical = parsed.getBaseName()
    if not _CANONICAL.fullmatch(canonical):
        raise ValueError(f"not a locale tag: {locale!r}")
    return canonical


def locale_chain(locale: str) -> tuple[str, ...]:
    """The locales ``locale`` looks in, most specific first, ending at ``root``.

    ``"en_US"`` -> ``("en_US", "en", "root")``; the tag is canonicalized first
    (:func:`canonical_locale`), so ``"EN-us"`` walks the same chain.
    """
    canonical = canonical_locale(locale)
    if canonical == ROOT:
        return (ROOT,)
    tags = canonical.split("_")
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


def measured_directory(name: str, locale: str) -> Traversable | None:
    """The measured data directory ``name`` for ``locale`` (``data/<locale>/<name>/``,
    such as the context trees), or ``None`` when its chain has none; like a measured
    table, never read from ``root``."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"not a directory name: {name!r}")
    for tag in locale_chain(locale):
        if tag == ROOT:
            break
        resource = _data().joinpath(tag, name)
        if resource.is_dir():
            return resource
    return None


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
