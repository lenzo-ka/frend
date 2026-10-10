"""Process-level helpers for callers that keep frend loaded and resolve many texts.

Nothing here runs on import: changing the interpreter's garbage collector is the
caller's decision, made once its own setup is done.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable

from frend.locale_data import canonical_locale

_DATE_PROBES = {
    "de_DE": "02.03.2003",
    "en_US": "March 3, 2020",
    "es_ES": "15/03/2024",
    "es_MX": "15/03/2024",
    "fr_FR": "02.03.2003",
    "it_IT": "02/03/2003",
    "ja_JP": "2024/01/30",
    "ko_KR": "2024. 6. 30.",
    "pt_BR": "15/03/2024",
    "pt_PT": "01/01/2000",
    "zh_CN": "2019/2/10",
}


def prewarm(locales: Iterable[str], *, probe_text: str = "123") -> tuple[str, ...]:
    """Load and compile the production path once per canonical locale, in input order.

    This is an optional startup operation.  Normalization remains lazy by default,
    and prewarming never freezes the garbage collector.  The first normalization
    leaves that locale's retained icukit detector gang and its shared scan tables
    resident for later requests.
    """
    if isinstance(locales, (str, bytes)):
        raise TypeError("locales must be an iterable of locale tags, not str or bytes")

    from frend.normalize import normalize

    order = tuple(dict.fromkeys(canonical_locale(locale) for locale in locales))
    for locale in order:
        normalize(probe_text, locale=locale)
        if date_probe := _DATE_PROBES.get(locale):
            normalize(date_probe, locale=locale)
            if locale != "en_US":
                # Lowering is locale-bound and must be resident even when a caller's
                # sentence breaker ranks a narrower numeric path for the written probe.
                from frend.date_rules import load_date_rule_bundle

                load_date_rule_bundle(locale)
        if locale != "en_US":
            from frend.fraction_rules import GENERATED_FRACTION_LOCALES

            if locale in GENERATED_FRACTION_LOCALES:
                normalize("3/7", locale=locale)
                normalize("45%", locale=locale)
    return order


def freeze_after_setup() -> int:
    """Move every object alive now into the collector's permanent generation.

    Call it once, after setup: after the readers are built (``reading_detectors``),
    the priors and tables are loaded, and a first text has been resolved and
    verbalized so every lazy cache is filled. Those objects live for the process, yet
    each full collection walks all of them; frozen, they are never scanned again, so
    a later collection pays only for what a resolve allocates. In a server that
    forks workers, call it in the parent before forking: frozen objects are also left
    untouched by the collector in the children, so their pages stay shared.

    It collects first, so garbage made during setup is freed rather than frozen, and
    returns how many objects are now frozen (``gc.get_freeze_count()``). Objects
    made later are collected as usual. ``gc.unfreeze()`` undoes it.
    """
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()
