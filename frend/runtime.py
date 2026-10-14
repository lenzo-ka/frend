"""Process-level helpers for callers that keep frend loaded and resolve many texts.

Nothing here runs on import: changing the interpreter's garbage collector is the
caller's decision, made once its own setup is done.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable

from frend.generated_rule_locales import locales_for
from frend.locale_data import canonical_locale

_ENGLISH_DATE_PROBE = "March 3, 2020"


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
        if locale == "en_US":
            normalize(_ENGLISH_DATE_PROBE, locale=locale)
        else:
            # Load compiled paths directly. Several locale date probes use slashes
            # and also form valid fraction candidates; normalizing them here would
            # defeat the fraction path's first-use laziness.
            from frend.date_rules import load_date_rule_bundle
            from frend.electronic_rules import load_electronic_rule_bundle
            from frend.money_rules import load_money_rule_bundle
            from frend.telephone_rules import regional_telephone_formats
            from frend.time_rules import load_time_rule_bundle

            if locale in locales_for("date"):
                load_date_rule_bundle(locale)
            if locale in locales_for("time"):
                load_time_rule_bundle(locale)
            if locale in locales_for("money"):
                load_money_rule_bundle(locale)
            if locale in locales_for("telephone"):
                regional_telephone_formats(locale)
            if locale in locales_for("electronic"):
                load_electronic_rule_bundle(locale)
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
