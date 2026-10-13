"""Single capability declaration for generated linguistic rule bundles."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

RuleFamily = Literal["date", "time", "fraction", "money", "telephone"]

GENERATED_RULE_CAPABILITIES: Mapping[str, frozenset[RuleFamily]] = MappingProxyType(
    {
        "es_MX": frozenset(("date", "time", "fraction", "money")),
        "es_ES": frozenset(("date", "time", "fraction")),
        "fr_FR": frozenset(("date", "time", "fraction", "money", "telephone")),
        "de_DE": frozenset(("date", "time", "fraction")),
        "pt_BR": frozenset(("date", "time", "fraction", "money", "telephone")),
        "pt_PT": frozenset(("date", "time", "fraction", "telephone")),
        "it_IT": frozenset(("date", "time", "fraction", "money")),
        "zh_CN": frozenset(("date", "time", "fraction")),
        "ko_KR": frozenset(("date", "time", "fraction", "money")),
        "ja_JP": frozenset(("date", "time", "fraction")),
    }
)


def locales_for(family: RuleFamily) -> tuple[str, ...]:
    """Return locales with a generated bundle for ``family`` in declaration order."""
    return tuple(
        locale
        for locale, capabilities in GENERATED_RULE_CAPABILITIES.items()
        if family in capabilities
    )
