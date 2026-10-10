"""Run the locale normalization recipe's reflective capability probe.

Only the ``probe`` runner contract is implemented here.  Later recipe slices own
bundle construction and gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import icu

_GENDER_MARKERS = ("feminine", "masculine", "neuter")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _locale_source(formatter: Any) -> str:
    return formatter.getLocaleID(icu.ULocDataLocaleType.ACTUAL_LOCALE)


def _ruleset_names(formatter: icu.RuleBasedNumberFormat) -> list[str]:
    return [formatter.getRuleSetName(index) for index in range(formatter.getNumberOfRuleSetNames())]


def _rbnf_capability(locale: icu.Locale) -> tuple[dict[str, Any], dict[str, str]]:
    spellout = icu.RuleBasedNumberFormat(icu.URBNFRuleSetTag.SPELLOUT, locale)
    ordinal = icu.RuleBasedNumberFormat(icu.URBNFRuleSetTag.ORDINAL, locale)
    source_locale = _locale_source(spellout)
    own_language = source_locale.split("_", 1)[0] == locale.getLanguage()
    rule_sets = _ruleset_names(spellout)
    ordinal_rule_sets = _ruleset_names(ordinal)
    capability = {
        "available": True,
        "default_rule_set": spellout.getDefaultRuleSetName(),
        "fallback_source": None if own_language else source_locale,
        "gendered_rule_sets": [
            name for name in rule_sets if any(marker in name for marker in _GENDER_MARKERS)
        ],
        "ordinal": {
            "available": True,
            "default_rule_set": ordinal.getDefaultRuleSetName(),
            "rule_sets": ordinal_rule_sets,
            "source_locale": _locale_source(ordinal),
        },
        "own_language": own_language,
        "rule_sets": rule_sets,
        "source_locale": source_locale,
    }
    hashes = {
        "rbnf_rules_sha256": _sha256(spellout.getRules()),
        "rbnf_ordinal_rules_sha256": _sha256(ordinal.getRules()),
    }
    return capability, hashes


def _cldr_capability(factory: Callable[[], Any], exercise: Callable[[Any], None]) -> dict[str, Any]:
    try:
        formatter = factory()
        exercise(formatter)
    except (icu.ICUError, ValueError):
        return {"available": False, "source_locale": None}
    return {"available": True, "source_locale": _locale_source(formatter)}


def _cldr_capabilities(locale: icu.Locale) -> dict[str, dict[str, Any]]:
    return {
        "currency": _cldr_capability(
            lambda: icu.NumberFormat.createCurrencyInstance(locale),
            lambda formatter: formatter.format(1234.5),
        ),
        "date": _cldr_capability(
            lambda: icu.DateFormat.createDateInstance(icu.DateFormat.MEDIUM, locale),
            lambda formatter: formatter.format(datetime(2000, 2, 29, tzinfo=UTC)),
        ),
        "number": _cldr_capability(
            lambda: icu.NumberFormat.createInstance(locale),
            lambda formatter: formatter.format(1234.5),
        ),
        "unit": _cldr_capability(
            lambda: icu.MeasureFormat(locale, icu.UMeasureFormatWidth.WIDE),
            lambda formatter: formatter.formatMeasure(
                icu.Measure(2, icu.MeasureUnit.forIdentifier("meter"))
            ),
        ),
    }


def build_probe(locale_name: str) -> dict[str, Any]:
    """Return a deterministic capability document for one ICU locale."""
    locale = icu.Locale.createCanonical(locale_name.replace("-", "_"))
    if locale.isBogus() or not locale.getLanguage():
        raise ValueError(f"invalid locale: {locale_name!r}")

    rbnf, rbnf_hashes = _rbnf_capability(locale)
    # PyICU mutates the Locale passed to addLikelySubtags, so expand a copy.
    likely = icu.Locale.addLikelySubtags(icu.Locale(locale.getName()))
    capabilities = {
        "cldr": _cldr_capabilities(locale),
        "plural_categories": sorted(icu.PluralRules.forLocale(locale).getKeywords()),
        "rbnf": rbnf,
        "script": likely.getScript(),
    }
    document = {
        "schema": "frend-locale-capability-probe/1",
        "locale": locale.getName(),
        "versions": {
            "icu": icu.ICU_VERSION,
            "icukit": _package_version("icukit"),
            "pyicu": icu.VERSION,
            "unicode": icu.UNICODE_VERSION,
        },
        "capabilities": capabilities,
    }
    document["hashes"] = {
        **rbnf_hashes,
        "probe_payload_sha256": _sha256(_canonical_json(document)),
    }
    return document


def serialize(document: dict[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe", help="write ICU/icukit locale capabilities")
    probe.add_argument("--locale", required=True, help="ICU locale identifier")
    probe.add_argument("--out", required=True, type=Path, help="JSON output path")
    args = parser.parse_args(argv)

    document = build_probe(args.locale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(serialize(document), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
