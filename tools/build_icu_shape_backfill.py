"""Build the reflective ICU shape-likelihood count table.

This is generated evidence, not corpus evidence: a fixed cross-locale panel and
fixed values are formatted through ICU, reduced with :func:`irn.shape.shape`, and
tallied as integer ``P(shape | class)`` support counts. Scientific and compact
notation are deliberately outside v1 because each is a separate notational
family needing its own class, values, and recipes; folding either into cardinal
or decimal would make those likelihoods ambiguous.

Usage::

    python tools/build_icu_shape_backfill.py
    python tools/build_icu_shape_backfill.py --check
    python tools/build_icu_shape_backfill.py --out /tmp/backfill.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import icu
from icukit.locale import format_currency, format_number, format_ordinal, format_percent
from icukit.measure import format_measure

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "irn" / "data" / "icu_shape_backfill.json"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from irn.shape import shape  # noqa: E402 -- make direct script execution work

CLASSES = (
    "cardinal",
    "decimal",
    "percent",
    "date",
    "time",
    "money",
    "fraction",
    "ordinal",
    "measure",
)

# The panel is intentionally small and reviewable while spanning scripts, native
# digits, grouping systems, currencies, calendars, and 12/24-hour conventions.
LOCALE_PANEL: tuple[tuple[str, str], ...] = (
    ("en_US", "Latin baseline; comma grouping, USD, and 12-hour time."),
    ("de_DE", "German conventions; decimal comma and 24-hour time."),
    ("fr_FR", "French spacing/grouping and localized unit/currency placement."),
    ("es_ES", "A second major Romance convention and European currency forms."),
    ("ru_RU", "Cyrillic script and Russian plural categories."),
    ("el_GR", "Greek script with European number and currency conventions."),
    ("ar_EG", "Arabic script, Arabic digits, RTL layout, and EGP."),
    ("fa_IR", "Persian script/digits, Persian calendar default, and IRR."),
    ("hi_IN", "Indian grouping in a Latin-digit locale."),
    ("hi_IN@numbers=deva", "Explicit Devanagari digits plus Indian grouping."),
    ("bn_BD", "Bengali digits/script and BDT currency."),
    ("th_TH", "Thai script and Buddhist-calendar locale behavior."),
    ("ja_JP", "Japanese script, JPY zero-minor-unit currency, and date order."),
    ("zh_CN", "Simplified Han date/unit forms and CNY currency."),
    ("ko_KR", "Hangul date/unit forms and KRW zero-minor-unit currency."),
)

# Explicit values probe sign, digit width, grouping thresholds, and common
# cardinal boundary patterns without any sampling or randomness.
CARDINAL_VALUES = (
    0,
    1,
    2,
    5,
    9,
    10,
    11,
    12,
    20,
    21,
    99,
    100,
    101,
    999,
    1000,
    1001,
    1234,
    10000,
    100000,
    1000000,
    -1,
    -1234,
)
# Explicit fractional parts probe precision, trailing widths, grouping, and sign.
DECIMAL_VALUES = (0.1, 0.5, 1.2, 1.25, 1.234, 12.05, 1234.5, -0.5)
# Ratios deliberately include zero, common percentages, fractions, and >100%.
PERCENT_VALUES = (0, 0.01, 0.05, 0.10, 0.125, 0.5, 0.99, 1.0, 1.25)
# Dates vary day/month widths, month names, weekdays, and years at fixed GMT.
DATETIME_VALUES = (
    datetime(1999, 1, 2, 0, 5, 7, tzinfo=UTC),
    datetime(2001, 11, 12, 9, 8, 6, tzinfo=UTC),
    datetime(2024, 2, 29, 13, 5, 9, tzinfo=UTC),
    datetime(2030, 12, 31, 23, 59, 58, tzinfo=UTC),
)
DATE_SKELETONS = ("yMd", "Md", "yMMMd", "MMMd", "yMMMEd")
TIME_SKELETONS = ("Hm", "Hms", "hm", "hms")
# Default currency is crossed with common amount forms; the controlled ISO set
# checks symbol/code and minor-unit differences without enumerating currencies.
MONEY_VALUES = (
    (0, None),
    (1, None),
    (12, None),
    (12.5, None),
    (1234.56, None),
    (-12.5, None),
    (1234.56, "USD"),
    (1234.56, "EUR"),
    (1234.56, "JPY"),
    (1234.56, "INR"),
)
# Proper, improper, mixed, and one/two-digit numerator/denominator structures.
FRACTION_VALUES = (
    (None, 1, 2),
    (None, 2, 3),
    (None, 9, 10),
    (None, 11, 12),
    (None, 13, 7),
    (None, 25, 12),
    (1, 1, 2),
    (2, 3, 4),
    (12, 11, 16),
)
# English suffix boundaries also expose punctuation/word behavior elsewhere.
ORDINAL_VALUES = (1, 2, 3, 4, 10, 11, 12, 13, 20, 21, 22, 23, 100, 101)
# Values cover zero/one/two/fractional plural behavior; units span five domains.
MEASURE_VALUES = (0, 1, 2, 5, 1.5)
MEASURE_UNITS = ("meter", "kilogram", "hour", "celsius", "liter")
MEASURE_WIDTHS = ("WIDE", "SHORT", "NARROW")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _skeleton_surface(value: datetime, locale: str, skeleton: str) -> str:
    loc = icu.Locale(locale)
    pattern = icu.DateTimePatternGenerator.createInstance(loc).getBestPattern(skeleton)
    formatter = icu.SimpleDateFormat(pattern, loc)
    formatter.setTimeZone(icu.TimeZone.createTimeZone("GMT"))
    return formatter.format(value.timestamp())


def _fraction_surface(value: tuple[int | None, int, int], locale: str) -> str:
    """Assemble a fraction from locale digits, as the runtime detector does."""
    integer, numerator, denominator = value
    formatter = icu.NumberFormat.createInstance(icu.Locale(locale))
    formatter.setGroupingUsed(False)
    parts = [formatter.format(numerator), "/", formatter.format(denominator)]
    if integer is not None:
        parts[:0] = [formatter.format(integer), " "]
    return "".join(parts)


def _formatters(class_name: str, locale: str, manifests: dict[str, Any]) -> list[Callable[[], str]]:
    """Return one deferred call per combination so failures stay granular."""
    if class_name == "cardinal":
        return [partial(format_number, value, locale) for value in manifests[class_name]]
    if class_name == "decimal":
        return [partial(format_number, value, locale) for value in manifests[class_name]]
    if class_name == "percent":
        return [partial(format_percent, value, locale) for value in manifests[class_name]]
    if class_name in ("date", "time"):
        return [
            partial(_skeleton_surface, value, locale, skeleton)
            for value in manifests[class_name]
            for skeleton in manifests[f"{class_name}_skeletons"]
        ]
    if class_name == "money":
        return [
            partial(format_currency, value, locale, currency)
            for value, currency in manifests[class_name]
        ]
    if class_name == "fraction":
        return [partial(_fraction_surface, value, locale) for value in manifests[class_name]]
    if class_name == "ordinal":
        return [partial(format_ordinal, value, locale) for value in manifests[class_name]]
    if class_name == "measure":
        return [
            partial(format_measure, value, unit, locale, width)
            for value in manifests[class_name]
            for unit in manifests["measure_units"]
            for width in manifests["measure_widths"]
        ]
    raise ValueError(f"unknown class: {class_name}")


def default_manifests() -> dict[str, Any]:
    """Return the fixed v1 recipe inputs in an injectable form for fast tests."""
    return {
        "cardinal": CARDINAL_VALUES,
        "decimal": DECIMAL_VALUES,
        "percent": PERCENT_VALUES,
        "date": DATETIME_VALUES,
        "date_skeletons": DATE_SKELETONS,
        "time": DATETIME_VALUES,
        "time_skeletons": TIME_SKELETONS,
        "money": MONEY_VALUES,
        "fraction": FRACTION_VALUES,
        "ordinal": ORDINAL_VALUES,
        "measure": MEASURE_VALUES,
        "measure_units": MEASURE_UNITS,
        "measure_widths": MEASURE_WIDTHS,
    }


def build_document(
    locales: tuple[str, ...] | None = None, manifests: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Format the fixed panel and return a canonical generated-count document."""
    locale_ids = locales or tuple(locale for locale, _rationale in LOCALE_PANEL)
    recipes = manifests or default_manifests()
    counts: dict[str, Counter[str]] = {class_name: Counter() for class_name in CLASSES}
    skipped = Counter()
    for class_name in CLASSES:
        for locale in locale_ids:
            for formatter in _formatters(class_name, locale, recipes):
                try:
                    surface = formatter()
                    counts[class_name][shape(surface)] += 1
                except Exception as error:  # ICU support varies by locale/formatter
                    skipped[f"{class_name}:{type(error).__name__}"] += 1

    canonical_counts = {
        class_name: dict(sorted(counts[class_name].items())) for class_name in CLASSES
    }
    return {
        "schema_version": 1,
        "kind": "icu-generated-shape-likelihoods",
        "counts_by_class": canonical_counts,
        "totals_by_class": {
            class_name: sum(canonical_counts[class_name].values()) for class_name in CLASSES
        },
        "diagnostics": {
            "skipped": sum(skipped.values()),
            "skipped_by_class_and_error": dict(sorted(skipped.items())),
        },
        "provenance": {
            "source": "icu-reflective-generation",
            "status": "generated-estimate",
            "icu_version": icu.ICU_VERSION,
            "unicode_version": icu.UNICODE_VERSION,
            "icukit_version": _package_version("icukit"),
            "locale_manifest": "fixed-diversity-panel-v1",
            "value_manifest_version": 1,
            "recipe_version": 1,
            "note": "Generated ICU estimate P(shape|class); NOT measured corpus evidence.",
        },
    }


def serialize(document: dict[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit nonzero if the output drifts")
    parser.add_argument("--out", type=Path, default=_OUT, help="output path")
    args = parser.parse_args(argv)
    rendered = serialize(build_document())
    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != rendered:
            print(
                f"drift: {args.out} is missing or out of date; rerun build_icu_shape_backfill.py",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {args.out} is up to date")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
