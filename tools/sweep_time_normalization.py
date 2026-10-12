"""Exercise every clock value and ICU 12/24-hour period spelling without exceptions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import icu

from frend import prewarm, resolve_lattice
from frend.normalize import _reading_detectors
from frend.time_rules import GENERATED_TIME_LOCALES
from frend.verbalize import verbalize_lattice

LOCALES = (*GENERATED_TIME_LOCALES, "en_US")


def _formatter(locale: str, skeleton: str) -> icu.SimpleDateFormat:
    pattern = icu.DateTimePatternGenerator.createInstance(icu.Locale(locale)).getBestPattern(
        skeleton
    )
    formatter = icu.SimpleDateFormat(pattern, icu.Locale(locale))
    formatter.setTimeZone(icu.TimeZone.createTimeZone("UTC"))
    return formatter


def cases(*, full: bool, locales=LOCALES):
    seconds = range(60) if full else (0, 1, 5, 59)
    for locale in locales:
        formatters = {
            skeleton: _formatter(locale, skeleton) for skeleton in ("hm", "Bhm", "hms", "Bhms")
        }
        zone = icu.TimeZone.createTimeZone("UTC")
        for hour in range(24):
            for minute in range(60):
                minute_calendar = icu.GregorianCalendar(zone, icu.Locale(locale))
                minute_calendar.clear()
                minute_calendar.set(2000, 0, 1, hour, minute, 0)
                yield locale, f"{hour:02d}:{minute:02d}"
                for skeleton in ("hm", "Bhm"):
                    yield locale, formatters[skeleton].format(minute_calendar)
                for second in seconds:
                    calendar = icu.GregorianCalendar(zone, icu.Locale(locale))
                    calendar.clear()
                    calendar.set(2000, 0, 1, hour, minute, second)
                    yield locale, f"{hour:02d}:{minute:02d}:{second:02d}"
                    for skeleton in ("hms", "Bhms"):
                        yield locale, formatters[skeleton].format(calendar)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use four second values instead of the release-gate exhaustive 0..59 sweep",
    )
    parser.add_argument("--child-locale", choices=LOCALES)
    args = parser.parse_args()
    if args.child_locale is None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        processes = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--child-locale",
                    locale,
                    *(["--quick"] if args.quick else []),
                ],
                env=environment,
                stdout=subprocess.PIPE,
                text=True,
            )
            for locale in LOCALES
        ]
        reports = []
        for process in processes:
            stdout, _stderr = process.communicate()
            if process.returncode:
                print(stdout, end="")
                return process.returncode
            reports.append(json.loads(stdout))
        count = sum(report["count"] for report in reports)
        failures = sum(report["failures"] for report in reports)
        if failures:
            print(f"FAILED: {failures} exceptions across {count} cases")
            return 1
        print(f"OK: {count} cases across {len(LOCALES)} locales, zero exceptions")
        return 0
    locales = (args.child_locale,)
    prewarm(locales)
    count = 0
    failures: list[tuple[str, str, str]] = []
    registries = {locale: _reading_detectors(locale) for locale in locales}
    for locale, text in cases(full=not args.quick, locales=locales):
        count += 1
        try:
            lattice = resolve_lattice(
                registries[locale].detect(text), source_text=text, locale=locale
            )
            spoken = verbalize_lattice(lattice, locale=locale)
            if any(len(unit.alternatives) > 128 for path in spoken.paths for unit in path.units):
                raise ValueError("more than 128 alternatives")
            if any(
                edge.start < 0 or edge.end > len(text) or edge.start >= edge.end
                for edge in lattice.edges
            ):
                raise ValueError("invalid or unstable edge span")
        except Exception as exc:  # noqa: BLE001 -- exception-safety sweep
            failures.append((locale, text, f"{type(exc).__name__}: {exc}"))
    if failures:
        for locale, text, error in failures[:100]:
            print(f"{locale}\t{text}\t{error}")
        print(f"FAILED: {len(failures)} exceptions across {count} cases")
        return 1
    print(json.dumps({"locale": args.child_locale, "count": count, "failures": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
