"""Crash-sweep money and adjacent measure/range paths across all shipped locales."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import icu

from frend import resolve_lattice
from frend.normalize import _currencies, _reading_detectors
from frend.verbalize import verbalize_lattice

VALUES = (0, 1, 2, 9, 10, 11, 99, 100, 101, 999, 10**3, 10**6, 10**9, 10**12)
LOCALES = (
    "en_US",
    "es_MX",
    "es_ES",
    "fr_FR",
    "de_DE",
    "pt_BR",
    "it_IT",
    "zh_CN",
    "ko_KR",
    "ja_JP",
    "pt_PT",
)


def _surfaces(locale: str, code: str, amount: Decimal) -> tuple[str, ...]:
    formatter = icu.NumberFormat.createCurrencyInstance(icu.Locale(locale))
    formatter.setCurrency(code)
    rendered = formatter.format(float(amount))
    return tuple(dict.fromkeys((f"{code} {amount}", f"{amount} {code}", rendered)))


def _cases(locale: str):
    for code in _currencies(locale):
        for value in VALUES:
            for places in (0, 1, 2):
                amount = Decimal(value) + (Decimal(1) / (Decimal(10) ** places) if places else 0)
                for text in _surfaces(locale, code, amount):
                    yield "money", text
                    yield "range", f"{text}–{text}"
    for text in ("1 kg", "1–2 kg", "1 ft 2 in", "not-money USDx 2", "$", "R$ x"):
        yield "adjacent", text


def _child(locale: str) -> dict:
    registry = _reading_detectors(locale)
    failures = []
    count = 0
    maximum = 0
    for kind, text in _cases(locale):
        count += 1
        try:
            lattice = resolve_lattice(registry.detect(text), source_text=text, locale=locale)
            spoken = verbalize_lattice(lattice, locale=locale)
            alternatives = [len(unit.alternatives) for path in spoken.paths for unit in path.units]
            maximum = max((maximum, *alternatives))
            if any(value > 128 for value in alternatives):
                raise ValueError("more than 128 alternatives")
            if any(
                edge.start < 0 or edge.end > len(text) or edge.start >= edge.end
                for edge in lattice.edges
            ):
                raise ValueError("invalid edge span")
        except Exception as error:  # noqa: BLE001 -- exception-safety sweep
            failures.append((kind, text, f"{type(error).__name__}: {error}"))
    return {
        "locale": locale,
        "count": count,
        "failures": failures,
        "max_alternatives": maximum,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-locale", choices=LOCALES)
    args = parser.parse_args()
    if args.child_locale:
        print(json.dumps(_child(args.child_locale), ensure_ascii=False))
        return 0
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONDONTWRITEBYTECODE="1"
    )
    reports = []
    for locale in LOCALES:
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--child-locale", locale],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        reports.append(json.loads(completed.stdout))
    failures = [failure for report in reports for failure in report["failures"]]
    if failures:
        for failure in failures[:100]:
            print("\t".join(failure))
        print(f"FAILED: {len(failures)} exceptions")
        return 1
    count = sum(report["count"] for report in reports)
    maximum = max(report["max_alternatives"] for report in reports)
    print(
        f"OK: {count} money/range/measure cases across {len(reports)} locales, "
        f"zero exceptions, max {maximum} alternatives"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
