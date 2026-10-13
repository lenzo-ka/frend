#!/usr/bin/env python3
"""Fingerprint every declared generated-rule composition branch and output field."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Mapping
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(REPO))

from icukit.detectors import Capture, DateTimeValue, NumberValue  # noqa: E402

from frend.date_rules import generate_date_alternatives  # noqa: E402
from frend.fraction_rules import (  # noqa: E402
    generate_fraction_alternatives,
    generate_percent_alternatives,
)
from frend.generated_rule_locales import locales_for  # noqa: E402
from frend.money_rules import generate_money_alternatives, load_money_rule_bundle  # noqa: E402
from frend.time_rules import (  # noqa: E402
    _WRITTEN_UHR_MARKER,
    generate_time_alternatives,
    load_time_rule_bundle,
)

DEFAULT_BASELINE = REPO / "tests" / "data" / "rule_composition_fingerprint.json"


class Fingerprint:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.inputs = 0
        self.readings = 0

    def add(self, case_id: str, alternatives: Iterable[object]) -> None:
        rows = [
            [
                item.text,
                item.provenance,
                None if item.weight is None else str(item.weight),
                item.group,
                item.role,
                item.prior_provenance,
            ]
            for item in alternatives
        ]
        line = json.dumps([case_id, rows], ensure_ascii=False, separators=(",", ":")) + "\n"
        self.digest.update(line.encode())
        self.inputs += 1
        self.readings += len(rows)

    def summary(self) -> dict[str, object]:
        return {
            "sha256": self.digest.hexdigest(),
            "inputs": self.inputs,
            "readings": self.readings,
        }


def _capture(name: str, text: str, value: object, form: str = "numeric") -> Capture:
    return Capture(name, 0, len(text), text, value, form)


def _date_fingerprint() -> dict[str, object]:
    result = Fingerprint()
    first = date(2000, 1, 1)
    for locale in locales_for("date"):
        for offset in range(366):
            current = first + timedelta(days=offset)
            for kind, fields in (
                ("Md", (("M", current.month), ("d", current.day))),
                ("yMd", (("y", current.year), ("M", current.month), ("d", current.day))),
            ):
                value = DateTimeValue(fields, "gregorian")
                result.add(
                    f"date:{locale}:{kind}:{current.isoformat()}",
                    generate_date_alternatives(value, {"value": value, "captures": ()}, locale),
                )
    return result.summary()


def _fraction_detection(
    numerator: int, denominator: int, *, whole: int | None = None, negative: bool = False
) -> dict[str, object]:
    captures = []
    if negative:
        captures.append(_capture("sign", "-", None, "symbol"))
    if whole is not None:
        captures.append(_capture("whole", str(whole), whole))
    captures.extend(
        (
            _capture("numerator", str(numerator), numerator),
            _capture("denominator", str(denominator), denominator),
        )
    )
    return {"captures": tuple(captures)}


def _fraction_fingerprint() -> dict[str, object]:
    result = Fingerprint()
    for locale in locales_for("fraction"):
        for denominator in range(1, 1201):
            for numerator in range(1, 13):
                result.add(
                    f"fraction:{locale}:{numerator}/{denominator}",
                    generate_fraction_alternatives(
                        _fraction_detection(numerator, denominator), locale
                    ),
                )
            result.add(
                f"fraction:{locale}:2 3/{denominator}",
                generate_fraction_alternatives(
                    _fraction_detection(3, denominator, whole=2), locale
                ),
            )
        for amount in range(1001):
            result.add(
                f"percent:{locale}:{amount}",
                generate_percent_alternatives(Decimal(amount), {"captures": ()}, locale),
            )
        result.add(
            f"fraction:{locale}:-3/7",
            generate_fraction_alternatives(_fraction_detection(3, 7, negative=True), locale),
        )
    return result.summary()


def _time_detection(
    shape: str,
    hour: int,
    minute: int = 0,
    second: int = 0,
    *,
    zone: Capture | None = None,
    written_uhr: bool = False,
) -> tuple[DateTimeValue, dict[str, object]]:
    fields: list[tuple[str, int]] = [("H", hour)]
    if "M" in shape:
        fields.append(("m", minute))
    if "S" in shape:
        fields.append(("s", second))
    captures = []
    if shape.startswith("P"):
        written_hour = 12 if hour % 12 == 0 else hour % 12
        captures.extend(
            (
                _capture("H", str(written_hour), written_hour),
                _capture("day-period", " PM" if hour >= 12 else " AM", None, "symbol"),
            )
        )
    if zone is not None:
        captures.append(zone)
    value = DateTimeValue(tuple(fields), "gregorian")
    detection: dict[str, object] = {"value": value, "captures": tuple(captures)}
    if written_uhr:
        detection[_WRITTEN_UHR_MARKER] = True
    return value, detection


def _time_fingerprint() -> dict[str, object]:
    result = Fingerprint()
    scenario_counts = {
        "day_period": 0,
        "long_zone": 0,
        "short_zone": 0,
        "iana_zone": 0,
        "written_uhr_zero": 0,
        "written_uhr_nonzero": 0,
        "refusal_non_german_uhr": 0,
        "refusal_uhr_seconds": 0,
        "refusal_incongruent_period": 0,
    }
    for locale in locales_for("time"):
        bundle = load_time_rule_bundle(locale)
        for shape, recipes in bundle.recipes.items():
            if not recipes:
                continue
            minutes = range(60) if "M" in shape else (0,)
            seconds = (0, 1, 5, 30, 59) if "S" in shape else (0,)
            for hour in range(24):
                for minute in minutes:
                    for second in seconds:
                        value, detection = _time_detection(shape, hour, minute, second)
                        result.add(
                            f"time:{locale}:{shape}:{hour:02}:{minute:02}:{second:02}",
                            generate_time_alternatives(value, detection, locale),
                        )
                        scenario_counts["day_period"] += shape.startswith("P")

    tail_cases = (
        (
            "long_zone",
            "es_ES",
            Capture("time-zone", 0, 22, " Eastern Standard Time", None, None),
        ),
        ("short_zone", "es_ES", Capture("time-zone", 0, 3, " ET", "ET", None)),
        (
            "iana_zone",
            "es_ES",
            Capture("time-zone", 0, 4, " IST", "Europe/Dublin", None),
        ),
    )
    for name, locale, zone in tail_cases:
        value, detection = _time_detection("PHM", 15, 7, zone=zone)
        alternatives = generate_time_alternatives(value, detection, locale)
        if not alternatives:
            raise AssertionError(f"{name} produced no alternatives")
        result.add(f"time-capture:{name}", alternatives)
        scenario_counts[name] += 1

    for minute, name in ((0, "written_uhr_zero"), (7, "written_uhr_nonzero")):
        value, detection = _time_detection("HM", 8, minute, written_uhr=True)
        alternatives = generate_time_alternatives(value, detection, "de_DE")
        result.add(f"time-capture:{name}", alternatives)
        scenario_counts[name] += 1

    refusal_cases: tuple[tuple[str, str, DateTimeValue, Mapping[str, object]], ...] = ()
    value, detection = _time_detection("HM", 8, 0, written_uhr=True)
    refusal_cases += (("refusal_non_german_uhr", "es_ES", value, detection),)
    value, detection = _time_detection("HMS", 8, 0, 1, written_uhr=True)
    refusal_cases += (("refusal_uhr_seconds", "de_DE", value, detection),)
    value = DateTimeValue((("H", 15), ("m", 7)), "gregorian")
    detection = {
        "value": value,
        "captures": (
            _capture("H", "2", 2),
            _capture("day-period", " PM", None, "symbol"),
        ),
    }
    refusal_cases += (("refusal_incongruent_period", "es_ES", value, detection),)
    for name, locale, value, detection in refusal_cases:
        try:
            generate_time_alternatives(value, detection, locale)
        except NotImplementedError:
            scenario_counts[name] += 1
            result.add(f"time-capture:{name}", ())
        else:
            raise AssertionError(f"{name} did not refuse")
    if not all(scenario_counts.values()):
        raise AssertionError(f"missing time scenarios: {scenario_counts}")
    return {**result.summary(), "scenario_counts": scenario_counts}


def _money_fingerprint() -> dict[str, object]:
    result = Fingerprint()
    amounts = {
        "major": ("0", "1", "2", "9", "10", "11", "99", "100", "101", "999", "1000"),
        "major_minor": ("1.01", "1.02", "2.01", "2.02", "10.11", "-1.09"),
        "minor_only": ("0.01", "0.02", "0.11", "-0.53"),
    }
    for locale in locales_for("money"):
        for key in load_money_rule_bundle(locale).recipes:
            shape, currency = key.split(":", 1)
            for amount in amounts[shape]:
                value = NumberValue(amount, currency=currency)
                result.add(
                    f"money:{locale}:{key}:{amount}",
                    generate_money_alternatives(value, {"captures": ()}, locale),
                )
    return result.summary()


_FAMILIES: Mapping[str, Callable[[], dict[str, object]]] = {
    "date": _date_fingerprint,
    "fraction": _fraction_fingerprint,
    "time": _time_fingerprint,
    "money": _money_fingerprint,
}


def fingerprint(families: Iterable[str] = _FAMILIES) -> dict[str, object]:
    selected = tuple(families)
    return {
        "schema": 2,
        "row_fields": ["text", "provenance", "weight", "group", "role", "prior_provenance"],
        "families": {family: _FAMILIES[family]() for family in selected},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", action="append", choices=tuple(_FAMILIES))
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    document = fingerprint(args.family or _FAMILIES)
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.write_baseline:
        args.baseline.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    expected = json.loads(args.baseline.read_text(encoding="utf-8"))
    if document != expected:
        print(rendered, end="", file=sys.stderr)
        return 1
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
