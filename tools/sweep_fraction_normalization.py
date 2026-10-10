"""Exercise fraction, mixed-number, and percent normalization without exceptions."""

from __future__ import annotations

from frend import normalize, prewarm
from frend.fraction_rules import GENERATED_FRACTION_LOCALES

LOCALES = (*GENERATED_FRACTION_LOCALES, "en_US")


def cases():
    for locale in LOCALES:
        for denominator in range(1, 1201):
            for numerator in range(1, 13):
                yield locale, f"{numerator}/{denominator}"
            yield locale, f"2 3/{denominator}"
        for amount in range(1001):
            yield locale, f"{amount}%"
        yield locale, "-3/7"


def main() -> int:
    prewarm(LOCALES)
    count = 0
    failures: list[tuple[str, str, str]] = []
    for locale, text in cases():
        count += 1
        try:
            normalize(text, locale=locale)
        except Exception as exc:  # noqa: BLE001 -- this is an exception-safety sweep
            failures.append((locale, text, f"{type(exc).__name__}: {exc}"))
    if failures:
        for locale, text, error in failures:
            print(f"{locale}\t{text}\t{error}")
        print(f"FAILED: {len(failures)} exceptions across {count} cases")
        return 1
    print(f"OK: {count} cases across {len(LOCALES)} locales, zero exceptions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
