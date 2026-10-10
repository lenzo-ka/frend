"""Recognize common English scientific-notation expressions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from icukit.detectors import Capture, MeasureValue

from frend.locale_data import canonical_locale

__all__ = ["ScientificNotationDetector", "ScientificValue"]

_MANTISSA = r"[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)"
_EXPONENT = r"[+\-−]?\d+"
_SUPERSCRIPT_DIGITS = r"[\u2070¹²³⁴-⁹]+"
_SUPERSCRIPT_EXPONENT = rf"(?:[⁺⁻]{_SUPERSCRIPT_DIGITS}|[+\-−]?{_SUPERSCRIPT_DIGITS})"
_E_NOTATION = re.compile(
    rf"(?<![\w.])(?P<mantissa>{_MANTISSA})(?P<operator>[eE])"
    rf"(?P<exponent>{_EXPONENT})(?!\w|\.\w)"
)
_TIMES_TEN_CARET = re.compile(
    rf"(?<![\w.])(?P<mantissa>{_MANTISSA})"
    rf"(?P<operator>\s*[×x·*]\s*10\s*\^\s*)"
    rf"(?P<exponent>{_EXPONENT})(?!\w|\.\w)"
)
_TIMES_TEN_SUPERSCRIPT = re.compile(
    rf"(?<![\w.])(?P<mantissa>{_MANTISSA})"
    rf"(?P<operator>\s*[×x·*]\s*10\s*)"
    rf"(?P<exponent>{_SUPERSCRIPT_EXPONENT})(?!\w|\.\w)"
)
_SIGNS = "+-−"
_SUPERSCRIPT_SIGNS = "⁺⁻"
_SUPERSCRIPT_TRANSLATION = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


class _Detector(Protocol):
    def detect(self, text: str) -> list[dict]: ...


@dataclass(frozen=True)
class ScientificValue:
    """A written mantissa and its integer power-of-ten exponent."""

    mantissa: str
    exponent: int
    exponent_sign: str | None
    unit: str | None = None


def _plain_exponent(exponent: str) -> str:
    return exponent.translate(_SUPERSCRIPT_TRANSLATION).replace("−", "-")


def _capture(name: str, start: int, text: str, value: object, form: str) -> Capture:
    return Capture(name, start, start + len(text), text, value, form)


def _captures(match: re.Match[str]) -> tuple[Capture, ...]:
    mantissa = match.group("mantissa")
    mantissa_start = match.start("mantissa")
    sign = mantissa[0] if mantissa[0] in _SIGNS else ""
    unsigned = mantissa[len(sign) :]
    unsigned_start = mantissa_start + len(sign)
    integer, dot, fraction = unsigned.partition(".")
    captures: list[Capture] = [_capture("mantissa", mantissa_start, mantissa, mantissa, "numeric")]
    if sign:
        captures.append(_capture("sign", mantissa_start, sign, sign, "symbol"))
    captures.append(_capture("integer", unsigned_start, integer, integer, "numeric"))
    if dot:
        separator_start = unsigned_start + len(integer)
        captures.extend(
            (
                _capture("decimal-separator", separator_start, dot, dot, "symbol"),
                _capture("fraction", separator_start + 1, fraction, fraction, "numeric"),
            )
        )
    captures.append(
        _capture(
            "scientific-operator",
            match.start("operator"),
            match.group("operator"),
            None,
            "symbol",
        )
    )
    exponent = match.group("exponent")
    exponent_sign = exponent[0] if exponent[0] in _SIGNS + _SUPERSCRIPT_SIGNS else ""
    if exponent_sign:
        captures.append(
            _capture(
                "exponent-sign",
                match.start("exponent"),
                exponent_sign,
                exponent_sign,
                "symbol",
            )
        )
    exponent_digits = exponent[len(exponent_sign) :]
    captures.append(
        _capture(
            "exponent",
            match.start("exponent") + len(exponent_sign),
            exponent_digits,
            int(_plain_exponent(exponent_digits)),
            "numeric",
        )
    )
    return tuple(captures)


class ScientificNotationDetector:
    """Detect ``6.02e23`` and ``1.5×10^6`` only for English locales."""

    def __init__(
        self, locale: str = "en_US", *, measure_detectors: tuple[_Detector, ...] = ()
    ) -> None:
        self.locale = canonical_locale(locale)
        self.measure_detectors = measure_detectors

    def _with_unit(self, text: str, detection: dict) -> list[dict]:
        """Reuse ICU's measure grammar to attach a suffix without duplicating its units."""
        if not self.measure_detectors:
            return []
        start, end = detection["start"], detection["end"]
        masked = text[:start] + "0" * (end - start) + text[end:]
        combined = []
        for detector in self.measure_detectors:
            for measure in detector.detect(masked):
                value = measure.get("value")
                if (
                    measure.get("start") != start
                    or measure.get("end", 0) <= end
                    or not isinstance(value, MeasureValue)
                ):
                    continue
                unit = next(
                    (
                        capture
                        for capture in measure.get("captures", ())
                        if capture.name == "unit" and capture.start >= end
                    ),
                    None,
                )
                if unit is None:
                    continue
                scientific = detection["value"]
                combined.append(
                    {
                        **detection,
                        "text": text[start : measure["end"]],
                        "end": measure["end"],
                        "value": ScientificValue(
                            scientific.mantissa,
                            scientific.exponent,
                            scientific.exponent_sign,
                            value.unit,
                        ),
                        "captures": (*detection["captures"], unit),
                        "writes_unit": True,
                    }
                )
        return combined

    def detect(self, text: str) -> list[dict]:
        if self.locale != "en_US":
            return []
        detections = []
        for pattern in (_E_NOTATION, _TIMES_TEN_CARET, _TIMES_TEN_SUPERSCRIPT):
            for match in pattern.finditer(text):
                exponent = match.group("exponent")
                exponent_sign = exponent[0] if exponent[0] in _SIGNS + _SUPERSCRIPT_SIGNS else None
                detection = {
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "type": "number:scientific",
                    "value": ScientificValue(
                        match.group("mantissa"), int(_plain_exponent(exponent)), exponent_sign
                    ),
                    "captures": _captures(match),
                }
                detections.extend((detection, *self._with_unit(text, detection)))
        return sorted(detections, key=lambda detection: (detection["start"], detection["end"]))
