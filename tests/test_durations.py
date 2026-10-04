"""Training-resolved ambiguity in ICU numeric durations."""

import pytest
from icukit import FlexibleNumericDurationDetector

from frend.durations import NumericDurationDetector


def _signature(detector, written: str) -> list[tuple]:
    return [
        (
            detection["start"],
            detection["end"],
            detection["text"],
            detection["type"],
            detection["value"],
            tuple(detection.get("captures", ())),
        )
        for detection in detector.detect(written)
    ]


def test_fractional_colon_keeps_the_elapsed_seconds_parse():
    found = NumericDurationDetector("en_US").detect("2:08.34")
    assert [detection["value"].unit for detection in found] == ["second"]


@pytest.mark.parametrize(
    "written",
    [
        "1.2:3.4",  # decimal ratio
        "3-2",  # score with a dash
        "3:2",  # score with a colon
        "10:30",  # clock
        "10:30.5 am",  # fractional clock plus day period
        "1/2",  # fraction
        "1:2.3",  # version-like form
        "lap 2:08.34 result",  # the target shape embedded in text
        "2:08.3",  # not hundredths
        "2:08.345",  # not hundredths
    ],
)
def test_out_of_scope_forms_are_exactly_icus_detector_result(written):
    assert _signature(NumericDurationDetector("en_US"), written) == _signature(
        FlexibleNumericDurationDetector("en_US"), written
    )


def test_non_english_locale_is_exactly_icus_detector_result():
    written = "2:08.34"
    assert _signature(NumericDurationDetector("fr_FR"), written) == _signature(
        FlexibleNumericDurationDetector("fr_FR"), written
    )
