"""Training-resolved ambiguity in ICU numeric durations."""

import pytest
from icukit import FlexibleNumericDurationDetector

from frend import durations
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
    ("preferred_unit", "preferred_count", "other_count"),
    [
        ("fortnight", 2, 1),
        ("second", True, 0),
        ("second", 2, False),
        ("second", -1, 0),
        ("second", 2, -1),
        ("second", 1.5, 0),
        ("second", 2, 0.5),
    ],
)
def test_malformed_duration_prior_abstains(
    monkeypatch, preferred_unit, preferred_count, other_count
):
    monkeypatch.setattr(
        durations,
        "measured_table",
        lambda *_args: {
            "numeric_duration": {
                "preferred_unit": preferred_unit,
                "preferred_count": preferred_count,
                "other_count": other_count,
            }
        },
    )
    durations._preferred_unit.cache_clear()
    assert durations._preferred_unit("en_US") is None
    found = NumericDurationDetector("en_US").detect("2:08.34")
    assert [detection["value"].unit for detection in found] == ["minute", "second"]
    durations._preferred_unit.cache_clear()


def test_valid_duration_prior_requires_strictly_greater_support(monkeypatch):
    record = {
        "preferred_unit": "second",
        "preferred_count": 2,
        "other_count": 1,
    }
    monkeypatch.setattr(
        durations,
        "measured_table",
        lambda *_args: {"numeric_duration": record},
    )
    durations._preferred_unit.cache_clear()
    assert durations._preferred_unit("en_US") == "second"
    record["other_count"] = 2
    durations._preferred_unit.cache_clear()
    assert durations._preferred_unit("en_US") is None
    durations._preferred_unit.cache_clear()


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
