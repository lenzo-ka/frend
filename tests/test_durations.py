"""Training-resolved ambiguity in ICU numeric durations."""

from frend.durations import NumericDurationDetector


def _units(written: str) -> list[str]:
    return [
        detection["value"].unit
        for detection in NumericDurationDetector("en_US").detect(written)
        if detection["start"] == 0 and detection["end"] == len(written)
    ]


def test_fractional_colon_keeps_the_elapsed_seconds_parse():
    assert _units("1:09.12") == ["second"]


def test_plain_colon_keeps_icus_clock_duration_ambiguity():
    assert _units("10:30") == ["minute", "second"]


def test_invalid_duration_is_not_invented():
    assert _units("16:79.5") == []
