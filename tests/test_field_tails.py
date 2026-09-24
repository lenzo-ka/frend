"""Detection captures retain their surface order in tiergraph field tails."""

from __future__ import annotations

from icukit.recognize import FlexibleDateDetector, FlexibleNumberDetector

from frend.field_tails import build_field_tails, ordered_fields


def _date(locale: str, text: str):
    return FlexibleDateDetector(locale).detect(text)[0]


def test_month_first_date_reads_in_surface_order():
    detection = _date("en_US", "1/3/2026")
    tails = build_field_tails([detection])

    assert tuple(field.name for field in ordered_fields(tails, tails.candidates[0])) == (
        "M",
        "d",
        "y",
    )


def test_date_field_order_is_non_commutative():
    month_first = _date("en_US", "1/3/2026")
    year_first = _date("ja_JP", "2026/1/3")
    tails = build_field_tails([month_first, year_first])

    orders = [
        tuple(field.name for field in ordered_fields(tails, candidate))
        for candidate in tails.candidates
    ]
    assert orders == [("M", "d", "y"), ("y", "M", "d")]
    assert orders[0] != orders[1]


def test_number_fields_read_in_surface_order():
    detection = FlexibleNumberDetector("en_US").detect("1,234.50")[0]
    tails = build_field_tails([detection])

    assert tuple(field.name for field in ordered_fields(tails, tails.candidates[0])) == (
        "integer",
        "decimal-separator",
        "fraction",
    )


def test_field_values_round_trip_as_strings_through_traversal():
    detection = FlexibleNumberDetector("en_US").detect("-1,234.50")[0]
    tails = build_field_tails([detection])

    actual = tuple(field.value for field in ordered_fields(tails, tails.candidates[0]))
    expected = tuple(str(capture.value) for capture in detection["captures"])
    assert actual == expected


def test_ordered_fields_reads_the_graph_snapshot_not_the_live_input():
    # Mutating the input detection after the tail is built must not change the
    # read: ordered_fields resolves through the graph, not the source captures.
    detection = dict(_date("en_US", "1/3/2026"))
    tails = build_field_tails([detection])
    detection["captures"] = ()
    assert tuple(field.name for field in ordered_fields(tails, tails.candidates[0])) == (
        "M",
        "d",
        "y",
    )


def test_empty_captures_build_and_read_as_no_fields():
    detection = dict(_date("en_US", "1/3/2026"))
    detection["captures"] = ()
    tails = build_field_tails([detection])
    assert ordered_fields(tails, tails.candidates[0]) == ()


def test_multiple_detections_have_independent_ordered_tails():
    detections = [
        _date("ja_JP", "2026/1/3"),
        FlexibleNumberDetector("en_US").detect("1,234.50")[0],
        _date("en_US", "1/3/2026"),
    ]
    tails = build_field_tails(detections)

    assert [
        tuple(field.name for field in ordered_fields(tails, candidate))
        for candidate in tails.candidates
    ] == [
        ("y", "M", "d"),
        ("integer", "decimal-separator", "fraction"),
        ("M", "d", "y"),
    ]
