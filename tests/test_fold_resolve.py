"""The fold-backed resolver ranks by coverage, parsimony, then capture count.

These exercise the wiring on the deposited candidate universe (overlapping and
nested detections), not on a hand-built fixture: the 1-best must drop the digit
fragments and keep the date, and keep the currency over the bare decimal.
"""

from __future__ import annotations

import pytest
from icukit.detectors import detect
from icukit.recognize import (
    FlexibleCurrencyDetector,
    FlexibleDateDetector,
    FlexibleNumberDetector,
)

from frend import fold_resolve
from frend.fold_resolve import CoverMargin, CoverScore, resolve, resolve_cover


# A minimal detection for the synthetic tie cases: resolve reads start/end/type/
# captures/value; these use no captures.
def _det(start: int, end: int, type_: str, captures=()) -> dict:
    return {
        "text": "x" * (end - start),
        "start": start,
        "end": end,
        "type": type_,
        "value": None,
        "captures": tuple(captures),
    }


DETECTORS = [
    FlexibleDateDetector("en_US"),
    FlexibleNumberDetector("en_US"),
    FlexibleCurrencyDetector("en_US", "USD"),
]


def _key(detection) -> tuple[str, int, int, str]:
    return (detection["type"], detection["start"], detection["end"], detection["text"])


def _keys(detections) -> list[tuple[str, int, int, str]]:
    return [_key(d) for d in detections]


def _patterns(detections):
    return sorted(d["spec"].pattern for d in detections if d["type"] == "date:flexible")


def test_date_beats_its_digit_fragments():
    dets = detect("1/3/2026", DETECTORS)
    # The deposited universe overlaps: the whole date read both month-first and
    # day-first (one reading per CLDR structure), plus three digit fragments.
    assert _patterns(dets) == ["M/d/yy", "dd/MM/y"]
    assert sum(d["type"] == "number:decimal" for d in dets) == 3
    cover = resolve_cover(dets)
    assert _keys(cover.best) == [("date:flexible", 0, 8, "1/3/2026")]
    assert cover.score == CoverScore(coverage=8, span_count=1, capture_count=3)


def test_date_and_currency_both_kept():
    text = "on 1/3/2026 we paid $1,234.50"
    dets = detect(text, DETECTORS)
    cover = resolve_cover(dets)
    assert _keys(cover.best) == [
        ("date:flexible", 3, 11, "1/3/2026"),
        ("number:currency:USD", 20, 29, "$1,234.50"),
    ]
    assert cover.score == CoverScore(
        coverage=8 + 9,
        span_count=2,
        capture_count=sum(len(d.get("captures", ())) for d in cover.best),
    )


def test_best_is_non_overlapping_and_sorted():
    dets = detect("on 1/3/2026 we paid $1,234.50", DETECTORS)
    best = resolve_cover(dets).best
    spans = [(d["start"], d["end"]) for d in best]
    assert spans == sorted(spans)
    for (a_start, a_end), (b_start, _b_end) in zip(spans, spans[1:], strict=False):
        assert a_end <= b_start  # pairwise non-overlapping, in source order
        assert a_start <= a_end


def test_empty_input():
    cover = resolve_cover([])
    assert cover.best == ()
    assert cover.score == CoverScore(0, 0, 0)
    resolution = resolve([])
    assert resolution.best == ()
    assert resolution.covers == ((),)
    assert resolution.margin == CoverMargin(0, 0, 0)
    assert resolution.ambiguous is False


def test_no_detections_in_text():
    dets = detect("no numbers here at all", DETECTORS)
    assert dets == []
    cover = resolve_cover(dets)
    assert cover.best == ()
    assert cover.score == CoverScore(0, 0, 0)


def test_score_includes_capture_specificity_and_prefers_capture_rich_reading():
    dates = [d for d in detect("1/3/2026", DETECTORS) if d["type"] == "date:flexible"]
    assert len(dates) == 2  # month-first and day-first
    for date in dates:
        assert len(date["captures"]) == 3
        capture_poor = _det(0, 8, "capture-poor")
        cover = resolve_cover([capture_poor, date], feature_sources=())
        assert cover.score == CoverScore(coverage=8, span_count=1, capture_count=3)
        assert cover.best == (date,)


def test_real_slash_date_is_ambiguous_between_its_two_structures():
    """ "1/3/2026" is January 3 or March 1; the two readings tie on every geometry axis."""
    for text in ["1/3/2026", "on 1/3/2026 we paid $1,234.50"]:
        dets = detect(text, DETECTORS)
        res = resolve(dets)
        assert res.ambiguous is True
        assert res.margin == CoverMargin(0, 0, 0)
        first, second = res.covers[0], res.covers[1]
        assert _score(first) == _score(second)
        assert [d for d in first if d["type"] != "date:flexible"] == [
            d for d in second if d["type"] != "date:flexible"
        ]
        assert _patterns(first) + _patterns(second) == ["M/d/yy", "dd/MM/y"]
        assert len(res.covers) == 8
        assert res.covers[0] == res.best
        scores = [_score(cover) for cover in res.covers]
        assert scores == sorted(
            scores,
            key=lambda score: (-score.coverage, score.span_count, -score.capture_count),
        )


def _score(cover) -> CoverScore:
    return CoverScore(
        coverage=sum(int(d["end"]) - int(d["start"]) for d in cover),
        span_count=len(cover),
        capture_count=sum(len(d.get("captures", ())) for d in cover),
    )


def test_parsimony_prefers_fewer_spans_not_ambiguous():
    dets = [_det(0, 2, "A"), _det(0, 1, "B"), _det(1, 2, "C")]
    res = resolve(dets)
    assert res.ambiguous is False
    assert _keys(res.best) == [("A", 0, 2, "xx")]
    assert _score(res.covers[0]) == CoverScore(2, 1, 0)
    assert _score(res.covers[1]) == CoverScore(2, 2, 0)
    assert res.margin == CoverMargin(0, 1, 0)


def test_honest_same_span_tie():
    dets = [_det(0, 4, "date"), _det(0, 4, "fraction")]
    res = resolve(dets)
    assert res.ambiguous is True
    assert res.margin == CoverMargin(0, 0, 0)
    assert {_keys(cover)[0][0] for cover in res.covers[:2]} == {"date", "fraction"}
    assert [_score(cover) for cover in res.covers[:2]] == [CoverScore(4, 1, 0)] * 2


def test_same_span_capture_rich_date_wins_before_prior_and_cannot_be_resurrected():
    date = _det(0, 4, "date:Md", captures=("month", "day", "year"))
    fraction = _det(0, 4, "number:fraction")
    date["text"] = fraction["text"] = "3/24"

    structural = resolve([date, fraction], feature_sources=())
    assert structural.best == (date,)
    assert _score(structural.best) == CoverScore(4, 1, 3)
    # The margin exposes the decisive capture advantage, not a dead structural tie.
    assert structural.margin == CoverMargin(0, 0, 3)

    with_prior = resolve([date, fraction])
    assert with_prior.best == (date,)
    assert tuple(reading.detection for reading in with_prior.spans[0].readings) == (date,)
    assert with_prior.semantic_ambiguous is False


def test_coverage_strictly_primary():
    fragments = [_det(i, i + 1, f"unit-{i}") for i in range(10)]
    res = resolve([_det(0, 9, "long"), *fragments])
    assert _score(res.best).coverage == 10


def test_decomposable_weight():
    dets = detect("on 1/3/2026 we paid $1,234.50", DETECTORS)
    for cover in resolve(dets).covers:
        score = _score(cover)
        assert score.coverage == sum(int(d["end"]) - int(d["start"]) for d in cover)
        assert score.span_count == len(cover)
        assert score.capture_count == sum(len(d.get("captures", ())) for d in cover)


def test_negative_position_spans_are_dropped_keeping_the_invariant():
    # Out-of-contract negative-position spans are dropped, so the lexicographic
    # invariant span_count <= span_end < M holds for everything scored. A cover of
    # negative-position units must never outrank a longer nonnegative span.
    dets = [
        _det(0, 4, "long"),
        _det(-4, -3, "u1"),
        _det(-3, -2, "u2"),
        _det(-2, -1, "u3"),
        _det(-1, 0, "u4"),
    ]
    res = resolve(dets)
    assert _keys(res.best) == [("long", 0, 4, "xxxx")]
    assert _score(res.best) == CoverScore(4, 1, 0)


def test_dedupe_prevents_false_ambiguity():
    # two content-identical detections must not look like a tie between them.
    dets = [_det(0, 3, "A"), _det(0, 3, "A")]
    res = resolve(dets)
    assert res.ambiguous is False
    assert len(res.covers) == 2


def test_nbest_ordered_by_coverage_then_parsimony():
    cases = [
        detect("1/3/2026", DETECTORS),
        detect("on 1/3/2026 we paid $1,234.50", DETECTORS),
        [_det(0, 2, "A"), _det(0, 1, "B"), _det(1, 2, "C")],
    ]
    for dets in cases:
        res = resolve(dets)
        scores = [_score(cover) for cover in res.covers]
        assert scores == sorted(
            scores,
            key=lambda score: (-score.coverage, score.span_count, -score.capture_count),
        )
        assert res.best == res.covers[0]
        assert _score(res.best) == scores[0]


def test_enumeration_refuses_rather_than_returning_a_prefix(monkeypatch):
    """REGRESSION: a truncated enumeration is refused, not returned as complete.

    The resolver asks the fold for more witnesses than any practical input
    produces and relies on the fold's own truncation flag to know the answer is
    whole. If that bound were ever reached, returning what came back would hand
    canonical selection a prefix that looks exactly like a complete answer --
    which is the failure the single call replaced a cap-doubling loop to avoid.

    The cap is lowered here rather than a pathological input constructed, because
    the condition under test is the response to truncation, not the size at which
    it occurs.
    """
    monkeypatch.setattr(fold_resolve, "_COVER_ENUMERATION_CAP", 1)
    detections = [
        _det(0, 4, "date"),
        _det(0, 2, "number"),
        _det(2, 4, "number"),
        _det(0, 1, "number"),
        _det(1, 4, "number"),
    ]
    with pytest.raises(ValueError) as caught:
        resolve_cover(detections)
    message = str(caught.value)
    assert "witness bound" in message
    assert "prefix" in message
