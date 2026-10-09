"""The fold-backed resolver ranks by coverage, parsimony, then capture count.

These exercise the wiring on the deposited candidate universe (overlapping and
nested detections), not on a hand-built fixture: the 1-best must drop the digit
fragments and keep the date, and keep the currency over the bare decimal.
"""

from __future__ import annotations

import random

import pytest
from icukit.detectors import detect
from icukit.recognize import (
    FlexibleCurrencyDetector,
    FlexibleDateDetector,
    FlexibleNumberDetector,
)
from tiergraph import WorkBudget, WorkMeter

from frend import DEFAULT_TIERGRAPH_WORK_BUDGET, BudgetExhausted, fold_resolve
from frend.fold_resolve import CoverMargin, CoverScore, resolve, resolve_cover


# A minimal detection for the synthetic tie cases: resolve reads start/end/type/
# captures/value; these use no captures.
def _det(start: int, end: int, type_: str, captures=()) -> dict:
    return {
        "text": "x" * min(end - start, 8),
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


def test_cover_count_matches_the_ranked_enumeration():
    """The counting fold admits exactly the covers the ranked fold enumerates, and
    within the bound the ranked fold is asked for exactly that many."""
    detections = [
        _det(0, 4, "date"),
        _det(0, 2, "number"),
        _det(2, 4, "number"),
        _det(0, 1, "number"),
        _det(1, 4, "number"),
        _det(5, 7, "number"),
    ]
    graph, roots, id_to_index = fold_resolve.build_lattice(detections)
    count = fold_resolve._count_covers(graph, roots)
    scored, truncated = fold_resolve._fold_covers(detections, graph, roots, id_to_index, 1 << 16)
    assert not truncated
    assert count == len(scored) == len({tuple(map(id, cover)) for _value, cover in scored})
    exact, exact_truncated = fold_resolve._fold_covers(detections, graph, roots, id_to_index, count)
    assert not exact_truncated
    assert exact == scored


def _tied_universe() -> list[dict]:
    # Five disjoint spans, each read two ways and also covered in halves: 32 covers
    # share the top geometry, more than the fold's first ranked request.
    detections = []
    for block in range(5):
        start = 3 * block
        detections += [
            _det(start, start + 2, "number"),
            _det(start, start + 2, "date"),
            _det(start, start + 1, "number"),
            _det(start + 1, start + 2, "number"),
        ]
    return detections


@pytest.mark.parametrize("n", [1, 2, 3, 31, 32, 33, 200])
def test_level_prefix_resolves_as_the_complete_enumeration(monkeypatch, n):
    """Folding only the whole geometry levels the n best covers need gives exactly
    what ranking every cover gives: best, covers, margin, flags and truncation."""
    detections = _tied_universe()
    prefix = resolve(detections, n=n)
    monkeypatch.setattr(fold_resolve, "_FIRST_RANKED_REQUEST", 1 << 16)
    complete = resolve(detections, n=n)
    assert prefix == complete
    assert prefix.structural_ambiguous is False
    assert prefix.semantic_ambiguous is True


def test_level_prefix_widens_until_the_needed_level_is_whole(monkeypatch):
    """A request that ends inside the needed geometry level is widened, never used."""
    requests = []
    ranked = fold_resolve._fold_covers

    def recording(detections, graph, roots, id_to_index, output_cap, work_budget):
        requests.append(output_cap)
        return ranked(detections, graph, roots, id_to_index, output_cap, work_budget)

    monkeypatch.setattr(fold_resolve, "_fold_covers", recording)
    scored, count = fold_resolve._gather_top_geometry(_tied_universe(), 2)
    assert requests == [16, 64]
    assert count == 6**5
    assert len(scored) == 32 and len({value[0] for value, _cover in scored}) == 1


def _complete_enumeration(detections) -> list:
    """Every cover the fold ranks, asked for all at once: the reference the level
    prefix is held to."""
    graph, roots, id_to_index = fold_resolve.build_lattice(detections)
    count = fold_resolve._count_covers(graph, roots)
    scored, truncated = fold_resolve._fold_covers(detections, graph, roots, id_to_index, count)
    assert not truncated
    return scored


def _selection(detections, n):
    context = fold_resolve.ResolveContext(None, tuple(detections))
    return fold_resolve._select(detections, (), context, n)


@pytest.mark.parametrize("n", [1, 2, 31, 32, 33, 6**5 - 1, 6**5, 6**5 + 1])
def test_selection_truncation_matches_the_complete_enumeration(monkeypatch, n):
    """``truncated`` says whether covers exist past the ``n`` returned, counted over
    the complete enumeration -- not over the level prefix the selection gathered.
    At n=32 the prefix holds exactly the 32 covers of the top level, so a flag read
    off the prefix would say nothing was cut from a lattice of 7,776 covers."""
    detections = _tied_universe()
    total = len(_complete_enumeration(detections))
    prefix = _selection(detections, n)
    assert prefix.truncated is (total > n)
    monkeypatch.setattr(fold_resolve, "_FIRST_RANKED_REQUEST", 1 << 16)
    assert prefix == _selection(detections, n)


def _lone_top_cover() -> list[dict]:
    # One span over everything, and the same text in six unit pieces: the top
    # geometry level holds the one long cover, and 65 covers exist in all -- more
    # than the fold's first ranked request.
    return [_det(0, 6, "date")] + [_det(i, i + 1, "number") for i in range(6)]


def test_single_best_margin_reads_the_level_below_a_lone_top_cover(monkeypatch):
    """At n=1 the margin still needs the runner-up, which lies in the second level
    when the top level holds one cover. Gathering only the top level would report a
    dead (0, 0, 0) margin for what is a five-span advantage."""
    detections = _lone_top_cover()
    graph, roots, _ids = fold_resolve.build_lattice(detections)
    assert fold_resolve._count_covers(graph, roots) == 65 > fold_resolve._FIRST_RANKED_REQUEST
    prefix = resolve(detections, n=1)
    assert prefix.margin == CoverMargin(coverage=0, span_count=5, capture_count=0)
    monkeypatch.setattr(fold_resolve, "_FIRST_RANKED_REQUEST", 1 << 16)
    assert prefix == resolve(detections, n=1)


def _tied_universe_past_the_old_bound() -> list[dict]:
    # The tied universe (7,776 covers, 32 of them sharing the top geometry) beside
    # four adjacent unit candidates, each taken or skipped: 7,776 * 2**4 = 124,416
    # covers, past the 65,536 at which the resolver used to refuse.
    return _tied_universe() + [_det(i, i + 1, "number") for i in range(15, 19)]


@pytest.mark.parametrize("n", [1, 33])
def test_past_the_old_cover_bound_resolves_as_the_complete_enumeration(monkeypatch, n):
    """No cover bound refuses: a lattice admitting 124,416 covers resolves, and to
    exactly what ranking every one of them gives -- best, covers, margin, flags and
    ``truncated``, which the exact count decides."""
    detections = _tied_universe_past_the_old_bound()
    graph, roots, _ids = fold_resolve.build_lattice(detections)
    assert fold_resolve._count_covers(graph, roots) == 7_776 * 2**4 > 1 << 16
    prefix = _selection(detections, n)
    assert prefix.truncated is True
    monkeypatch.setattr(fold_resolve, "_FIRST_RANKED_REQUEST", 1 << 30)
    assert prefix == _selection(detections, n)


def test_an_astronomical_lattice_resolves_by_whole_levels(monkeypatch):
    """Ninety-six adjacent unit candidates, each taken or skipped, admit 2**96 covers
    (the spaced-Cyrillic sentence's count) and one of them at the best geometry. The
    count is an exact int, never a float nor clamped, and it is what the gathering
    returns; the ranked fold is asked only for the witnesses the top levels need (the
    lone full cover, then the 96 that drop one unit, shown whole by a witness past
    them), never a request sized by the count."""
    detections = [_det(i, i + 1, "number") for i in range(96)]
    graph, roots, _ids = fold_resolve.build_lattice(detections)
    count = fold_resolve._count_covers(graph, roots)
    assert type(count) is int and count == 2**96
    assert fold_resolve._count_top_level(graph, roots) == 1

    requests = []
    ranked = fold_resolve._fold_covers

    def recording(detections, graph, roots, id_to_index, output_cap, work_budget):
        requests.append(output_cap)
        return ranked(detections, graph, roots, id_to_index, output_cap, work_budget)

    monkeypatch.setattr(fold_resolve, "_fold_covers", recording)
    assert fold_resolve._gather_top_geometry(detections, 2)[1] == 2**96
    assert requests == [16, 64, 256]
    assert resolve_cover(detections).best == tuple(detections)
    assert _selection(detections, 1).truncated is True


def test_top_level_count_is_the_top_level_of_the_complete_enumeration():
    """The top-level count is exactly how many covers the complete enumeration puts at
    its best geometry, on a tied level and on a lone top cover."""
    for detections, expected in ((_tied_universe(), 32), (_lone_top_cover(), 1)):
        graph, roots, _ids = fold_resolve.build_lattice(detections)
        scored = _complete_enumeration(detections)
        best = scored[0][0][0]
        assert sum(value[0] == best for value, _cover in scored) == expected
        assert fold_resolve._count_top_level(graph, roots) == expected


def _tied_chain(k: int) -> list[dict]:
    # Two-code-point spans starting at every position of 2k, each read two ways: one
    # component (every span overlaps the next), whose top level is the k disjoint
    # spans tiling the text, 2**k ways.
    detections = []
    for start in range(2 * k - 1):
        detections += [_det(start, start + 2, "number"), _det(start, start + 2, "date")]
    return detections


def test_top_level_bound_admits_its_size_and_refuses_one_past_it(monkeypatch):
    """A component's top level of exactly the bound's size resolves; lower the bound
    by one and the same lattice is refused, by the count, before any witness is
    ranked."""
    detections = _tied_chain(3)
    graph, roots, _ids = fold_resolve.build_lattice(detections)
    assert fold_resolve._count_top_level(graph, roots) == 8
    monkeypatch.setattr(fold_resolve, "_DIRECT_ENUMERATION_MAX", 0)
    monkeypatch.setattr(fold_resolve, "_RANKED_LEVEL_BOUND", 8)
    assert resolve(detections, n=1).best
    monkeypatch.setattr(fold_resolve, "_RANKED_LEVEL_BOUND", 7)

    def ranked(*_args, **_kwargs):
        raise AssertionError("the ranked fold ran past the top-level bound")

    monkeypatch.setattr(fold_resolve, "_fold_covers", ranked)
    with pytest.raises(ValueError, match="top geometry level holds 8 covers"):
        resolve(detections, n=1)


def test_a_real_tied_top_level_past_the_bound_is_refused_before_ranking(monkeypatch):
    """One component whose top level holds 2**17 covers -- twice the real bound, not
    a lowered one. The top-level count refuses it, and no witness is ranked."""
    detections = _tied_chain(17)
    candidates, _end = fold_resolve._candidates(detections)
    assert len(fold_resolve._components(candidates)) == 1
    graph, roots, _ids = fold_resolve.build_lattice(detections)
    assert fold_resolve._count_top_level(graph, roots) == 2**17 > fold_resolve._RANKED_LEVEL_BOUND

    def ranked(*_args, **_kwargs):
        raise AssertionError("the ranked fold ran past the top-level bound")

    monkeypatch.setattr(fold_resolve, "_fold_covers", ranked)
    with pytest.raises(ValueError, match=f"holds {2**17} covers"):
        resolve_cover(detections)


def test_tied_top_levels_in_separate_components_are_never_formed(monkeypatch):
    """Seventeen disjoint spans, each read two ways, put 2**17 covers at the best
    geometry, but as seventeen components of two: each is enumerated alone, the
    ranked fold never runs, and the sentence resolves."""
    detections = []
    for block in range(17):
        detections += [_det(2 * block, 2 * block + 1, "number")]
        detections += [_det(2 * block, 2 * block + 1, "date")]
    graph, roots, _ids = fold_resolve.build_lattice(detections)
    assert fold_resolve._count_top_level(graph, roots) == 2**17

    def ranked(*_args, **_kwargs):
        raise AssertionError("the ranked fold ran on a small component")

    monkeypatch.setattr(fold_resolve, "_fold_covers", ranked)
    resolution = resolve(detections, n=3)
    assert len(resolution.best) == 17 and resolution.semantic_ambiguous
    assert resolution.margin == CoverMargin(0, 0, 0)
    assert _selection(detections, 3).truncated is True


def test_widening_past_the_bound_for_a_lower_level_is_refused(monkeypatch):
    """The top level fits, but the cover list needs the next level too, and that level
    runs more than the bound past the last cover asked for: refused, not a prefix.
    The chain's two best levels hold 16 and 80 covers."""
    monkeypatch.setattr(fold_resolve, "_DIRECT_ENUMERATION_MAX", 0)
    monkeypatch.setattr(fold_resolve, "_RANKED_LEVEL_BOUND", 64)
    assert len(resolve(_tied_chain(4), n=16).covers) == 16
    assert len(resolve(_tied_chain(4), n=32).covers) == 32
    with pytest.raises(ValueError, match="holding cover 31 of the cover order runs more than 64"):
        resolve(_tied_chain(4), n=31)


def _whole_sentence(monkeypatch, detections, n):
    """The selection as one lattice: a single component, gathered by the ranked fold
    and sorted whole, as the resolver ran before components."""
    with monkeypatch.context() as patched:
        patched.setattr(fold_resolve, "_components", lambda candidates: [list(candidates)])
        patched.setattr(fold_resolve, "_DIRECT_ENUMERATION_MAX", 0)
        return _selection(detections, n)


def _every_cover(detections) -> list[tuple]:
    """Every non-overlapping subset of the valid spans, each in span order: a plain
    recursion that shares nothing with the resolver's components or folds."""
    valid = sorted(
        (d for d in detections if 0 <= int(d["start"]) < int(d["end"])),
        key=lambda d: (int(d["start"]), int(d["end"])),
    )
    covers: list[tuple] = []

    def extend(index, free_from, chosen):
        if index == len(valid):
            covers.append(tuple(chosen))
            return
        extend(index + 1, free_from, chosen)
        if int(valid[index]["start"]) >= free_from:
            extend(index + 1, int(valid[index]["end"]), [*chosen, valid[index]])

    extend(0, 0, [])
    return covers


def _oracle_margin(ordered) -> CoverMargin:
    """The top cover's advantage over the second in the full cover order."""
    if len(ordered) < 2:
        return CoverMargin(0, 0, 0)
    top, second = (fold_resolve._cover_score(cover) for cover in ordered[:2])
    return CoverMargin(
        top.coverage - second.coverage,
        second.span_count - top.span_count,
        top.capture_count - second.capture_count,
    )


def _assert_matches_every_cover_sorted(detections, n):
    """The selection against the definition, computed independently: every cover
    enumerated, sorted whole by the cover order (geometry, span signature, per-span
    reading rank, content key), and the top level, s*, flags, margin, cover list and
    truncation read off that list, as the resolver did before components."""
    context = fold_resolve.ResolveContext(None, tuple(detections))
    order = {}
    for span in {(int(d["start"]), int(d["end"])) for d in detections}:
        at_span = [d for d in detections if (int(d["start"]), int(d["end"])) == span]
        ranked = fold_resolve._rank_span_readings(at_span, (), context, None)
        most = max(len(d.get("captures", ())) for d in at_span)
        order[span] = [
            fold_resolve._content_key(r.detection)
            for r in ranked
            if len(r.detection.get("captures", ())) == most
        ]

    def key(cover):
        score = fold_resolve._cover_score(cover)
        ranks = []
        for d in cover:
            keys = order[(int(d["start"]), int(d["end"]))]
            content = fold_resolve._content_key(d)
            ranks.append(keys.index(content) if content in keys else len(keys))
        return (
            (-score.coverage, score.span_count, -score.capture_count),
            tuple((int(d["start"]), int(d["end"])) for d in cover),
            tuple(ranks),
            tuple(fold_resolve._content_key(d) for d in cover),
        )

    ordered = sorted(_every_cover(detections), key=key)
    top = [cover for cover in ordered if key(cover)[0] == key(ordered[0])[0]]
    signatures = sorted({key(cover)[1] for cover in top})
    selection = _selection(detections, n)
    assert selection.covers == tuple(ordered[:n])
    assert selection.best == ordered[0]
    assert selection.margin == _oracle_margin(ordered)
    assert selection.truncated is (len(ordered) > n)
    assert selection.structural_ambiguous is (len(signatures) > 1)
    assert tuple((span.start, span.end) for span in selection.spans) == signatures[0]


def _random_universe(rng: random.Random) -> list[dict]:
    """A random candidate universe: a few clusters of overlapping spans, separated by
    uncovered gaps or touching, with repeated geometry across clusters (so top levels
    and runner-up losses tie across components), two-way readings at some spans, a
    motif whose best covers differ in span signature at one geometry (structural
    ambiguity), and now and then a cluster wide enough to be gathered by the fold."""
    detections: list[dict] = []
    at = rng.randrange(0, 3)
    shapes = [rng.choice([(0, 2), (1, 3), (0, 1), (0, 3), (2, 3)]) for _ in range(3)]
    for _cluster in range(rng.randrange(1, 5)):
        wide = rng.random() < 0.15
        width = rng.randrange(6, 10) if wide else rng.randrange(2, 6)
        spans = set()
        for _ in range(rng.randrange(8, 13) if wide else rng.randrange(1, 6)):
            start = rng.randrange(0, width)
            spans.add((start, rng.randrange(start + 1, min(width, start + 4) + 1)))
        if rng.random() < 0.5:
            spans |= {(s, e) for s, e in shapes if e <= width}
        if width >= 3 and rng.random() < 0.3:
            spans = {(s, e) for s, e in spans if e - s < 3}
            spans |= {(0, 1), (1, 3), (0, 2), (2, 3)}
        for start, end in sorted(spans):
            captures = tuple(range(rng.choice([0, 0, 1])))
            detections.append(_det(at + start, at + end, "number", captures))
            if rng.random() < 0.4:
                detections.append(_det(at + start, at + end, "date", captures))
        at += width + rng.choice([0, 0, 1, 2])
    return detections


@pytest.mark.parametrize("seed", range(60))
def test_components_resolve_as_the_whole_sentence(monkeypatch, seed):
    """Resolving each component alone and assembling gives exactly the whole-sentence
    selection -- best, s*, spans, flags, margin, the cover list and its priors, and
    ``truncated`` -- against the ranked fold over the whole lattice and against
    every cover enumerated and sorted whole, for several n."""
    detections = fold_resolve._dedupe(_random_universe(random.Random(seed)))
    candidates, _end = fold_resolve._candidates(detections)
    total = fold_resolve._count_local(candidates)
    for n in (1, 2, 3, 8, 50):
        split = _selection(detections, n)
        assert split == _whole_sentence(monkeypatch, detections, n)
        if total <= 20_000:
            _assert_matches_every_cover_sorted(detections, n)
    if len(fold_resolve._components(candidates)) > 1:
        direct = _selection(detections, 8)
        with monkeypatch.context() as patched:
            patched.setattr(fold_resolve, "_DIRECT_ENUMERATION_MAX", 0)
            assert _selection(detections, 8) == direct


def test_the_random_universes_split_and_tie_across_components():
    """The equivalence above is not vacuous: the random universes split into several
    components; top levels and runner-up losses tie across components; some are
    structurally ambiguous; some components admit more covers than the smallest
    ranked-fold lattices do; and most are small enough to check against every
    cover."""
    seen = {"split": 0, "tied": 0, "loss_tied": 0, "structural": 0, "folded": 0, "brute": 0}

    def key(cover):
        return (fold_resolve._geometry_rank(cover), fold_resolve._span_signature(cover))

    for seed in range(60):
        detections = fold_resolve._dedupe(_random_universe(random.Random(seed)))
        candidates = fold_resolve._candidates(detections)[0]
        components = fold_resolve._components(candidates)
        locals_ = [fold_resolve._resolve_component(c, 2, key) for c in components]
        losses = [
            tuple(
                b - t
                for b, t in zip(
                    local.second_geometry, fold_resolve._geometry_rank(local.covers[0]), strict=True
                )
            )
            for local in locals_
        ]
        seen["split"] += len(components) > 1
        seen["tied"] += sum(local.top_size > 1 for local in locals_) > 1
        seen["loss_tied"] += len(losses) > 1 and losses.count(min(losses)) > 1
        seen["structural"] += _selection(detections, 1).structural_ambiguous
        seen["folded"] += any(fold_resolve._count_local(c) > 64 for c in components)
        seen["brute"] += fold_resolve._count_local(candidates) <= 20_000
    assert seen["split"] >= 45 and seen["tied"] >= 15 and seen["loss_tied"] >= 8
    assert seen["structural"] >= 5 and seen["folded"] >= 3 and seen["brute"] >= 50


@pytest.mark.parametrize(
    "detections",
    [
        _lone_top_cover(),
        _tied_universe(),
        [
            _det(0, 4, "date", captures=("m", "d", "y")),
            _det(0, 4, "number"),
            _det(0, 2, "number", captures=("i",)),
            _det(2, 4, "number"),
            _det(1, 3, "number", captures=("i", "f")),
            _det(0, 1, "number"),
            _det(5, 7, "number"),
            _det(6, 9, "date", captures=("m",)),
        ],
    ],
    ids=["lone-top", "tied", "captures-and-gaps"],
)
def test_fold_cost_and_geometry_rank_correspond_one_to_one(detections):
    """Every cover's fold cost determines its geometry rank and the reverse, and
    the two order covers alike: the level-whole gathering reads a geometry level's
    boundary off the cost, so a coarser or finer cost would cut a level."""
    scored = _complete_enumeration(detections)
    pairs = {(value[0], fold_resolve._geometry_rank(cover)) for value, cover in scored}
    assert len(pairs) == len({cost for cost, _rank in pairs}) == len({r for _c, r in pairs})
    assert sorted(pairs) == sorted(pairs, key=lambda pair: pair[1])


@pytest.mark.parametrize("seed", range(20))
def test_a_lattice_over_boundaries_only_gathers_the_same_levels(seed):
    """Positions only at candidate boundaries, one skip between neighbors: the same
    count, top level and ranked levels as the lattice with a position per code point,
    on a lattice several times smaller where text runs unbounded."""
    detections = fold_resolve._dedupe(_random_universe(random.Random(seed)))
    detections += [_det(40, 90, "electronic"), _det(40, 60, "number"), _det(70, 90, "number")]

    def levels(scored_count_beyond):
        # A witness value is (cost, paths); the paths name positions and skips,
        # which differ by construction, so the cost and the decoded cover are kept.
        scored, count, beyond = scored_count_beyond
        return [(value[0], cover) for value, cover in scored], count, beyond

    full = levels(fold_resolve._gather_levels(detections, 8))
    compact = levels(fold_resolve._gather_levels(detections, 8, boundaries_only=True))
    assert compact == full
    graph, roots, _ids = fold_resolve.build_lattice(detections, boundaries_only=True)
    whole, whole_roots, _ids = fold_resolve.build_lattice(detections)
    assert fold_resolve._count_covers(graph, roots) == fold_resolve._count_covers(
        whole, whole_roots
    )
    assert fold_resolve._count_top_level(graph, roots) == fold_resolve._count_top_level(
        whole, whole_roots
    )
    assert 2 * len(graph.tiers[0].items) < len(whole.tiers[0].items)


def _chain(k: int, offset: int = 0) -> list[dict]:
    # k two-code-point spans, each overlapping the next: one component whose covers
    # (Fibonacci in k) pass the direct-enumeration limit well before k = 14.
    return [_det(offset + i, offset + i + 2, "number") for i in range(k)]


def test_resolver_work_budget_is_shared_typed_and_output_preserving():
    assert DEFAULT_TIERGRAPH_WORK_BUDGET.steps == 1 << 24
    detections = _chain(14)
    default = repr(resolve(detections, n=8)).encode()
    meter = WorkMeter(WorkBudget(steps=1 << 30))
    ample = repr(resolve(detections, n=8, work_budget=meter)).encode()
    assert default == ample
    assert meter.spent > 0

    with pytest.raises(BudgetExhausted) as caught:
        resolve(detections, n=8, work_budget=WorkBudget(steps=1))
    assert caught.value.operation == "fold.run"
    assert caught.value.spent > caught.value.budget.steps
    assert "exhausted its work budget" in str(caught.value)


def test_a_cover_list_past_the_bound_is_answered_on_a_folded_component():
    """The bound limits a folded component's top level, never the n asked for: n=300
    on a 14-span chain (987 covers, gathered by the ranked fold, a top level of 8) is
    the first 300 covers of the full cover order, as main gave."""
    detections = _chain(14)
    candidates, _end = fold_resolve._candidates(detections)
    assert fold_resolve._count_local(candidates) == 987 > fold_resolve._DIRECT_ENUMERATION_MAX
    graph, roots, _ids = fold_resolve.build_lattice(detections)
    assert fold_resolve._count_top_level(graph, roots) == 8
    for n in (256, 257, 300):
        _assert_matches_every_cover_sorted(detections, n)
    assert len(resolve(detections, n=300).covers) == 300


def test_folds_stay_exact_however_long_the_spans():
    """One span of 10**30 over ten of 10**29, each read two ways: weights near 10**60,
    past Decimal's default 28 digits. Rounded, the long span's cost ties with the
    1,024 ten-span covers and the top level reads 1,025 (refused). The folds run in
    a context wide enough to keep them apart: the top level is the long span alone,
    nine spans ahead of the runner-up."""
    unit = 10**29
    detections = [_det(0, 10 * unit, "number")]
    for i in range(10):
        detections += [_det(i * unit, (i + 1) * unit, kind) for kind in ("number", "date")]
    graph, roots, _ids = fold_resolve.build_lattice(detections, boundaries_only=True)
    assert fold_resolve._count_covers(graph, roots) == 3**10 + 1
    assert fold_resolve._count_top_level(graph, roots) == 1
    selection = _selection(detections, 1)
    assert selection.best == (detections[0],)
    assert selection.margin == CoverMargin(coverage=0, span_count=9, capture_count=0)
