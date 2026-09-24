"""Layer-2 corpus base-rate prior: shape signature, table build, and post-rank tiebreak.

These pin the reconciled Layer-2 design: geometry is provably primary and the
corpus prior is one post-rank axis reached only among geometry-equal covers. The
build-from-fixture tests use a tiny inline corpus, not NeMo's exact numbers, so
they survive swapping the placeholder corpus for a stronger basis.
"""

from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest

from irn.fold_resolve import CoverMargin, resolve, resolve_cover
from irn.shape import shape
from irn.type_priors import (
    MIN_N,
    CorpusPrior,
    PriorTable,
    ReadingFeature,
    ReadingPrior,
    ResolveContext,
    corpus_classes,
)

_REPO = Path(__file__).resolve().parents[1]


def _load_build_tool():
    """Load tools/build_type_priors.py by path (tools is not an installed package)."""
    spec = importlib.util.spec_from_file_location(
        "build_type_priors", _REPO / "tools" / "build_type_priors.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _det(start: int, end: int, type_: str, text: str) -> dict:
    return {
        "text": text,
        "start": start,
        "end": end,
        "type": type_,
        "value": None,
        "captures": (),
    }


def _geometry(cover) -> tuple[int, int]:
    return (sum(int(d["end"]) - int(d["start"]) for d in cover), len(cover))


# --------------------------------------------------------------------------- test 1


def test_corpus_breaks_a_same_geometry_tie():
    """date:Md vs number:fraction on "3/24": the higher-P group (fraction) wins,
    a decomposable ReadingPrior is attached, and the geometry is byte-identical
    to the pre-Layer-2 (prior-free) resolution."""
    dets = [_det(0, 4, "date:Md", "3/24"), _det(0, 4, "number:fraction", "3/24")]

    res = resolve(dets)
    assert res.best[0]["type"] == "number:fraction"
    assert res.ambiguous is False
    # margin/CoverMargin stay geometric facts: the geometry is still a dead tie.
    assert res.margin == CoverMargin(0, 0, 0)

    # The prior is present and fully decomposed on each cover, in rank order.
    by_type = {
        cover[0]["type"]: priors[0]
        for cover, priors in zip(res.covers, res.priors, strict=True)
        if cover
    }
    fraction_prior = by_type["number:fraction"]
    date_prior = by_type["date:Md"]
    assert isinstance(fraction_prior, ReadingPrior)
    assert fraction_prior.group == "fraction" and fraction_prior.shape == "N/N"
    assert fraction_prior.supported is True
    assert fraction_prior.p > 0 and fraction_prior.n >= MIN_N
    # date is attested-zero on N/N in the placeholder: unsupported, no fabricated
    # value (p is None), yet the sample size stays observable (the shape was seen).
    assert date_prior.group == "date"
    assert date_prior.supported is False
    assert date_prior.p is None and date_prior.n >= MIN_N
    # fraction is supported and date is not, so fraction wins the same-span tie.
    assert isinstance(date_prior, ReadingPrior)

    # Geometry unchanged vs pre-Layer-2 (feature_sources=() disables the prior):
    # both same-span readings are covers of coverage 4, span 1.
    baseline = resolve(dets, feature_sources=())
    top_two = {_geometry(c) for c in res.covers if c}
    assert top_two == {_geometry(c) for c in baseline.covers if c} == {(4, 1)}


# --------------------------------------------------------------------------- test 2


def test_corpus_cannot_break_a_same_group_tie():
    """date:MMMd vs date:yMMM on "Jan 5": both map to class date, so the prior is
    identical, the tie falls through to the content key, ambiguous stays True, and
    no winner is fabricated."""
    dets = [_det(0, 5, "date:MMMd", "Jan 5"), _det(0, 5, "date:yMMM", "Jan 5")]
    res = resolve(dets)

    assert res.ambiguous is True
    assert res.margin == CoverMargin(0, 0, 0)
    priors = {
        cover[0]["type"]: p[0] for cover, p in zip(res.covers[:2], res.priors[:2], strict=True)
    }
    a, b = priors["date:MMMd"], priors["date:yMMM"]
    assert a.group == b.group == "date"
    assert a.p == b.p and a.n == b.n  # identical prior => genuinely untied by corpus


# --------------------------------------------------------------------------- test 3


def test_geometry_is_never_overridden_by_the_prior():
    """A higher-coverage whole-span reading whose prior is unsupported still beats
    a lower-coverage competitor whose prior is supported and perfect."""
    whole = _det(0, 4, "date:Md", "3/24")  # coverage 4, date unsupported on N/N
    shorter = _det(0, 3, "number:fraction", "3/2")  # coverage 3, supported p=1 on N/N
    res = resolve([whole, shorter])

    assert res.best[0]["type"] == "date:Md"
    assert _geometry(res.best) == (4, 1)
    # The winner carries the unsupported prior; geometry won in spite of it.
    assert res.priors[0][0].supported is False
    assert res.priors[0][0].p is None


# --------------------------------------------------------------------------- test 4


def test_no_prior_when_corpus_is_absent_leaves_baseline_untouched():
    """A shape with no corpus rows yields None from lookup: the order is identical
    to the prior-free baseline and ambiguity is unchanged, with no fabricated
    scores anywhere."""
    # "xxxx"/"yyyy" both have shape "A", which the placeholder never attests.
    assert shape("xxxx") == "A"
    dets = [_det(0, 4, "date", "xxxx"), _det(0, 4, "fraction", "yyyy")]

    res = resolve(dets)
    baseline = resolve(dets, feature_sources=())

    order = [tuple(_key(d) for d in cover) for cover in res.covers]
    baseline_order = [tuple(_key(d) for d in cover) for cover in baseline.covers]
    assert order == baseline_order
    assert res.ambiguous == baseline.ambiguous is True
    # No support => every reading is unsupported with no fabricated value. The
    # shape is absent, so the sample size is honestly unknown too (n is None).
    for priors in res.priors:
        for p in priors:
            assert p is not None and p.supported is False
            assert p.p is None and p.n is None

    # A below-floor shape (n < MIN_N) also withholds rather than speak from too few:
    # unsupported, no fabricated p, and no sample size (it never reached the floor).
    # Pinned to an inline table so this tests the floor mechanic, not a particular
    # corpus (in the shipped Google-TN table "N.N.N" is well above the floor).
    table = PriorTable({"N.N.N": {"date": 2}}, {"source": "fixture"})  # n=2 < MIN_N=3
    sparse = table.reading_prior(_det(0, 5, "date", "1.2.3"))  # shape "N.N.N", sparse
    assert sparse is not None and sparse.supported is False
    assert sparse.p is None and sparse.n is None


def _key(detection) -> tuple:
    return (detection["type"], detection["start"], detection["end"], detection["text"])


# --------------------------------------------------------------------------- test 5


def test_table_build_counts_to_probabilities_and_check_mode(tmp_path):
    """From a tiny fixture corpus: counts -> P and n are correct, the artifact
    carries raw counts (not ratios) plus provenance, and --check passes on the
    emitted file and fails on a mutated one."""
    build = _load_build_tool()
    pairs = [
        ("1/2", "fraction"),
        ("3/4", "fraction"),
        ("1/3", "fraction"),
        ("jan 5", "date"),
    ]
    counts = build.build_counts(pairs)
    assert counts == {"A N": {"date": 1}, "N/N": {"fraction": 3}}

    table = PriorTable(counts, {"source": "fixture"})
    assert table.n("N/N") == 3
    # counts -> P: 3 of 3 N/N rows are fraction, 0 are date.
    assert table.prior(("fraction",), "N/N") == (Decimal(1), 3)
    assert table.prior(("date",), "N/N") == (Decimal(0), 3)
    assert table.n("A N") is None  # below MIN_N floor
    assert table.prior(("date",), "A N") is None

    document = build.build_document(pairs)
    # Artifact carries raw integer COUNTS, not derived ratios, plus provenance.
    assert document["counts"]["N/N"]["fraction"] == 3
    assert isinstance(document["counts"]["N/N"]["fraction"], int)
    assert document["provenance"]["source"] == "provided-pairs"

    # --check round-trips: build to a temp artifact, verify it, then mutate it. The
    # checked-in Google-TN fixture is selected so this stays a fast, corpus-free unit test
    # (the default corpus is the multi-GB Google-TN tree the shipped table is built
    # from, which --check re-derives from when present).
    fixture = _REPO / "tests" / "data" / "google_tn"
    out = tmp_path / "type_priors.json"
    assert build.main(["--corpus", str(fixture), "--out", str(out)]) == 0
    built = json.loads(out.read_text(encoding="utf-8"))
    assert built["provenance"]["source"] == f"google-tn-en_with_types:{fixture}"
    assert build.main(["--corpus", str(fixture), "--check", "--out", str(out)]) == 0
    out.write_text(out.read_text(encoding="utf-8").replace("}", "} ", 1), encoding="utf-8")
    assert build.main(["--corpus", str(fixture), "--check", "--out", str(out)]) == 1


# --------------------------------------------------------------------------- test 6


def test_shape_is_reflective_over_scripts():
    """Shape is drawn from Unicode categories, not a digit range: Arabic-Indic
    digits shape identically to ASCII digits."""
    assert shape("٣/٤") == shape("3/24") == "N/N"
    assert shape("$2.00") == "¤N.N"
    assert shape("23rd") == "NA"
    assert shape("Jan 5") == "A N"
    assert shape("01.10.2010") == "N.N.N"


def test_single_uppercase_shape_is_reflective_and_unambiguous():
    from irn.shape import SINGLE_UPPERCASE_SHAPE, is_single_uppercase

    assert is_single_uppercase("I")
    assert is_single_uppercase("Ω")
    assert not is_single_uppercase("i")
    assert not is_single_uppercase("IV")
    assert shape("I") == shape("Ω") == SINGLE_UPPERCASE_SHAPE
    assert shape("hello") != SINGLE_UPPERCASE_SHAPE


def test_decomposed_single_uppercase_agrees_with_its_shape():
    from irn.shape import SINGLE_UPPERCASE_SHAPE, is_single_uppercase

    decomposed = "O\u0304\u0301"
    assert is_single_uppercase(decomposed)
    assert shape(decomposed) == SINGLE_UPPERCASE_SHAPE


def test_shape_is_nfc_stable():
    """shape() NFC-normalizes first, so a combining sequence and its precomposed
    form key the same shape -- the builder and runtime cannot diverge by
    normalization form (blocker 5)."""
    precomposed = "\u00e9"  # e-acute as one precomposed code point
    decomposed = "e\u0301"  # e + combining acute accent
    assert precomposed != decomposed  # genuinely different code-point sequences
    assert shape(precomposed) == shape(decomposed) == "A"


# ------------------------------------------------------------------ blockers 1-4, 6-7


def test_different_span_geometry_tie_is_structurally_ambiguous():
    """Finding-1 (HIGH): two covers of equal geometry but DIFFERENT span boundaries
    are a STRUCTURAL ambiguity the prior may not resolve. Here [0,4] date is
    unsupported and [1,5] fraction is supported; the two distinct span signatures
    at the top geometry level make the resolution ambiguous, with the min signature
    [0,4] as the canonical representative -- even though the competing [1,5] cover
    is the supported one. The prior never gets to reorder across span sets."""
    dets = [_det(0, 4, "date:Md", "3/24"), _det(1, 5, "number:fraction", "1/2")]
    res = resolve(dets)

    # Both are coverage-4, one-span covers, but over different spans.
    top_two = [c for c in res.covers if c][:2]
    assert {_geometry(c) for c in top_two} == {(4, 1)}
    signatures = [tuple((d["start"], d["end"]) for d in c) for c in top_two]
    assert signatures == [((0, 4),), ((1, 5),)]  # by span signature, ascending

    # Two span signatures at equal geometry => structurally ambiguous. This was
    # WRONGLY False under the old cover-aggregate prior; it must be True now.
    assert res.ambiguous is True
    assert res.structural_ambiguous is True
    assert res.semantic_ambiguous is False

    # The min-signature [0,4] date cover is the canonical 1-best despite date being
    # UNSUPPORTED and the competing [1,5] fraction cover being SUPPORTED: the prior
    # did not reorder them across span sets.
    assert res.best[0]["type"] == "date:Md"
    assert res.best[0]["start"] == 0 and res.best[0]["end"] == 4
    # s* has one span [0,4]; its lone candidate is unambiguous at the span level.
    assert [(s.start, s.end, s.ambiguous) for s in res.spans] == [(0, 4, False)]


def test_one_best_is_independent_of_requested_n():
    """Blocker 2: the prior re-rank must see the COMPLETE same-span equivalence
    class, not a truncated n-best. Three readings share span [0,4]; the supported
    one is listed LAST, so a truncated candidate set (cap = max(n,2) = 2) would
    drop it and pick the wrong 1-best. The true prior winner must be returned for
    every requested n, and resolve_cover must agree."""
    dets = [
        _det(0, 4, "date:Md", "3/24"),  # unsupported (attested-zero on N/N)
        _det(0, 4, "script:latin", "3/24"),  # unmapped -> unsupported
        _det(0, 4, "number:fraction", "1/2"),  # SUPPORTED on N/N, listed last
    ]
    winners = {resolve(dets, n=n).best[0]["type"] for n in range(1, 9)}
    assert winners == {"number:fraction"}  # identical winner regardless of n
    assert resolve_cover(dets).best[0]["type"] == "number:fraction"


def test_supported_reading_beats_unmapped_and_is_unambiguous():
    """Blocker 3 / 7: a measured reading beats an unmapped one, and the tie is not
    ambiguous. "1" as date (supported on shape "N") vs script:latin (unmapped, so
    unsupported): date wins with support_count 1 > 0 and ambiguous is False."""
    dets = [_det(0, 1, "date", "1"), _det(0, 1, "script:latin", "1")]
    res = resolve(dets)

    assert res.best[0]["type"] == "date"
    assert res.ambiguous is False
    by_type = {c[0]["type"]: p[0] for c, p in zip(res.covers, res.priors, strict=True) if c}
    assert by_type["date"].supported is True and by_type["date"].p > 0
    # script:latin maps to no corpus class, so it carries explicit unsupported
    # provenance rather than a fabricated probability.
    assert by_type["script:latin"].tier == "unsupported"
    assert by_type["script:latin"].p is None


def test_two_supported_readings_compare_by_probability():
    """Blocker 4: among two SUPPORTED same-span readings the higher base rate wins.
    On shape "N", cardinal (9/16) outweighs date (7/16), so number:cardinal wins
    and the tie is broken (not ambiguous).

    Pinned to an inline table so this tests the ranking mechanic, not the shipped
    corpus (where "N" is date-majority: date 0.553 over cardinal 0.428)."""
    table = PriorTable({"N": {"cardinal": 9, "date": 7}}, {"source": "fixture"})
    dets = [_det(0, 1, "date", "1"), _det(0, 1, "number:cardinal", "1")]
    res = resolve(dets, feature_sources=[CorpusPrior(table)])

    assert res.best[0]["type"] == "number:cardinal"
    assert res.ambiguous is False
    by_type = {c[0]["type"]: p[0] for c, p in zip(res.covers, res.priors, strict=True) if c}
    assert by_type["number:cardinal"].p > by_type["date"].p


def test_two_unsupported_readings_stay_ambiguous():
    """Blocker 4 / 7: two UNSUPPORTED same-span readings tie honestly. date and
    fraction on shape "A" (absent from the corpus) are both unsupported, so
    support_count and summed log P are equal (0 and 0) and ambiguous is True."""
    dets = [_det(0, 4, "date", "xxxx"), _det(0, 4, "fraction", "yyyy")]
    res = resolve(dets)

    assert res.ambiguous is True
    for priors in res.priors:
        for p in priors:
            assert p is not None and p.supported is False


def test_per_span_winners_are_chosen_by_strength():
    """Per-span resolution: two independent spans, each with two SUPPORTED readings,
    resolve to the higher-base-rate reading at each span independently. On shape
    "N", cardinal (9/16) beats date (7/16), so each span picks cardinal and neither
    span is ambiguous.

    Pinned to an inline table so this tests the per-span strength mechanic, not the
    shipped corpus (where "N" is date-majority)."""
    table = PriorTable({"N": {"cardinal": 9, "date": 7}}, {"source": "fixture"})
    dets = [
        _det(0, 1, "date", "1"),
        _det(0, 1, "number:cardinal", "1"),
        _det(1, 2, "date", "2"),
        _det(1, 2, "number:cardinal", "2"),
    ]
    res = resolve(dets, feature_sources=[CorpusPrior(table)])

    assert [d["type"] for d in res.best] == ["number:cardinal", "number:cardinal"]
    assert [(d["start"], d["end"]) for d in res.best] == [(0, 1), (1, 2)]
    assert res.ambiguous is False
    assert [(s.start, s.end, s.winner["type"], s.ambiguous) for s in res.spans] == [
        (0, 1, "number:cardinal", False),
        (1, 2, "number:cardinal", False),
    ]


def test_breadth_does_not_beat_strength_and_exposes_span_ambiguity():
    """Finding-3 (fugu counterexample): the type choice is per span, not an
    aggregate count across the cover. Span [0,1] has one strongly-supported reading
    (cardinal) beside an unmapped one; it resolves unambiguously by strength. Span
    [1,2] has two unsupported readings (shape "A", absent) with no basis to choose,
    so that span is ambiguous -- and the whole resolution is therefore ambiguous.
    The old cover-aggregate count would have silently broken this to False."""
    dets = [
        _det(0, 1, "number:cardinal", "1"),  # supported on "N"
        _det(0, 1, "script:latin", "1"),  # unmapped -> unsupported
        _det(1, 2, "date", "z"),  # shape "A", absent -> unsupported
        _det(1, 2, "fraction", "z"),  # shape "A", absent -> unsupported
    ]
    res = resolve(dets)

    # Winner at each span is the per-span argmax: span 0 chooses cardinal by
    # strength; span 1 has no basis, so its winner is a deterministic representative.
    assert res.best[0]["type"] == "number:cardinal"
    assert res.best[0]["start"] == 0 and res.best[1]["start"] == 1

    # The per-span ambiguity is exposed and lifts the overall flag; it is SEMANTIC,
    # not structural (all top covers share the one signature {(0,1),(1,2)}).
    spans = {(s.start, s.end): s for s in res.spans}
    assert spans[(0, 1)].ambiguous is False
    assert spans[(1, 2)].ambiguous is True
    assert res.structural_ambiguous is False
    assert res.semantic_ambiguous is True
    assert res.ambiguous is True


def test_best_ambiguous_and_spans_are_independent_of_n():
    """N-independence: the 1-best cover, the ambiguity flags, and the per-span
    alternatives are identical for resolve(n=1), n=2, n=8, and resolve_cover(),
    even with many same-span alternatives listed after the supported one, and even
    when a unique-best-geometry cover sits above a wide second-level class."""

    def snapshot(res):
        return (
            tuple((d["type"], d["start"], d["end"]) for d in res.best),
            res.ambiguous,
            res.structural_ambiguous,
            res.semantic_ambiguous,
            tuple(
                (
                    s.start,
                    s.end,
                    s.winner["type"],
                    s.ambiguous,
                    tuple(r.detection["type"] for r in s.readings),
                )
                for s in res.spans
            ),
        )

    # (a) Seven readings share span [0,4]; only the fraction is supported and it is
    # listed LAST. A truncated n-best would drop it, but the per-span view gathers
    # the whole span, so fraction wins for every n. The per-span alternatives are a
    # complete, ranked, n-independent view.
    wide = [
        _det(0, 4, "date:Md", "3/24"),  # unsupported (attested-zero on N/N)
        _det(0, 4, "script:latin", "3/24"),  # unmapped
        _det(0, 4, "foo:a", "3/24"),  # unmapped
        _det(0, 4, "bar:b", "3/24"),  # unmapped
        _det(0, 4, "baz:c", "3/24"),  # unmapped
        _det(0, 4, "qux:d", "3/24"),  # unmapped
        _det(0, 4, "number:fraction", "1/2"),  # SUPPORTED on N/N, listed last
    ]
    snaps = {snapshot(resolve(wide, n=n)) for n in (1, 2, 8)}
    assert len(snaps) == 1  # identical across n
    only = next(iter(snaps))
    assert only[0] == (("number:fraction", 0, 4),)  # fraction is the 1-best
    assert only[1] is False  # unambiguous: unique supported winner
    assert resolve_cover(wide).best[0]["type"] == "number:fraction"
    # The per-span alternatives are the whole span, seven candidates, fraction first.
    (span,) = resolve(wide, n=1).spans
    assert len(span.readings) == 7 and span.readings[0].detection["type"] == "number:fraction"

    # (b) A unique-best-geometry cover [0,7] (coverage 7) sits above a seven-way
    # class of coverage-6 alternatives. The top level is a single cover regardless
    # of n, so best/ambiguous/spans are stable and the geometric margin is 1.
    tall = [_det(0, 7, "date", "1234567")]
    for i in range(7):
        tall.append(_det(0, 6, f"num{i}:cardinal", "123456"))
    r1, r2, r8 = resolve(tall, n=1), resolve(tall, n=2), resolve(tall, n=8)
    assert snapshot(r1) == snapshot(r2) == snapshot(r8)
    assert r2.best[0]["type"] == "date" and r2.ambiguous is False
    assert r2.margin == CoverMargin(1, 0, 0)  # coverage 7 vs 6, same span count
    assert resolve_cover(tall).best[0]["type"] == "date"


def test_exact_probability_ordering_survives_float_rounding():
    """Fugu re-review HIGH: the per-span winner and the tie decision compare EXACT
    Decimal probabilities, never float(p). Two supported readings on shape "N" whose
    probabilities differ only in digits far below float precision (both round to
    0.5) must still order by the exact value -- cardinal (the strictly higher p)
    wins and the span is NOT ambiguous. Through float they would collapse to an
    equal log and the tie-break would wrongly pick the lower-p reading."""
    table = PriorTable({"N": {"cardinal": 10**20 + 1, "date": 10**20}}, {"source": "fixture"})
    dets = [_det(0, 1, "date", "1"), _det(0, 1, "number:cardinal", "1")]
    res = resolve(dets, feature_sources=[CorpusPrior(table)])

    # Both base rates round to the same float, so the comparison must be exact.
    p_card = table.reading_prior(_det(0, 1, "number:cardinal", "1")).p
    p_date = table.reading_prior(_det(0, 1, "date", "1")).p
    assert float(p_card) == float(p_date) == 0.5  # indistinguishable in float
    assert p_card > p_date  # but strictly ordered as exact Decimals

    assert res.best[0]["type"] == "number:cardinal"
    assert res.ambiguous is False
    assert res.spans[0].ambiguous is False

    # A genuinely equal base rate is still an honest tie -- exact equality holds.
    even = PriorTable({"N": {"cardinal": 5, "date": 5}}, {"source": "fixture"})
    tie = resolve(dets, feature_sources=[CorpusPrior(even)])
    assert tie.ambiguous is True
    assert tie.spans[0].ambiguous is True


def test_non_positive_or_non_integer_n_is_rejected():
    """Fugu re-review LOW: n <= 0 would yield an empty covers list beside a populated
    best, breaking covers[0] == best. resolve rejects a non-positive or non-integer
    n with a clear error; a positive n keeps the contract."""
    dets = [_det(0, 4, "date:Md", "3/24"), _det(0, 4, "number:fraction", "3/24")]
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            resolve(dets, n=bad)
    res = resolve(dets, n=1)
    assert res.covers[0] == res.best  # contract holds for the smallest valid n


class _NanSource:
    """A feature source that emits a non-finite contribution, to prove the boundary
    guard rejects it rather than raising decimal.InvalidOperation during sort."""

    def features(self, detection, context):
        return (ReadingFeature(name="bad", contribution=Decimal("NaN"), detail="", n=None),)


def test_non_finite_feature_contribution_is_rejected():
    """Blocker 6: a NaN feature contribution is rejected with a clear error at the
    feature boundary, not left to corrupt the ordering."""
    dets = [_det(0, 4, "date:Md", "3/24"), _det(0, 4, "number:fraction", "3/24")]
    with pytest.raises(ValueError, match="non-finite"):
        resolve(dets, feature_sources=[_NanSource()])


def test_malformed_prior_table_is_rejected():
    """Blocker 6: PriorTable validates on load -- non-negative counts and positive
    totals -- with a clear error, so a corrupt table cannot poison a base rate."""
    with pytest.raises(ValueError, match="negative"):
        PriorTable({"N/N": {"fraction": -1}}, {"source": "fixture"})
    with pytest.raises(ValueError, match="non-positive total"):
        PriorTable({"N/N": {"fraction": 0}}, {"source": "fixture"})
    # Non-string keys never match a shape/class lookup, so reject them on load.
    with pytest.raises(ValueError, match="shape key is not a string"):
        PriorTable({1: {"fraction": 3}}, {"source": "fixture"})
    with pytest.raises(ValueError, match="class key .* is not a string"):
        PriorTable({"N/N": {2: 3}}, {"source": "fixture"})
    # A well-formed table still loads and reads back.
    ok = PriorTable({"N/N": {"fraction": 3}}, {"source": "fixture"})
    assert ok.n("N/N") == 3


def test_supported_base_rate_log_is_finite_under_astronomical_totals():
    """Minor boundary: a supported reading always has p > 0, but float(p) can
    underflow to 0.0 for an astronomically large sample, which would make
    math.log raise a domain error. The feature falls back to exact Decimal ln, so
    the contribution stays finite rather than crashing (corpus sizes are nowhere
    near this; the boundary is simply closed)."""
    huge = 10**400
    table = PriorTable({"N": {"cardinal": 1, "date": huge - 1}}, {"source": "fixture"})
    prior = table.reading_prior(_det(0, 1, "number:cardinal", "1"))
    assert prior is not None and prior.supported is True
    assert float(prior.p) == 0.0  # the base rate underflows in float
    context = ResolveContext(source_text=None, all_detections=())
    feats = CorpusPrior(table).features(_det(0, 1, "number:cardinal", "1"), context)
    assert len(feats) == 1 and feats[0].contribution.is_finite()


# --------------------------------------------------------------------------- seams


def test_corpus_prior_ignores_context_and_withholds_unsupported():
    # Pinned to an inline table so the sample size is fixed by the test, not the
    # shipped corpus (where n(N/N) is ~1e5, all fraction).
    prior = CorpusPrior(PriorTable({"N/N": {"fraction": 7}}, {"source": "fixture"}))
    context = ResolveContext(source_text="on 3/24 we met", all_detections=())
    # Supported reading yields exactly one base_rate feature carrying its sample size.
    feats = prior.features(_det(0, 4, "number:fraction", "3/24"), context)
    assert len(feats) == 1
    assert feats[0].name == "base_rate" and feats[0].n == 7
    # A type mapping to no corpus class yields no feature at all.
    assert corpus_classes("script:latin") == ()
    assert prior.features(_det(0, 3, "script:latin", "abc"), context) == ()


def test_isolated_letter_readings_map_to_corpus_classes():
    assert corpus_classes("letter:name") == ("letters",)
    assert corpus_classes("word:single-letter") == ("plain",)
    assert corpus_classes("number:cardinal:roman") == ("cardinal",)


def test_resolve_cover_applies_the_prior_and_exposes_it():
    cover = resolve_cover([_det(0, 4, "date:Md", "3/24"), _det(0, 4, "number:fraction", "3/24")])
    assert cover.best[0]["type"] == "number:fraction"
    assert cover.priors[0].group == "fraction"


def test_plural_numerals_take_the_corpus_date_class():
    """The corpus files "1990s" and "'90s" as DATE, so decades are priored there."""
    assert corpus_classes("number:plural") == ("date",)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
