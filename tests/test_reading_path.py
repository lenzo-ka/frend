"""IRN's consumer path profile addresses a real ranked resolution."""

from __future__ import annotations

import pytest
from icukit.detectors import detect
from icukit.recognize import (
    FlexibleCurrencyDetector,
    FlexibleDateDetector,
    FlexibleNumberDetector,
)
from tiergraph import ItemRef
from tiergraph.path import (
    AlternativeRef,
    CanonicalPath,
    PathKind,
    PathRefusal,
    PathRefusalCode,
    ResolvedAlternative,
    ResolvedItem,
    resolve_path,
)

from irn.fold_resolve import _POS, build_lattice, resolve
from irn.reading_path import IrnReadingProfile

DETECTORS = [
    FlexibleDateDetector("en_US"),
    FlexibleNumberDetector("en_US"),
    FlexibleCurrencyDetector("en_US", "USD"),
]


def _profile(text: str = "on 1/3/2026 we paid $1,234.50"):
    detections = detect(text, DETECTORS)
    return detections, IrnReadingProfile(detections)


def _refuses(code: PathRefusalCode, operation) -> PathRefusal:
    with pytest.raises(PathRefusal) as caught:
        operation()
    assert caught.value.code is code
    return caught.value


def test_ranked_readings_resolve_to_real_covers():
    detections, profile = _profile()
    expected = resolve(detections)

    first = resolve_path(profile.graph, profile, "/reading/0")
    second = resolve_path(profile.graph, profile, "/reading/1")
    assert isinstance(first, ResolvedAlternative)
    assert first.value == expected.covers[0] == expected.best
    assert isinstance(second, ResolvedAlternative)
    assert second.value == expected.covers[1]

    _refuses(
        PathRefusalCode.ALTERNATIVE_OUT_OF_RANGE,
        lambda: resolve_path(profile.graph, profile, f"/reading/{len(profile.covers)}"),
    )


def test_offset_and_reading_kinds_are_distinct():
    _detections, profile = _profile("1/3/2026")

    offset = resolve_path(profile.graph, profile, "/offset/3", require=PathKind.ITEM)
    assert isinstance(offset, ResolvedItem)
    assert offset.current == ItemRef(_POS, 3)
    assert isinstance(
        resolve_path(profile.graph, profile, "/reading/0", require=PathKind.ALTERNATIVE),
        ResolvedAlternative,
    )

    _refuses(
        PathRefusalCode.WRONG_KIND,
        lambda: resolve_path(profile.graph, profile, "/reading/0", require=PathKind.ITEM),
    )
    _refuses(
        PathRefusalCode.WRONG_KIND,
        lambda: resolve_path(profile.graph, profile, "/offset/3", require=PathKind.ALTERNATIVE),
    )


def test_spell_round_trips_both_forms():
    _detections, profile = _profile("1/3/2026")
    for text in ("/offset/3", "/reading/1"):
        binding = profile.bind(CanonicalPath.parse(text), profile.graph)
        assert str(profile.spell(binding, profile.graph)) == text


def test_unknown_form_and_integer_refusals():
    _detections, profile = _profile("1/3/2026")
    _refuses(
        PathRefusalCode.UNKNOWN_FORM,
        lambda: resolve_path(profile.graph, profile, "/cover/0"),
    )
    refusal = _refuses(
        PathRefusalCode.NONCANONICAL_SEGMENT,
        lambda: resolve_path(profile.graph, profile, "/reading/01"),
    )
    assert refusal.offender.segment_index == 1
    assert refusal.offender.segment == "01"


def test_profile_is_guarded_to_its_graph_snapshot():
    detections, profile = _profile("1/3/2026")
    other_graph, _roots, _indices = build_lattice(detections)
    _refuses(
        PathRefusalCode.PROFILE_REFUSED,
        lambda: resolve_path(other_graph, profile, "/reading/0"),
    )


def test_alternatives_refuse_wrong_owner_and_relation():
    _detections, profile = _profile("1/3/2026")
    reading = profile.bind(CanonicalPath.parse("/reading/0"), profile.graph)
    assert isinstance(reading, AlternativeRef)
    _refuses(
        PathRefusalCode.PROFILE_REFUSED,
        lambda: profile.alternatives(
            ItemRef(_POS, 1),
            reading.relation,
            profile.graph,
        ),
    )
    _refuses(
        PathRefusalCode.PROFILE_REFUSED,
        lambda: profile.alternatives(
            reading.owner,
            reading.owner.tier,
            profile.graph,
        ),
    )


def test_empty_detections_addresses_the_sole_empty_reading():
    # No detections: the only reading is the empty cover; p0 still exists.
    profile = IrnReadingProfile([])
    reading = resolve_path(profile.graph, profile, "/reading/0")
    assert isinstance(reading, ResolvedAlternative)
    assert reading.value == ()
    offset = resolve_path(profile.graph, profile, "/offset/0", require=PathKind.ITEM)
    assert isinstance(offset, ResolvedItem)
    assert offset.current == ItemRef(_POS, 0)
    _refuses(
        PathRefusalCode.ALTERNATIVE_OUT_OF_RANGE,
        lambda: resolve_path(profile.graph, profile, "/reading/1"),
    )


def test_negative_and_out_of_range_offset_are_typed():
    _detections, profile = _profile("1/3/2026")  # span_end 8: positions p0..p8
    _refuses(
        PathRefusalCode.INVALID_SEGMENT,
        lambda: resolve_path(profile.graph, profile, "/offset/-1"),
    )
    _refuses(
        PathRefusalCode.OUT_OF_RANGE,
        lambda: resolve_path(profile.graph, profile, "/offset/99"),
    )


def test_direct_spell_and_alternatives_guard_the_snapshot():
    detections, profile = _profile("1/3/2026")
    other_graph, _roots, _indices = build_lattice(detections)
    reading = profile.bind(CanonicalPath.parse("/reading/0"), profile.graph)
    _refuses(
        PathRefusalCode.UNSPELLABLE,
        lambda: profile.spell(reading, other_graph),
    )
    _refuses(
        PathRefusalCode.PROFILE_REFUSED,
        lambda: profile.alternatives(reading.owner, reading.relation, other_graph),
    )
