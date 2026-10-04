"""Ranges ICU writes: icukit's range and date-interval readers in the spoken profile,
each read as one range, its ends read as frend reads each value alone and joined by the
locale's connector patterns (``frend.ranges``, ``frend.verbalize._spoken_range``)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from icukit.detectors import NumberValue, detect  # noqa: E402

from frend import resolve_lattice  # noqa: E402
from frend.context import TextContext  # noqa: E402
from frend.ranges import RangeValue, has_icu_ranges  # noqa: E402
from frend.spoken_priors import normalize_spoken  # noqa: E402
from frend.verbalize import _fill_slot, _spoken_range, verbalize_lattice  # noqa: E402

needs_icu_ranges = pytest.mark.skipif(
    not has_icu_ranges(),
    reason="this icukit builds no range readers (icukit.engine.range_detectors)",
)


def _readings(text: str, count: int | None = 2) -> list[tuple[str, str]]:
    """The first ``count`` readings of ``text`` as the evaluator reads a token (alone, its
    own context), each with its source: a span read as one unit gives that unit's
    readings; a span read as several gives its one first reading, sourced "(units)"."""
    from reading_profile import reading_detectors

    detections = list(detect(text, reading_detectors("en_US")))
    verbalized = verbalize_lattice(
        resolve_lattice(detections, source_text=text), context=TextContext(text)
    )
    units = verbalized.best_path.units
    if len(units) == 1:
        alternatives = units[0].alternatives
        if count is not None:
            alternatives = alternatives[:count]
        return [(normalize_spoken(item.text), item.provenance) for item in alternatives]
    said = "".join(
        u.best.text if u.best.provenance == "surface:passthrough" else f" {u.best.text} "
        for u in units
    )
    return [(normalize_spoken(said), "(units)")]


def _assert_range_reads(text: str, said: str) -> None:
    readings = _readings(text)
    assert said in [reading for reading, _ in readings], readings
    assert all(source.startswith("range:") for _, source in readings), readings


# ------------------------------------------------------------------ ICU's own ranges


@needs_icu_ranges
def test_icu_en_dash_years():
    """CLDR's range separator between two years reads as one range of years (frend
    #43 already says "to" for the en dash between the three units it read)."""
    _assert_range_reads("1990–1995", "nineteen ninety to nineteen ninety five")


@needs_icu_ranges
def test_icu_spaced_interval():
    """CLDR's interval fallback, spaced ("{0} – {1}"), reads as the same range."""
    _assert_range_reads("1990 – 1995", "nineteen ninety to nineteen ninety five")


@needs_icu_ranges
def test_icu_measure_range_moves_the_unit():
    """A unit written once, after the right end, is said once, after it."""
    _assert_range_reads("5–10 km", "five to ten kilometers")


@needs_icu_ranges
def test_icu_range_follows_the_table(no_context_trees):
    """ "1990–95" reads as the range table orders a four-digit, two-digit range (J's
    ``dash:4+2`` row, the en dash pooled with the hyphen), no longer by the interim
    four-digit-year rule: the same first reading, from a measured source."""
    (first, source), *_ = _readings("1990–95", 1)
    assert first == "one thousand nine hundred ninety ninety five", (first, source)
    assert source == "range:cardinal+silent+cardinal"


@needs_icu_ranges
def test_money_range_offers_the_currency_in_place_and_at_the_end():
    """ "$5–10": the currency is written on the left only; both placements are offered,
    unmeasured (the ranges plan's (R))."""
    said = [reading for reading, _ in _readings("$5–10", 16)]
    assert "five dollars to ten" in said
    assert "five to ten dollars" in said


@needs_icu_ranges
def test_date_interval_ends_are_cut_where_icu_writes_them():
    """ICU writes "May 3 – 5, 2020" with the month on the left end and the year on the
    right: each end says what it writes."""
    assert _readings("May 3–5, 2020", 1)[0][0] == "may third to fifth twenty twenty"
    assert _readings("10:00 AM – 2:00 PM", 1)[0][0] == "ten a m to two p m"


def test_approximately_is_not_in_the_profile():
    """icukit builds an approximately reader beside each range reader; "~3" is no range,
    and none of them is read (filtering "number:approximately" alone would leave
    "measure:approximately" on "~3 km")."""
    from reading_profile import reading_detectors

    detectors = reading_detectors("en_US")
    assert not any(str(getattr(d, "type", "")).endswith(":approximately") for d in detectors)
    for text in ("~3", "~3 km"):
        types = [str(d["type"]) for d in detect(text, detectors)]
        assert not any(t.endswith(":approximately") for t in types), (text, types)


# ------------------------------------------------------------------ connector slots

_OT_DO = {
    "id": "ot-do",
    "pattern": "от {0} до {1}",
    "slots": {
        "0": {"case": "genitive", "gender": "masculine"},
        "1": {"case": "genitive", "gender": "masculine"},
    },
}


def _numbers(left: str, right: str) -> RangeValue:
    def end(decimal: str) -> dict:
        return {"type": "number:decimal", "text": decimal, "value": NumberValue(decimal=decimal)}

    return RangeValue((end(left),), "–", "range", (end(right),))


def test_russian_genitive_slot_fills_from_rbnf():
    """A slot's case and gender pick the locale's RBNF rule set (fixture pattern: ru has
    no range connector in its lexical table yet)."""
    said = _spoken_range(_numbers("5", "10"), "ru_RU", apply_source_priors=False, patterns=[_OT_DO])
    assert [item.text for item in said] == ["от пяти до десяти"]


def test_unfillable_slot_skips_the_pattern():
    """A slot the locale has no rule set for is not said in another case: the pattern is
    left out."""
    wrong = {**_OT_DO, "slots": {"0": {"case": "genitive", "gender": "xx"}, "1": {}}}
    assert _fill_slot({"value": NumberValue(decimal="5")}, wrong["slots"]["0"], "ru_RU") is None
    said = _spoken_range(_numbers("5", "10"), "ru_RU", apply_source_priors=False, patterns=[wrong])
    assert said == ()


# ------------------------------------------------------------------ ends read as written


def _first(text: str) -> str:
    return _readings(text, 1)[0][0]


def _all(text: str) -> list[str]:
    return [reading for reading, _ in _readings(text, 64)]


@needs_icu_ranges
def test_decimal_ends_keep_their_written_zeros():
    """An end is whole only where it writes no fraction: "1.00" is said "one point o o",
    as frend reads it alone (and as the base read each end), not "one"."""
    assert _first("1.00–2.00") == "one point o o to two point o o"
    assert _first("1.50–2.00") == "one point five o to two point o o"


@needs_icu_ranges
def test_percent_ends_keep_their_written_fraction_digits():
    """A percent end is read from its own captures ("79.20%": fraction "20"), so its
    written digits are said; "80.00%" keeps "eighty point o o percent" among them."""
    assert _first("79.20%–80.00%").startswith("seventy nine point two o percent to ")
    assert any(r.endswith("to eighty point o o percent") for r in _all("79.20%–80.00%"))


@needs_icu_ranges
def test_an_interval_writing_fields_its_skeleton_does_not_name_is_not_a_range():
    """ICU's "h" interval writes each end's date when the days differ; that reading is
    dropped, so the span keeps the readings frend has without it, never an unsupported
    one."""
    from reading_profile import reading_detectors

    text = "5/1/2020, 10 AM – 5/2/2020, 10 AM"
    lattice = resolve_lattice(list(detect(text, reading_detectors("en_US"))), source_text=text)
    units = verbalize_lattice(lattice, context=TextContext(text)).best_path.units
    sources = [unit.best.provenance for unit in units]
    assert "surface:unsupported" not in sources, sources
    assert not any(source.startswith("range:") for source in sources), sources


@needs_icu_ranges
def test_year_readings_are_for_plain_numbers_only():
    """Year readings (R8) are offered for plain whole ends only; a range with a unit or
    currency keeps its kind's order and is not sourced as a date."""
    for text in ("1990–95 km", "$1000–2000"):
        (first, source), *_ = _readings(text, 1)
        assert not first.startswith("nineteen ninety"), (text, first)
        assert "date" not in source, (text, source)


@needs_icu_ranges
def test_interval_fields_are_cut_in_code_points_not_utf16_units():
    """ICU's field positions are UTF-16; Adlam's digits are outside the BMP, so each
    field's text must be cut through UTF-16 ("𞥑𞥐" for 10, "𞥓𞥐" for 30)."""
    import icu

    from frend.ranges import date_interval_readers, from_icukit

    locale = "ff_Adlm_GN"
    digits = icu.NumberFormat.createInstance(icu.Locale(locale))
    (reader,) = [r for r in date_interval_readers(locale) if r.type == "date-interval:hm"]
    formatter = icu.DateIntervalFormat.createInstance("hm", icu.Locale(locale))
    calendar = icu.Calendar.createInstance(icu.Locale(locale))
    calendar.clear()
    calendar.set(2000, 0, 1, 10, 0)
    early = calendar.getTime()
    calendar.clear()
    calendar.set(2000, 0, 1, 14, 30)
    text = str(formatter.format(icu.DateInterval(early, calendar.getTime())))
    (found,) = reader.detect(text)
    value = from_icukit(found, locale)["value"]
    left = {c.name: c.text for c in value.left[0]["captures"]}
    right = {c.name: c.text for c in value.right[0]["captures"]}
    assert left["H"] == digits.format(10)
    assert right["m"] == digits.format(30)
    assert all(left.values()) and all(right.values())


# ------------------------------------------------------------------ ranges written in text
#
# The ranges plan's P6: a range written in running text ("5-10", "16:79", "3x4") is one
# span (frend.ranges.RangeDetector), emitted where the range table's E says so, and its
# readings ordered by J (frend/data/en/range_priors.json). Rules R1-R12 are tested as
# rules; E and J are measured.


def _units(text: str):
    from reading_profile import reading_detectors

    detections = list(detect(text, reading_detectors("en_US")))
    lattice = resolve_lattice(detections, source_text=text)
    return verbalize_lattice(lattice, context=TextContext(text)).best_path.units


def _range_unit(text: str):
    (unit,) = [u for u in _units(text) if u.best.provenance.startswith("range:")]
    return unit


def _said(text: str) -> str:
    out = ""
    for unit in _units(text):
        best = unit.best
        out += best.text if best.provenance == "surface:passthrough" else f" {best.text} "
    return normalize_spoken(out)


def _token_in_sentence(before: str, token: str, after: str):
    """``token`` read alone with its sentence as context, as the evaluator reads a corpus
    token: its units' alternatives."""
    from reading_profile import reading_detectors

    head = f"{before} " if before else ""
    tail = f" {after}" if after else ""
    detections = list(detect(token, reading_detectors("en_US")))
    verbalized = verbalize_lattice(
        resolve_lattice(detections, source_text=token),
        context=TextContext(f"{head}{token}{tail}", len(head)),
    )
    return verbalized.best_path.units


def test_written_hyphen_range_is_one_span():
    """ "5-10" is one unit, every reading a range source; "to" and the silent form are
    both offered (main read it as the units "5" and "-10")."""
    readings = _readings("5-10", 16)
    assert all(source.startswith("range:") for _, source in readings), readings
    said = [reading for reading, _ in readings]
    assert "five to ten" in said and "five ten" in said


def test_year_range_prior_order(no_context_trees):
    """J's ``dash:4+4`` leader is ``range:date+to+date`` (main, no trees: "nineteen
    ninety minus one thousand nine hundred ninety five")."""
    (first, source), *_ = _readings("1990-1995", 1)
    assert first == "nineteen ninety to nineteen ninety five"
    assert source == "range:date+to+date"


def test_four_plus_two_prior_order(no_context_trees):
    """R6 refined keeps a silent row when both ends are correct; the rebuilt
    ``dash:4+2`` prior therefore leads with its measured silent cardinal pair."""
    readings = _readings("1992-93", 16)
    assert readings[0] == (
        "one thousand nine hundred ninety two ninety three",
        "range:cardinal+silent+cardinal",
    )
    assert "nineteen ninety two to ninety three" in [r for r, _ in readings]


def test_joined_year_span_uses_the_range_tree_in_running_text():
    """The retained ``range:range`` tree fixes a joined span from the held-out probe."""
    before = "Count Alessandro Contini Bonacossi ( 18 March"
    after = "October 1955 ) was an Italian politician , art collector , dealer and philatelist ."
    (unit,) = _token_in_sentence(before, "1878-22", after)
    assert normalize_spoken(unit.best.text) == "eighteen seventy eight to twenty two"
    assert unit.context is not None
    assert unit.context.problem == "range:range"
    assert unit.context.applied


def test_range_tree_shape_is_still_emitted_by_the_detector():
    """The duration exception leaves the retained ``range:range`` detector case alone."""
    from reading_profile import reading_detectors

    found = reading_detectors("en_US")[-1].detect("1878-22")
    assert [(item["type"], item["sub_key"]) for item in found] == [("range:dash", "dash:4+2")]


@pytest.mark.parametrize(
    "text",
    ["1.2:3.4", "3-2", "3:2", "10:30", "10:30.5 am", "1/2", "1:2.3", "1878-22"],
)
def test_duration_exception_does_not_change_out_of_scope_range_detections(text, monkeypatch):
    """Decimal ratios, scores, clocks, fractions, versions, and range-tree cases keep
    the detector result they have when the duration exception is disabled."""
    from reading_profile import reading_detectors

    detector = reading_detectors("en_US")[-1]
    found = detector.detect(text)
    monkeypatch.setattr(detector, "_fractional_duration", lambda _text: False)
    assert detector.detect(text) == found


def test_silence_is_offered_inside_a_written_range():
    """The silent connector is offered inside a written range (main offered "to" and the
    minus sign only)."""
    said = [reading for reading, _ in _readings("1917-1918", 16)]
    assert "one thousand nine hundred seventeen one thousand nine hundred eighteen" in said


def test_range_ends_are_read_together(no_context_trees):
    """The two ends are one joint reading (main, no trees: "eighteen eighty three minus
    one thousand nine hundred seventy five")."""
    assert _said("( 1883-1975 )") == "eighteen eighty three to nineteen seventy five"


def test_measure_range_reads_to():
    """A measure end is a range end (R7; main: "five minus ten kilograms")."""
    assert _said("5-10 kg") == "five to ten kilograms"


def test_money_ranges_offer_both_placements():
    """R10: a money range keeps P5's order and both unit placements; J does not reorder
    it (no first reading is asserted: the corpus holds no money range reading)."""
    said = [reading for reading, _ in _readings("$5-10", 16)]
    assert "five dollars to ten" in said and "five to ten dollars" in said
    said = [reading for reading, _ in _readings("$15,000-$25,000", 16)]
    assert "fifteen thousand dollars to twenty five thousand dollars" in said
    assert "fifteen thousand dollars twenty five thousand dollars" in said


def test_invalid_clock_colon_reads_to():
    """A colon that is no clock (R4a) is a ratio range read "to" (main: "sixteen seventy
    nine", "twenty nine hours forty six minutes")."""
    assert _said("16:79") == "sixteen to seventy nine"
    assert _said("29:46") == "twenty nine to forty six"


def test_dimension_reads_by():
    """ "x" between numbers is a dimension (main: "three x four", one reading)."""
    readings = _readings("3x4", 16)
    assert readings[0][0] == "three by four"
    assert "three x four" in [reading for reading, _ in readings]
    assert _said("4x400m") == "four by four hundred meters"


def test_lone_colon_between_numbers_is_offered_to():
    """R12: a lone ":" between numbers (the corpus token read alone) is offered "to",
    first by the range table's ratio class share (main: silence and "colon" only)."""
    (unit,) = _token_in_sentence("aspect ratio 16", ":", "9 .")
    assert unit.alternatives[0].text == "to"
    assert "" in [a.text for a in unit.alternatives]


def test_emit_decision():
    """E: a valid clock (R4a) and a phone's local shape (``dash:3+4``: 31 range triples
    against 4,004 single tokens) emit no span; an invalid clock does; a sub-key never
    observed reads its class row. Break: with the table's readings emptied, nothing is
    emitted."""
    from reading_profile import reading_detectors

    from frend.ranges import RangeDetector, RangePriorTable, load_range_priors

    detector = reading_detectors("en_US")[-1]
    assert isinstance(detector, RangeDetector)
    assert detector.detect("10:30") == []
    assert detector.detect("555-1212") == []
    (found,) = detector.detect("16:79")
    assert found["type"] == "range:ratio" and found["sub_key"] == "ratio:2+2"
    table = load_range_priors("en_US")
    assert table.emission("dash:3+4") == (31, 4004)
    assert table.emission("dash:31+29") == table.emission("dash")
    document = {"readings": {}, "kinds": {}}
    empty = RangeDetector("en_US", detector.endpoints, table=RangePriorTable(document))
    assert empty.detect("5-10") == []


def test_minus_reading_survives_in_the_choice_lattice():
    """The one span wins by geometry; the old edges ("5" and "-10") stay in the choice
    lattice, so the minus reading survives for alignment."""
    from reading_profile import reading_detectors

    from frend.lattice import resolve_choices

    text = "5-10"
    lattice = resolve_choices(list(detect(text, reading_detectors("en_US"))), source_text=text)
    spans = {(e.start, e.end, e.detection["type"]) for e in lattice.edges if e.kind == "reading"}
    assert (1, 4, "number:decimal") in spans
    assert (0, 4, "range:dash") in spans


def test_valid_clock_is_still_a_time():
    readings = _readings("10:30", 16)
    assert readings[0][0] == "ten thirty"
    assert not any(source.startswith("range:") for _, source in readings)


def test_fractional_elapsed_time_beats_the_ratio_range():
    """Training's M:SS.hh race times use ICU's minutes/seconds parse, not a ratio."""
    readings = _readings("2:08.34", None)
    assert readings[0][0] == "two minutes eight seconds and thirty four milliseconds"
    assert not any(source.startswith("range:") for _, source in readings)


def test_iso_date_is_not_a_range():
    for unit in _units("2008-09-30"):
        assert not any(a.provenance.startswith("range:") for a in unit.alternatives)


def test_a_chain_is_not_a_range():
    """R3: three or more digit groups are an identifier."""
    for text in ("1-2-3", "978-0-19-960563-7"):
        for unit in _units(text):
            for alternative in unit.alternatives:
                assert not alternative.provenance.startswith("range:"), text
        assert " to " not in f" {_said(text)} ", text


def test_range_reading_prior_is_measured():
    """The range span's reading prior is E's measurement: p = range / (range + single) of
    its emit key, tier measured. Break: a table whose reading_prior returns None fails
    it."""
    from decimal import Decimal

    from reading_profile import reading_detectors

    from frend.ranges import load_range_priors
    from frend.type_priors import CorpusPrior

    table = load_range_priors("en_US")
    found = reading_detectors("en_US")[-1].detect("16:79")[0]
    prior = CorpusPrior(range_table=table).reading_prior(found)
    positives, negatives = table.emission("ratio:2+2")
    assert prior.tier == "measured" and prior.supported
    assert prior.p == Decimal(positives) / (positives + negatives)
    assert prior.n == positives + negatives


def test_leading_zero_right_end_reads_digits(no_context_trees):
    """R4b/R11: a right end written with a leading zero is its own sub-key
    (``dash:4+2:0``), whose leader says the digits (main: "from twenty twenty minus five
    on")."""
    assert _said("from 2020-05 on") == "from twenty twenty five on"
    assert _range_unit("from 2020-05 on").best.provenance == "range:date+silent+cardinal"


def test_punctuation_dash_filter():
    """R6 (kal A): a dash said as nothing after a year-shaped cardinal, or with a money
    end, is punctuation; a dash said "to", or silent between plain cardinals, is not.
    The shipped table records the filter it applied."""
    import json

    from frend.ranges import load_range_priors, punctuation_dash

    assert punctuation_dash(
        ("CARDINAL", "1994"), ("PUNCT", "-", "sil"), ("CARDINAL", "95"), ends_match=False
    )
    assert not punctuation_dash(
        ("CARDINAL", "1994"), ("PUNCT", "-", "sil"), ("CARDINAL", "95"), ends_match=True
    )
    assert not punctuation_dash(
        ("CARDINAL", "1994"), ("PLAIN", "-", "to"), ("CARDINAL", "95"), ends_match=False
    )
    relevance = load_range_priors("en_US").provenance["relevance"]
    assert relevance["dropped_filter"].startswith("punctuation_dash:")
    assert relevance["wider_not_counted"]["punctuation_dash"] > 0
    json.dumps(relevance)


def test_no_table_reproduces_main(monkeypatch):
    """R9: with no range table the detector emits nothing and #43's joined path reads
    every range as main reads it."""
    from frend import ranges, verbalize

    monkeypatch.setattr(ranges, "load_range_priors", lambda locale="en_US": None)
    monkeypatch.setattr(verbalize, "load_range_priors", lambda locale="en_US": None)
    from reading_profile import reading_detectors

    assert reading_detectors("en_US")[-1].__class__.__name__ == "RangeDetector"
    detector = ranges.RangeDetector("en_US", reading_detectors("en_US")[:-1])
    assert detector.detect("5-10") == []
    assert _said("pages 5-10") == "pages five to ten"
    assert _said("from 1990-1995 the club") == (
        "from nineteen ninety to nineteen ninety five the club"
    )
    assert not any(
        a.provenance.startswith("range:") for u in _units("pages 5-10") for a in u.alternatives
    )


def test_relevance_predicates():
    """R1-R3 decide which corpus triples are evidence for E and J: a date whose right
    end starts with a word, and a chain, are not."""
    from frend.ranges import emit_relevant

    assert not emit_relevant("Oct. 29, 1951", "-", "April 28, 1953")
    assert not emit_relevant("1-2", "-", "3")
    assert emit_relevant("1990", "-", "95")
    assert emit_relevant("15", "-", "$25")


def test_date_day_ranges_are_one_typed_range_and_complete_an_abbreviated_day():
    assert _readings("June 26-27", 1)[0] == (
        "june twenty sixth to twenty seventh",
        "range:date+to+date",
    )
    assert _readings("June 26-7", 1)[0] == (
        "june twenty sixth to twenty seventh",
        "range:date+to+date",
    )


def test_month_and_date_hyphens_are_ranges():
    assert _readings("June-July 2020", 1)[0][0] == "june to july twenty twenty"
    assert _readings("Jun-Jul", 1)[0][0] == "june to july"
    assert _readings("June 26-July 3", 1)[0][0] == "june twenty sixth to july third"


def test_elided_years_and_their_ranges_are_read():
    assert _readings("'94", 1)[0][0] == "ninety four"
    assert _readings("’94", 1)[0][0] == "ninety four"
    assert _readings("'94-'95", 1)[0][0] == "ninety four to ninety five"


# ------------------------------------------------------------------ fugu review fixes


def test_grouped_numbers_count_their_digits():
    """C1: a number grouped by commas keys by its digits ("1,000-2,000" is ``dash:4+4``,
    not the sparse, mostly measure ``dash:other``). How it is then said is J's: the
    corpus writes such a range with a silent VERBATIM dash (``dash:4+4``
    ``cardinal+silent+cardinal`` 8, ``cardinal+to+cardinal`` 0)."""
    from frend.ranges import written_sub_key

    assert written_sub_key("range", "1,000", "-", "2,000") == "dash:4+4"
    assert written_sub_key("range", "1,00", "-", "2") == "dash:other"
    (found,) = __import__("reading_profile").reading_detectors("en_US")[-1].detect("1,000-2,000")
    assert found["sub_key"] == "dash:4+4"


def test_a_sparse_sub_key_reads_its_class_row():
    """F3: a sub-key seen fewer than RANGE_MIN_N times does not decide E alone."""
    from frend.ranges import RANGE_MIN_N, RangePriorTable, load_range_priors

    document = {
        "readings": {
            "dash:4+10": {"range": 4, "single_token": {}},
            "dash": {"range": 10, "single_token": {"TELEPHONE": 100}},
        },
        "kinds": {},
    }
    table = RangePriorTable(document)
    assert RANGE_MIN_N == 50
    assert table.emission("dash:4+10") == (10, 100)
    assert not table.emits("dash:4+10")
    shipped = load_range_priors("en_US")
    assert shipped.emission("dash:4+10") == shipped.emission("dash")


def test_a_spaced_chain_is_not_a_range():
    """F1: R3 counts a chain spaced alike ("1 - 2 - 3", a date "2008 - 09 - 30",
    "1 x 2 x 3"): no range span is formed."""
    from reading_profile import reading_detectors

    detector = reading_detectors("en_US")[-1]
    for text in ("1 - 2 - 3", "2008 - 09 - 30", "1 x 2 x 3", "412 - 555 - 1212"):
        assert detector.detect(text) == [], text
    assert len(detector.detect("5 - 10")) == 1


def test_range_triples_skip_chain_fragments():
    """F2: a corpus chain ("1 - 2 - 3") yields no range triple for the builders."""
    from google_tn_rows import range_triples

    chain = [("CARDINAL", "1", "one"), ("PLAIN", "-", "to"), ("CARDINAL", "2", "two"),
             ("PLAIN", "-", "to"), ("CARDINAL", "3", "three")]  # fmt: skip
    assert list(range_triples([chain])) == []
    assert len(list(range_triples([chain[:3]]))) == 1


def test_spacing_is_no_range_feature():
    """C2: the corpus's triples record no spacing, so no tree can learn it: family R has
    no ``r_spaced``."""
    from frend.context import range_features

    assert set(range_features("-", "5", "10")) == {"r_sep", "r_ldig", "r_rdig", "r_lead0"}


def test_combination_parameters_are_the_plans():
    """C3: stage 2 found no gain beyond noise, so the plan's values stand."""
    from decimal import Decimal

    from frend.context import RANGE_CONTEXT_THRESHOLD
    from frend.ranges import EMIT_RATIO, RANGE_SUB_KEY_STRENGTH

    assert (EMIT_RATIO, RANGE_SUB_KEY_STRENGTH, RANGE_CONTEXT_THRESHOLD) == (
        Decimal(1),
        Decimal(5),
        0.7,
    )
