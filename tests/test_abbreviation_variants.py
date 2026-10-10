"""Abbreviations as running text writes them ("Mr", "st", "Vol", "E.G."), read and ranked
by how the corpus says them (``frend.abbreviation_variants``, ``abbreviation_priors.json``)."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest
from icukit.detectors import detect

from frend import compose_choices, resolve_choices, resolve_lattice
from frend.align_graph import build_align_graph
from frend.spoken_priors import SUB_KEY_PRIOR_STRENGTH, normalize_spoken
from frend.verbalize import verbalize_lattice

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))


def _detectors():
    from reading_profile import reading_detectors

    return reading_detectors("en_US")


def _units(text: str):
    """The best path's reading units of ``text`` through the evaluator's profile."""
    lattice = resolve_lattice(list(detect(text, _detectors())), source_text=text)
    return [
        unit
        for unit in verbalize_lattice(lattice).best_path.units
        if unit.best.provenance != "surface:passthrough"
    ]


def _first(text: str) -> str:
    """frend's first reading of ``text``, joined as the evaluator joins it."""
    lattice = resolve_lattice(list(detect(text, _detectors())), source_text=text)
    out = ""
    for unit in verbalize_lattice(lattice).best_path.units:
        passthrough = unit.best.provenance == "surface:passthrough"
        out += unit.best.text if passthrough else f" {unit.best.text} "
    return normalize_spoken(out)


def _alternatives(text: str) -> list[str]:
    return [normalize_spoken(item.text) for unit in _units(text) for item in unit.alternatives]


def _types(text: str) -> set[str]:
    return {str(detection["type"]) for detection in detect(text, _detectors())}


def _key_share(key: str, category: str) -> Decimal:
    from frend.abbreviation_variants import abbreviation_priors

    rows = abbreviation_priors()[key]
    total = sum(sum(counts.values()) for counts in rows.values())
    return Decimal(sum(counts.get(category, 0) for counts in rows.values())) / total


def test_period_less_title_reads_mister():
    """ "Mr" is title case, which the corpus never writes for mr, so it ranks by the key,
    where mister leads."""
    assert _first("Mr McVeigh") == "mister mcveigh"


def test_a_key_never_written_in_title_case_ranks_by_its_key():
    """A case the corpus writes has its own row ("No": number 947 of 11,847 title rows);
    one it never writes ("Mr") reads the key's share."""
    from frend.abbreviation_variants import abbreviation_priors, abbreviation_weights

    table = abbreviation_priors()
    assert "title" not in table["mr"] and "title" in table["no"]
    assert abbreviation_weights("Mr", ["Mister", "Mr"]) == (_key_share("mr", "mister"), None)
    title = table["no"]["title"]
    blended = (title["number"] + SUB_KEY_PRIOR_STRENGTH * _key_share("no", "number")) / (
        sum(title.values()) + SUB_KEY_PRIOR_STRENGTH
    )
    number = abbreviation_weights("No", ["Number"])[0]
    assert number == blended
    assert number != _key_share("no", "number")
    assert abbreviation_weights("Mr.", ["Mister"]) == abbreviation_weights("Mr", ["Mister"])


@pytest.mark.parametrize(
    "counts",
    [
        {"as-written": -1, "spelled": 2},
        {"as-written": True, "spelled": 2},
        {"as-written": 1.5, "spelled": 2},
        {"as-written": 0, "spelled": 0},
    ],
)
def test_malformed_abbreviation_prior_counts_raise(monkeypatch, counts):
    from frend import abbreviation_variants

    monkeypatch.setattr(
        abbreviation_variants,
        "abbreviation_priors",
        lambda **_kwargs: {"mr": {"lower": counts}},
    )
    with pytest.raises(ValueError):
        abbreviation_variants.abbreviation_weights("mr", ["mr"])


def test_valid_abbreviation_prior_counts_produce_bounded_weights(monkeypatch):
    from frend import abbreviation_variants

    monkeypatch.setattr(
        abbreviation_variants,
        "abbreviation_priors",
        lambda **_kwargs: {
            "mr": {
                "lower": {"as-written": 1, "spelled": 2},
                "upper": {"as-written": 2, "spelled": 1},
            }
        },
    )
    weights = abbreviation_variants.abbreviation_weights("mr", ["mr", "m r"])
    assert weights is not None
    assert all(weight is not None and Decimal(0) <= weight <= Decimal(1) for weight in weights)


def test_corpus_surface_st_reads_saint_first():
    assert _first("st Paul") == "saint paul"
    assert "street" in _alternatives("st Paul")


def test_title_case_vol_reads_volume():
    assert _first("Vol 3") == "volume three"


@pytest.mark.parametrize(("written", "said"), [("ltd", "limited"), ("vs", "versus")])
def test_lowercase_ltd_and_vs(written, said):
    assert _first(written) == said


def test_mrs_keeps_as_written_first_and_offers_missus():
    """The corpus says "mrs" as written, so it leads; "missus" is kept for a speaker who
    says it."""
    assert _first("Mrs Hall") == "mrs hall"
    assert "missus" in _alternatives("Mrs Hall")


@pytest.mark.parametrize("written", ["in", "or", "me", "ok", "hi", "IN", "OR"])
def test_state_codes_and_acronyms_are_not_folded(written):
    """Capitals-only lexicon surfaces ("IN", "OK") are not variant sources."""
    assert "abbreviation:variant" not in _types(written)


@pytest.mark.parametrize("written", ["p", "c", "v"])
def test_single_letter_folds_make_no_variant(written):
    """ "p.", "c.", "v." fold to one letter, which the corpus mostly says as the letter."""
    assert "abbreviation:variant" not in _types(written)


@pytest.mark.parametrize(
    ("written", "said"),
    [("LT", "l t"), ("DR", "doctor"), ("MR", "mister"), ("MR Smith", "mister smith")],
)
def test_a_capitals_run_reads_as_its_upper_row_says(written, said):
    """An all-capitals form is a letter run, not a variant; where the corpus measures the
    key in capitals, that row ranks its readings ("MR": mister 562 of 611, "DR": doctor
    436 of 800, "LT": spelled 155 of 155)."""
    assert "abbreviation:variant" not in _types(written)
    assert _first(written) == said


@pytest.mark.parametrize(
    ("uppercase", "title_case"),
    [("DR. SMITH.", "Dr. Smith."), ("MR. JONES.", "Mr. Jones.")],
)
def test_a_winning_capitals_expansion_consumes_its_abbreviation_period(uppercase, title_case):
    """A measured capitals expansion occupies its abbreviation period before a word;
    the following sentence period remains separate, as it does for the title-case source.
    """
    from frend import normalize

    assert normalize(uppercase) == normalize(title_case)


@pytest.mark.parametrize("written", ["PA. ROAD", "ZZ. TOP"])
def test_a_nonexpanding_capitals_run_leaves_its_period(written):
    """A state code and an unknown capitals run do not claim the following period."""
    from frend import normalize

    assert "." in normalize(written)


@pytest.mark.parametrize("written", ["SAT. MORNING", "MON. MORNING"])
def test_an_as_written_capitals_run_leaves_its_period(written):
    """A borrowed expansion does not claim the period when the uppercase corpus row
    ranks the run as written first, even if that expansion outranks the spelling.
    """
    from frend import normalize

    assert "." in normalize(written)


@pytest.mark.parametrize(
    ("written", "expansions"),
    [("MR", ["mister"]), ("DR", ["doctor", "drive"]), ("LT", ["lieutenant"])],
)
def test_a_capitals_run_also_offers_its_lexicon_expansions(written, expansions):
    """A run that is a lexicon abbreviation in capitals is offered its expansions beside
    its letters."""
    alternatives = _alternatives(written)
    assert " ".join(written.lower()) in alternatives
    for said in expansions:
        assert said in alternatives


def test_a_capitals_run_takes_its_upper_row_weights():
    """ "MR": mister and the letters take the upper row's shares (blended toward the key
    as every case row is); the run read as a word, which the corpus never says for mr,
    follows with no weight, its acronym weight dropped."""
    from frend.abbreviation_variants import abbreviation_weights

    (unit,) = _units("MR")
    weights = {normalize_spoken(item.text): item.weight for item in unit.alternatives}
    assert (weights["mister"], weights["m r"]) == abbreviation_weights("MR", ["Mister", "m r"])
    assert weights["mister"] > weights["m r"]
    assert weights["mr"] is None
    assert normalize_spoken(unit.alternatives[-1].text) == "mr"


def test_a_capitals_run_with_no_upper_row_keeps_the_letter_readers_order():
    """ "PKWY" is the capitals of "Pkwy." but the corpus has no capitals row for pkwy, so
    the letter reader's readings lead and the expansion follows, as before."""
    from frend.abbreviation_variants import measured_case

    assert measured_case("MR") and measured_case("LT")
    assert not measured_case("PKWY")
    alternatives = _alternatives("PKWY")
    assert alternatives[-1] == "parkway"
    assert alternatives[0] in ("p k w y", "pkwy")


def test_a_capitals_run_that_is_its_own_lexicon_surface_borrows_nothing():
    """ "PA" is the lexicon's own state code, so it borrows nothing from "Pa."."""
    from frend.abbreviation_variants import upper_variant_expansions

    assert upper_variant_expansions("PA") == ()
    assert [item.text for item in upper_variant_expansions("MR")] == ["Mister"]


@pytest.mark.parametrize(
    ("written", "said"),
    [("sun", "sun"), ("sat", "sat"), ("No man", "no man"), ("no", "no"), ("miss", "miss")]
    + [("mass", "mass")],
)
def test_words_stay_words_first(written, said):
    """A variant that is also a word reads as the word where the corpus says the word."""
    assert _first(written) == said


def test_no_before_a_digit_stays_as_written_until_context(no_context_trees):
    """ "No 5" reads as the corpus says "No" at large; "number five" is kept for context
    (``test_context.py``: the context tree reads "number five" before a digit)."""
    assert _first("No 5") == "no five"
    assert "number" in _alternatives("No 5")


@pytest.mark.parametrize(
    ("written", "said"),
    [("Ch 5", "c h five"), ("Ind", "i n d"), ("Rt", "r t"), ("Lt", "l t"), ("ft", "f t")],
)
def test_a_key_the_corpus_spells_reads_spelled(written, said):
    """A variant whose key the corpus only spells ("ch" 1,644 of 1,644, "lt" 2,040, "ind",
    "rt", "ft") reads its letters first, ranked by the table like its other readings;
    the token as written and the expansions follow."""
    assert _first(written) == said


def test_a_measured_spelled_reading_is_offered_and_weighted():
    """ "Mr" also offers "m r" (the corpus spells mr 135 of 8,756), weighted by its share,
    after mister; a key the corpus never spells ("blvd") offers no letters."""
    from frend.abbreviation_variants import abbreviation_weights, measures_spelled

    (unit,) = _units("Mr")
    weights = {normalize_spoken(item.text): item.weight for item in unit.alternatives}
    assert weights["m r"] == abbreviation_weights("Mr", ["m r"])[0]
    assert weights["mister"] > weights["m r"]
    assert measures_spelled("Mr") and not measures_spelled("Blvd")
    assert "b l v d" not in _alternatives("Blvd")


@pytest.mark.parametrize("written", ["no", "No", "Vol"])
def test_letters_follow_the_tokens_own_case_row(written):
    """Letters are offered only where the token's own case row measures spelled: "no" is
    spelled 38 times, all in capitals, and never in its lower or title rows; "Vol" is
    spelled once, in capitals. A case with no row reads the key ("Mr": "m r")."""
    from frend.abbreviation_variants import measures_spelled

    letters = " ".join(written.lower())
    assert not measures_spelled(written)
    assert letters not in _alternatives(written)
    assert measures_spelled("Mr") and "m r" in _alternatives("Mr")


def test_exact_st_period_is_ranked_by_measure():
    """An exact lexicon surface ("St.") is ranked by the same table: title case with a
    period reads the st key, saint far ahead of street."""
    (unit,) = _units("St.")
    weights = {normalize_spoken(item.text): item.weight for item in unit.alternatives}
    assert weights["saint"] is not None and weights["street"] is not None
    assert weights["saint"] > weights["street"]
    assert unit.alternatives[0].text == "Saint"


def test_a_locale_without_a_lexicon_reads_no_variants():
    from frend.abbreviation_variants import AbbreviationVariantDetector

    assert AbbreviationVariantDetector("ru_RU").detect("Mr McVeigh on st Paul") == []
    assert AbbreviationVariantDetector("en_US").detect("Mr McVeigh on st Paul")


def test_variant_readings_are_unit_weight_in_the_align_graph():
    """Saint and Street are spoken forms of one reading: their branches carry no factor
    (kal, 2026-09-14: shares never weight the alignment graph)."""
    from frend.abbreviation_variants import AbbreviationVariantDetector

    text = "st Paul"
    choices = compose_choices(
        resolve_choices(detect(text, [AbbreviationVariantDetector("en_US")]), source_text=text)
    )
    graph = build_align_graph(choices)
    words = {item.tokens[0]: item for item in graph.items.values() if item.role == "word"}
    assert {"saint", "street"} <= set(words)
    for said in ("saint", "street"):
        assert not words[said].weight.scored
        assert words[said].weight.log_weight == 0.0


@pytest.mark.parametrize(
    ("written", "first", "also"),
    [
        ("e.g.", "e g", "for example"),
        ("E.G.", "e g", "for example"),
        ("i.e.", "i e", "that is"),
        ("j.r.r.", "j r r", None),
    ],
)
def test_a_dotted_chain_is_spelled_first_in_any_case(written, first, also):
    """The corpus spells every dotted chain it writes ("e.g." e g 2,432 of 2,437 lower rows,
    "E.G." e g 81 of 83); a chain in another case than the lexicon's borrows its
    expansions after the letters."""
    assert _first(written) == first
    (unit,) = _units(written)
    assert unit.best.provenance != "surface:unsupported"
    if also is not None:
        assert also in _alternatives(written)
