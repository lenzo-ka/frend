"""Context: the context trees rank a span's readings by the running text around it, and a
range separator between numbers is offered the locale's range connector ("to")."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from icukit.detectors import detect  # noqa: E402

from frend import resolve_lattice  # noqa: E402
from frend.spoken_priors import normalize_spoken  # noqa: E402
from frend.verbalize import verbalize_lattice  # noqa: E402


def _verbalized(text: str):
    from reading_profile import reading_detectors

    detections = list(detect(text, reading_detectors("en_US")))
    return verbalize_lattice(resolve_lattice(detections, source_text=text))


def _said(text: str) -> str:
    """frend's first reading of running text, each unit said as its best alternative."""
    out = ""
    for unit in _verbalized(text).best_path.units:
        best = unit.best
        out += best.text if best.provenance == "surface:passthrough" else f" {best.text} "
    return normalize_spoken(out)


def _unit(text: str, surface: str):
    from reading_profile import reading_detectors

    detections = list(detect(text, reading_detectors("en_US")))
    lattice = resolve_lattice(detections, source_text=text)
    verbalized = verbalize_lattice(lattice)
    edges = {edge.id: edge for edge in lattice.edges}
    for unit in verbalized.best_path.units:
        edge = edges[unit.edge_id]
        if text[edge.start : edge.end] == surface:
            return unit
    raise AssertionError(f"no unit for {surface!r} in {text!r}")


# ------------------------------------------------------------------ the range connector


@pytest.mark.parametrize(
    ("text", "said"),
    [
        # A lone hyphen between numbers (frend main: "pages five ten of the book").
        ("pages 5 - 10 of the book", "pages five to ten of the book"),
        # A range written as one word (main: "pages five minus ten of the book").
        ("pages 5-10 of the book", "pages five to ten of the book"),
        # CLDR's own range separator, the en dash (main: "pages five ten of the book").
        ("pages 5–10 of the book", "pages five to ten of the book"),
        # Years: the right end read as the left (main: "... minus one thousand nine
        # hundred ninety five the club").
        ("from 1990-1995 the club", "from nineteen ninety to nineteen ninety five the club"),
        # A time range (main: "open ten thirty eleven forty five daily").
        ("open 10:30-11:45 daily", "open ten thirty to eleven forty five daily"),
    ],
)
def test_a_separator_between_numbers_reads_to_where_the_tree_says_so(text, said):
    """The separator's tree (trained on the corpus's lone hyphens between numbers) says
    "to" here, and it leads; the same tree reads a separator written inside a range."""
    assert _said(text) == said


def test_to_is_offered_beside_silence_and_minus():
    """The connector joins the readings frend already offers: silence and the symbol's
    names for a lone hyphen (the corpus token read alone, its sentence as context); a
    range written in the text is one span, offered "to" and the silent form."""
    (lone,) = _token_units("pages 5", "-", "10 of")
    assert {"", "to", "minus"} <= {a.text for a in lone.alternatives}
    inside = _unit("pages 5-10 of", "5-10").alternatives
    assert {"five to ten", "five ten"} <= {a.text for a in inside}
    assert all(a.provenance.startswith("range:") for a in inside)


def test_a_separator_not_between_numbers_offers_no_connector():
    assert "to" not in {a.text for a in _unit("well - known", "-").alternatives}
    assert "to" not in {a.text for a in _unit("the 5 - fold", "-").alternatives}


def _token_units(before: str, token: str, after: str):
    """``token``'s units read alone with its sentence as context (the evaluator's view)."""
    from reading_profile import reading_detectors

    from frend.context import TextContext

    head = f"{before} " if before else ""
    tail = f" {after}" if after else ""
    detections = list(detect(token, reading_detectors("en_US")))
    return verbalize_lattice(
        resolve_lattice(detections, source_text=token),
        context=TextContext(f"{head}{token}{tail}", len(head)),
    ).best_path.units


def _token_in_sentence(before: str, token: str, after: str):
    """``token`` read alone with its sentence as context, as the evaluator reads a
    corpus token: (its first reading, every reading any of its units offers)."""
    from reading_profile import reading_detectors

    from frend.context import TextContext

    head = f"{before} " if before else ""
    tail = f" {after}" if after else ""
    detections = list(detect(token, reading_detectors("en_US")))
    verbalized = verbalize_lattice(
        resolve_lattice(detections, source_text=token),
        context=TextContext(f"{head}{token}{tail}", len(head)),
    )
    first = ""
    offered = set()
    for unit in verbalized.best_path.units:
        best = unit.best
        first += best.text if best.provenance == "surface:passthrough" else f" {best.text} "
        offered |= {a.text for a in unit.alternatives}
    return normalize_spoken(first), offered


def _offers_to(offered) -> bool:
    return any(text == "to" or text.startswith("to ") for text in offered)


@pytest.mark.parametrize(
    ("before", "token", "after"),
    [
        # A phone number's local shape, three digits then four (e273c24: "five hundred
        # fifty five to one thousand two hundred twelve", the tree at 0.76).
        ("Call", "555-1212", "now"),
        ("Phone:", "555-1234", ""),
        # A chain of digit groups: an ISBN (e273c24: "to" at its first hyphen, p=1.0)
        # and a phone number with its area code.
        ("ISBN", "978-1-234-56789-7", ""),
        ("Call", "1-800-555-1212", "now"),
    ],
)
def test_an_identifiers_digit_groups_are_no_range(before, token, after):
    """A hyphen between an identifier's digit groups is no range separator: no "to" is
    offered, in running text or for the token read in its sentence."""
    first, offered = _token_in_sentence(before, token, after)
    assert not _offers_to(offered), (first, offered)
    assert " to " not in f" {first} "
    text = " ".join(part for part in (before, token, after) if part)
    assert " to " not in f" {_said(text)} "
    running = {a.text for unit in _verbalized(text).best_path.units for a in unit.alternatives}
    assert not _offers_to(running)


@pytest.mark.parametrize(
    ("before", "token", "after"),
    [
        # A signed number after a number, the sign spaced on one side only (e273c24:
        # "values ten to five and three", p=1.0; "pages five to ten").
        ("values 10", "-5", "and 3"),
        ("pages 5", "-10", ""),
    ],
)
def test_a_sign_spaced_on_one_side_is_no_range(before, token, after):
    """A range's separator is spaced alike on both sides ("5-10", "5 - 10"): a sign
    spaced before only writes a signed number, and is offered no "to"."""
    first, offered = _token_in_sentence(before, token, after)
    assert not _offers_to(offered), (first, offered)
    assert " to " not in f" {first} "
    text = f"{before} {token} {after}".strip()
    assert " to " not in f" {_said(text)} "


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("pages 5-10", "pages five to ten"),
        ("pages 5 - 10", "pages five to ten"),
    ],
)
def test_a_separator_spaced_alike_on_both_sides_still_reads_to(text, said):
    assert _said(text) == said


def test_digits_frend_does_not_read_are_no_range_ends():
    """Arabic-Indic digits pass through unread, so the hyphen between them is offered no
    "to" (e273c24: "٥ to ١ ٠", the tree at 0.88); for now only ASCII digits end a range."""
    first, offered = _token_in_sentence("pages", "٥-١٠", "")
    assert not _offers_to(offered), (first, offered)
    running = {
        a.text for unit in _verbalized("pages ٥-١٠").best_path.units for a in unit.alternatives
    }
    assert not _offers_to(running)


def test_a_locale_without_the_connector_offers_none(monkeypatch):
    """With the Russian table (no forms) the feature is off: no connector, no separator,
    no "to" anywhere; a pattern with words outside its ends is not said by a separator."""
    from frend import verbalize as verbalize_module
    from frend.context import connector_words
    from frend.locale_data import lexical_forms
    from frend.ranges import RangeDetector

    assert verbalize_module.range_connector("en_US").words == "to"
    assert verbalize_module.range_separators("en_US") == frozenset({"-", "–"})
    assert RangeDetector("ru_RU").detect("5-10") == []
    assert connector_words("от {0} до {1}") is None
    ru = dict(lexical_forms("ru_RU"))
    monkeypatch.setattr(verbalize_module, "_lexical_for", lambda locale: ru)
    assert verbalize_module.range_connector("en_US") is None
    assert verbalize_module.range_separators("en_US") == frozenset()
    spoken = {
        a.text
        for unit in _verbalized("pages 5 - 10 and 5-10").best_path.units
        for a in unit.alternatives
    }
    assert not {text for text in spoken if text == "to" or text.startswith("to ")}


# ------------------------------------------------------------------ the context trees


@pytest.mark.parametrize(
    ("text", "said"),
    [
        # A Roman numeral after a regnal name (main: "king louis fourteen of france").
        ("King Louis XIV of France", "king louis the fourteenth of france"),
        ("Henry VIII was king", "henry the eighth was king"),
        # A rating's ".0" (main: "it was rated four point o out of five").
        ("It was rated 4.0 out of five.", "it was rated four point zero out of five"),
        # "No" before a number (main: "he lived at no five main street").
        ("He lived at No 5 Main Street.", "he lived at number five main street"),
    ],
)
def test_the_context_tree_puts_the_reading_the_text_favors_first(text, said):
    assert _said(text) == said


def test_the_tree_overrides_at_seventy_percent():
    """ "UEFA" after "by" is said as a word at 0.73 (main spells it, "u e f a"): at or
    over the threshold, the tree's reading leads."""
    unit = _unit("It was played by UEFA in May.", "UEFA")
    assert unit.context.label == "measured:acronym-word"
    assert 0.7 <= unit.context.probability < 0.75
    assert unit.context.applied
    assert _said("It was played by UEFA in May.") == "it was played by uefa in may"


def test_below_the_threshold_frend_keeps_its_first_choice():
    """For "UNICEF" the tree leans to its letters at 0.60, under the threshold: frend's
    own first choice, the word, stays."""
    unit = _unit("the UNICEF team", "UNICEF")
    assert unit.context.label == "measured:acronym-spelled"
    assert 0.6 <= unit.context.probability < 0.7
    assert not unit.context.applied
    assert _said("the UNICEF team") == "the unicef team"


def test_the_threshold_is_seventy_percent():
    from frend.context import CONTEXT_THRESHOLD

    assert CONTEXT_THRESHOLD == 0.7


def test_reordering_changes_no_weight():
    """The tree moves a reading; every reading keeps its text, provenance and weight."""
    from frend.verbalize import verbalize_edge

    text = "King Louis XIV of France"
    from reading_profile import reading_detectors

    lattice = resolve_lattice(list(detect(text, reading_detectors("en_US"))), source_text=text)
    edge = next(e for e in lattice.edges if text[e.start : e.end] == "XIV" and e.kind == "reading")
    ranked = verbalize_edge(edge, source_text=text)
    own = verbalize_edge(edge, source_text=text, rerank_by_context=False)
    assert ranked.context.applied
    assert ranked.alternatives != own.alternatives
    assert sorted(ranked.alternatives, key=repr) == sorted(own.alternatives, key=repr)


def test_a_token_read_alone_takes_its_sentence_as_context():
    """The evaluator reads one corpus token at a time: with its sentence as context the
    lone hyphen reads "to"; alone it is silent, as before."""
    import evaluate_google_tn

    assert evaluate_google_tn._score_text("-", "to", "pages 5", "10 of the book")[0]
    assert evaluate_google_tn._score_text("-", "")[0]
    assert evaluate_google_tn._score_text("XIV", "the fourteenth", "King Louis", "of France")[0]


def test_features_read_the_text_by_offsets():
    """A span's features come from the text around it, not from any tokenization: far
    text changes nothing, a near word does; a window's cut edge reads as padding."""
    from functools import partial

    from frend.context import context_model
    from frend.context import features as bare

    model = context_model("en_US")
    features = partial(bare, frequent=model.frequent, curated=model.curated)
    near = "King Louis XIV of France"
    far = "x" * 200 + " " + near
    a = features(near, 11, 14, first="f", first_weight=None)
    b = features(far, 201 + 11, 201 + 14, first="f", first_weight=None)
    assert a == b
    c = features("Pope Leo XIV of Rome", 9, 12, first="f", first_weight=None)
    assert a["c_rexname-1"] == "1" and c["c_rexname-1"] == "1"
    assert a["c_rextitle"] == "1" and c["c_rextitle"] == "1"
    assert a["f_tpg-2"] == "other" and a["f_tpg+1"] == "w:of"
    assert features("Louis XIV", 6, 9, first="f", first_weight=None)["f_tpg-2"] == "<BOS>"
    cut = features("Louis XIV", 6, 9, first="f", first_weight=None, bos=False)
    assert cut["f_tpg-2"] == "<PAD>" and cut["w_wb-3"] != "<BOS>"


def test_a_locale_without_trees_keeps_frends_order(no_context_trees):
    """With no trees, frend's own order: the range table's for a range written in the
    text ("5 - 10" is one span, J's ``dash:1+2`` leader "to"), and the lone hyphen's
    silence first, "to" offered, for the corpus token read alone."""
    assert _said("It was rated 4.0 out of five.") == "it was rated four point o out of five"
    assert _said("pages 5 - 10 of the book") == "pages five to ten of the book"
    (lone,) = _token_units("pages 5", "-", "10 of the book")
    assert lone.best.text == ""
    assert "to" in {a.text for a in lone.alternatives}
