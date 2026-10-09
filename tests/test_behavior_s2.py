"""Behavior schema S2 regression and characterization contract."""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from itertools import permutations
from pathlib import Path
from types import MappingProxyType

import icu
import pytest
from icukit import break_grapheme_spans

import frend
from frend import BehaviorLoadError, SpokenAlternative
from frend.align_export import to_fsg_text
from frend.align_graph import build_align_graph
from frend.behavior import resolve_behavior
from frend.lattice import ChoiceGraph, resolve_choices
from frend.profiles import CHAR_DETAIL, GroupSetting, validate_groups
from frend.symbols import SymbolDetector, symbol_names
from frend.verbalize import (
    _char_detail,
    _rank_final,
    _verbalize_edge,
    verbalize_edge,
    verbalize_lattice,
)

DATA = Path("tests/data/behaviors")
GOLDEN = Path("tests/data/normalize_golden.json")
SR = resolve_behavior([DATA / "user/screen-reader.json"]).kwargs["groups"]
SR_NUMBER = resolve_behavior([DATA / "user/screen-reader-number.json"]).kwargs["groups"]
INERT = {"char-detail": ["reading", "named", "spelled"]}

EXAMPLES = (
    ("E1", "Hello, world.", "en_US", " hello  comma   world  period "),
    (
        "E2",
        "Yes!!!!!",
        "en_US",
        " yes  exclamation mark  exclamation mark  exclamation mark  exclamation mark  "
        "exclamation mark ",
    ),
    (
        "E3",
        "See ****  here.",
        "en_US",
        " see   asterisk asterisk asterisk asterisk    here  period ",
    ),
    ("E6", ":-)", "en_US", " colon  hyphen-minus  close parenthesis "),
    (
        "E7",
        "See “this”.",
        "en_US",
        " see   left quotation mark  this  right quotation mark  period ",
    ),
    ("E8", "Oui, non.", "fr_FR", "Oui virgule  non point "),
    ("E10", "don't", "en_US", "don typewriter apostrophe t"),
    ("E11", "* * * *", "en_US", " asterisk asterisk asterisk asterisk "),
    ("E12", "$42 is 50%", "en_US", " dollar four two   is   five zero percent "),
    ("E14", ",", "en_US", " comma "),
    ("E16", "$43.50", "en_US", " dollar four three period five zero "),
    ("E17", "1,234.56", "en_US", " one comma two three four period five six "),
    ("E18", "3:15:30", "en_US", " three colon one five colon three zero "),
    ("E19", "user@x.co", "en_US", " user at-sign x period co "),
    (
        "E20",
        "https://x.co/a?b=1",
        "en_US",
        " https colon slash slash x period co slash a  question mark  b  equal  one ",
    ),
    ("E21", "2/14/67", "en_US", " two slash one four slash six seven "),
    (
        "E22",
        "2026-10-05",
        "en_US",
        " two zero two six hyphen-minus one zero hyphen-minus zero five ",
    ),
    (
        "E23",
        "14.02.1967",
        "en_US",
        " one four period zero two period one nine six seven ",
    ),
    ("E24", "user@x.co", "fr_FR", " user arobase x point co "),
)

NUMBER_EXAMPLES = (
    ("E12", "$42 is 50%", " dollar forty-two   is   fifty percent "),
    ("E16", "$43.50", " dollar forty-three period fifty "),
    ("E17", "1,234.56", " one comma two hundred thirty-four period fifty-six "),
    ("E18", "3:15:30", " three colon fifteen colon thirty "),
    ("E21", "2/14/67", " two slash fourteen slash sixty-seven "),
    (
        "E22",
        "2026-10-05",
        " two thousand twenty-six hyphen-minus ten hyphen-minus five ",
    ),
    (
        "E23",
        "14.02.1967",
        " fourteen period two period one thousand nine hundred sixty-seven ",
    ),
)


def _golden_rows():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["rows"]


def _verbalized(text, *, groups=SR, locale="en_US", fold="typographic", **kwargs):
    lattice = frend.resolve_lattice(source_text=text, locale=locale, fold=fold)
    result = verbalize_lattice(lattice, groups=groups, locale=locale, **kwargs)
    edges = {edge.id: edge for edge in lattice.edges}
    return lattice, result, edges


def _error(path):
    with pytest.raises(BehaviorLoadError) as caught:
        resolve_behavior([path])
    return caught.value


def _document(name, groups):
    return {
        "schema_version": 1,
        "kind": "behavior-schema",
        "name": name,
        "version": 1,
        "extends": [],
        "provenance": {"source": "frend tests"},
        "sections": {"frend": {"schema_version": 1, "groups": groups}},
    }


def test_c1_char_detail_permutations_and_refusals_are_exact():
    roles = ("named", "reading", "spelled")
    for order in permutations(roles):
        setting = dict(validate_groups({"char-detail": order}))["char-detail"]
        assert setting.order == order

    with pytest.raises(ValueError) as caught:
        validate_groups({"char-detial": roles})
    assert str(caught.value) == (
        "unknown verbalization group 'char-detial'; known groups: 'tts-sanity', 'char-detail'"
    )
    with pytest.raises(ValueError) as caught:
        validate_groups({"char-detail": ["named", "reading"]})
    assert str(caught.value) == (
        "group 'char-detail' order must list each of 'named', 'reading', 'spelled' exactly once"
    )
    assert str(_error(DATA / "invalid/char-detail-partial.json")) == (
        "INVALID_VALUE: tests/data/behaviors/invalid/char-detail-partial.json: "
        "sections.frend.groups: group 'char-detail' order must list each of 'named', "
        "'reading', 'spelled' exactly once"
    )


@pytest.mark.parametrize(("_label", "text", "locale", "expected"), EXAMPLES)
def test_c2_authored_examples_are_exact(_label, text, locale, expected):
    assert frend.normalize(text, locale=locale, groups=SR) == expected


def test_c2_custom_orders_and_offsets_are_exact():
    both = {
        "char-detail": ["named", "reading", "spelled"],
        "tts-sanity": ["described", "named", "silent"],
    }
    spelled = {"char-detail": ["spelled", "named", "reading"]}
    assert frend.normalize("See ****  here.", groups=both) == (
        " see   line of asterisk    here  period "
    )
    assert frend.normalize("NASA 42.", groups=spelled) == " n a s a   four two  period "
    comma = frend.normalize(",", groups=SR, offsets=True)
    nasa = frend.normalize("NASA", groups=spelled, offsets=True)
    assert comma.units[0].provenance == "cldr-symbol:comma"
    assert nasa.units[0].provenance == "icu:letter-name"


def test_c3_group_precedence_places_complete_slots_and_keeps_roles():
    _, first, _ = _verbalized("****", fold=None)
    later_tts = {
        "char-detail": ["named", "reading", "spelled"],
        "tts-sanity": ["described", "named", "silent"],
    }
    _, second, _ = _verbalized("****", groups=later_tts, fold=None)
    assert [(item.group, item.role) for item in first.best_path.units[0].alternatives] == [
        ("char-detail", "named"),
        ("tts-sanity", "described"),
        ("tts-sanity", "named"),
        ("tts-sanity", "silent"),
    ]
    assert [(item.group, item.role) for item in second.best_path.units[0].alternatives] == [
        ("tts-sanity", "described"),
        ("tts-sanity", "named"),
        ("tts-sanity", "silent"),
        ("char-detail", "named"),
    ]


def test_c4_composition_keeps_every_base_alternative_by_identity(monkeypatch):
    import frend.verbalize as verbalize_module

    original = verbalize_module._composed

    def checked(base, detail, settings):
        result = original(base, detail, settings)
        assert all(any(item is candidate for candidate in result) for item in base)
        return result

    monkeypatch.setattr(verbalize_module, "_composed", checked)
    for row in _golden_rows():
        lattice = frend.resolve_lattice(source_text=row["text"])
        verbalize_lattice(lattice, groups=SR)


def test_c5_reading_first_is_byte_identical_for_every_golden_row():
    for row in _golden_rows():
        text = row["text"]
        assert frend.normalize(text, groups=INERT) == frend.normalize(text)
        assert repr(frend.normalize(text, groups=INERT, offsets=True)) == repr(
            frend.normalize(text, offsets=True)
        )


def test_c6_registry_order_merge_and_key_setters(tmp_path):
    resolved = resolve_behavior(
        ["screen-reader", "screen-reader-symbols", "screen-reader"],
        search=[DATA / "user"],
    )
    assert list(resolved.kwargs["groups"]) == ["tts-sanity", "char-detail"]
    assert resolved.setters["groups.tts-sanity"] == "screen-reader-symbols"
    assert resolved.setters["groups.char-detail"] == "screen-reader"

    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    groups = {
        "char-detail": {"order": ["named", "reading", "spelled"]},
        "tts-sanity": {"order": ["described", "named", "silent"]},
    }
    (left / "both.json").write_text(json.dumps(_document("both", groups)))
    reversed_groups = dict(reversed(groups.items()))
    (right / "both.json").write_text(json.dumps(_document("both", reversed_groups)))
    one = resolve_behavior([left / "both.json"])
    two = resolve_behavior([right / "both.json"])
    assert one.digest == two.digest
    assert (
        list(one.kwargs["groups"])
        == list(two.kwargs["groups"])
        == [
            "tts-sanity",
            "char-detail",
        ]
    )


def test_c7_roles_are_hidden_from_repr_and_survive_final_ranking():
    _, result, _ = _verbalized("****", fold=None)
    alternatives = result.best_path.units[0].alternatives
    assert [item.role for item in alternatives] == ["named", "described", "named", "silent"]
    assert "role=" not in repr(alternatives)
    original = SpokenAlternative("four", "icu-rbnf:%spellout-numbering", group="g", role="r")
    ranked = _rank_final((original,), "cardinal", None, "en_US")
    assert ranked[0].group == "g"
    assert ranked[0].role == "r"


def test_c8_lattice_size_is_fixed_and_detail_adds_at_most_two_forms():
    for _label, text, locale, _expected in EXAMPLES:
        lattice = frend.resolve_lattice(source_text=text, locale=locale)
        base = verbalize_lattice(lattice, locale=locale)
        composed = verbalize_lattice(lattice, locale=locale, groups=SR)
        assert len(base.paths) == len(composed.paths)
        assert len(base.best_path.edge_ids) == len(composed.best_path.edge_ids)
        for before, after in zip(base.best_path.units, composed.best_path.units, strict=True):
            assert len(after.alternatives) - len(before.alternatives) <= 2


def test_c9_locale_detail_uses_locale_authoritative_sources():
    cases = [
        (",", "fr_FR", "virgule", "cldr-symbol:virgule"),
        (",", "haw_US", "comma", "icu-name:symbol"),
        ("😀", "fr_FR", "grinning face", "icu-name:symbol"),
    ]
    for text, locale, expected, provenance in cases:
        _, result, _ = _verbalized(text, locale=locale)
        named = next(
            item
            for item in result.best_path.units[0].alternatives
            if item.group == CHAR_DETAIL and item.role == "named"
        )
        assert (named.text, named.provenance) == (expected, provenance)

    _, letters, _ = _verbalized("A", locale="fr_FR")
    assert not any(
        item.group == CHAR_DETAIL and item.role == "spelled"
        for item in letters.best_path.units[0].alternatives
    )
    spelled_first = {"char-detail": ["spelled", "named", "reading"]}
    assert frend.normalize("4", locale="fr_FR", groups=spelled_first) == " quatre "
    assert frend.normalize("user@x.co", locale="fr_FR", groups=SR) == (" user arobase x point co ")


def _grapheme_kind(grapheme):
    if icu.Char.isdigit(grapheme[0]):
        return "digit"
    if icu.Char.isalpha(grapheme[0]):
        return "letter"
    return "symbol"


def test_c10_every_symbol_in_a_unit_is_named_in_source_order():
    samples = [(row["text"], "en_US") for row in _golden_rows()]
    samples.extend((text, locale) for _label, text, locale, _expected in EXAMPLES)
    for text, locale in samples:
        lattice, result, edges = _verbalized(text, locale=locale)
        raw = lattice.raw_source_text or lattice.source_text or ""
        for edge_id, unit in zip(result.best_path.edge_ids, result.best_path.units, strict=True):
            edge = edges[edge_id]
            graphemes = [
                span["text"]
                for span in break_grapheme_spans(raw[edge.start : edge.end], "root")
                if not span["text"].isspace()
            ]
            symbols = [item for item in graphemes if _grapheme_kind(item) == "symbol"]
            named = [
                item
                for item in unit.alternatives
                if item.group == CHAR_DETAIL and item.role == "named"
            ]
            assert bool(named) is bool(symbols)
            if not symbols:
                continue
            offset = 0
            for symbol in symbols:
                names = symbol_names(symbol, locale)[:1]
                expected = names[0][0] if names else symbol
                found = named[0].text.find(expected, offset)
                assert found >= offset
                offset = found + len(expected)

    for text in ("!!!!!", ":-)", "`code`", "https://x.co/a?b=1"):
        lattice = frend.resolve_lattice(source_text=text)
        selected = {edge.id: edge for edge in lattice.edges}
        assert all(
            not (
                selected[edge_id].detection is not None
                and selected[edge_id].detection.get("type") == "symbol:run"
            )
            for edge_id in lattice.best_path.edge_ids
        )


def test_c11_fallback_and_raw_surface_paths_are_composed():
    lattice, result, edges = _verbalized("a–b", fold=None)
    units = {
        (edges[edge_id].start, edges[edge_id].end): unit
        for edge_id, unit in zip(result.best_path.edge_ids, result.best_path.units, strict=True)
    }
    assert [(item.text, item.role) for item in units[0, 1].alternatives] == [
        ("a", None),
        ("a", "spelled"),
    ]
    assert units[1, 2].best.text == "en dash"

    folded = frend.resolve_lattice(source_text="“")
    edge = next(edge for edge in folded.edges if edge.id in folded.best_path.edge_ids)
    direct = verbalize_edge(edge, source_text=folded.source_text, groups=SR)
    raw = verbalize_edge(
        edge,
        source_text=folded.source_text,
        raw_source_text="“",
        groups=SR,
    )
    assert direct.best.text == "quotation mark"
    assert raw.best.text == "left quotation mark"


def test_c12b_curated_reading_stays_inside_the_reading_slot():
    lattice = frend.resolve_lattice(source_text="****", fold=None)
    supplement = SpokenAlternative("line of asterisk", "curated:t", Decimal(1))
    result = verbalize_lattice(
        lattice,
        supplements={("symbol:run", "****"): (supplement,)},
        groups=SR,
    )
    alternatives = result.best_path.units[0].alternatives
    assert [(item.text, item.group, item.role) for item in alternatives] == [
        ("asterisk asterisk asterisk asterisk", "char-detail", "named"),
        ("line of asterisk", None, None),
        ("asterisk asterisk asterisk asterisk", "tts-sanity", "named"),
        ("", "tts-sanity", "silent"),
    ]
    assert alternatives[1] is supplement


def test_c13_export_carries_both_groups_but_not_internal_roles():
    text = "* * * *"
    lattice = resolve_choices(SymbolDetector().detect(text), source_text=text)
    units = tuple(
        _verbalize_edge(
            edge,
            source_text=text,
            raw_source_text=text,
            groups=SR,
        )
        for edge in lattice.edges
    )
    _text, manifest = to_fsg_text(
        build_align_graph(ChoiceGraph(lattice, units)),
        "character-detail-groups",
    )
    alternatives = [
        alternative
        for transition in manifest.transitions
        for alternative in transition.alternatives
    ]
    assert all("role" not in alternative for alternative in alternatives)
    grouped = {
        (alternative["text"], alternative.get("group"))
        for alternative in alternatives
        if alternative.get("group") is not None
    }
    assert ("asterisk asterisk asterisk asterisk", "char-detail") in grouped
    assert {group for _text, group in grouped} == {"char-detail", "tts-sanity"}


def test_c14_latency_is_a_measurement_without_a_runtime_mode():
    lattice = frend.resolve_lattice(source_text="Hello, world.")
    before = (len(lattice.edges), len(lattice.paths))
    verbalize_lattice(lattice, groups=SR)
    assert (len(lattice.edges), len(lattice.paths)) == before


def test_c15_google_tn_respelling_rewrites_only_passthrough(monkeypatch):
    import frend.verbalize as verbalize_module

    lattice = frend.resolve_lattice([], source_text="colour")
    captured = {}
    original = verbalize_module._respell_passthrough_words

    def spy(units, edge_ids, edges, source_text, table):
        captured["before"] = units
        result = original(units, edge_ids, edges, source_text, table)
        captured["after"] = result
        return result

    monkeypatch.setattr(verbalize_module, "_acronym_surface_priors", lambda **_kwargs: None)
    monkeypatch.setattr(verbalize_module, "_google_tn_britishisms", lambda **_kwargs: object())
    monkeypatch.setattr(
        verbalize_module,
        "respell_from_table",
        lambda word, _table: "color" if word == "colour" else word,
    )
    monkeypatch.setattr(verbalize_module, "_respell_passthrough_words", spy)
    result = verbalize_lattice(lattice, profile="google-tn", groups=SR)
    assert [unit.best.text for unit in result.best_path.units] == ["c", "o", "l", "o", "", "r"]

    for before, after in zip(captured["before"], captured["after"], strict=True):
        preserved = [
            item for item in before.alternatives if item.provenance != "surface:passthrough"
        ]
        assert all(any(item is candidate for candidate in after.alternatives) for item in preserved)
        rewritten = next(
            item for item in after.alternatives if item.provenance == "surface:passthrough"
        )
        old = next(item for item in before.alternatives if item.provenance == "surface:passthrough")
        assert rewritten is not old


def test_c16_digit_stretches_keep_zeros_and_spelling_is_locale_bounded():
    assert "zero five" in frend.normalize("2026-10-05", groups=SR)
    assert "zero two" in frend.normalize("14.02.1967", groups=SR)
    _, english, _ = _verbalized("user@x.co")
    english_spelled = [
        item.text
        for item in english.best_path.units[0].alternatives
        if item.group == CHAR_DETAIL and item.role == "spelled"
    ]
    assert english_spelled == ["u s e r at-sign x period c o"]
    _, french, _ = _verbalized("user@x.co", locale="fr_FR")
    assert not any(
        item.group == CHAR_DETAIL and item.role == "spelled"
        for unit in french.best_path.units
        for item in unit.alternatives
    )


def test_c17_group_objects_options_and_refusals_are_exact(tmp_path):
    order = ("named", "reading", "spelled")
    sequence = dict(validate_groups({"char-detail": order}))["char-detail"]
    object_default = dict(validate_groups({"char-detail": {"order": order}}))["char-detail"]
    numbered = dict(validate_groups({"char-detail": {"order": order, "digits": "number"}}))[
        "char-detail"
    ]
    assert isinstance(numbered, GroupSetting)
    assert sequence.options == object_default.options == {"digits": "each"}
    assert numbered.options == {"digits": "number"}
    assert dict(validate_groups({"char-detail": numbered}))["char-detail"] is not numbered
    source_options = {"digits": "number"}
    source = GroupSetting(order, source_options)
    frozen = dict(validate_groups({"char-detail": source}))["char-detail"]
    source_options["digits"] = "each"
    assert frozen.options == {"digits": "number"}
    with pytest.raises(TypeError):
        frozen.options["digits"] = "each"  # type: ignore[index]
    full_order = GroupSetting(order, {})
    assert dict(validate_groups({"char-detail": full_order}))["char-detail"].options == {
        "digits": "each"
    }
    with pytest.raises(ValueError) as caught:
        validate_groups({"char-detail": GroupSetting(("named",), {})})
    assert str(caught.value) == (
        "group 'char-detail' order must list each of 'named', 'reading', 'spelled' exactly once"
    )
    with pytest.raises(ValueError) as caught:
        validate_groups({"char-detail": GroupSetting(order, {"digits": "digit"})})
    assert str(caught.value) == (
        "group 'char-detail' digits must be one of 'each', 'number'; got 'digit'"
    )
    assert (
        dict(validate_groups({"tts-sanity": {"order": ["described", "named", "silent"]}}))[
            "tts-sanity"
        ].options
        == {}
    )

    invalid = [
        (
            {"char-detail": {"order": order, "digits": "digit"}},
            "group 'char-detail' digits must be one of 'each', 'number'; got 'digit'",
        ),
        (
            {"char-detail": {"order": order, "digitz": "each"}},
            "group 'char-detail' has unknown option 'digitz'; known options: 'digits'",
        ),
        (
            {"tts-sanity": {"order": ["described", "named", "silent"], "digits": "each"}},
            "group 'tts-sanity' has unknown option 'digits'; known options: none",
        ),
        (
            {"char-detail": {"digits": "each"}},
            "group 'char-detail' must be an order sequence or an object with 'order'; "
            "got dict without 'order'",
        ),
    ]
    for groups, message in invalid:
        with pytest.raises(ValueError) as caught:
            validate_groups(groups)
        assert str(caught.value) == message

    path = tmp_path / "bad-group-object.json"
    path.write_text(
        json.dumps(
            _document(
                "bad-group-object",
                {"tts-sanity": {"order": ["described", "named", "silent"], "digits": "each"}},
            )
        )
    )
    assert str(_error(path)).endswith(
        "sections.frend.groups: group 'tts-sanity' has unknown option 'digits'; known options: none"
    )


def test_c18_behavior_options_compose_key_by_key_and_round_trip(tmp_path):
    search = [DATA / "user"]
    first = resolve_behavior(["screen-reader-number", "screen-reader"], search=search)
    second = resolve_behavior(["screen-reader", "screen-reader-number"], search=search)
    for resolved in (first, second):
        setting = resolved.kwargs["groups"]["char-detail"]
        assert isinstance(setting, MappingProxyType)
        assert setting == {
            "order": ("named", "reading", "spelled"),
            "digits": "number",
        }
    assert first.setters["groups.char-detail"] == "screen-reader"
    assert first.setters["groups.char-detail.digits"] == "screen-reader-number"
    assert second.setters["groups.char-detail"] == "screen-reader-number"
    assert second.setters["groups.char-detail.digits"] == "screen-reader-number"

    reset = tmp_path / "screen-reader-each.json"
    reset.write_text(
        json.dumps(
            _document(
                "screen-reader-each",
                {
                    "char-detail": {
                        "order": ["named", "reading", "spelled"],
                        "digits": "each",
                    }
                },
            )
        )
    )
    restored = resolve_behavior(
        ["screen-reader", "screen-reader-number", reset],
        search=search,
    )
    assert restored.kwargs["groups"]["char-detail"] == (
        "named",
        "reading",
        "spelled",
    )
    once = dict(validate_groups(second.kwargs["groups"]))
    twice = dict(validate_groups(once))
    assert twice["char-detail"] is not once["char-detail"]
    assert twice["char-detail"] == once["char-detail"]


@pytest.mark.parametrize(("_label", "text", "expected"), NUMBER_EXAMPLES)
def test_c19_number_option_outputs_are_exact(_label, text, expected):
    assert frend.normalize(text, groups=SR_NUMBER) == expected


def test_c19_number_option_leaves_letters_and_spelled_digits_unchanged():
    for text in ("user@x.co", "https://x.co/a?b=1"):
        assert frend.normalize(text, groups=SR_NUMBER) == frend.normalize(text, groups=SR)
    expected = {
        "$43.50": "dollar four three period five zero",
        "2026-10-05": "two zero two six hyphen-minus one zero hyphen-minus zero five",
    }
    for text, spelled_text in expected.items():
        _, number, _ = _verbalized(text, groups=SR_NUMBER)
        number_spelled = [
            item.text
            for unit in number.best_path.units
            for item in unit.alternatives
            if item.group == CHAR_DETAIL and item.role == "spelled"
        ]
        assert number_spelled == [spelled_text]


def test_c19_number_option_uses_only_exact_digit_run_lengths():
    at_limit = _char_detail("$" + "9" * 15, "en_US", ("named",), "number")[0]
    assert at_limit.text == (
        "dollar nine hundred ninety-nine trillion nine hundred ninety-nine billion "
        "nine hundred ninety-nine million nine hundred ninety-nine thousand "
        "nine hundred ninety-nine"
    )

    past_limit = _char_detail("$" + "9" * 16, "en_US", ("named",), "number")[0]
    assert past_limit.text == "dollar " + " ".join(["nine"] * 16)

    very_long = _char_detail("$" + "1" * 5000, "en_US", ("named",), "number")[0]
    assert very_long.text == "dollar " + " ".join(["one"] * 5000)


def test_c20_inert_order_is_identical_under_both_digit_options():
    groups = [
        {"char-detail": {"order": ["reading", "named", "spelled"], "digits": "each"}},
        {"char-detail": {"order": ["reading", "named", "spelled"], "digits": "number"}},
    ]
    for _label, text, locale, _expected in EXAMPLES:
        plain = frend.normalize(text, locale=locale)
        offsets = repr(frend.normalize(text, locale=locale, offsets=True))
        for setting in groups:
            assert frend.normalize(text, locale=locale, groups=setting) == plain
            assert (
                repr(frend.normalize(text, locale=locale, groups=setting, offsets=True)) == offsets
            )


def test_c21_marks_scripts_and_nfc_named_stretches():
    for text in ("$é", "$e\u0301"):
        detail = _char_detail(text, "en_US", ("named", "reading", "spelled"), "each")
        assert (detail[0].text, detail[0].provenance) == (
            "dollar é",
            "cldr-symbol:dollar+surface:passthrough",
        )
    template = frend.resolve_lattice([], source_text="$").best_path
    lattice = frend.resolve_lattice([], source_text="$")
    edge = next(item for item in lattice.edges if item.id in template.edge_ids)
    for text in ("$é", "$e\u0301"):
        unit = verbalize_edge(replace(edge, end=len(text)), source_text=text, groups=SR)
        assert unit.best.text == "dollar é"
        assert unit.best.provenance == "cldr-symbol:dollar+surface:passthrough"
    arabic = verbalize_edge(replace(edge, end=3), source_text="$١٢", groups=SR)
    assert arabic.best.text == "dollar one two"
    # Main's code-point passthrough edges (lattice.py) split decomposed é/١٢; fix is out of lane.
    assert frend.normalize("$é", groups=SR) == " dollar é"
    assert frend.normalize("$e\u0301", groups=SR) == " dollar e combining acute accent "
    assert frend.normalize("$١٢", groups=SR) == " dollar ١٢"

    _, cafe, _ = _verbalized("cafe\u0301")
    assert any(
        item.text == "c a f é" and item.group == CHAR_DETAIL and item.role == "spelled"
        for item in cafe.best_path.units[0].alternatives
    )
    _, mark, _ = _verbalized("\u0301")
    assert mark.best_path.units[0].best.text == "combining acute accent"
    assert mark.best_path.units[0].best.provenance == "icu-name:symbol"


def test_c22_character_detail_provenance_lists_only_sources_used():
    expected = {
        ",": "cldr-symbol:comma",
        "$43.50": (
            "cldr-symbol:dollar+icu-rbnf:%spellout-numbering+"
            "icu-rbnf-fallback:en+cldr-symbol:period"
        ),
        "user@x.co": "surface:passthrough+cldr-symbol:at-sign+cldr-symbol:period",
    }
    for text, provenance in expected.items():
        _, result, _ = _verbalized(text)
        named = next(
            item
            for unit in result.best_path.units
            for item in unit.alternatives
            if item.group == CHAR_DETAIL and item.role == "named"
        )
        assert named.provenance == provenance


def test_r1_s1_symbol_group_behavior_is_unchanged():
    groups = {"tts-sanity": ["named", "silent", "described"]}
    _, result, _ = _verbalized("****", groups=groups, fold=None)
    assert [item.text for item in result.best_path.units[0].alternatives] == [
        "asterisk asterisk asterisk asterisk",
        "",
        "line of asterisk",
    ]


def test_r2_retired_unknown_group_message_is_replaced_by_c1():
    with pytest.raises(ValueError) as caught:
        validate_groups({"tts-santy": ["described", "named", "silent"]})
    assert str(caught.value).endswith("known groups: 'tts-sanity', 'char-detail'")


def test_r3_s1_weighted_deduplication_is_unchanged():
    lattice = frend.resolve_lattice(source_text="****", fold=None)
    supplement = SpokenAlternative("line of asterisk", "curated:t", Decimal(1))
    result = verbalize_lattice(
        lattice,
        supplements={("symbol:run", "****"): (supplement,)},
        groups={"tts-sanity": ["named", "silent", "described"]},
    )
    alternatives = result.best_path.units[0].alternatives
    assert alternatives[0] is supplement
    assert [item.text for item in alternatives].count("line of asterisk") == 1


def test_r4_empty_group_mapping_is_no_schema_byte_identical():
    for row in _golden_rows():
        text = row["text"]
        assert frend.normalize(text, groups={}) == frend.normalize(text)
        assert repr(frend.normalize(text, groups={}, offsets=True)) == repr(
            frend.normalize(text, offsets=True)
        )


def test_r5_google_tn_no_group_respelling_still_replaces_the_single_form(monkeypatch):
    import frend.verbalize as verbalize_module

    lattice = frend.resolve_lattice([], source_text="colour")
    monkeypatch.setattr(verbalize_module, "_acronym_surface_priors", lambda **_kwargs: None)
    monkeypatch.setattr(verbalize_module, "_google_tn_britishisms", lambda **_kwargs: object())
    monkeypatch.setattr(
        verbalize_module,
        "respell_from_table",
        lambda word, _table: "color" if word == "colour" else word,
    )
    result = verbalize_lattice(lattice, profile="google-tn")
    assert "".join(unit.best.text for unit in result.best_path.units) == "color"


def test_r6_s1_kwargs_and_sequence_group_shape_are_unchanged():
    resolved = resolve_behavior(["google-tn"])
    assert resolved.kwargs == {
        "profile": "google-tn",
        "groups": {"tts-sanity": ("silent", "described", "named")},
    }
    setting = dict(validate_groups({"tts-sanity": ["silent", "described", "named"]}))["tts-sanity"]
    assert setting.order == ("silent", "described", "named")
