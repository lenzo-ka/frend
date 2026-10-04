"""Standalone symbols and letters of other scripts, read by CLDR's and ICU's names."""

from __future__ import annotations

import pytest
from icukit.detectors import detect

import frend
from frend import compose_choices, resolve_choices, resolve_lattice
from frend.align_graph import build_align_graph
from frend.context import TextContext
from frend.normalize import _reading_detectors
from frend.profiles import GOOGLE_TN
from frend.spoken_priors import normalize_spoken
from frend.symbols import SymbolDetector, SymbolRunValue
from frend.verbalize import verbalize_edge, verbalize_lattice


def _forms(text):
    (detection,) = SymbolDetector().detect(text)
    lattice = resolve_lattice([detection], source_text=text)
    edge = next(e for e in lattice.edges if e.kind == "reading")
    # The measured order, before any context tree (the symbol stands alone here).
    unit = verbalize_edge(edge, source_text=text, rerank_by_context=False)
    assert unit.unspoken == ()
    return [normalize_spoken(a.text) for a in unit.alternatives]


@pytest.mark.parametrize(
    ("text", "first"),
    [("&", "and"), ("#", "number"), (".", ""), (",", ""), ("-", ""), ("α", "alpha"), ("風", "")],
)
def test_symbol_reads_as_the_corpus_measures_it(text, first):
    """CLDR's names and silence are offered; the corpus picks "and" for "&", nothing for
    punctuation and for a character of a script it does not read, a Greek letter's name."""
    assert _forms(text)[0] == first


def test_every_cldr_name_and_silence_are_offered():
    forms = _forms("&")
    assert {"ampersand", "and", ""} <= set(forms)


@pytest.mark.parametrize("text", ["ᵋ", "々"])
def test_measured_property_classes_are_silent_first_with_name_fallback(text):
    (detection,) = SymbolDetector().detect(text)
    assert detection["type"] == "symbol:property"
    lattice = resolve_lattice([detection], source_text=text)
    edge = next(edge for edge in lattice.edges if edge.kind == "reading")
    unit = verbalize_edge(edge, source_text=text)

    assert unit.alternatives[0].text == ""
    assert unit.alternatives[0].provenance == "surface:silence"
    assert unit.alternatives[1].text
    assert unit.alternatives[1].provenance == "icu-name:property"


def test_property_rule_applies_only_to_whole_standalone_tokens():
    assert SymbolDetector().detect("aᵋb") == []
    assert SymbolDetector().detect("時々") == []
    assert [item["type"] for item in SymbolDetector().detect("ᵋᵋ")] == ["symbol:property"]


def test_property_silence_uses_the_normal_context_reranker(monkeypatch):
    import importlib

    verbalize_module = importlib.import_module("frend.verbalize")

    (detection,) = SymbolDetector().detect("ᵋ")
    lattice = resolve_lattice([detection], source_text="ᵋ")
    edge = next(edge for edge in lattice.edges if edge.kind == "reading")
    calls = []

    def reverse(alternatives, *_args, **_kwargs):
        calls.append(tuple(item.text for item in alternatives))
        return tuple(reversed(alternatives)), None

    monkeypatch.setattr(verbalize_module, "_rank_final", lambda alternatives, *_args: alternatives)
    monkeypatch.setattr(verbalize_module, "rerank", reverse)
    unit = verbalize_edge(
        edge,
        source_text="ᵋ",
        context=TextContext("ᵋ"),
    )

    assert calls == [("", "modifier letter small open e")]
    assert unit.alternatives[0].text == "modifier letter small open e"


def test_property_rule_preserves_words_like_main_and_silences_a_standalone_token():
    assert frend.normalize("時々", fold=None) == "時々"
    assert frend.normalize("aᵋb", fold=None) == " aᵋb "
    assert frend.normalize("ᵋ", fold=None).strip() == ""
    assert frend.normalize("ᵋᵋ", fold=None).strip() == ""


def test_mixed_modifier_class_is_report_only():
    from frend.symbols import silent_property_class

    assert silent_property_class("ʻ") is None
    assert silent_property_class("ー") is None


@pytest.mark.parametrize("text", ["R&D", "AT&T", "a-b", "x.y", "3.14", "αβ"])
def test_a_character_inside_a_word_is_left_alone(text):
    assert SymbolDetector().detect(text) == []


def _symbol_run_unit(text: str, *, threshold: int = 3):
    detections = SymbolDetector(run_threshold=threshold).detect(text)
    run = next(item for item in detections if item["type"] == "symbol:run")
    choices = resolve_choices(detections, source_text=text)
    graph = compose_choices(choices)
    edge = next(edge for edge in choices.edges if edge.detection == run)
    return run, next(unit for unit in graph.units if unit.edge_id == edge.id)


def test_four_repeated_symbols_are_one_described_unit_with_all_alternatives():
    detection, unit = _symbol_run_unit("****")
    assert (detection["start"], detection["end"]) == (0, 4)
    assert [alternative.text for alternative in unit.alternatives] == [
        "line of asterisk symbols",
        "asterisk asterisk asterisk asterisk",
        "",
    ]
    assert {alternative.group for alternative in unit.alternatives} == {"tts-sanity"}
    assert frend.normalize("****", fold=None).strip() == "line of asterisk symbols"


def test_three_symbols_stay_verbatim_and_the_threshold_is_configurable():
    assert not any(item["type"] == "symbol:run" for item in SymbolDetector().detect("***"))
    assert not any(
        item["type"] == "symbol:run"
        for item in SymbolDetector(run_threshold=4).detect("****")
    )
    assert SymbolDetector(run_threshold=2).detect("***")[0]["type"] == "symbol:run"
    assert frend.normalize("***", fold=None, symbol_run_threshold=2).strip() == (
        "line of asterisk symbols"
    )


def test_a_space_separated_run_is_one_unit():
    detection, unit = _symbol_run_unit("*       *       *       *")
    assert detection["text"] == "*       *       *       *"
    assert unit.best.text == "line of asterisk symbols"


@pytest.mark.parametrize(
    ("text", "description"),
    [("😀😃😄😁", "line of emoji"), ("×+÷=", "line of symbols")],
)
def test_mixed_runs_use_their_coarse_name(text, description):
    detection, unit = _symbol_run_unit(text)
    assert isinstance(detection["value"], SymbolRunValue)
    assert unit.best.text == description


@pytest.mark.parametrize(
    ("text", "description", "source"),
    [
        ("😀😀😀😀", "line of grinning face symbols", "icu-name:symbol"),
        ("❤️❤️❤️❤️", "line of red heart symbols", "cldr-symbol:red heart"),
        ("𝄞𝄞𝄞𝄞", "line of musical symbol g clef symbols", "icu-name:symbol"),
    ],
)
def test_repeated_pictographs_use_cldr_then_icu_names(text, description, source):
    _detection, unit = _symbol_run_unit(text)
    assert unit.best.text == description
    assert source in unit.best.provenance


def test_profile_callers_rank_the_run_without_dropping_alternatives(monkeypatch):
    import frend.verbalize as verbalize_module

    monkeypatch.setattr(verbalize_module, "_google_tn_britishisms", lambda **_kwargs: None)
    monkeypatch.setattr(verbalize_module, "_acronym_surface_priors", lambda **_kwargs: None)
    lattice = resolve_lattice(source_text="****", fold=None)
    default = verbalize_lattice(lattice)
    google = verbalize_lattice(lattice, profile=GOOGLE_TN)
    assert [item.text for item in default.best_path.units[0].alternatives] == [
        "line of asterisk symbols",
        "asterisk asterisk asterisk asterisk",
        "",
    ]
    assert [item.text for item in google.best_path.units[0].alternatives] == [
        "",
        "line of asterisk symbols",
        "asterisk asterisk asterisk asterisk",
    ]
    assert frend.normalize("****", fold=None).strip() == "line of asterisk symbols"
    assert frend.normalize("****", fold=None, profile=GOOGLE_TN).strip() == ""


def test_variation_selectors_are_attached_to_their_base_and_never_pass_through():
    heart = "\u2764\ufe0f"
    detections = SymbolDetector().detect(heart)
    assert [(item["text"], item["start"], item["end"]) for item in detections] == [
        (heart, 0, 2)
    ]
    lattice = resolve_lattice(detections, source_text=heart)
    assert len(lattice.best_path.edge_ids) == 1
    assert "\ufe0f" not in frend.normalize(heart, fold=None)
    assert frend.normalize("A\ufe0f", fold=None).strip() == "A"


@pytest.mark.parametrize(
    "text",
    [
        "--",
        "...",
        "***",
        "! ? ! ?",
        "!!!!",
        "....",
        "\u0589\u0589\u0589\u0589",
        ":-):-):-):-)",
        "*#=~",
        "**bold**",
        "https://example.org/----",
        "1--2",
    ],
)
def test_non_separator_sequences_do_not_become_symbol_runs(text):
    assert not any(item["type"] == "symbol:run" for item in SymbolDetector().detect(text))


@pytest.mark.parametrize(
    "text",
    ["`----`", "before `----` after", "```\n----\n```", "~~~\n====\n~~~", "    ----"],
)
def test_code_context_does_not_become_a_symbol_run(text):
    assert not any(item["type"] == "symbol:run" for item in SymbolDetector().detect(text))
    assert "line of" not in frend.normalize(text, fold=None)


def test_disallowed_mixed_punctuation_keeps_only_its_repeated_subrun():
    runs = [item for item in SymbolDetector().detect("****#") if item["type"] == "symbol:run"]
    assert [(item["text"], item["value"].symbols) for item in runs] == [
        ("****", ("*", "*", "*", "*"))
    ]


@pytest.mark.parametrize("text", ["-", "=", "_", "#", "~"])
def test_same_nonterminal_punctuation_can_form_a_run(text):
    run = next(item for item in SymbolDetector().detect(text * 4) if item["type"] == "symbol:run")
    assert run["value"].symbols == (text,) * 4


def test_icu_graphemes_are_single_units_and_zwj_runs_retain_components():
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466"
    (single,) = SymbolDetector().detect(family)
    assert (single["text"], single["start"], single["end"]) == (family, 0, len(family))

    detections = SymbolDetector().detect(family * 4)
    run = next(item for item in detections if item["type"] == "symbol:run")
    assert run["value"].symbols == (family,) * 4
    assert [item["text"] for item in detections if item["type"] != "symbol:run"] == [family] * 4

    heart = "\u2764\ufe0f"
    heart_run = next(
        item for item in SymbolDetector().detect(heart * 4) if item["type"] == "symbol:run"
    )
    assert heart_run["value"].symbols == (heart,) * 4
    keycap = "#\ufe0f\u20e3"
    (keycap_detection,) = SymbolDetector().detect(keycap)
    assert (keycap_detection["text"], keycap_detection["start"], keycap_detection["end"]) == (
        keycap,
        0,
        3,
    )


def _alignment_token_paths(alignment):
    plan = alignment.count_plan()

    def walk(index, tokens):
        item = alignment.items[plan.labels[index]]
        tokens = (*tokens, *item.tokens)
        if not plan.children[index]:
            yield tokens
        for child in plan.children[index]:
            yield from walk(child, tokens)

    return {tokens for root in plan.roots for tokens in walk(root, ())}


def test_run_is_additional_and_alignment_retains_partial_component_readings():
    text = "****"
    detections = list(detect(text, _reading_detectors("en_US")))
    choices = resolve_choices(detections, source_text=text)
    reading_types = [
        edge.detection["type"] for edge in choices.edges if edge.kind == "reading"
    ]
    assert reading_types.count("symbol:run") == 1
    assert reading_types.count("symbol:cldr") == 4

    paths = _alignment_token_paths(build_align_graph(compose_choices(choices)))
    assert ("line", "of", "asterisk", "symbols") in paths
    assert () in paths
    assert {sum(token == "asterisk" for token in path) for path in paths} >= {0, 1, 2, 3, 4}


def test_locale_without_symbol_run_phrase_keeps_names_and_silence():
    text = "****"
    detections = SymbolDetector("ru").detect(text)
    run = next(item for item in detections if item["type"] == "symbol:run")
    choices = resolve_choices([run], locale="ru", source_text=text)
    unit = compose_choices(choices).units[0]
    assert len(unit.alternatives) == 2
    assert unit.alternatives[0].text == " ".join(name for name, _source in run["value"].names)
    assert unit.alternatives[1].text == ""


def test_a_standalone_symbol_in_running_text_is_read():
    detections = SymbolDetector().detect("Tom & Jerry")
    assert [(d["type"], d["text"]) for d in detections] == [("symbol:cldr", "&")]


def test_the_locale_scripts_come_from_icu():
    """Likely subtags' script first, the exemplars' scripts, then Common and Inherited."""
    from frend.symbols import locale_scripts

    assert locale_scripts("en_US") == ("Latn", "Zinh", "Zyyy")
    assert locale_scripts("ru")[0] == "Cyrl"
    assert {"Hani", "Hira", "Kana"} <= set(locale_scripts("ja_JP"))


def test_a_letter_of_the_locale_script_is_not_read_by_name():
    """Cyrillic is another script for en_US and the locale's own for ru."""
    assert [d["type"] for d in SymbolDetector("en_US").detect("Ж")] == ["symbol:letter"]
    assert SymbolDetector("ru").detect("Ж") == []


def test_spaced_letters_of_another_script_are_one_span():
    text = "« Т Е С Т - а б »."
    runs = [d for d in SymbolDetector().detect(text) if d["type"] == "symbol:script-run"]
    assert [(d["text"], d["start"], d["end"]) for d in runs] == [
        ("Т Е С Т", 2, 9),
        ("а б", 12, 15),
    ]
    assert not any(d["type"] == "symbol:letter" for d in SymbolDetector().detect(text))


def test_a_word_is_no_run_and_does_not_join_its_neighbors():
    detections = SymbolDetector().detect("Москва и Петербург")
    assert [(d["type"], d["text"]) for d in detections] == [("symbol:letter", "и")]
    detections = SymbolDetector().detect("а б Москва в г")
    assert [(d["type"], d["text"]) for d in detections] == [
        ("symbol:script-run", "а б"),
        ("symbol:script-run", "в г"),
    ]


def _run_unit(text, locale="en_US"):
    (detection,) = SymbolDetector(locale).detect(text)
    lattice = resolve_lattice([detection], source_text=text)
    edge = next(e for e in lattice.edges if e.kind == "reading")
    return detection, verbalize_edge(edge, source_text=text, rerank_by_context=False)


def test_a_run_reads_by_transliteration_names_as_written_and_silence():
    detection, unit = _run_unit("Θ Α")
    assert detection["value"].transform == "Greek-Latin; Latin-ASCII"
    readings = {(a.text, a.provenance) for a in unit.alternatives}
    assert readings == {
        ("th a", "icu-transliteration:Greek-Latin; Latin-ASCII"),
        ("theta alpha", "icu-name:letter"),
        ("Θ Α", "surface:as-written"),
        ("", "surface:silence"),
    }
    # Greek is measured by name, so the names lead, as they do for a lone letter.
    assert unit.alternatives[0].text == "theta alpha"


def test_a_cyrillic_run_is_silent_first_as_its_lone_letters_are():
    _, unit = _run_unit("Т Е С Т")
    assert unit.alternatives[0].text == ""
    assert ("t e s t", "icu-transliteration:Cyrillic-Latin; Latin-ASCII") in {
        (a.text, a.provenance) for a in unit.alternatives
    }


@pytest.mark.parametrize(
    ("text", "spoken"),
    [("ж з", "z z"), ("風 雨", "feng yu"), ("Ё Щ", "e s")],
)
def test_a_latin_transliteration_holds_only_ascii_letters(text, spoken):
    (detection,) = SymbolDetector().detect(text)
    readings = {source: name for name, source in detection["value"].names}
    transliterated = next(v for k, v in readings.items() if k.startswith("icu-transliteration"))
    assert transliterated == spoken
    assert transliterated.isascii()


def test_the_transform_is_one_icu_lists_else_any():
    from frend.symbols import transform_id

    assert transform_id("Cyrl", "en_US") == "Cyrillic-Latin; Latin-ASCII"
    assert transform_id("Latn", "ru") == "Latin-Cyrillic"


def test_a_run_joins_only_letters_of_one_script():
    detections = SymbolDetector().detect("Т Α")
    assert [(d["type"], d["text"]) for d in detections] == [
        ("symbol:letter", "Т"),
        ("symbol:letter", "Α"),
    ]
    detections = SymbolDetector().detect("а б Θ Α")
    assert [(d["type"], d["text"]) for d in detections] == [
        ("symbol:script-run", "а б"),
        ("symbol:script-run", "Θ Α"),
    ]


@pytest.mark.parametrize(
    ("text", "letters"),
    [("Т Е С Т5", ["Т", "Е", "С"]), ("xА Б В", ["Б", "В"])],
)
def test_a_run_touching_a_word_keeps_its_letters_one_by_one(text, letters):
    detections = SymbolDetector().detect(text)
    assert [(d["type"], d["text"]) for d in detections] == [
        ("symbol:letter", letter) for letter in letters
    ]


@pytest.mark.parametrize("text", ["Т\nЕ", "Т  Е", "Т\tЕ"])
def test_a_run_joins_across_one_space_only(text):
    detections = SymbolDetector().detect(text)
    assert [d["type"] for d in detections] == ["symbol:letter", "symbol:letter"]


@pytest.mark.parametrize(
    ("text", "names", "spoken"),
    [
        ("А́ Б", "a be", "a b"),
        ("й́ к", "short i ka", "j k"),
        ("ά β", "alpha with tonos beta", "a b"),
    ],
)
def test_a_run_names_its_letters_without_their_marks(text, names, spoken):
    (detection,) = SymbolDetector().detect(text)
    readings = {source.split(":")[0]: name for name, source in detection["value"].names}
    assert readings["icu-name"] == names
    assert readings["icu-transliteration"] == spoken


def test_a_locale_naming_no_script_writes_its_language_s_scripts():
    """Serbian is written in Cyrillic and Latin: "sr" reads neither as another script."""
    from frend.symbols import locale_scripts

    assert {"Cyrl", "Latn"} <= set(locale_scripts("sr"))
    assert "Latn" not in locale_scripts("sr_Cyrl")
    assert SymbolDetector("sr").detect("A B") == []
    assert SymbolDetector("sr").detect("А Б") == []


def test_a_lone_latin_letter_reads_as_written_in_any_locale():
    assert SymbolDetector("ru").detect("A") == []
    assert SymbolDetector("sr_Cyrl").detect("A") == []


def test_a_transliteration_outside_the_locale_scripts_is_dropped():
    """ICU's Any-Hira leaves Greek as Greek; ja reads no Greek, so no such reading."""
    (detection,) = SymbolDetector("ja_JP").detect("α β")
    sources = [source for _, source in detection["value"].names]
    assert not any(source.startswith("icu-transliteration") for source in sources)


def test_spaced_letters_resolve_as_one_choice():
    """Ninety-six spaced letters were 2^96 covers; as one run they are a handful."""
    from icukit.detectors import detect

    text = " ".join("абвгдежзиклмнопрстуфхцчшщэюя" * 4)
    lattice = resolve_lattice(list(detect(text, [SymbolDetector()])), source_text=text)
    readings = [e for e in lattice.edges if e.kind == "reading"]
    assert len(readings) == 1 and readings[0].end - readings[0].start == len(text)
