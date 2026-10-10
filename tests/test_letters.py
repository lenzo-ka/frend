"""Runs of capitals and initials, spelled or said as measured; Roman numerals left to icukit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from icukit.abbreviation_recognize import AbbreviationDetector
from icukit.detectors import detect
from icukit.recognize import FlexibleNumberDetector

from frend import normalize, resolve_lattice
from frend.letters import LettersDetector, cv_pattern, is_roman, is_spelled_token, numeral_share
from frend.verbalize import verbalize_lattice

_DETECTORS = (FlexibleNumberDetector("en_US"), AbbreviationDetector("en_US"), LettersDetector())


def _read(text: str) -> list[list[str]]:
    lattice = resolve_lattice(list(detect(text, _DETECTORS)), source_text=text)
    return [
        [alternative.text for alternative in unit.alternatives]
        for unit in verbalize_lattice(lattice).best_path.units
        if unit.best.provenance != "surface:passthrough"
        and not any("spelled-token" in item.provenance for item in unit.alternatives)
        and unit.best.provenance != "surface:word"
    ]


def _spans(text: str) -> list[tuple[str, str]]:
    return [
        (d["type"], d["text"])
        for d in LettersDetector().detect(text)
        if d["type"] != "letters:token"
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the ATM", [("letters:run", "ATM")]),
        ("UFOs and AFI's", [("letters:run", "UFOs"), ("letters:run", "AFI's")]),
        ("Jane S. Smith", [("letters:initial", "S.")]),
        (
            "J.R.R. Tolkien",
            [("letters:initial", "J."), ("letters:initial", "R."), ("letters:initial", "R.")],
        ),
        ("USA.", [("letters:run", "USA")]),
    ],
)
def test_detects_runs_initials_and_suffixes(text, expected):
    assert _spans(text) == expected


@pytest.mark.parametrize("apostrophe", ["'", "’"])
def test_uppercase_acronym_possessive_suffix_matches_lowercase(apostrophe):
    uppercase = f"FBI{apostrophe}S"
    lowercase = f"FBI{apostrophe}s"

    assert _spans(uppercase) == [("letters:run", uppercase)]
    assert normalize(uppercase, fold=None) == normalize(lowercase, fold=None) == " f b i's "


@pytest.mark.parametrize(
    ("uppercase", "lowercase", "expected"), [("JR'S", "JR's", " jr's "), ("KY'S", "KY's", " ky's ")]
)
def test_uppercase_possessive_uses_lowercase_suffix_dictionary_row(uppercase, lowercase, expected):
    assert normalize(uppercase, fold=None) == normalize(lowercase, fold=None) == expected


@pytest.mark.parametrize("stem", ["IT", "WHAT", "THAT", "SHE"])
@pytest.mark.parametrize("apostrophe", ["'", "’"])
def test_uppercase_contractions_are_not_acronym_possessives(stem, apostrophe):
    text = f"{stem}{apostrophe}S"

    assert _spans(text) == []
    assert normalize(text, fold=None) == text


@pytest.mark.parametrize("text", ["A&I", "I saw it", "McDonald", "ATMs2", "U.S.A", "GRA-B", "II"])
def test_leaves_mixed_single_and_numeral_runs(text):
    assert _spans(text) == []


@pytest.mark.parametrize("text", ["pdf", "pc", "fMRI", "Meic"])
def test_short_locale_script_tokens_offer_spelling(text):
    found = [d for d in LettersDetector().detect(text) if d["type"] == "letters:token"]
    assert [(d["start"], d["end"], d["text"]) for d in found] == [(0, len(text), text)]
    assert is_spelled_token(text)


@pytest.mark.parametrize("text", ["a", "toolong", "NASA", "AΒ"])
def test_short_token_rule_is_bounded(text):
    assert not is_spelled_token(text)


@pytest.mark.parametrize(("text", "locale"), [("мир", "ru_RU"), ("hola", "es_ES")])
def test_spell_or_say_requires_locale_names_and_measured_data(text, locale):
    detections = LettersDetector(locale).detect(text)
    lattice = resolve_lattice(detections, locale=locale, source_text=text)
    units = verbalize_lattice(lattice).best_path.units

    assert detections == []
    assert all(unit.best.provenance == "surface:passthrough" for unit in units)
    assert all(unit.best.provenance != "surface:unsupported" for unit in units)


def test_unattested_rule_say_token_still_offers_both_readings():
    assert is_spelled_token("word")
    found = [d for d in LettersDetector().detect("word") if d["type"] == "letters:token"]
    lattice = resolve_lattice(found, source_text="word")
    unit = verbalize_lattice(lattice).best_path.units[0]
    assert [item.text for item in unit.alternatives] == ["word", "w o r d"]


def test_consonant_only_rule_spells_first():
    found = [d for d in LettersDetector().detect("xyz") if d["type"] == "letters:token"]
    lattice = resolve_lattice(found, source_text="xyz")
    unit = verbalize_lattice(lattice).best_path.units[0]
    assert [item.text for item in unit.alternatives] == ["x y z", "xyz"]


def test_spellout_dictionary_precedes_the_vowel_rule(monkeypatch):
    from frend import verbalize

    monkeypatch.setattr(
        verbalize,
        "spellout_dictionary_entry",
        lambda token, locale, **_kwargs: {
            "counts": {"say": 99, "spell": 1},
            "decision": "say",
            "spell_share": 0.01,
        },
    )
    lattice = resolve_lattice(list(detect("BBC", _DETECTORS)), source_text="BBC")
    unit = verbalize_lattice(lattice).best_path.units[0]
    assert [item.text for item in unit.alternatives[:2]] == ["bbc", "b b c"]


def test_unattested_acronym_keeps_the_measured_fallback(monkeypatch):
    from frend import verbalize

    monkeypatch.setattr(verbalize, "spellout_dictionary_entry", lambda *_args, **_kwargs: None)
    consonants = resolve_lattice(list(detect("BBC", _DETECTORS)), source_text="BBC")
    vowel = resolve_lattice(list(detect("FBI", _DETECTORS)), source_text="FBI")
    assert verbalize_lattice(consonants).best_path.units[0].best.text == "b b c"
    assert verbalize_lattice(vowel).best_path.units[0].best.text == "f b i"


def test_source_label_without_counts_abstains(monkeypatch):
    from frend import verbalize

    monkeypatch.setattr(
        verbalize,
        "spellout_dictionary_entry",
        lambda token, locale, **_kwargs: {
            "decision": "say",
            "label": "wikipedia:say",
            "source": "wikimedia/enwiki-lists-of-acronyms",
        },
    )
    lattice = resolve_lattice(list(detect("BBC", _DETECTORS)), source_text="BBC")
    assert verbalize_lattice(lattice).best_path.units[0].best.text == "b b c"


def test_dictionary_case_variant_lookup_is_opt_in(monkeypatch):
    from frend import letters

    monkeypatch.setattr(
        letters,
        "_spellout_dictionary",
        lambda _locale: (
            {"FBI": ("spell", 0, 10), "NASA": ("say", 10, 0)},
            {"fbi": "FBI"},
            "google/tn-en_with_types",
        ),
    )
    assert letters.spellout_dictionary_entry("NASA")["decision"] == "say"
    assert letters.spellout_dictionary_entry("Fbi") is None
    assert letters.spellout_dictionary_entry("Fbi", case_variant_lookup=True)["decision"] == "spell"


@pytest.fixture
def load_spellout_table(monkeypatch):
    from frend import letters, locale_data

    measured_table = locale_data.measured_table

    def load(tokens, casefold=()):
        letters._spellout_dictionary.cache_clear()
        monkeypatch.setattr(
            locale_data,
            "measured_table",
            lambda name, locale: (
                {
                    "tokens": tokens,
                    "casefold": casefold,
                    "provenance": {"source": "probe"},
                }
                if name == "spellout_dictionary"
                else measured_table(name, locale)
            ),
        )
        return letters._spellout_dictionary("en_US")

    yield load
    letters._spellout_dictionary.cache_clear()


def test_spellout_dictionary_refuses_duplicate_exact_surfaces(load_spellout_table):
    with pytest.raises(ValueError, match="duplicate spell-out dictionary surface.*ABC"):
        load_spellout_table(
            [["ABC", "say", 9, 1], ["ABC", "spell", 1, 9]],
        )


def test_spellout_dictionary_refuses_duplicate_casefold_aliases(load_spellout_table):
    with pytest.raises(ValueError, match="duplicate spell-out dictionary casefold alias.*abc"):
        load_spellout_table(
            [["ABC", "spell", 0, 5], ["Abc", "spell", 0, 6]],
            [["abc", "ABC"], ["abc", "Abc"]],
        )


def test_spellout_dictionary_retains_unique_builder_rows(load_spellout_table):
    from frend.letters import spelled_token_rule

    tokens = [["ABC", "spell", 0, 5], ["scrypt", "say", 13, 0]]
    assert all(decision != spelled_token_rule(surface) for surface, decision, *_ in tokens)
    assert load_spellout_table(tokens, [["abc", "ABC"]]) == (
        {"ABC": ("spell", 0, 5), "scrypt": ("say", 13, 0)},
        {"abc": "ABC"},
        "probe",
    )


@pytest.mark.parametrize(
    "row",
    [
        ("unknown", 1, 1),
        ("say", -1, 2),
        ("spell", True, 2),
        ("spell", 0, 0),
    ],
)
def test_spellout_dictionary_refuses_invalid_rows(monkeypatch, row):
    from frend import letters

    monkeypatch.setattr(
        letters,
        "_spellout_dictionary",
        lambda _locale: ({"ABC": row}, {}, "probe"),
    )
    with pytest.raises(ValueError, match="spell-out dictionary"):
        letters.spellout_dictionary_entry("ABC")


def test_spellout_dictionary_retains_a_valid_measured_share(monkeypatch):
    from frend import letters

    monkeypatch.setattr(
        letters,
        "_spellout_dictionary",
        lambda _locale: ({"ABC": ("say", 3, 1)}, {}, "probe"),
    )
    assert letters.spellout_dictionary_entry("ABC") == {
        "counts": {"say": 3, "spell": 1},
        "decision": "say",
        "source": "probe",
        "spell_share": 0.25,
    }


def test_ordinary_word_is_not_changed_to_spelling_by_the_dictionary():
    for word in ("word", "zijn", "échec"):
        found = [d for d in LettersDetector().detect(word) if d["type"] == "letters:token"]
        lattice = resolve_lattice(found, source_text=word)
        assert verbalize_lattice(lattice).best_path.units[0].best.text == word


def test_lowercase_context_words_stay_words_with_case_variants_enabled():
    from frend import normalize

    assert " ".join(normalize("it is", case_variant_lookup=True).split()) == "it is"
    assert " ".join(normalize("or else", case_variant_lookup=True).split()) == "or else"


def test_dictionary_precedes_the_vowel_rule_for_a_lowercase_token(monkeypatch):
    from frend import verbalize

    monkeypatch.setattr(
        verbalize,
        "spellout_dictionary_entry",
        lambda token, locale, **_kwargs: {
            "counts": {"say": 1, "spell": 9},
            "decision": "spell",
            "source": "google/tn-en_with_types",
            "spell_share": 0.9,
        },
    )
    lattice = resolve_lattice(list(detect("pdf", (LettersDetector(),))), source_text="pdf")
    unit = verbalize_lattice(lattice).best_path.units[0]
    assert [item.text for item in unit.alternatives] == ["p d f", "pdf"]


def test_a_consonant_run_spells_first():
    assert _read("at the GWR")[0][0] == "g w r"


def test_a_plural_rides_on_the_last_letter():
    assert _read("UFOs")[0] == ["u f o's", "ufos"]


@pytest.mark.parametrize("apostrophe", ["'", "’"])
def test_a_plural_possessive_keeps_the_plural_acronym_readings(apostrophe):
    text = f"UFOs{apostrophe}"
    assert _spans(text) == [("letters:run", text)]
    assert _read(text)[0] == _read("UFOs")[0]


def test_a_word_shaped_run_is_said_first():
    # "cvc" is said more than spelled in the corpus; "GUS" reads as written.
    assert cv_pattern("GUS") == "cvc"
    assert _read("GUS")[0][0] == "gus"


def test_an_initial_is_its_letter_not_an_abbreviation():
    assert _read("Jane S. Smith") == [["s"]]


@pytest.mark.parametrize(
    ("suffix", "spoken"),
    [("", " s "), ("'s", " s's "), ("’s", " s's ")],
)
def test_a_single_dotted_initial_keeps_its_possessive_suffix(suffix, spoken):
    text = f"S.{suffix}"

    assert _spans(text) == [("letters:initial", text)]
    assert normalize(text, fold=None) == spoken


def test_roman_numerals_follow_the_corpus_per_surface():
    assert is_roman("II") and is_roman("CD") and not is_roman("ATM")
    assert numeral_share("II") > 0.9 > 0.1 > numeral_share("CD")
    assert _read("World War II")[0][0] == "two"
    assert _read("a CD")[0][0] == "c d"


def test_a_lexicon_acronym_keeps_its_own_measure():
    assert _read("NASA")[0][0] == "nasa"
    assert _read("FBI")[0][0] == "f b i"


def _profile_table(path, surfaces, *, minimum_support=1):
    path.write_text(
        json.dumps(
            {
                "locale": "en",
                "profile": "google-tn",
                "provenance": {
                    "source_shards": [
                        {
                            "relative_path": f"output-{index:05d}-of-00100",
                            "sha256": "0" * 64,
                        }
                        for index in range(90)
                    ]
                },
                "schema_version": 1,
                "selection": {"minimum_support": minimum_support, "parent_strength": 1},
                "surfaces": surfaces,
            }
        ),
        encoding="utf-8",
    )
    path.with_name("britishisms.json").write_text(
        json.dumps(
            {
                "locale": "en",
                "pairs": {},
                "profile": "google-tn",
                "provenance": {
                    "source_shards": [
                        {
                            "relative_path": f"output-{index:05d}-of-00100",
                            "sha256": "0" * 64,
                        }
                        for index in range(90)
                    ]
                },
                "rules": {},
                "schema_version": 1,
                "selection": {
                    "admitted_classes": [],
                    "case_folding": {"enabled": False},
                    "class_minimum_support": {},
                    "rule_rate_threshold": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_profile_off_is_the_main_output_byte_for_byte(monkeypatch):
    """The default cannot even consult profile data, and its baseline bytes stay fixed."""
    from frend import verbalize

    monkeypatch.setattr(
        verbalize,
        "_acronym_surface_priors",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("profile table consulted")),
    )
    assert repr((_read("AIDS"), _read("CE"), _read("PAR's"))).encode() == (
        b"([['a i d s', 'aids']], [['c e', 'ce']], [[\"par's\", \"p a r's\"]])"
    )


def test_google_tn_profile_uses_the_external_surface_table(tmp_path, monkeypatch):
    from frend import verbalize

    monkeypatch.setattr(
        verbalize,
        "spellout_dictionary_entry",
        lambda *_args, **_kwargs: {
            "counts": {"say": 0, "spell": 10},
            "decision": "spell",
            "source": "google/tn-en_with_types",
            "spell_share": 1.0,
        },
    )
    table = tmp_path / "acronym_surfaces.json"
    _profile_table(table, {"GWR": {"bare": {"word": 5}}})
    monkeypatch.setenv("FREND_GOOGLE_TN_PROFILE_PATH", str(table))
    verbalize._acronym_surface_priors_for.cache_clear()
    lattice = resolve_lattice(list(detect("GWR", _DETECTORS)), source_text="GWR")
    assert verbalize_lattice(lattice).best_path.units[0].best.text == "g w r"
    unit = verbalize_lattice(lattice, profile="google-tn").best_path.units[0]
    assert unit.best.text == "gwr"


def test_google_tn_profile_requires_its_external_table(tmp_path, monkeypatch):
    from frend import verbalize

    missing = tmp_path / "missing.json"
    monkeypatch.setenv("FREND_GOOGLE_TN_PROFILE_PATH", str(missing))
    verbalize._acronym_surface_priors_for.cache_clear()
    lattice = resolve_lattice(list(detect("XYZ", _DETECTORS)), source_text="XYZ")
    with pytest.raises(FileNotFoundError, match="profile data is missing.*--profile-out"):
        verbalize_lattice(lattice, profile="google-tn")


def test_google_tn_profile_reloads_a_replaced_table(tmp_path, monkeypatch):
    from frend import verbalize

    table = tmp_path / "acronym_surfaces.json"
    replacement = tmp_path / "replacement.json"
    _profile_table(table, {"GWR": {"bare": {"word": 5}}})
    monkeypatch.setenv("FREND_GOOGLE_TN_PROFILE_PATH", str(table))
    verbalize._acronym_surface_priors_for.cache_clear()
    assert verbalize._acronym_surface_priors()[0]["GWR"]["bare"] == {"word": 5}

    _profile_table(replacement, {"GWR": {"bare": {"spelled": 5}}})
    replacement.replace(table)
    assert verbalize._acronym_surface_priors()[0]["GWR"]["bare"] == {"spelled": 5}


def test_google_tn_profile_detects_deletion_after_loading(tmp_path, monkeypatch):
    from frend import verbalize

    table = tmp_path / "acronym_surfaces.json"
    _profile_table(table, {"GWR": {"bare": {"word": 5}}})
    monkeypatch.setenv("FREND_GOOGLE_TN_PROFILE_PATH", str(table))
    verbalize._acronym_surface_priors_for.cache_clear()
    verbalize._acronym_surface_priors()
    table.unlink()

    with pytest.raises(FileNotFoundError, match="profile data is missing"):
        verbalize._acronym_surface_priors()


def test_surface_prior_abstains_below_its_selected_support(monkeypatch):
    """A sparse contrary row must not perturb the established shape/CV fallback."""
    from frend import verbalize

    fallback = {"*": {"spelled": 0, "word": 10}}
    monkeypatch.setattr(verbalize, "spellout_dictionary_entry", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verbalize, "_acronym_priors", lambda **_kwargs: fallback)
    monkeypatch.setattr(
        verbalize,
        "_acronym_surface_priors",
        lambda **_kwargs: ({"XYZ": {"bare": {"spelled": 4}}}, 5, 0),
    )
    forms = verbalize._with_acronym_readings("XYZ", (), "en_US", profile="google-tn")
    assert max(forms, key=lambda item: item.weight).text == "xyz"

    monkeypatch.setattr(
        verbalize,
        "_acronym_surface_priors",
        lambda **_kwargs: ({"XYZ": {"bare": {"spelled": 5}}}, 5, 0),
    )
    forms = verbalize._with_acronym_readings("XYZ", (), "en_US", profile="google-tn")
    assert max(forms, key=lambda item: item.weight).text == "x y z"


def test_acronym_surface_prior_keeps_suffix_subkeys_separate(monkeypatch):
    """Bare, plural and possessive evidence each reaches only its own reading."""
    from frend import verbalize
    from frend.letters import LettersValue

    fallback = {"*": {"spelled": 0, "word": 10}}
    monkeypatch.setattr(verbalize, "spellout_dictionary_entry", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verbalize, "_acronym_priors", lambda **_kwargs: fallback)
    surfaces = {
        "ABC": {
            "bare": {"word": 5},
            "plural": {"spelled": 5},
            "possessive": {"word": 5},
        }
    }
    monkeypatch.setattr(verbalize, "_acronym_surface_priors", lambda **_kwargs: (surfaces, 5, 0))
    bare = verbalize._spoken_letters(LettersValue("ABC", "ABC", ""), "en_US", "google-tn")
    plural = verbalize._spoken_letters(LettersValue("ABCs", "ABC", "s"), "en_US", "google-tn")
    possessive = verbalize._spoken_letters(LettersValue("ABC's", "ABC", "'s"), "en_US", "google-tn")

    def best(forms):
        return max(forms, key=lambda item: item.weight).provenance

    assert best(bare) == "measured:acronym-word"
    assert best(plural) == "measured:acronym-spelled"
    assert best(possessive) == "measured:acronym-word"


def test_dotted_acronym_never_uses_the_bare_surface_row(monkeypatch):
    from frend import verbalize

    monkeypatch.setattr(
        verbalize, "_acronym_priors", lambda **_kwargs: {"*": {"spelled": 10, "word": 0}}
    )
    monkeypatch.setattr(
        verbalize,
        "_acronym_surface_priors",
        lambda **_kwargs: ({"US": {"bare": {"word": 100}}}, 1, 1),
    )
    forms = verbalize._with_acronym_readings("U.S.", (), "en_US", profile="google-tn")
    assert [(form.text, form.weight) for form in forms] == [("u s", 1)]


def test_capitals_the_lexicon_lists_without_expansion_are_spelled():
    # icukit's lexicon lists "J.R.R." (a sentence-break exception) with no expansion.
    assert _read("J.R.R. Tolkien") == [["j r r"]]


def test_a_chain_of_initials_reads_one_initial_at_a_time():
    # Without icukit's abbreviation reader, the chain is frend's own: each initial alone.
    text = "X.Q.Z. Smith"
    lattice = resolve_lattice(list(detect(text, (LettersDetector(),))), source_text=text)
    units = verbalize_lattice(lattice).best_path.units
    spoken = [
        u.best.text
        for u in units
        if u.best.provenance not in {"surface:passthrough", "surface:word"}
        and not any("spelled-token" in item.provenance for item in u.alternatives)
    ]
    assert spoken == ["x", "q", "z"]


@pytest.mark.parametrize("suffix", ["'s", "’s"])
def test_a_dotted_acronym_keeps_its_possessive_suffix(suffix):
    text = f"F.B.I.{suffix}"
    detections = [
        detection
        for detection in LettersDetector().detect(text)
        if detection["type"] != "letters:token"
    ]

    assert [(detection["type"], detection["text"]) for detection in detections] == [
        ("letters:run", text)
    ]
    assert normalize(text, fold=None) == " f b i's "


@pytest.mark.parametrize("suffix", ["s", "s'", "s’"])
def test_a_dotted_plural_matches_its_undotted_acronym(suffix):
    dotted = f"F.B.I.{suffix}"
    undotted = f"FBI{suffix}"

    assert _spans(dotted) == [("letters:run", dotted)]
    assert normalize(dotted, fold=None) == normalize(undotted, fold=None) == " f b i's "


@pytest.mark.parametrize("suffix", ["s", "s'", "s’"])
def test_a_dotted_plural_uses_the_same_exact_dictionary_row_as_its_undotted_twin(suffix):
    assert normalize(f"P.C.{suffix}", fold=None) == normalize(f"PC{suffix}", fold=None) == " pcs "


@pytest.mark.parametrize("suffix", ["s", "s'", "s’"])
def test_a_plural_suffix_does_not_turn_a_single_initial_into_a_detection(suffix):
    assert _spans(f"A.{suffix}") == []


@pytest.mark.parametrize("apostrophe", ["'", "’"])
def test_a_dotted_uppercase_possessive_matches_lowercase(apostrophe):
    uppercase = f"F.B.I.{apostrophe}S"
    lowercase = f"F.B.I.{apostrophe}s"

    assert _spans(uppercase) == [("letters:run", uppercase)]
    assert normalize(uppercase, fold=None) == normalize(lowercase, fold=None) == " f b i's "


@pytest.mark.parametrize("suffix", ["'s", "’s"])
def test_a_word_favored_dotted_acronym_is_still_spelled(suffix):
    assert normalize(f"N.A.S.A.{suffix}", fold=None) == " n a s a's "


def test_a_bare_dotted_acronym_remains_a_chain_of_initials():
    assert _spans("F.B.I.") == [
        ("letters:initial", "F."),
        ("letters:initial", "B."),
        ("letters:initial", "I."),
    ]
    assert normalize("F.B.I.", fold=None) == " f b i "


# Capitals by Unicode general category Lu, in one ICU script (dualplan-ranges P3).


def test_a_cyrillic_capital_run_is_a_letter_run():
    assert _spans("СССР") == [("letters:run", "СССР")]


def test_a_latin_capital_with_accent_joins_a_run():
    assert _spans("ÉCU") == [("letters:run", "ÉCU")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ATM", [("letters:run", "ATM")]),
        ("UFOs", [("letters:run", "UFOs")]),
        ("S.", [("letters:initial", "S.")]),
        (
            "J.R.R.",
            [("letters:initial", "J."), ("letters:initial", "R."), ("letters:initial", "R.")],
        ),
    ],
)
def test_latin_runs_and_initials_are_unchanged(text, expected):
    assert _spans(text) == expected


@pytest.mark.parametrize("text", ["AΒC", "X.Β."])
def test_capitals_of_two_scripts_are_not_one_run(text):
    # Latin A and C around Greek Beta; a Latin and a Greek initial.
    assert _spans(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ABC東京", [("letters:run", "ABC")]),
        ("東京ABC", [("letters:run", "ABC")]),
        ("ABCМосква", [("letters:run", "ABC")]),
        ("МоскваABC", [("letters:run", "ABC")]),
        (
            "ABCАБВ",
            [("letters:run", "ABC"), ("letters:run", "АБВ")],
        ),
    ],
)
def test_script_transitions_bound_same_script_acronyms(text, expected):
    assert _spans(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "サーバーABC",
        "ユーザーID",
        "हिंदीABC",
    ],
)
def test_neutral_letters_and_combining_marks_inherit_the_adjacent_word_script(text):
    expected = "ID" if text.endswith("ID") else "ABC"
    assert _spans(text) == [("letters:run", expected)]


@pytest.mark.parametrize("text", ["wordABC", "ABCword", "wordABCword", "cafe\u0301ABC"])
def test_same_script_word_internal_capitals_remain_unread(text):
    assert _spans(text) == []


def _acronym_builder():
    import importlib.util
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "build_acronym_priors", tools / "build_acronym_priors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Tokens the builder and the reader must agree on, one by one: decomposed and
# precomposed accents, a Common capital beside a Latin one, other scripts, and the
# refusals (two real scripts, a non-capital, a lower case letter).
_POPULATION = ["ÉCO", "E\u0301CO", "AB\u0301CD", "СССР", "ℂA", "ℂℍ", "AΒ", "AB漢", "ATM", "Ab"]


def _counted(tmp_path, token: str) -> bool:
    shard = tmp_path / token.encode("unicode_escape").decode("ascii").replace("\\", "_")
    shard.mkdir()
    (shard / "output-00000-of-00001").write_text(
        f"LETTERS\t{token}\t{' '.join(token.lower())}\n<eos>\t<eos>\n", encoding="utf-8"
    )
    return "*" in _acronym_builder().build_document(shard)["keys"]


def test_the_acronym_builder_counts_what_the_reader_matches(tmp_path):
    """One predicate: the builder counts a token exactly when the letters reader matches
    it whole, token by token, so no count stands for a population the reader never
    ranks. A builder that counted "AΒ" and dropped "СССР" keeps the total and fails here."""
    counted = {token for token in _POPULATION if _counted(tmp_path, token)}
    matched = {token for token in _POPULATION if _spans(token) == [("letters:run", token)]}
    assert counted == matched
    assert matched == {"ÉCO", "E\u0301CO", "AB\u0301CD", "СССР", "ℂA", "ℂℍ", "ATM"}


def test_the_builder_counts_the_numerals_the_reader_defers(tmp_path):
    """The one stated difference: a run the corpus reads as a Roman numeral ("II") is
    counted by the builder (under ``roman:II`` too) and left by the reader to icukit."""
    assert _counted(tmp_path, "II")
    assert _spans("II") == []


def test_the_builder_strips_and_separates_acronym_suffixes(tmp_path):
    shard = tmp_path / "output-00000-of-00001"
    shard.write_text(
        "PLAIN\tABC\tabc\n"
        "LETTERS\tABCs\ta b c s\n"
        "LETTERS\tABCs'\ta b c s\n"
        "LETTERS\tABCs’\ta b c s\n"
        "PLAIN\tABC's\tabc s\n",
        encoding="utf-8",
    )
    builder = _acronym_builder()
    assert "surfaces" not in builder.build_document(tmp_path)
    assert builder.build_profile_document(tmp_path)["surfaces"] == {
        "ABC": {
            "bare": {"word": 1},
            "plural": {"spelled": 3},
            "possessive": {"word": 1},
        }
    }


def test_profile_builder_records_verified_shard_digests(tmp_path):
    import hashlib

    builder = _acronym_builder()
    from corpus_inputs import VerifiedInput

    shard = tmp_path / "output-00000-of-00100"
    shard.write_text("PLAIN\tABC\tabc\n", encoding="utf-8")
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    verified = VerifiedInput(
        "google/tn-en_with_types",
        shard.name,
        digest,
        "shippable-share-alike",
        shard,
    )
    profile = builder.build_profile_document(tmp_path, inputs=[verified])
    assert profile["schema_version"] == 1
    assert profile["provenance"]["source_shards"] == [
        {"relative_path": shard.name, "sha256": digest}
    ]
    assert profile["selection"]["selection_training_shards"][0] == "output-00000-of-00100"


def test_profile_builder_refuses_a_supplied_shard_95(tmp_path):
    builder = _acronym_builder()
    from corpus_inputs import VerifiedInput

    shard = tmp_path / "output-00095-of-00100"
    shard.write_text("PLAIN\tABC\tabc\n", encoding="utf-8")
    supplied = VerifiedInput("source", shard.name, "unused", "license", shard)
    with pytest.raises(ValueError, match="training shards 00-89.*output-00095"):
        builder.build_profile_document(tmp_path, inputs=[supplied])


def test_profile_output_must_be_outside_the_repository(monkeypatch):
    builder = _acronym_builder()
    destination = Path(__file__).resolve().parents[1] / "frend" / "data" / "profile.json"
    monkeypatch.setattr(
        builder,
        "_default_corpus_dir",
        lambda: (_ for _ in ()).throw(AssertionError("corpus must not be opened")),
    )
    with pytest.raises(ValueError, match="--profile-out must be outside"):
        builder.main(
            [
                "--locale",
                "en_US",
                "--source-id",
                "google/tn-en_with_types",
                "--pool",
                "training",
                "--receipt",
                str(destination.with_suffix(".receipt.json")),
                "--profile-out",
                str(destination),
            ]
        )


# Decomposed input is one run, with exact spans (fugu P3 review, finding 1).


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the E\u0301CO fund", [("letters:run", "E\u0301CO", 4, 8)]),
        ("AB\u0301CD", [("letters:run", "AB\u0301CD", 0, 5)]),
        ("E\u0301COs", [("letters:run", "E\u0301COs", 0, 5)]),
        ("Jane E\u0301. Smith", [("letters:initial", "E\u0301.", 5, 8)]),
        ("e\u0301AB", []),
        ("ABe\u0301", []),
    ],
)
def test_combining_marks_stay_with_their_capital(text, expected):
    found = [
        (d["type"], d["text"], d["start"], d["end"])
        for d in LettersDetector().detect(text)
        if d["type"] != "letters:token"
    ]
    assert found == expected
    assert all(text[start:end] == surface for _, surface, start, end in found)


def test_a_decomposed_run_reads_as_the_precomposed_one():
    assert _read("E\u0301CO") == _read("ÉCO")


# Each capital is spelled by its own lower case (fugu P3 review, finding 2).


def _letters_alternatives(text: str) -> list[str]:
    lattice = resolve_lattice(list(detect(text, (LettersDetector(),))), source_text=text)
    units = verbalize_lattice(lattice).best_path.units
    return [
        a.text for u in units if u.best.provenance != "surface:passthrough" for a in u.alternatives
    ]


def test_each_capital_is_spelled_by_its_own_lower_case():
    # Not "i ̇ b" (the full lowering of İ leaves a free dot) nor "ο ς" (final sigma).
    assert "i b" in _letters_alternatives("İB")
    assert "ο σ" in _letters_alternatives("ΟΣ")
    assert "e\u0301 c o".replace("e\u0301", "é") in _letters_alternatives("E\u0301CO")


def test_english_spelled_forms_are_unchanged():
    assert "a t m" in _letters_alternatives("ATM")
    assert _read("UFOs")[0] == ["u f o's", "ufos"]
    assert _read("J.R.R. Tolkien") == [["j r r"]]


# Script runs per UAX #24: Common capitals take their neighbors' script (finding 3).


@pytest.mark.parametrize("text", ["ℂA", "ℂℍ", "Aℂ"])
def test_a_common_capital_takes_its_neighbors_script(text):
    assert _spans(text) == [("letters:run", text)]


@pytest.mark.parametrize("text", ["AΒ", "AℂΒ"])
def test_two_real_scripts_are_still_not_one_run(text):
    assert _spans(text) == []


def test_the_vowels_are_the_locales():
    """``cv_pattern`` and ``letter_key`` read ``letter.vowels``; a locale whose table has
    none (ru's is empty) forms no vowel key."""
    from frend.electronic import letter_key

    assert cv_pattern("GUS") == cv_pattern("GUS", "en_US") == "cvc"
    assert letter_key("NASA") == "upper:4:v"
    assert letter_key("GWR") == "upper:3:nv"
    # English "y" is a vowel: "MYTH" has one.
    assert letter_key("MYTH") == "upper:4:v"
    assert cv_pattern("MYTH") == "cvcc"
    assert cv_pattern("GUS", "ru_RU") is None
    assert letter_key("NASA", "ru_RU") == "upper:4"


def test_a_changed_vowel_set_changes_the_keys(monkeypatch):
    """Verify by breaking: the keys follow the table, not a string in the code."""
    from frend import letters
    from frend.electronic import letter_key

    real = letters.lexical_forms

    def only_a(locale):
        forms = dict(real(locale))
        forms["letter.vowels"] = {"why": "test", "value": "a"}
        return forms

    monkeypatch.setattr(letters, "lexical_forms", only_a)
    letters._vowels_for.cache_clear()
    try:
        assert cv_pattern("GUS") == "ccc"
        assert letter_key("GUS") == "upper:3:nv"
    finally:
        letters._vowels_for.cache_clear()
