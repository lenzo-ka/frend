"""Guard and new-behavior checks for sourced regional telephone rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from icukit.detectors import detect

from frend import resolve_lattice
from frend.generated_rule_locales import locales_for
from frend.normalization_schema import validate_bundle
from frend.telephone import TelephoneDetector
from frend.telephone_rules import (
    generate_telephone_alternatives,
    regional_telephone_formats,
    telephone_rule_events,
)
from frend.verbalize import verbalize_edge
from frend.written_forms import WrittenFormsDetector

REPO = Path(__file__).resolve().parents[1]


def _detection(text: str, locale: str):
    matches = [
        item
        for item in detect(text, (WrittenFormsDetector(locale),))
        if item["type"] == "telephone:number" and item["start"] == 0 and item["end"] == len(text)
    ]
    assert len(matches) == 1
    return matches[0]


def _alternatives(text: str, locale: str):
    detection = _detection(text, locale)
    lattice = resolve_lattice([detection], source_text=text, locale=locale)
    edge = next(item for item in lattice.edges if item.kind == "reading")
    return verbalize_edge(edge, source_text=text, locale=locale).alternatives


# new: every row fails behaviorally on main because non-English locales have no phone route.
@pytest.mark.parametrize(
    ("locale", "written", "spoken", "rule"),
    [
        (
            "fr_FR",
            "01 23 45 67 89",
            "zéro un sil vingt-trois sil quarante-cinq sil soixante-sept sil quatre-vingt-neuf",
            "telephone-fr-fixed",
        ),
        (
            "fr_FR",
            "06 12 34 56 78",
            "zéro six sil douze sil trente-quatre sil cinquante-six sil soixante-dix-huit",
            "telephone-fr-mobile",
        ),
        (
            "pt_BR",
            "(11) 96123-4567",
            "um um sil nove seis um dois três sil quatro cinco seis sete",
            "telephone-br-mobile",
        ),
        (
            "pt_PT",
            "21 234 5678",
            "dois um sil dois três quatro sil cinco seis sete oito",
            "telephone-pt-lisbon",
        ),
        (
            "pt_PT",
            "21 203 0405",
            "dois um sil dois zero três sil zero quatro zero cinco",
            "telephone-pt-lisbon",
        ),
    ],
)
def test_new_supported_rows_offer_sourced_readings(locale, written, spoken, rule):
    detection = _detection(written, locale)
    assert detection["value"].matched_format.rule == rule
    alternatives = _alternatives(written, locale)
    assert alternatives[0].text == spoken
    assert " o " not in f" {alternatives[0].text} "
    assert "source:vendor-documentation:provisional" in alternatives[0].provenance
    assert "measured:" not in alternatives[0].provenance


# guard: kal explicitly excluded these three routes.
@pytest.mark.parametrize(
    ("locale", "written"),
    [("de_DE", "030 1234567"), ("de_DE", "+49 4 07 88 45"), ("pt_BR", "0800 12 3456")],
)
def test_guard_ruling_drops_unsupported_routes(locale, written):
    assert TelephoneDetector(locale).detect(written) == []


# guard: English stays on the existing measured branch with its exact first reading.
def test_guard_english_legacy_route_is_unchanged():
    detection = _detection("212-868-4444", "en_US")
    assert not hasattr(detection["value"], "matched_format")
    assert _alternatives("212-868-4444", "en_US")[0].text == (
        "two one two sil eight six eight sil four four four four"
    )


@pytest.mark.parametrize("locale", ("fr_FR", "pt_BR", "pt_PT"))
@pytest.mark.parametrize("written", ("15/03/2024", "12345/67", "12:34:56"))
def test_regional_non_phone_shapes_skip_libphonenumber(monkeypatch, locale, written):
    import frend.telephone as telephone

    def unexpected_matcher(*_args, **_kwargs):
        raise AssertionError("non-phone input reached libphonenumber")

    monkeypatch.setattr(telephone, "PhoneNumberMatcher", unexpected_matcher)
    assert TelephoneDetector(locale).detect(written) == []


def test_regional_candidate_guard_allows_phone_in_running_text():
    matches = TelephoneDetector("pt_PT").detect("ligue 21 234 5678 agora")
    assert [match["text"] for match in matches] == ["21 234 5678"]


def test_new_runtime_fails_closed_on_descriptor_drift(monkeypatch):
    import frend.telephone as telephone

    monkeypatch.setattr(telephone, "_descriptor_sha256", lambda _descriptor: "0" * 64)
    assert TelephoneDetector("fr_FR").detect("01 23 45 67 89") == []


def test_newer_phonenumbers_release_loads_but_descriptor_still_gates(monkeypatch):
    import phonenumbers

    regional_telephone_formats.cache_clear()
    monkeypatch.setattr(phonenumbers, "__version__", "9.1.0")
    assert regional_telephone_formats("fr_FR")
    regional_telephone_formats.cache_clear()


def test_new_runtime_loads_only_one_requested_locale():
    regional_telephone_formats.cache_clear()
    before = dict(telephone_rule_events())
    regional_telephone_formats("pt_PT")
    after = dict(telephone_rule_events())
    assert after["load"] == before["load"] + 1
    assert after["lower"] == before["lower"] == 0
    assert after["generate"] == before["generate"]
    assert regional_telephone_formats.cache_info().currsize == 1


def test_new_direct_generator_refuses_129_derivations(monkeypatch):
    import frend.telephone_rules as rules

    detection = _detection("21 234 5678", "pt_PT")
    from frend.verbalize import SpokenAlternative

    many = tuple(SpokenAlternative(str(index), "test") for index in range(129))
    monkeypatch.setattr(rules, "_telephone_group", lambda *_args: many, raising=False)
    # The generator imports from verbalize, so replace that bound function for this seam.
    import frend.verbalize as verbalize

    monkeypatch.setattr(verbalize, "_telephone_group", lambda *_args: many)
    with pytest.raises(ValueError, match="derivations"):
        generate_telephone_alternatives(detection["value"], "pt_PT")


def test_new_records_validate_and_projection_contains_no_copied_metadata():
    import tools.build_telephone_rules as builder

    assert locales_for("telephone") == ("fr_FR", "pt_BR", "pt_PT")
    for locale in locales_for("telephone"):
        authority = json.loads(
            (REPO / "frend/data/normalization" / f"telephone_{locale}.json").read_text("utf-8")
        )
        validate_bundle(authority, receipt_index=builder._receipt_index(authority))
        assert all(rule["source_category"] == "vendor documentation" for rule in authority["rules"])
        assert all(rule["provenance_status"] == "provisional" for rule in authority["rules"])
        projection = json.loads(
            (REPO / "frend/data/telephone_rules" / f"{locale}.json").read_text("utf-8")
        )
        serialized = json.dumps(projection)
        assert "pattern" not in serialized
        assert "quoted_passage" not in serialized
        assert "receipt" not in serialized
        assert "measured:" not in serialized


def test_new_vendor_receipts_name_the_manuals_and_record_unknown_license():
    for locale in ("fr-FR", "pt-BR", "pt-PT"):
        receipt = json.loads(
            (REPO / "frend/data/normalization" / f"acapela-telephone-{locale}.json").read_text(
                "utf-8"
            )
        )
        assert receipt["source"] == "acapela/tts-language-manuals"
        assert receipt["locator"].endswith(f"AcapelaLanguageManual_{locale}.pdf")
        assert receipt["fetch_method"] == "vendor documentation download"
        assert receipt["license"] == "not stated"


def test_new_builder_check_is_deterministic():
    import tools.build_telephone_rules as builder

    for locale in locales_for("telephone"):
        expected = builder._bytes(builder.build_bundle(locale))
        observed = (REPO / "frend/data/telephone_rules" / f"{locale}.json").read_text("utf-8")
        assert observed == expected
