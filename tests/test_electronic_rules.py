from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from frend import normalize, resolve_lattice, verbalize_edge
from frend.electronic import ElectronicValue
from frend.electronic_rules import (
    ElectronicInputRefusal,
    electronic_grammar_input,
    electronic_letter_names,
    electronic_rule_events,
    generate_electronic_alternatives,
    load_electronic_rule_bundle,
)
from frend.generated_rule_locales import locales_for
from frend.letters import letter_names
from frend.normalization_schema import bundle_refusals, validate_bundle
from frend.normalize import _reading_detectors

REPO = Path(__file__).resolve().parents[1]
CLAIMS = REPO.parent / "agents/untracked/frend/lanes/nine-locales/pr9/source-claims.tsv"


def _email_detection(text: str, locale: str = "de_DE"):
    matches = [
        item
        for item in _reading_detectors(locale).detect(text)
        if item.get("type") == "electronic:email"
        and item["start"] == 0
        and item["end"] == len(text)
    ]
    assert matches
    return matches[0]


def _public_alternatives(text: str, locale: str = "de_DE"):
    detection = _email_detection(text, locale)
    lattice = resolve_lattice([detection], source_text=text, locale=locale)
    edge = next(item for item in lattice.edges if item.kind == "reading")
    return verbalize_edge(edge, source_text=text, locale=locale).alternatives


def _source_document():
    return json.loads((REPO / "frend/data/normalization/electronic_de_DE.json").read_text("utf-8"))


def _write_claims(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_new_de_email_offers_goethe_then_cldr_separator_readings():
    alternatives = _public_alternatives("a-b@c.de")
    sourced = [item for item in alternatives if "electronic-rule:de_DE:email" in item.provenance]

    assert [item.text for item in sourced] == [
        "a Minus be at ce Punkt de e",
        "a Minus be At-Zeichen ce Punkt de e",
        "a Bindestrich-Minus be at ce Punkt de e",
        "a Bindestrich-Minus be At-Zeichen ce Punkt de e",
    ]
    assert all(item.weight is None for item in sourced)
    assert all("measured:" not in item.provenance for item in sourced)


def test_new_de_email_isolated_letters_and_digits_use_private_sourced_units():
    alternatives = _public_alternatives("Q7@x.de")
    assert any(
        item.text == "ku sieben at ix Punkt de e"
        and "normalization-record:letter-Q" in item.provenance
        for item in alternatives
    )


def test_new_email_grammar_input_retains_each_labeled_run_once_in_order():
    value = _email_detection("ab@c.de")["value"]
    [token] = electronic_grammar_input(value, "de_DE").tokens
    assert token.symbol == "FIELD:RUNS"
    assert [item.tokens for item in token.realization] == [
        ("a be at ce Punkt de e",),
        ("a be At-Zeichen ce Punkt de e",),
    ]


def test_guard_surface_reading_is_retained_and_total_cap_includes_it():
    value = _email_detection("a-b@c.de")["value"]
    alternatives = generate_electronic_alternatives(value, "de_DE")
    assert alternatives[0].text == "a-b@c.de"
    assert len(alternatives) == 5
    assert len(alternatives) <= 8


def test_guard_unknown_email_separator_is_a_typed_refusal_and_public_surface():
    value = ElectronicValue(
        "email",
        "b.de",
        (("letters", "a"), ("separator", "+"), ("letters", "x@b")),
    )
    with pytest.raises(ElectronicInputRefusal):
        generate_electronic_alternatives(value, "de_DE")
    assert normalize("a+x@b.de", locale="de_DE").strip() == "a+x@b.de"


def test_guard_bare_symbols_keep_their_non_email_cldr_names():
    assert normalize("@", locale="de_DE").strip() == "At-Zeichen"
    assert normalize("-", locale="de_DE").strip() == "Bindestrich-Minus"


def test_guard_general_german_letter_names_remain_unsupported():
    assert letter_names("AB", "de_DE") is None
    assert electronic_letter_names("AB", "de_DE")[0].tokens == ("a be",)


def test_guard_unsupported_kind_does_not_load_electronic_bundle(monkeypatch):
    import frend.electronic_rules as rules

    def unexpected(_locale):
        raise AssertionError("unsupported electronic kind loaded a private bundle")

    monkeypatch.setattr(rules, "load_electronic_rule_bundle", unexpected)
    assert normalize("example.de", locale="de_DE").strip() == "example.de"


def test_guard_english_electronic_candidates_are_byte_identical():
    alternatives = _public_alternatives("jane.doe@example.org", "en_US")
    signature = [
        (item.text, item.provenance, item.prior_provenance, str(item.weight))
        for item in alternatives
    ]
    assert signature == [
        (
            "jane dot doe at example dot org",
            "measured:electronic+lexical:en_US",
            None,
            "0.7341764842685862225145740363",
        ),
        (
            "j a n e dot doe at example dot org",
            "measured:electronic+lexical:en_US",
            None,
            "0.1230081585104572821942909331",
        ),
        (
            "jane dot d o e at example dot org",
            "measured:electronic+lexical:en_US",
            None,
            "0.1125242794117217704376038999",
        ),
        (
            "j a n e dot d o e at example dot org",
            "measured:electronic+lexical:en_US",
            None,
            "0.01885296613925380029043488105",
        ),
        (
            "jane dot doe at e x a m p l e dot org",
            "measured:electronic+lexical:en_US",
            None,
            "0.008381906103877870690910862369",
        ),
        (
            "j a n e dot doe at e x a m p l e dot org",
            "measured:electronic+lexical:en_US",
            None,
            "0.001404352845314490778819000448",
        ),
        (
            "jane dot d o e at e x a m p l e dot org",
            "measured:electronic+lexical:en_US",
            None,
            "0.001284661065350231868127180593",
        ),
        (
            "j a n e dot d o e at e x a m p l e dot org",
            "measured:electronic+lexical:en_US",
            None,
            "0.0002152395171254271256257585202",
        ),
    ]


def test_guard_electronic_capability_is_only_de_de():
    assert locales_for("electronic") == ("de_DE",)


def test_guard_runtime_projection_matches_source_and_has_no_source_prose():
    import tools.build_electronic_rules as builder

    document = _source_document()
    validate_bundle(document, receipt_index=builder._receipt_index(document))
    runtime = json.loads((REPO / "frend/data/electronic_rules/de_DE.json").read_text("utf-8"))
    assert runtime["recipes"] == builder.compile_recipes(
        builder.compile_grammar("de_DE", document), document
    )
    assert runtime["records"] == [
        {
            key: record[key]
            for key in (
                "id",
                "kind",
                "value",
                "forms",
                "electronic_kind",
                "priority",
                "derived_from",
            )
            if key in record
        }
        for record in document["resources"]
    ]
    assert (
        runtime["normalization_sha256"]
        == hashlib.sha256(
            json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    compact = json.dumps(runtime, ensure_ascii=False)
    assert "quoted_passage" not in compact
    assert '"receipt"' not in compact
    assert "https://" not in compact


def test_guard_rule_schema_rejects_missing_or_cross_kind_discriminator():
    document = _source_document()
    document["rules"][0]["electronic_kind"] = "telephone"
    assert any("invalid-rule" in refusal for refusal in bundle_refusals(document))


def test_guard_claim_join_rejects_cross_kind_and_unapproved_rows(tmp_path):
    import tools.build_electronic_rules as builder

    rows = list(csv.DictReader(CLAIMS.open(encoding="utf-8"), delimiter="\t"))
    claim = next(row for row in rows if row["record_id"] == "de_DE:electronic-email")
    claim["electronic_kind"] = "url"
    path = tmp_path / "claims.tsv"
    _write_claims(path, rows)
    with pytest.raises(ValueError, match="cross-kind"):
        builder.validate_source_claims("de_DE", _source_document(), path)

    claim["electronic_kind"] = "email"
    claim["verdict"] = "NOT CLAIMED"
    _write_claims(path, rows)
    with pytest.raises(ValueError, match="not approved"):
        builder.validate_source_claims("de_DE", _source_document(), path)


def test_guard_target_must_emit_each_hole_exactly_once():
    import tools.build_electronic_rules as builder

    document = _source_document()
    document["rules"][0]["target"] = ["RUNS", "RUNS"]
    with pytest.raises(ValueError, match="consume and emit each field once"):
        builder.compile_grammar("de_DE", document)


def test_guard_runtime_is_lazy_and_uses_precompiled_projection_only():
    load_electronic_rule_bundle.cache_clear()
    before = dict(electronic_rule_events())
    load_electronic_rule_bundle("de_DE")
    after = dict(electronic_rule_events())
    assert after["load"] == before["load"] + 1
    assert after["input"] == before["input"]
    assert after["generate"] == before["generate"]
    assert load_electronic_rule_bundle.cache_info().currsize == 1


def test_guard_runtime_rejects_projection_and_policy_drift(tmp_path, monkeypatch):
    import frend.electronic_rules as rules

    runtime_dir = tmp_path / "data/electronic_rules"
    normalization_dir = tmp_path / "data/normalization"
    runtime_dir.mkdir(parents=True)
    normalization_dir.mkdir(parents=True)
    runtime = json.loads((REPO / "frend/data/electronic_rules/de_DE.json").read_text("utf-8"))
    runtime["policy"]["max_total_alternatives"] = 9
    (runtime_dir / "de_DE.json").write_text(json.dumps(runtime), encoding="utf-8")
    (normalization_dir / "electronic_de_DE.json").write_text(
        json.dumps(_source_document()), encoding="utf-8"
    )
    monkeypatch.setattr(rules, "files", lambda _package: tmp_path)
    load_electronic_rule_bundle.cache_clear()
    with pytest.raises(ValueError, match="policy"):
        load_electronic_rule_bundle("de_DE")
    load_electronic_rule_bundle.cache_clear()
