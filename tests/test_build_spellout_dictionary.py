from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _builder():
    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "build_spellout_dictionary", tools / "build_spellout_dictionary.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_google_decisions_differing_from_the_fallback_ship(tmp_path):
    builder = _builder()
    (tmp_path / "output-00000-of-00001").write_text(
        "LETTERS\tABC\ta b c\n" * 5
        + "PLAIN\tword\t<self>\n" * 5
        + "LETTERS\tFBI\tf b i\n" * 5
        + "PLAIN\tNASA\tnasa\n" * 5,
        encoding="utf-8",
    )
    document = builder.build_document(tmp_path)

    assert document["tokens"] == [["ABC", "spell", 0, 5], ["FBI", "spell", 0, 5]]
    assert document["provenance"]["source"] == builder.GOOGLE_SOURCE


def test_case_variants_remain_exact_surface_rows(tmp_path):
    builder = _builder()
    (tmp_path / "output-00000-of-00001").write_text(
        ("LETTERS\tFBI\tf b i\n" + "LETTERS\tfBi\tf b i\n") * 5,
        encoding="utf-8",
    )
    document = builder.build_document(tmp_path)
    assert document["tokens"] == [["FBI", "spell", 0, 5], ["fBi", "spell", 0, 5]]
    assert dict(document["casefold"])["fbi"] == "fBi"


def test_only_literal_spell_and_say_rows_are_counted(tmp_path):
    builder = _builder()
    (tmp_path / "output-00000-of-00001").write_text(
        "LETTERS\tpdf\tp d f\n"
        "LETTERS\txyz\tex why zee\n"
        "LETTERS\tABC\talphabet\n"
        "PLAIN\tword\t<self>\n"
        "PLAIN\tother\tchanged\n",
        encoding="utf-8",
    )

    counts = builder.google_counts(tmp_path)

    assert counts == {"pdf": {"spell": 1}, "word": {"say": 1}}


def test_wiktionary_word_senses_filter_spell_artifacts(tmp_path):
    builder = _builder()
    shard = tmp_path / "output-00000-of-00001"
    surfaces = (
        "zijn",
        "échec",
        "ZIJN",
        "Brno",
        "Léon",
        "TNA",
        "IBGE",
        "iOS",
        "eds",
        "cDNAs",
        "MLAs",
        "ve",
        "Amt",
        "Tas",
        "aka",
        "it",
        "IT",
    )
    shard.write_text(
        "".join(
            f"LETTERS\t{surface}\t{' '.join(surface.casefold())}\n" * 5 for surface in surfaces
        ),
        encoding="utf-8",
    )
    root = tmp_path / "lex"
    lexicon = root / "en" / "wiktionary.tsv"
    lexicon.parent.mkdir(parents=True)
    lexicon.write_text(
        "# lex/1  source=wiktionary  locale=en  "
        "columns=form,kind,case,sense,expansion,tags\n"
        "IT\tinitialism\texact\tinformation technology\tinformation technology\t"
        "initialism\n"
        "IBGE\tabbrev\texact\t\tisobutylgermane\tabbreviation\n"
        "AKA\tinitialism\texact\t\talso known as\tinitialism\n"
        "Tas.\tabbrev\texact\t\tTasmania\tabbreviation\n"
        "Leon\tabbrev\texact\t\tLeon County\tabbreviation\n"
        "cDNA\tinitialism\texact\tcomplementary DNA\tcomplementary DNA\tinitialism\n"
        "MLA\tinitialism\texact\tmember of the legislative assembly\t"
        "member of the legislative assembly\tinitialism\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "wiktionary.jsonl"
    artifact.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {"word": "zijn", "pos": "verb", "senses": [{"tags": []}]},
                {"word": "échec", "pos": "noun", "senses": [{"tags": []}]},
                {"word": "it", "pos": "pron", "senses": [{"tags": []}]},
                {
                    "word": "failure",
                    "pos": "noun",
                    "senses": [{"tags": []}],
                    "translations": [{"lang": "German", "word": "Amt"}],
                },
                {"word": "IT", "pos": "noun", "senses": [{"tags": []}]},
                {"word": "Brno", "pos": "name", "senses": [{"tags": []}]},
                {
                    "word": "Léon",
                    "pos": "name",
                    "senses": [{"tags": ["alt-of"], "alt_of": [{"word": "Leon"}]}],
                },
                {
                    "word": "Leon",
                    "pos": "name",
                    "senses": [
                        {"tags": ["abbreviation"]},
                        {"tags": [], "glosses": ["A male given name."]},
                    ],
                },
                {
                    "word": "TNA",
                    "pos": "name",
                    "senses": [{"tags": [], "glosses": ["A railway station code."]}],
                },
                {
                    "word": "IBGE",
                    "pos": "noun",
                    "senses": [{"tags": ["abbreviation"]}],
                },
                {"word": "iOS", "pos": "name", "senses": [{"tags": []}]},
                {
                    "word": "Tas",
                    "pos": "name",
                    "senses": [{"tags": ["alt-of"], "alt_of": [{"word": "Tas."}]}],
                },
                {"word": "aka", "pos": "noun", "senses": [{"tags": []}]},
                {
                    "word": "aka",
                    "pos": "prep",
                    "senses": [{"tags": ["alt-of"], "alt_of": [{"word": "AKA"}]}],
                },
                {
                    "word": "eds",
                    "pos": "noun",
                    "senses": [{"tags": ["form-of", "plural"], "form_of": [{"word": "ed"}]}],
                },
                {
                    "word": "cDNAs",
                    "pos": "noun",
                    "senses": [{"tags": ["form-of", "plural"], "form_of": [{"word": "cDNA"}]}],
                },
                {
                    "word": "MLAs",
                    "pos": "noun",
                    "senses": [{"tags": ["form-of", "plural"], "form_of": [{"word": "MLA"}]}],
                },
                {
                    "word": "ve",
                    "pos": "noun",
                    "senses": [{"tags": [], "glosses": ["The name of the Cyrillic letter."]}],
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = root / "sources" / "wiktionary" / "RECEIPT"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(f"artifact: {artifact}\n", encoding="utf-8")

    document = builder._build_documents(tmp_path, wiktionary=lexicon)

    rows = {row[0]: row[1:] for row in document["tokens"]}
    for surface in ("zijn", "échec", "Brno", "Léon", "it"):
        assert surface not in rows
    for surface in (
        "ZIJN",
        "IT",
        "TNA",
        "IBGE",
        "iOS",
        "eds",
        "cDNAs",
        "MLAs",
        "ve",
        "Amt",
        "Tas",
        "aka",
    ):
        assert rows[surface][0] == "spell"
    assert "zijn" not in dict(document["casefold"])
    assert "it" not in dict(document["casefold"])
    assert document["provenance"]["selection"]["ordinary_word_spell_rows_dropped"] == 5


def test_exact_headword_with_any_abbreviation_sense_is_not_ordinary(tmp_path):
    builder = _builder()
    artifact = tmp_path / "wiktionary.jsonl"
    artifact.write_text(
        json.dumps(
            {
                "word": "dual",
                "pos": "noun",
                "senses": [{"tags": []}, {"tags": ["abbreviation"]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    root = tmp_path / "lex"
    lexicon = root / "en" / "wiktionary.tsv"
    lexicon.parent.mkdir(parents=True)
    lexicon.write_text("", encoding="utf-8")
    receipt = root / "sources" / "wiktionary" / "RECEIPT"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(f"artifact: {artifact}\n", encoding="utf-8")

    assert builder.wiktionary_ordinary_words(lexicon, {"dual"}) == set()


def test_every_dictionary_row_is_context_free(tmp_path):
    builder = _builder()
    (tmp_path / "output-00000-of-00001").write_text("LETTERS\tABC\ta b c\n" * 5, encoding="utf-8")
    document = builder.build_document(tmp_path)
    assert document["provenance"]["columns"] == [
        "surface",
        "decision",
        "say_count",
        "spell_count",
    ]
    assert all(len(row) == 4 for row in document["tokens"])
    assert "context" not in repr(document).casefold()
    assert "wikipedia" not in repr(document).casefold()


def test_check_reproduces_counted_fixture_and_detects_changes(tmp_path, monkeypatch):
    builder = _builder()
    shard = tmp_path / "output-00000-of-00100"
    shard.write_text("LETTERS\tABC\ta b c\n" * 5, encoding="utf-8")
    item = SimpleNamespace(relative_path=shard.name, path=shard)
    out = tmp_path / "dictionary.json"
    receipt = tmp_path / "receipt.json"

    import corpus_inputs

    monkeypatch.setattr(builder, "full_training_set", lambda _paths: [shard])
    monkeypatch.setattr(corpus_inputs, "verified_inputs", lambda *_args, **_kwargs: (item,))
    monkeypatch.setattr(
        corpus_inputs,
        "open_verified",
        lambda verified: verified.path.open(encoding="utf-8"),
    )
    monkeypatch.setattr(corpus_inputs, "write_verification_receipt", lambda *_args, **_kwargs: {})
    document = builder.build_document(tmp_path, inputs=(item,), workers=1)
    assert document["tokens"] == [["ABC", "spell", 0, 5]]
    out.write_text(builder._render(document), encoding="utf-8")
    args = [
        "--corpus-dir",
        str(tmp_path),
        "--locale",
        "en_US",
        "--source-id",
        "google/tn-en_with_types",
        "--pool",
        "training",
        "--receipt",
        str(receipt),
        "--out",
        str(out),
        "--check",
        "--workers",
        "1",
    ]
    assert builder.main(args) == 0
    out.write_text(out.read_text(encoding="utf-8") + "altered\n", encoding="utf-8")
    assert builder.main(args) == 1
