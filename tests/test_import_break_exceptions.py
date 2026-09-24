from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from icukit.breaker import break_sentence_spans, break_word_spans
from icukit.exceptions import load_exception_inventory

from irn.exceptions import english_break_exceptions


def _load_importer():
    path = Path(__file__).resolve().parent.parent / "tools" / "import_break_exceptions.py"
    spec = importlib.util.spec_from_file_location("import_break_exceptions", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


importer = _load_importer()
_REPO = Path(__file__).resolve().parent.parent
_INVENTORY = _REPO / "irn" / "data" / "exceptions" / "en.json"


def test_curation_keeps_conditioned_titles_and_audits_ambiguous_and_regex_drops() -> None:
    entries = importer.parse_seed("Mr.\nMon.\nPt.\n:re [A-Z]\\.\n:re (?i:No\\.)\n")

    rules, drops = importer.curate(entries)

    assert [rule["surface"] for rule in rules] == ["Mr."]
    assert rules[0]["level"] == ["word", "sentence"]
    condition = rules[0]["conditions"][0]
    assert condition["set"] == "[[:Lu:]]"
    assert condition["skip"] == {"kind": "whitespace", "max": None}
    reasons = {drop.value: drop.reason for drop in drops}
    assert "real sentence endings" in reasons["Mon."]
    assert "ambiguous" in reasons["Pt."]
    assert reasons[r"[A-Z]\."] == "regex seed dropped as over-broad"
    assert reasons[r"(?i:No\.)"] == "regex seed dropped as over-broad"


def test_vendored_inventory_loads_and_passes_transactional_witness_gate() -> None:
    document = json.loads(_INVENTORY.read_text(encoding="utf-8"))

    loaded = load_exception_inventory(document)

    assert loaded.corpus == "break-exceptions-en-curated"
    assert len(document["rules"]) == 28
    assert all(rule["witnesses"]["positive"] for rule in document["rules"])
    assert all(rule["witnesses"]["near_miss"] for rule in document["rules"])


def test_english_layer_suppresses_false_title_sentence_boundary() -> None:
    text = "I met Mr. Smith today. He left."

    vanilla = break_sentence_spans(text, "en_US")
    curated = english_break_exceptions().break_spans(text, "sentence", "en_US")

    assert [span["text"] for span in vanilla] == ["I met Mr. ", "Smith today. ", "He left."]
    assert [span["text"] for span in curated] == ["I met Mr. Smith today. ", "He left."]


def test_single_rule_joins_abbreviation_at_word_and_sentence_levels(tmp_path) -> None:
    source = tmp_path / "en.txt"
    source.write_text("Dr.\n", encoding="utf-8")
    document, _drops = importer.build_document(source)
    loaded = load_exception_inventory(document)
    text = "I met Dr. Smith today. He left."

    vanilla_words = break_word_spans(text, "en_US")
    curated_words = loaded.break_spans(text, "word", "en_US")
    curated_sentences = loaded.break_spans(text, "sentence", "en_US")

    assert [span["text"] for span in vanilla_words[:8]] == [
        "I",
        " ",
        "met",
        " ",
        "Dr",
        ".",
        " ",
        "Smith",
    ]
    assert [span["text"] for span in curated_words[:7]] == [
        "I",
        " ",
        "met",
        " ",
        "Dr.",
        " ",
        "Smith",
    ]
    assert [span["text"] for span in curated_sentences] == [
        "I met Dr. Smith today. ",
        "He left.",
    ]


def test_check_passes_then_reports_drift(tmp_path, capsys) -> None:
    source = tmp_path / "en.txt"
    source.write_text("Mr.\nMon.\n:re [A-Z]\\.\n", encoding="utf-8")
    document, _drops = importer.build_document(source)
    out = tmp_path / "en.json"
    out.write_text(importer.serialize(document), encoding="utf-8")

    assert importer.main([str(source), "--check", "--out", str(out)]) == 0
    out.write_text("{}\n", encoding="utf-8")
    assert importer.main([str(source), "--check", "--out", str(out)]) == 1
    assert "drift:" in capsys.readouterr().err
