from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
    assert "casefold" not in document


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
    assert "wiktionary" not in repr(document).casefold()


def test_check_reproduces_the_builder_bytes(tmp_path, monkeypatch):
    builder = _builder()
    shard = tmp_path / "output-00000-of-00100"
    shard.write_text("LETTERS\tABC\ta b c\n", encoding="utf-8")
    out = tmp_path / "dictionary.json"
    receipt = tmp_path / "receipt.json"
    document = builder.build_document(tmp_path, inputs=[], workers=1)
    out.write_text(builder._render(document), encoding="utf-8")
    monkeypatch.setattr(builder, "full_training_set", lambda _paths: [shard])

    import corpus_inputs

    monkeypatch.setattr(corpus_inputs, "verified_inputs", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(corpus_inputs, "write_verification_receipt", lambda *_args, **_kwargs: {})
    assert (
        builder.main(
            [
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
        )
        == 0
    )
