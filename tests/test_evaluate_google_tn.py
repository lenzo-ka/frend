"""The evaluator's running-text rejoining: by written form only, never overlapping."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _evaluator():
    sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(
        "evaluate_google_tn", _TOOLS / "evaluate_google_tn.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_number_separator_number_triple_is_rejoined_as_written():
    sentence = [
        ("PLAIN", "pages", "<self>"),
        ("CARDINAL", "5", "five"),
        ("PLAIN", "-", "to"),
        ("CARDINAL", "10", "ten"),
    ]
    assert _evaluator()._running_text([sentence]) == [("-", "to", "5-10", "five to ten")]


def test_silence_and_non_numbers_and_overlaps():
    evaluate = _evaluator()
    silent = [("CARDINAL", "3", "three"), ("PUNCT", ":", "sil"), ("CARDINAL", "2", "two")]
    assert evaluate._running_text([silent]) == [(":", "(silence)", "3:2", "three  two")]
    words = [("PLAIN", "well", "<self>"), ("PUNCT", "-", "sil"), ("PLAIN", "known", "<self>")]
    assert evaluate._running_text([words]) == []
    chain = [
        ("CARDINAL", "1", "one"),
        ("PLAIN", "-", "to"),
        ("CARDINAL", "2", "two"),
        ("PLAIN", "-", "to"),
        ("CARDINAL", "3", "three"),
    ]
    assert [item[2] for item in evaluate._running_text([chain])] == ["1-2"]


# ------------------------------------------------------------------ held-out shards

# The corpus README's split: training 00-89, runtime eval 90-94, test 95-99.
_TRAINING = {f"output-{index:05d}-of-00100" for index in range(90)}
_RUNTIME_EVAL = {f"output-{index:05d}-of-00100" for index in range(90, 95)}
_TEST = {f"output-{index:05d}-of-00100" for index in range(95, 100)}
_HELD_OUT = _RUNTIME_EVAL | _TEST


def _tool(name: str):
    """Load ``tools/<name>.py`` by path under its own module name (tools is not a package)."""
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _corpus(root: Path, first: int) -> Path:
    """A tmp corpus of shards ``first`` .. 99 of 100, each one cardinal row."""
    root.mkdir()
    for index in range(first, 100):
        (root / f"output-{index:05d}-of-00100").write_text(
            "CARDINAL\t12\ttwelve\n<eos>\t<eos>\n", encoding="utf-8"
        )
    return root


def test_no_builder_reads_a_held_out_shard(tmp_path, monkeypatch):
    """Every corpus builder, run on a corpus of all 100 shards (where every tenth shard by
    position lands on 00090), opens at least one shard and never a held-out one (90-99)
    -- by what it opens, not by what it says it counted; and on one missing 00000-00004
    (where every tenth by position lands on 00095) it refuses to build rather than sample
    by position, again opening no held-out shard."""
    type_priors = _tool("build_type_priors")
    spoken = _tool("build_spoken_priors")
    builders = {
        "build_type_priors": lambda corpus: type_priors._build("google-tn", corpus, 1),
        "build_spoken_priors": spoken.build_document,
        "build_zero_priors": _tool("build_zero_priors").build_document,
        "build_acronym_priors": _tool("build_acronym_priors").build_document,
        "build_electronic_priors": _tool("build_electronic_priors").build_document,
        "build_abbreviation_priors": _tool("build_abbreviation_priors").build_document,
        "build_range_priors": _tool("build_range_priors").build_document,
    }
    corpora = {"corpus0": _corpus(tmp_path / "c0", 0), "corpus5": _corpus(tmp_path / "c5", 5)}

    opened: list[str] = []
    real_open = Path.open

    def recording_open(self, *args, **kwargs):
        opened.append(self.name)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    read_held_out, read_nothing = {}, []
    for builder, build in builders.items():
        for label, corpus in corpora.items():
            opened.clear()
            if label == "corpus5":
                with pytest.raises(FileNotFoundError, match="training shard"):
                    build(corpus)
            else:
                build(corpus)
            shards = [name for name in opened if name.startswith("output-")]
            if not shards and label == "corpus0":
                read_nothing.append(f"{builder}@{label}")
            held = sorted(set(shards) & _HELD_OUT)
            if held:
                read_held_out[f"{builder}@{label}"] = held
    monkeypatch.undo()
    assert read_held_out == {}
    assert read_nothing == []

    # Imported last, so the check above fails on what a builder reads, not on an import.
    google_tn_rows = _tool("google_tn_rows")
    assert google_tn_rows.HELD_OUT_SHARDS == frozenset(_HELD_OUT)
    assert _evaluator()._TEST_FILE in google_tn_rows.TEST_SHARDS


def test_the_shards_follow_the_corpus_split():
    """The three pools are the corpus README's (training 00-89, runtime eval 90-94, test
    95-99), they partition the 100 shards, and training_shards keeps exactly the
    training pool of a full corpus, and a fixture's shards of another split."""
    google_tn_rows = _tool("google_tn_rows")
    assert google_tn_rows.TRAINING_SHARDS == frozenset(_TRAINING)
    assert google_tn_rows.RUNTIME_EVAL_SHARDS == frozenset(_RUNTIME_EVAL)
    assert google_tn_rows.TEST_SHARDS == frozenset(_TEST)
    assert google_tn_rows.HELD_OUT_SHARDS == frozenset(_HELD_OUT)
    every = [Path(f"output-{index:05d}-of-00100") for index in range(100)]
    assert [path.name for path in google_tn_rows.training_shards(every)] == sorted(_TRAINING)
    fixture = [Path("output-00000-of-00002"), Path("output-00001-of-00002")]
    assert google_tn_rows.training_shards(fixture) == fixture


def test_builder_and_evaluator_share_one_triple_predicate():
    """The evaluator scores the very predicate the builders count: one object, not a copy."""
    evaluate = _evaluator()
    google_tn_rows = importlib.import_module("google_tn_rows")  # the module it imported
    assert evaluate._running_text is google_tn_rows.running_text


def test_held_out_shard_flag_reads_the_whole_named_shard(tmp_path):
    """--held-out-shard scores per token over the named shard's first 100,000 lines, as
    the paper cuts the test shard, and running text over all of it; the published
    test_set section is unchanged."""
    evaluate = _evaluator()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "output-00099-of-00100").write_text(
        "CARDINAL\t7\tseven\n<eos>\t<eos>\n", encoding="utf-8"
    )
    # Scored tokens on lines 1 and 100,000 (the window's last line); the triple after it.
    lines = ["CARDINAL\t12\ttwelve"] + ["<eos>\t<eos>"] * 99_998 + ["CARDINAL\t8\teight"]
    lines += ["CARDINAL\t2\ttwo", "PUNCT\t-\tto", "CARDINAL\t5\tfive"]  # lines 100,001-100,003
    name = "output-00095-of-00100"
    (corpus / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(*flags: str) -> dict:
        out = tmp_path / f"report{len(flags)}.json"
        base = ["--corpus-dir", str(corpus), "--workers", "1", "--json", str(out)]
        assert evaluate.main([*base, *flags]) == 0
        return json.loads(out.read_text(encoding="utf-8"))

    report = run("--held-out-shard", name)
    plain = run()

    # The published report renders exactly as before, the held-out section after it.
    assert evaluate._render(report).startswith(evaluate._render(plain))
    held = report.pop("held_out")
    assert report == plain
    assert held["shard"] == name
    assert held["per_token"]["tokens"] == 2  # a window of 99,999 or 100,001 lines is 1 or 3
    assert held["per_token"]["classes"] == {
        "CARDINAL": {"tokens": 2, "first_choice": 1.0, "any_reading": 1.0}
    }
    assert held["running_text"]["triples"] == 1
    rows = held["running_text"]["rows"]
    assert [(row["separator"], row["corpus_middle"], row["count"]) for row in rows] == [
        ("-", "to", 1)
    ]


def test_a_corpus_of_only_held_out_shards_fails_loudly(tmp_path):
    """With nothing but the held-out shards (90-99), the spoken-family builders refuse
    to build rather than measure nothing (every tenth of them is 00090)."""
    _evaluator()  # puts tools/ on sys.path
    import build_spoken_priors

    for name in sorted(_HELD_OUT):
        (tmp_path / name).write_text("CARDINAL\t7\tseven\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no training shards"):
        build_spoken_priors._files(tmp_path)
