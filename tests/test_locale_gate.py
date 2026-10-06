"""Scoring-contract tests use only tiny synthetic fixture files."""

from __future__ import annotations

import csv
import functools
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from frend import SpokenAlternative, normalize

REPO = Path(__file__).resolve().parents[1]


def _load_gate():
    name = "locale_gate_for_tests"
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "locale_gate.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _fixture(tmp_path: Path, language: str, name: str, text: str) -> Path:
    root = tmp_path / "nemo"
    directory = root / language / "data_text_normalization"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text, encoding="utf-8")
    return root


def test_german_files_read_spoken_then_written(tmp_path):
    root = _fixture(tmp_path, "de", "test_cases_cardinal.txt", "eins~1\n")
    [case] = gate.load_cases(root, "de", ("de_DE",))
    assert (case.written, case.targets) == ("1", ("eins",))


def test_audio_groups_score_any_admissible_form(tmp_path):
    root = _fixture(
        tmp_path,
        "en",
        "test_cases_normalize_with_audio.txt",
        "~1\none\nnot offered\n",
    )
    [case] = gate.load_cases(root, "en", ("en_US",))
    row = gate._evaluate(case, {})
    assert case.grouped
    assert row["target_hits"]["strict"] == [True, False]
    assert row["strict"]


def test_es_fixtures_load_under_both_es_locales(tmp_path):
    root = _fixture(tmp_path, "es", "test_cases_cardinal.txt", "1~uno\n")
    cases = gate.load_cases(root, "es", gate.LOCALES["es"])
    assert [case.locale for case in cases] == ["es_MX", "es_ES"]


def test_missing_final_newline_keeps_last_case(tmp_path):
    root = _fixture(
        tmp_path,
        "en",
        "test_cases_normalize_with_audio.txt",
        "~1\none\n~2\ntwo",
    )
    cases = gate.load_cases(root, "en", ("en_US",))
    assert [(case.written, case.targets) for case in cases] == [
        ("1", ("one",)),
        ("2", ("two",)),
    ]


def test_strict_form_deletes_format_characters():
    assert gate.strict_form("drei\u00adund\u200bzwanzig") == "dreiundzwanzig"


def test_strict_form_keeps_latin_and_hangul_spaces():
    assert gate.strict_form("w w w") != gate.strict_form("www")
    assert gate.strict_form("만 삼천") != gate.strict_form("만삼천")


def test_presentation_form_joins_only_breaks_between_letters_scripts():
    assert gate.presentation_form("漢 字") == "漢字"
    assert gate.presentation_form("か な") == "かな"
    assert gate.presentation_form("만 삼천") == "만 삼천"
    assert gate.presentation_form("a b") == "a b"


def test_unspaced_script_set_is_fixed_size():
    assert isinstance(gate._UNSPACED_SCRIPTS, frozenset)
    assert len(gate._UNSPACED_SCRIPTS) == 23


def test_upper_expansion_is_additive():
    case = gate.FixtureCase("en", "en_US", "word", "tiny.txt", 1, "x", ("NASA",), False)
    assert gate.admissible_targets(case, {}) == ("NASA", "N A S A")


def test_excluded_target_keyed_by_locale_case_target():
    case = gate.FixtureCase("en", "en_US", "cardinal", "tiny.txt", 1, "1", ("one", "first"), False)
    exclusions = {("en_US", "tiny.txt:1", "one"): "adjudicated"}
    assert gate.admissible_targets(case, exclusions) == ("first",)
    assert gate.admissible_targets(case, {("en_GB", "tiny.txt:1", "one"): "other"}) == (
        "one",
        "first",
    )


def test_mechanism_requires_complete_detection_for_C():
    case = gate.FixtureCase("en", "en_US", "cardinal", "tiny.txt", 1, "12x34", ("x",), False)
    partial = [{"type": "number:decimal", "start": 0, "end": 2}]
    complete = [{"type": "number:decimal", "start": 0, "end": 5}]
    assert gate.mechanism(case, partial) == "R-partial"
    assert gate.mechanism(case, complete) == "C"


def _tiny_graph():
    edges = (
        SimpleNamespace(start=0, end=1, kind="reading"),
        SimpleNamespace(start=0, end=1, kind="passthrough"),
        SimpleNamespace(start=1, end=2, kind="passthrough"),
    )
    units = (
        SimpleNamespace(
            alternatives=(
                SpokenAlternative("one", "test"),
                SpokenAlternative("won", "test"),
            )
        ),
        SimpleNamespace(alternatives=(SpokenAlternative("1", "surface:passthrough"),)),
        SimpleNamespace(alternatives=(SpokenAlternative("a", "surface:passthrough"),)),
    )
    return SimpleNamespace(lattice=SimpleNamespace(edges=edges, text_length=2), units=units)


def test_offers_dp_matches_bruteforce_on_tiny_graph():
    graph = _tiny_graph()
    brute = {gate.strict_form(value) for value in ("one a", "won a", "1a")}
    for wanted in ("one a", "won a", "1a", "onea", "two a"):
        assert gate.offers(graph, wanted, form=gate.strict_form) == (
            gate.strict_form(wanted) in brute
        )


def test_offers_spaces_spoken_alternative_on_passthrough_edge():
    edges = (
        SimpleNamespace(start=0, end=1, kind="reading"),
        SimpleNamespace(start=1, end=2, kind="passthrough"),
        SimpleNamespace(start=2, end=3, kind="reading"),
    )
    units = (
        SimpleNamespace(alternatives=(SpokenAlternative("five", "test:reading"),)),
        SimpleNamespace(
            alternatives=(
                SpokenAlternative("-", "surface:passthrough"),
                SpokenAlternative("to", "lexical:en_US"),
            )
        ),
        SimpleNamespace(alternatives=(SpokenAlternative("ten", "test:reading"),)),
    )
    graph = SimpleNamespace(lattice=SimpleNamespace(edges=edges, text_length=3), units=units)
    assert gate.offers(graph, "five to ten", form=gate.strict_form)
    assert not gate.offers(graph, "five toten", form=gate.strict_form)


def test_runtime_refuses_tracemalloc(monkeypatch):
    monkeypatch.setitem(sys.modules, "tracemalloc", SimpleNamespace())
    with pytest.raises(RuntimeError, match="tracemalloc"):
        gate.runtime(Path("unused"), Path("unused"), Path("unused"), 1)
    with pytest.raises(RuntimeError, match="tracemalloc"):
        gate._soak_child()


def test_unseen_soak_inputs_are_disjoint_across_passes():
    passes = [set(gate._unseen_inputs("en_US", pass_index, 20)) for pass_index in range(3)]
    assert all(left.isdisjoint(right) for left in passes for right in passes if left is not right)


def test_unseen_soak_exposes_unbounded_input_keyed_cache(monkeypatch):
    retained_bytes = 0

    @functools.cache
    def broken_normalize(text, *, locale):
        nonlocal retained_bytes
        retained_bytes += 1024
        return bytearray(1024), text, locale

    monkeypatch.setattr(gate, "GATE_LOCALES", ("en_US",))
    monkeypatch.setattr(gate, "normalize", broken_normalize)
    monkeypatch.setattr(gate, "rss_bytes", lambda: retained_bytes)
    result = gate._soak_child()
    assert result["pass3_minus_pass1_bytes"] > gate.GateBudget().rss_unseen_soak_delta_mib * 2**20


def _resource_documents(head_rss_mib: int, base_max_rss_mib: int):
    mib = 2**20

    def measurement(median, minimum=None, maximum=None):
        return {
            "median": median,
            "min": median if minimum is None else minimum,
            "max": median if maximum is None else maximum,
        }

    runtime_base = {
        "summary": {
            "first_hit_ns": {"en_US": measurement(1_000_000)},
            "sequential_first_hit_total_ns": measurement(1_000_000),
            "rss_loaded_bytes": measurement(100 * mib, 99 * mib, base_max_rss_mib * mib),
            "rss_workload_pass1_bytes": measurement(120 * mib),
            "rss_workload_pass2_bytes": measurement(121 * mib),
            "locales": {
                "en_US": {
                    "warm_p50_ns": measurement(1_000_000),
                    "warm_p95_ns": measurement(2_000_000),
                }
            },
        },
        "runs": [{"rss_workload_pass1_bytes": 120 * mib, "rss_workload_pass2_bytes": 121 * mib}],
    }
    runtime_head = json.loads(json.dumps(runtime_base))
    runtime_head["summary"]["rss_loaded_bytes"] = measurement(head_rss_mib * mib)
    soak = {"summary": {"pass3_minus_pass1_bytes": measurement(mib)}}
    fold = {
        "rows": [
            {
                "id": "short-en_US-001",
                "locale": "en_US",
                "mode": "plain",
                "timings_ns": {"resolve_k64": 1_000_000},
            }
        ]
    }
    return runtime_base, runtime_head, soak, fold


@pytest.mark.parametrize(
    ("head_rss_mib", "base_max_rss_mib", "expected_status"),
    ((109, 108, 1), (109, 110, 0)),
)
def test_resource_gate_fails_breach_but_passes_within_noise(
    tmp_path, head_rss_mib, base_max_rss_mib, expected_status
):
    runtime_base, runtime_head, soak, fold = _resource_documents(head_rss_mib, base_max_rss_mib)
    documents = {
        "base.json": {"cases": []},
        "head.json": {"cases": []},
        "base-runtime.json": runtime_base,
        "head-runtime.json": runtime_head,
        "base-soak.json": soak,
        "head-soak.json": soak,
        "base-fold.json": fold,
        "head-fold.json": fold,
    }
    for name, document in documents.items():
        (tmp_path / name).write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "compare.json"
    status = gate.main(
        [
            "compare",
            "--base",
            str(tmp_path / "base.json"),
            "--head",
            str(tmp_path / "head.json"),
            "--base-runtime",
            str(tmp_path / "base-runtime.json"),
            "--head-runtime",
            str(tmp_path / "head-runtime.json"),
            "--base-soak",
            str(tmp_path / "base-soak.json"),
            "--head-soak",
            str(tmp_path / "head-soak.json"),
            "--base-fold",
            str(tmp_path / "base-fold.json"),
            "--head-fold",
            str(tmp_path / "head-fold.json"),
            "--out",
            str(output),
        ]
    )
    assert status == expected_status
    payload = json.loads(output.read_text(encoding="utf-8"))
    failures = payload["resources"]["failures"]
    assert bool(failures) is bool(expected_status)
    assert next(
        check
        for check in payload["resources"]["checks"]
        if check["metric"] == "runtime.rss_loaded_bytes"
    )["outside_base_range"] is bool(expected_status)


def test_fixture_grep_finds_substrings(tmp_path):
    root = _fixture(tmp_path, "en", "test_cases_word.txt", "haystack needle suffix~x\n")
    result = gate.fixture_grep(root, ("needle", "absent"))
    assert [path.name for path in result["needle"]] == ["test_cases_word.txt"]
    assert result["absent"] == []


def test_no_unbounded_cache_is_keyed_on_input():
    assert gate.unbounded_cache_violations(REPO / "frend") == []


def test_checked_set_rows_all_carry_a_source_locator():
    path = REPO / "tests/data/locales/pt_PT_checked.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert all(row["source"].strip() for row in rows)
    cases = gate.load_checked_cases(path, "pt_PT")
    assert 35 <= len(cases) <= 50
    assert all(case.source == "checked.tsv" for case in cases)


def test_checked_set_loader_rejects_unsourced_rows(tmp_path):
    path = tmp_path / "checked.tsv"
    path.write_text("written\tspoken\tclass\tsource\n16\tdezasseis\tcardinal\t\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source locator"):
        gate.load_checked_cases(path, "pt_PT")


def test_pt_pt_icu_fields():
    outputs = [normalize(str(value), locale="pt_PT").casefold() for value in range(201)]
    outputs.extend(
        normalize(str(value), locale="pt_PT").casefold() for value in (1000, 10000, 1000000)
    )
    joined = " ".join(outputs)
    assert "dezasseis" in outputs[16]
    assert "dezassete" in outputs[17]
    assert "dezanove" in outputs[19]
    assert not {"dezesseis", "dezessete", "dezenove"} & set(joined.split())
