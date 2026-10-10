"""Scoring-contract tests use only tiny synthetic fixture files."""

from __future__ import annotations

import csv
import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from icukit.detectors import detect

from frend import SpokenAlternative, compose_choices, normalize, resolve_choices
from frend.normalize import _reading_detectors

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


@pytest.mark.parametrize(
    ("locale", "written", "other"),
    [
        ("zh_CN", "漢 字", "漢字"),
        ("zh_CN", "漢。字", "漢字"),
        ("zh_CN", "A漢", "A 漢"),
        ("zh_CN", "漢1字", "漢 初一 字"),
        ("ko_KR", "한1글", "한일글"),
    ],
)
def test_offers_matches_renderer_spacing(locale, written, other):
    detections = detect(written, _reading_detectors(locale))
    graph = compose_choices(resolve_choices(detections, source_text=written, locale=locale))
    rendered = normalize(written, locale=locale)

    assert gate.offers(graph, rendered, form=gate.strict_form)
    assert gate.strict_form(other) != gate.strict_form(rendered)
    assert not gate.offers(graph, other, form=gate.strict_form)


def test_runtime_refuses_tracemalloc(monkeypatch):
    monkeypatch.setitem(sys.modules, "tracemalloc", SimpleNamespace())
    with pytest.raises(RuntimeError, match="tracemalloc"):
        gate.runtime(Path("unused"), Path("unused"), Path("unused"), 1)
    with pytest.raises(RuntimeError, match="tracemalloc"):
        gate._soak_child()


def test_unseen_soak_inputs_are_disjoint_across_passes():
    passes = [set(gate._unseen_inputs("en_US", pass_index, 20)) for pass_index in range(3)]
    assert all(left.isdisjoint(right) for left in passes for right in passes if left is not right)


def test_unseen_soak_keeps_two_thousand_dates_per_locale_per_pass(monkeypatch):
    observed_counts = []

    def dates(_locale, _pass_index, count=2000):
        observed_counts.append(count)
        return ["not-a-date"]

    monkeypatch.setattr(gate, "GATE_LOCALES", ("en_US",))
    monkeypatch.setattr(gate, "_unseen_inputs", dates)
    monkeypatch.setattr(gate, "_unseen_fraction_inputs", lambda *_args: ["not-a-fraction"])
    monkeypatch.setattr(gate, "_reading_detectors", lambda _locale: ())
    monkeypatch.setattr(gate, "detect", lambda *_args: ())
    monkeypatch.setattr(
        gate,
        "normalize",
        lambda *_args, **_kwargs: SimpleNamespace(units=()),
    )
    monkeypatch.setattr(gate, "rss_bytes", lambda: 0)
    gate._soak_child()
    assert observed_counts == [2000, 2000, 2000]


def test_first_hit_times_only_the_lazy_default_request(monkeypatch):
    clock = iter((100, 250))
    calls = []
    monkeypatch.setattr(gate.time, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(gate, "normalize", lambda text, *, locale: calls.append((text, locale)))
    monkeypatch.setattr(gate, "_normalize_date_probe", lambda _locale: (10_000, {"date": True}))
    monkeypatch.setattr(
        gate,
        "_normalize_fraction_probes",
        lambda _locale: (20_000, {"fraction": True}),
    )
    result = gate._runtime_child("first", "fr_FR", Path("unused"), Path("unused"))
    assert result["first_hit_ns"] == 150
    assert calls == [("123", "fr_FR")]


def test_unseen_soak_exposes_unbounded_input_keyed_cache(monkeypatch):
    retained_bytes = 0

    @functools.cache
    def broken_normalize(text, *, locale, offsets=False):
        del offsets
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
        "runs": [
            {
                "run": 1,
                "rss_workload_pass1_bytes": 120 * mib,
                "rss_workload_pass2_bytes": 121 * mib,
                "date_probes": {
                    "en_US": {"date_units": 1, "selected_generated": False, "events": {}}
                },
                "fraction_probes": {
                    "en_US": {
                        "probes": {
                            "fraction": {"units": 1, "selected_generated": False},
                            "percent": {"units": 1, "selected_generated": False},
                        },
                        "events": {"load": 0, "lower": 0, "generate": 0},
                        "bundle_available": True,
                    }
                },
            }
        ],
    }
    runtime_head = json.loads(json.dumps(runtime_base))
    runtime_head["summary"]["rss_loaded_bytes"] = measurement(head_rss_mib * mib)
    soak_receipts = [
        {
            "locale": locale,
            "pass": pass_index,
            "count": 1,
            "successful_date_detections": 1,
            "selected_generated": 0 if locale == "en_US" else 1,
        }
        for locale in gate.GATE_LOCALES
        for pass_index in range(1, 4)
    ]
    soak = {
        "summary": {"pass3_minus_pass1_bytes": measurement(mib)},
        "runs": [
            {
                "run": 1,
                "date_receipts": soak_receipts,
                "fraction_receipts": [
                    {
                        "locale": row["locale"],
                        "pass": row["pass"],
                        "count": 1,
                        "successful_fraction_detections": 1,
                        "selected_generated": 0 if row["locale"] == "en_US" else 1,
                    }
                    for row in soak_receipts
                ],
            }
        ],
    }
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


def test_runtime_gate_rejects_empty_date_probe_receipts():
    runtime_base, runtime_head, _soak, _fold = _resource_documents(100, 100)
    runtime_head["runs"][0]["date_probes"] = {}

    result = gate.compare(
        {"cases": []},
        {"cases": []},
        base_runtime=runtime_base,
        head_runtime=runtime_head,
    )

    failures = {item["metric"] for item in result["resources"]["failures"]}
    assert "runtime.date_probe.receipt_contract.run1" in failures


def test_soak_gate_rejects_empty_date_receipts():
    _runtime_base, _runtime_head, soak, _fold = _resource_documents(100, 100)
    head_soak = json.loads(json.dumps(soak))
    head_soak["runs"][0]["date_receipts"] = []

    result = gate.compare(
        {"cases": []},
        {"cases": []},
        base_soak=soak,
        head_soak=head_soak,
    )

    failures = {item["metric"] for item in result["resources"]["failures"]}
    assert "soak.date_probe.receipt_contract.run1" in failures


def test_require_identical_checks_first_choice_and_all_offers():
    row = {
        "id": "en_US:fixture:1",
        "written": "1",
        "strict": True,
        "presentation": True,
        "insensitive": True,
        "first_strict": True,
        "first_presentation": True,
        "first": " one ",
        "offer_signature": [{"alternatives": [{"text": "one", "surface": False}]}],
        "error": None,
    }
    same = gate.compare({"cases": [row]}, {"cases": [row]}, require_identical=True)
    changed = json.loads(json.dumps(row))
    changed["offer_signature"][0]["alternatives"][0]["text"] = "won"
    different = gate.compare({"cases": [row]}, {"cases": [changed]}, require_identical=True)

    assert same["identity"]["passed"]
    assert not different["identity"]["passed"]
    assert different["identity"]["mismatches"][0]["changes"] == {
        "offer_signature": {
            "base": row["offer_signature"],
            "head": changed["offer_signature"],
        }
    }


def test_require_identical_rejects_documents_that_both_omit_offer_signature():
    row = {
        "id": "en_US:fixture:1",
        "written": "1",
        "strict": True,
        "presentation": True,
        "insensitive": True,
        "first_strict": True,
        "first_presentation": True,
        "first": " one ",
        "error": None,
    }
    result = gate.compare({"cases": [row]}, {"cases": [row]}, require_identical=True)

    assert not result["identity"]["passed"]
    assert result["identity"]["mismatches"][0]["changes"]["offer_signature"] == {
        "missing_on": ["base", "head"]
    }


def test_offer_signature_records_every_public_alternative_field():
    edge = SimpleNamespace(
        start=0,
        end=1,
        kind="reading",
        detection={"type": "date:yMd", "value": SimpleNamespace(fields=(("y", 2024),))},
    )
    alternative = SpokenAlternative("one", "icu:test", weight=None, prior_provenance="measured:old")
    graph = SimpleNamespace(
        lattice=SimpleNamespace(edges=(edge,)),
        units=(SimpleNamespace(alternatives=(alternative,)),),
    )
    [signature] = gate._offer_signature(graph)
    assert signature["type"] == "date:yMd"
    assert signature["fields"] == ["y"]
    assert signature["alternatives"] == [
        {
            "text": "one",
            "provenance": "icu:test",
            "prior_provenance": "measured:old",
            "weight": None,
            "surface": False,
        }
    ]


def test_rwandan_number_rbnf_is_counted_as_not_own_language():
    case = gate.FixtureCase("rw", "rw_RW", "cardinal", "tiny.txt", 1, "42", ("forty-two",), False)
    row = gate._evaluate(case, {})
    summary = gate._summarize([row])["total"]
    number_alternatives = [
        alternative
        for edge in row["offer_signature"]
        if str(edge["type"]).startswith("number:")
        for alternative in edge["alternatives"]
        if "icu-rbnf:" in alternative["provenance"]
    ]

    assert number_alternatives
    assert all(
        "icu-rbnf-fallback:en" in alternative["provenance"] for alternative in number_alternatives
    )
    assert summary["rbnf_own_language"] == 0
    assert summary["rbnf_not_own_language"] == len(number_alternatives)


def test_same_language_parent_rbnf_is_retained_and_counted_as_own_language():
    case = gate.FixtureCase("en", "en_US", "cardinal", "tiny.txt", 1, "42", ("forty-two",), False)
    row = gate._evaluate(case, {})
    summary = gate._summarize([row])["total"]
    rbnf_alternatives = [
        alternative
        for edge in row["offer_signature"]
        for alternative in edge["alternatives"]
        if "icu-rbnf:" in alternative["provenance"]
    ]

    assert rbnf_alternatives
    assert all(
        "icu-rbnf-fallback:en" in alternative["provenance"] for alternative in rbnf_alternatives
    )
    assert summary["rbnf_own_language"] == len(rbnf_alternatives)
    assert summary["rbnf_not_own_language"] == 0


def test_rbnf_counts_include_number_words_composed_into_other_reading_types():
    edge = SimpleNamespace(start=0, end=4, kind="reading", detection={"type": "measure:meter"})
    alternative = SpokenAlternative(
        "one meter",
        "icu-rbnf:%spellout-numbering+icu-rbnf-fallback:en+icu-measure:wide",
    )
    graph = SimpleNamespace(
        lattice=SimpleNamespace(edges=(edge,)),
        units=(SimpleNamespace(alternatives=(alternative,)),),
    )

    assert gate._rbnf_language_counts(graph, "rw_RW") == (0, 1)


def test_warm_p95_has_no_additive_or_base_spread_allowance():
    base_runtime, head_runtime, _soak, _fold = _resource_documents(100, 100)
    base_runtime["summary"]["locales"]["en_US"]["warm_p50_ns"] = {
        "median": 200_000,
        "min": 190_000,
        "max": 300_000,
    }
    head_runtime["summary"]["locales"]["en_US"]["warm_p50_ns"] = {
        "median": 290_000,
        "min": 290_000,
        "max": 290_000,
    }
    base_runtime["summary"]["locales"]["en_US"]["warm_p95_ns"] = {
        "median": 200_000,
        "min": 190_000,
        "max": 300_000,
    }
    head_runtime["summary"]["locales"]["en_US"]["warm_p95_ns"] = {
        "median": 290_000,
        "min": 290_000,
        "max": 290_000,
    }
    result = gate.compare(
        {"cases": []},
        {"cases": []},
        base_runtime=base_runtime,
        head_runtime=head_runtime,
    )
    failures = {item["metric"] for item in result["resources"]["failures"]}
    assert "runtime.locales.en_US.warm_p50_ns" not in failures
    assert "runtime.locales.en_US.warm_p95_ns" in failures


def test_require_no_negative_flips_is_an_executable_failure():
    base = {
        "id": "fr_FR:date:1",
        "locale": "fr_FR",
        "written": "x",
        "strict": True,
        "presentation": True,
        "insensitive": True,
        "first_strict": True,
        "first_presentation": True,
        "first": "x",
        "offer_signature": [],
        "error": None,
    }
    head = {**base, "strict": False}
    result = gate.compare({"cases": [base]}, {"cases": [head]}, require_no_negative_flips=True)
    assert not result["correctness"]["passed"]
    assert result["correctness"]["failures"][0]["kind"] == "negative-flips"


def test_expected_recoveries_cannot_pass_empty():
    result = gate.compare(
        {"cases": []},
        {"cases": []},
        expected_recoveries={
            "rows": [],
            "strict_total": 0,
            "insensitive_total": 0,
            "witness_sha256": hashlib.sha256(b"[]").hexdigest(),
        },
    )
    assert not result["correctness"]["passed"]
    assert result["correctness"]["failures"][0]["kind"] == "expected-recovery-minimum"


def test_fraction_manifest_routing_set_is_independent_of_date_rows():
    common = {
        "locale": "fr_FR",
        "written": "3/7",
        "strict": False,
        "presentation": False,
        "insensitive": False,
        "first_strict": False,
        "first_presentation": False,
        "first": "x",
        "error": None,
    }
    fraction = {
        **common,
        "id": "fr_FR:fraction:1",
        "offer_signature": [{"type": "number:fraction", "fields": []}],
    }
    date = {
        **common,
        "id": "fr_FR:date:1",
        "offer_signature": [{"type": "date:yMd", "fields": ["y", "M", "d"]}],
    }
    public_fields = (
        "strict",
        "presentation",
        "insensitive",
        "first",
        "first_strict",
        "first_presentation",
        "error",
        "offer_signature",
    )
    payload = {field: fraction.get(field) for field in public_fields}
    result = gate.compare(
        {"cases": [fraction, date]},
        {"cases": [fraction, date]},
        allowed_changes={
            "family": "fraction",
            "rows": [{"id": fraction["id"], "base": payload, "head": payload}],
        },
    )
    assert result["correctness"]["passed"]


def test_fraction_recovery_gate_rejects_removed_expected_positive():
    base_rows = []
    head_rows = []
    for index in range(4):
        base = {
            "id": f"fr_FR:fraction:{index}",
            "locale": "fr_FR",
            "written": "3/7",
            "strict": False,
            "presentation": False,
            "insensitive": False,
            "first_strict": False,
            "first_presentation": False,
            "targets": ["trois septièmes"],
            "target_hits": {
                "strict": [False],
                "presentation": [False],
                "insensitive": [False],
            },
            "offer_signature": [],
        }
        head = {
            **base,
            "strict": True,
            "presentation": True,
            "insensitive": True,
            "target_hits": {
                "strict": [True],
                "presentation": [True],
                "insensitive": [True],
            },
            "offer_signature": [
                {
                    "alternatives": [
                        {
                            "text": "trois septièmes",
                            "provenance": (
                                "fraction-rule:fr_FR+"
                                "normalization-record:frend/curated#fr-fractions"
                            ),
                        }
                    ]
                }
            ],
        }
        base_rows.append(base)
        head_rows.append(head)
    fields = ("strict", "presentation", "insensitive", "first_strict", "first_presentation")
    manifest_rows = [
        {
            "id": base_rows[index]["id"],
            "base": {field: base_rows[index][field] for field in fields},
            "head": {field: head_rows[index][field] for field in fields},
            "expected_speech": "trois septièmes",
            "source_record_id": "frend/curated#fr-fractions",
        }
        for index in range(3)
    ]
    witness_sha256 = hashlib.sha256(
        json.dumps(
            manifest_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    result = gate.compare(
        {"cases": base_rows},
        {"cases": head_rows},
        expected_recoveries={
            "rows": manifest_rows,
            "strict_total": 3,
            "insensitive_total": 3,
            "witness_sha256": witness_sha256,
        },
    )
    assert {failure["kind"] for failure in result["correctness"]["failures"]} == {
        "unexpected-positive-flips"
    }


@pytest.mark.parametrize(
    ("field", "replacement", "failure_kind"),
    [
        ("expected_speech", "nonsense", "expected-recovery-speech"),
        ("source_record_id", "nonsense", "expected-recovery-source-record"),
    ],
)
def test_fraction_recovery_gate_validates_witness_metadata(field, replacement, failure_kind):
    base = {
        "id": "fr_FR:fraction:1",
        "locale": "fr_FR",
        "written": "3/7",
        "strict": False,
        "presentation": False,
        "insensitive": False,
        "first_strict": False,
        "first_presentation": False,
        "targets": ["trois septièmes"],
        "target_hits": {name: [False] for name in ("strict", "presentation", "insensitive")},
        "offer_signature": [],
    }
    head = {
        **base,
        "strict": True,
        "presentation": True,
        "insensitive": True,
        "target_hits": {name: [True] for name in ("strict", "presentation", "insensitive")},
        "offer_signature": [
            {
                "alternatives": [
                    {
                        "text": "trois septièmes",
                        "provenance": (
                            "fraction-rule:fr_FR+normalization-record:frend/curated#fr-fractions"
                        ),
                    }
                ]
            }
        ],
    }
    fields = ("strict", "presentation", "insensitive", "first_strict", "first_presentation")
    declared = {
        "id": base["id"],
        "base": {name: base[name] for name in fields},
        "head": {name: head[name] for name in fields},
        "expected_speech": "trois septièmes",
        "source_record_id": "frend/curated#fr-fractions",
    }
    declared[field] = replacement
    witness_sha256 = hashlib.sha256(
        json.dumps([declared], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = gate.compare(
        {"cases": [base]},
        {"cases": [head]},
        expected_recoveries={
            "rows": [declared],
            "strict_total": 1,
            "insensitive_total": 1,
            "witness_sha256": witness_sha256,
        },
    )
    assert failure_kind in {item["kind"] for item in result["correctness"]["failures"]}


def test_fraction_recovery_source_must_offer_the_declared_speech():
    base = {
        "id": "fr_FR:fraction:1",
        "locale": "fr_FR",
        "written": "3/7",
        "strict": False,
        "presentation": False,
        "insensitive": False,
        "first_strict": False,
        "first_presentation": False,
        "targets": ["trois septièmes"],
        "target_hits": {name: [False] for name in ("strict", "presentation", "insensitive")},
        "offer_signature": [],
    }
    head = {
        **base,
        "strict": True,
        "presentation": True,
        "insensitive": True,
        "target_hits": {name: [True] for name in ("strict", "presentation", "insensitive")},
        "offer_signature": [
            {
                "alternatives": [
                    {"text": "trois septièmes", "provenance": "unrelated"},
                    {
                        "text": "une autre lecture",
                        "provenance": (
                            "fraction-rule:fr_FR+normalization-record:frend/curated#fr-fractions"
                        ),
                    },
                ]
            }
        ],
    }
    fields = ("strict", "presentation", "insensitive", "first_strict", "first_presentation")
    declared = {
        "id": base["id"],
        "base": {name: base[name] for name in fields},
        "head": {name: head[name] for name in fields},
        "expected_speech": "trois septièmes",
        "source_record_id": "frend/curated#fr-fractions",
    }
    witness_sha256 = hashlib.sha256(
        json.dumps([declared], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = gate.compare(
        {"cases": [base]},
        {"cases": [head]},
        expected_recoveries={
            "rows": [declared],
            "strict_total": 1,
            "insensitive_total": 1,
            "witness_sha256": witness_sha256,
        },
    )
    assert "expected-recovery-source-record" in {
        item["kind"] for item in result["correctness"]["failures"]
    }


def test_fraction_recovery_gate_validates_witness_hash():
    result = gate.compare(
        {"cases": []},
        {"cases": []},
        expected_recoveries={
            "rows": [],
            "strict_total": 0,
            "insensitive_total": 0,
            "witness_sha256": "0" * 64,
        },
    )
    assert "expected-recovery-witness-hash" in {
        item["kind"] for item in result["correctness"]["failures"]
    }


def test_allowed_change_gate_rejects_provenance_only_change_outside_routing_set():
    fraction = {
        "id": "fr_FR:fraction:1",
        "locale": "fr_FR",
        "written": "3/7",
        "strict": False,
        "presentation": False,
        "insensitive": False,
        "first_strict": False,
        "first_presentation": False,
        "first": "x",
        "offer_signature": [{"type": "number:fraction", "alternatives": []}],
        "error": None,
    }
    cardinal = {
        **fraction,
        "id": "fr_FR:cardinal:1",
        "written": "3",
        "offer_signature": [{"type": "number:decimal", "alternatives": [{"provenance": "base"}]}],
    }
    changed = json.loads(json.dumps(cardinal))
    changed["offer_signature"][0]["alternatives"][0]["provenance"] = "head"
    fields = (
        "strict",
        "presentation",
        "insensitive",
        "first",
        "first_strict",
        "first_presentation",
        "error",
        "offer_signature",
    )
    payload = {field: fraction[field] for field in fields}
    result = gate.compare(
        {"cases": [fraction, cardinal]},
        {"cases": [fraction, changed]},
        allowed_changes={
            "family": "fraction",
            "rows": [{"id": fraction["id"], "base": payload, "head": payload}],
        },
    )
    assert result["correctness"]["failures"][0]["kind"] == "change-outside-manifest"


def test_fraction_manifest_cannot_bless_changed_effective_ranking_input():
    common = {
        "id": "es_ES:fraction:1",
        "locale": "es_ES",
        "written": "2/7",
        "strict": True,
        "presentation": True,
        "insensitive": True,
        "first_strict": True,
        "first_presentation": True,
        "first": "dos séptimos",
        "error": None,
    }
    base = {
        **common,
        "offer_signature": [
            {
                "start": 0,
                "end": 3,
                "type": "fraction:flexible",
                "alternatives": [
                    {
                        "text": "dos séptimos",
                        "provenance": "base-public",
                        "prior_provenance": "legacy-measurement-key",
                        "weight": None,
                    }
                ],
            }
        ],
    }
    head = json.loads(json.dumps(base))
    head["offer_signature"][0]["alternatives"][0].update(
        provenance="fraction-rule:es_ES", prior_provenance="changed-measurement-key"
    )
    public_fields = (
        "strict",
        "presentation",
        "insensitive",
        "first",
        "first_strict",
        "first_presentation",
        "error",
        "offer_signature",
    )
    result = gate.compare(
        {"cases": [base]},
        {"cases": [head]},
        allowed_changes={
            "family": "fraction",
            "rows": [
                {
                    "id": base["id"],
                    "base": {field: base[field] for field in public_fields},
                    "head": {field: head[field] for field in public_fields},
                }
            ],
        },
    )
    assert "unchanged-reading-ranking" in {
        item["kind"] for item in result["correctness"]["failures"]
    }


def test_pr4_improvement_profile_enforces_stricter_runtime_and_fold_budgets():
    base_runtime, head_runtime, soak, _fold = _resource_documents(100, 100)
    head_runtime["summary"]["first_hit_ns"]["en_US"] = {
        "median": 1_001_000,
        "min": 1_001_000,
        "max": 1_001_000,
    }

    def fold(value):
        return {
            "data_dir": "/outside/fold-bench",
            "rows": [
                {
                    "id": "long-en_US-030",
                    "locale": "en_US",
                    "mode": "plain",
                    "timings_ns": {"resolve_k64": value},
                }
            ],
        }

    common = gate.compare(
        {"cases": []},
        {"cases": []},
        base_runtime=base_runtime,
        head_runtime=head_runtime,
        base_soak=soak,
        head_soak=soak,
        base_folds=[fold(1_000_000)],
        head_folds=[fold(847_000)],
    )
    strict = gate.compare(
        {"cases": []},
        {"cases": []},
        base_runtime=base_runtime,
        head_runtime=head_runtime,
        base_soak=soak,
        head_soak=soak,
        base_folds=[fold(1_000_000)],
        head_folds=[fold(847_000)],
        pr4_improvement=True,
    )

    assert not common["resources"]["failures"]
    assert {failure["metric"] for failure in strict["resources"]["failures"]} == {
        "runtime.first_hit_ns.en_US",
        "fold[fold-bench].resolve_k64.corpus_median_ns",
    }


def test_runtime_comparison_uses_base_prewarm_when_both_sides_have_api():
    base_runtime, head_runtime, _soak, _fold = _resource_documents(100, 100)
    base_runtime["summary"]["sequential_first_hit_total_ns"] = {
        "median": 2_000_000,
        "min": 2_000_000,
        "max": 2_000_000,
    }
    base_runtime["summary"]["prewarm_total_ns"] = {
        "median": 100_000,
        "min": 100_000,
        "max": 100_000,
    }
    head_runtime["summary"]["prewarm_total_ns"] = {
        "median": 1_000_000,
        "min": 1_000_000,
        "max": 1_000_000,
    }
    result = gate.compare(
        {"cases": []},
        {"cases": []},
        base_runtime=base_runtime,
        head_runtime=head_runtime,
    )

    failure = next(
        check
        for check in result["resources"]["failures"]
        if check["metric"] == "runtime.prewarm_total_ns_vs_base_prewarm"
    )
    assert failure["base"]["median"] == 100_000


def test_fold_row_gate_requires_ratio_and_absolute_noise_floor():
    def run(value):
        return {
            "data_dir": "/outside/fold-bench",
            "rows": [
                {
                    "id": "short-en_US-001",
                    "locale": "en_US",
                    "mode": "plain",
                    "timings_ns": {"resolve_k64": value},
                }
            ],
        }

    payload = gate.compare(
        {"cases": []},
        {"cases": []},
        base_folds=[run(100_000), run(110_000), run(90_000)],
        head_folds=[run(200_000), run(210_000), run(190_000)],
    )
    row = next(
        check for check in payload["resources"]["checks"] if check["metric"].endswith(".row")
    )
    assert row["head"]["median"] > row["ratio_limit"]
    assert row["head"]["median"] - row["base"]["median"] < row["absolute_floor_ns"]
    assert row["passed"]


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
