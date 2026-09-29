"""Regressions for the eight confirmed findings in the p8a locale review."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from icukit.detectors import Capture, NumberValue

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"


def _tool(name: str):
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_latency_worker_imports_the_selected_subject(tmp_path):
    """F1: the baseline worker must not import this candidate's tools or package."""
    benchmark = _tool("benchmark_latency")
    baseline = _REPO.parent / "frend"
    output = tmp_path / "baseline.json"
    args = argparse.Namespace(
        subject_root=baseline,
        expected_head=subprocess.check_output(
            ["git", "-C", str(baseline), "rev-parse", "HEAD"], text=True
        ).strip(),
        manifest=_TOOLS / "latency_inputs-v1.json",
        locales="en_US",
        max_words=10,
        warmup=0,
        runs=1,
        seed=1,
        output=output,
    )
    assert benchmark._run(args) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert Path(receipt["imports"]["reading_profile"]).is_relative_to(baseline)
    assert Path(receipt["imports"]["frend"]).is_relative_to(baseline)


def test_evaluator_rows_require_a_verified_input():
    """F2: a raw caller-selected path cannot cross the evaluator's row-opening boundary."""
    evaluator = _tool("evaluate_google_tn")
    rogue = Path("/wrong-store/output-00095-of-00100")
    with patch.object(Path, "open", return_value=io.StringIO("PLAIN\tFAKE\t<self>\n<eos>\n")):
        with pytest.raises((TypeError, ValueError), match="verified"):
            evaluator._rows(rogue)


def test_forged_receipt_cannot_launder_internal_ancestry():
    """F3: parent metadata and an unresolved fingerprint cannot hide internal leaves."""
    from corpus_inputs import VerifiedInput
    from corpus_receipts import DerivationReceipt, derive_receipt

    from frend.data_sources import shipping_refusals

    digest = "5d1b84c18b57c62e2c369ce93b3fdc607a2487de17acb74ec72503e242300034"
    first = derive_receipt(
        inputs=(VerifiedInput("icu/78.3/cldr", "probe", digest, "shippable"),),
        producer_commit="a",
        command=("one",),
        artifacts={},
        shipping=False,
    )
    forged = DerivationReceipt(
        first.inputs,
        first.ancestors,
        first.producer_commit,
        first.command,
        first.artifacts,
        "shippable",
        first.fingerprint,
    )
    with pytest.raises(ValueError, match="fingerprint|license class"):
        derive_receipt(
            inputs=(forged,),
            producer_commit="b",
            command=("two",),
            artifacts={},
            shipping=True,
        )
    assert shipping_refusals({"ancestors": [first.fingerprint]})


@pytest.mark.parametrize("locale", ["ru_RU", "es_ES"])
def test_non_english_profiles_reject_english_written_forms(locale):
    """F4: U.S. citations and English era spellings are not Russian or Spanish forms."""
    from frend.written_forms import WrittenFormsDetector

    reading_profile = _tool("reading_profile")
    detector = next(
        item
        for item in reading_profile.reading_detectors(locale)
        if isinstance(item, WrittenFormsDetector)
    )
    assert detector.detect("339 U.S. 629 and 500 B.C.") == []


def test_spanish_era_does_not_fabricate_letter_names():
    """F5: an unavailable locale letter spelling is omitted, leaving ICU's wide name."""
    from frend.verbalize import _era_names

    alternatives = _era_names(
        1,
        {"captures": (SimpleNamespace(name="era", text="AD", form="short"),)},
        "es_ES",
    )
    assert all(item.provenance != "surface:letters" for item in alternatives)
    assert alternatives


def test_english_em_dash_population_is_frozen_for_builder_and_evaluator():
    """F6: the frozen English builder and evaluator both exclude an em-dash triple."""
    rows = _tool("google_tn_rows")
    sentences = [[("CARDINAL", "5", "five"), ("PUNCT", "—", "sil"), ("CARDINAL", "10", "ten")]]
    assert list(rows.range_triples(sentences)) == []
    assert rows.running_text(sentences) == []


def test_latency_comparison_refuses_missing_locale_results():
    """F7: an English-only receipt cannot bypass the Russian and Spanish gates."""
    benchmark = _tool("benchmark_latency")
    identity = {
        key: "same"
        for key in (
            "hostname",
            "machine",
            "machine_model",
            "os",
            "cpu",
            "python",
            "icu",
            "pyicu",
            "icukit",
            "tiergraph",
            "power_mode",
        )
    }
    english = {
        "manifest_sha256": "x",
        "environment": identity,
        "warmup": 1,
        "runs": 3,
        "seed": 1,
        "max_words": 10,
        "denominators": {"en_US": 3},
        "summaries": {
            "en_US": {
                name: {"p50": 1, "p90": 1, "p99": 1}
                for name in ("end_to_end_ns", "resolve_only_ns")
            }
        },
    }

    class Receipt:
        def __init__(self, value):
            self.value = value

        def read_text(self, **_kwargs):
            return json.dumps(self.value)

    args = argparse.Namespace(
        compare=(Receipt(english), Receipt(copy.deepcopy(english))),
        english_max_regression=0.05,
    )
    with pytest.raises(SystemExit, match="ru_RU|Russian"):
        benchmark._compare(args)


def test_icukit_range_conversion_requires_the_resolver_locale(monkeypatch):
    """F8: endpoint and percent readers use an explicit locale, never an English default."""
    from frend import ranges

    detection = {
        "type": "number:range",
        "text": "5–10",
        "captures": (
            Capture("start", 0, 1, "5", NumberValue("5", None)),
            Capture("separator", 1, 2, "–"),
            Capture("end", 2, 4, "10", NumberValue("10", None)),
        ),
    }
    seen = []
    monkeypatch.setattr(
        ranges,
        "_read_whole",
        lambda text, kind, locale: (seen.append(locale), None)[1],
    )
    ranges.from_icukit(detection, "ru_RU")
    assert seen and set(seen) == {"ru_RU"}
    with pytest.raises(TypeError):
        ranges.from_icukit(detection)
