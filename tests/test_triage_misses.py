"""Small fixtures for the S0 miss classifier."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _triage():
    name = "triage_misses"
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _evidence(module, **changes):
    fields = {
        "corpus_class": "DATE",
        "written": "1/2/03",
        "expected": "january second two thousand three",
        "normalized_surface": "1 2 03",
        "expected_offered": False,
        "detection_spans": (),
        "detection_types": (),
        "exact_detection_types": (),
        "leaf_texts": (),
    }
    fields.update(changes)
    return module.MissEvidence(**fields)


def test_classifier_partitions_d_s_r_and_a():
    triage = _triage()
    assert triage.classify_miss(_evidence(triage)).miss_class == "D"
    wrong_span = _evidence(
        triage,
        detection_spans=((0, 3),),
        detection_types=("number:decimal",),
    )
    assert triage.classify_miss(wrong_span).miss_class == "S"
    assert triage.classify_miss(_evidence(triage, expected_offered=True)).miss_class == "R"
    artifact = _evidence(
        triage,
        corpus_class="PLAIN",
        written="Theatre",
        expected="theater",
        normalized_surface="theatre",
    )
    assert triage.classify_miss(artifact).miss_class == "A"


def test_v_family_and_fix_kinds():
    triage = _triage()
    template = _evidence(
        triage,
        corpus_class="MONEY",
        written="$3.50",
        expected="three dollars and fifty cents",
        normalized_surface="3 50",
        detection_spans=((0, 5),),
        detection_types=("currency:flexible",),
        exact_detection_types=("currency:flexible",),
        leaf_texts=("three dollars", "fifty cents"),
    )
    assert triage.classify_miss(template) == triage.Classification("V", "money", "template-fixable")

    recognition = _evidence(
        triage,
        corpus_class="MEASURE",
        written="57 ch",
        expected="fifty seven chains",
        normalized_surface="57 ch",
        detection_spans=((0, 5),),
        detection_types=("letters:token",),
        exact_detection_types=("letters:token",),
    )
    assert triage.classify_miss(recognition) == triage.Classification(
        "V", "measure", "recognition change"
    )

    entry = _evidence(
        triage,
        corpus_class="FRACTION",
        written="3/4",
        expected="three quarters",
        normalized_surface="3 4",
        detection_spans=((0, 3),),
        detection_types=("fraction:flexible",),
        exact_detection_types=("fraction:flexible",),
        leaf_texts=("three fourths", "three over four"),
    )
    assert triage.classify_miss(entry) == triage.Classification("V", "fraction", "new entry")
