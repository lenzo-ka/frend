"""Small fixtures for the S0 miss classifier."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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
        "readings": (),
        "detection_spans": (),
        "detection_types": (),
        "exact_detection_types": (),
        "corpus_non_reading": False,
        "verbalizer_exception": False,
        "leaf_texts": (),
    }
    fields.update(changes)
    return module.MissEvidence(**fields)


def test_classifier_partitions_d_s_r_a_p_o_and_e():
    triage = _triage()
    assert triage.classify_miss(_evidence(triage)).miss_class == "D"
    wrong_span = _evidence(
        triage,
        detection_spans=((0, 3),),
        detection_types=("number:decimal",),
    )
    assert triage.classify_miss(wrong_span).miss_class == "S"
    target = "january second two thousand three"
    assert triage.classify_miss(_evidence(triage, readings=("wrong", target))).miss_class == "R"
    artifact = _evidence(
        triage,
        corpus_non_reading=True,
        detection_spans=((0, 6),),
    )
    assert triage.classify_miss(artifact).miss_class == "A"
    respelling = _evidence(
        triage,
        corpus_class="PLAIN",
        written="Theatre",
        expected="theater",
        normalized_surface="theatre",
    )
    assert triage.classify_miss(respelling).miss_class == "P"
    other_plain = _evidence(
        triage,
        corpus_class="PLAIN",
        written="pdf",
        expected="p d f",
        normalized_surface="pdf",
    )
    assert triage.classify_miss(other_plain).miss_class == "O"
    exception = _evidence(triage, verbalizer_exception=True, detection_spans=((0, 6),))
    assert triage.classify_miss(exception).miss_class == "E"


def test_r_is_derived_from_bounded_readings_and_cap_is_reported():
    triage = _triage()
    alternatives = tuple(
        SimpleNamespace(text=f"reading {index}", provenance="test") for index in range(65)
    )
    path = SimpleNamespace(units=(SimpleNamespace(alternatives=alternatives),))
    readings, capped = triage._bounded_readings(path)
    normalized = tuple(item.strip() for item in readings)
    assert capped is True
    assert len(normalized) == 64
    assert (
        triage.classify_miss(
            _evidence(triage, expected="reading 63", readings=normalized)
        ).miss_class
        == "R"
    )
    assert (
        triage.classify_miss(
            _evidence(
                triage,
                expected="reading 64",
                readings=normalized,
                detection_spans=((0, 6),),
                exact_detection_types=("date:flexible",),
            )
        ).miss_class
        == "V"
    )


def test_ruling_a_indexes_reuse_the_range_training_verdict(monkeypatch):
    triage = _triage()
    import build_range_priors

    sentence = (
        ("CARDINAL", "1994", "one thousand nine hundred ninety four"),
        ("PUNCT", "-", "sil"),
        ("CARDINAL", "95", "ninety five"),
    )
    monkeypatch.setattr(
        build_range_priors,
        "credit",
        lambda *_rows: ("punctuation_dash", None, None),
    )
    assert triage._corpus_non_reading_indexes(sentence) == frozenset({0, 2})


def test_template_recombination_preserves_order_and_multiplicity():
    triage = _triage()
    assert triage._can_recombine("three dollars and fifty cents", ("three dollars", "fifty cents"))
    assert not triage._can_recombine("fifty cents three dollars", ("three dollars", "fifty cents"))
    assert not triage._can_recombine("one one", ("one",))
    assert not triage._can_recombine("one", ("one two",))


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
