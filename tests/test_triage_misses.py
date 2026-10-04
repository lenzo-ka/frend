"""Small fixtures for the S0 miss classifier."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        "verbalizer_exception": False,
        "leaf_texts": (),
    }
    fields.update(changes)
    return module.MissEvidence(**fields)


def test_classifier_partitions_d_s_r_v_p_o_and_e():
    triage = _triage()
    examples = [_evidence(triage)]
    wrong_span = _evidence(
        triage,
        detection_spans=((0, 3),),
        detection_types=("number:decimal",),
    )
    examples.append(wrong_span)
    target = "january second two thousand three"
    examples.append(_evidence(triage, readings=("wrong", target)))
    examples.append(
        _evidence(
            triage,
            detection_spans=((0, 6),),
            detection_types=("date:flexible",),
            exact_detection_types=("date:flexible",),
        )
    )
    respelling = _evidence(
        triage,
        corpus_class="PLAIN",
        written="Theatre",
        expected="theater",
        normalized_surface="theatre",
    )
    examples.append(respelling)
    other_plain = _evidence(
        triage,
        corpus_class="PLAIN",
        written="pdf",
        expected="p d f",
        normalized_surface="pdf",
    )
    examples.append(other_plain)
    exception = _evidence(triage, verbalizer_exception=True, detection_spans=((0, 6),))
    examples.append(exception)

    classes = tuple(triage.classify_miss(evidence).miss_class for evidence in examples)
    assert classes == triage.MISS_CLASSES == ("D", "S", "R", "V", "P", "O", "E")


def test_verbalizer_exception_surface_match_counts_as_e_miss(monkeypatch):
    triage = _triage()

    def fail_verbalization(*_args, **_kwargs):
        raise RuntimeError("fixture verbalizer failure")

    monkeypatch.setattr("frend.verbalize.verbalize_lattice", fail_verbalization)
    sentence = (("PLAIN", "Surface", "<self>"),)
    result = triage._score_sentence(("shard", 0, sentence, None, None))

    assert result["first"] == 0
    assert result["any"] == 0
    assert result["sentence_first"] is False
    assert result["miss_counts"] == {"E": 1}


def test_leaf_text_exception_counts_as_e_miss(monkeypatch):
    triage = _triage()
    alternative = SimpleNamespace(text="wrong", provenance="test")
    unit = SimpleNamespace(best=alternative, alternatives=(alternative,))
    path = SimpleNamespace(units=(unit,))
    verbalized = SimpleNamespace(best_path=path, paths=(path,))

    monkeypatch.setattr(triage, "_detectors", lambda locale: ())
    monkeypatch.setattr(
        "icukit.detectors.detect",
        lambda written, detectors: ({"start": 0, "end": 3, "type": "fraction:flexible"},),
    )
    monkeypatch.setattr(
        "frend.verbalize.verbalize_lattice",
        lambda lattice, context, profile: verbalized,
    )

    def fail_leaf_texts(detections, written):
        raise RuntimeError("leaf failure")

    monkeypatch.setattr(triage, "_leaf_texts", fail_leaf_texts)
    result = triage._score_token(("FRACTION", "3/4", "three quarters"), "", "")

    assert result[3] == triage.Classification("E")


def test_file_not_found_is_a_setup_error(monkeypatch):
    triage = _triage()

    monkeypatch.setattr(triage, "_detectors", lambda locale: ())

    def missing_corpus(written, detectors):
        raise FileNotFoundError("missing corpus")

    monkeypatch.setattr("icukit.detectors.detect", missing_corpus)

    with pytest.raises(FileNotFoundError, match="missing corpus"):
        triage._score_token(("PLAIN", "surface", "<self>"), "", "")


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
