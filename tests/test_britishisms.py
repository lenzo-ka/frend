from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from icukit.detectors import detect

from frend import resolve_lattice, verbalize_lattice
from frend.britishisms import apply_edit_rule

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _builder():
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(
        "build_britishisms", _TOOLS / "build_britishisms.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("rule", "written", "expected"),
    [
        ("our-to-or", "colours", "colors"),
        ("ise-to-ize", "organising", "organizing"),
        ("isation-to-ization", "organisation", "organization"),
        ("re-to-er", "theatre", "theater"),
        ("ogue-to-og", "synagogue", "synagog"),
        ("mme-to-m", "programme", "program"),
        ("doubled-l", "travelled", "traveled"),
    ],
)
def test_each_edit_type_changes_its_fixture(rule, written, expected):
    assert apply_edit_rule(rule, written) == expected


def test_builder_counts_converted_and_left_and_validates_on_80_89(tmp_path):
    for index in range(90):
        (tmp_path / f"output-{index:05d}-of-00100").write_text("", encoding="utf-8")
    training = tmp_path / "output-00000-of-00100"
    development = tmp_path / "output-00080-of-00100"
    training.write_text(
        "PLAIN\tcolour\tcolor\n" * 3 + "PLAIN\tcolour\t<self>\n" + "PLAIN\tflour\t<self>\n",
        encoding="utf-8",
    )
    development.write_text(
        "PLAIN\tcolours\tcolors\nPLAIN\tflour\t<self>\n",
        encoding="utf-8",
    )
    document = _builder().build_document(tmp_path)
    assert document["pairs"]["colour"] == {
        "target": "color",
        "converted": 3,
        "left": 1,
    }
    assert document["rules"]["our-to-or"]["converted"] == 3
    assert document["rules"]["our-to-or"]["eligible"] == 4
    assert document["rules"]["our-to-or"]["stems"] == ["col"]
    assert document["rules"]["our-to-or"]["enabled"] is True
    assert document["selection"]["development_shards"][0] == "output-00080-of-00100"


def _profile(path: Path, *, pairs, rules=None, support=1):
    path.write_text(
        json.dumps(
            {
                "locale": "en",
                "pairs": pairs,
                "profile": "google-tn",
                "provenance": {
                    "source_shards": [
                        {
                            "relative_path": f"output-{index:05d}-of-00100",
                            "sha256": "0" * 64,
                        }
                        for index in range(90)
                    ]
                },
                "rules": rules or {},
                "schema_version": 1,
                "selection": {
                    "minimum_support": support,
                    "rule_rate_threshold": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )


def _acronym_profile(path: Path):
    path.write_text(
        json.dumps(
            {
                "locale": "en",
                "profile": "google-tn",
                "provenance": {
                    "source_shards": [
                        {"relative_path": "output-00000-of-00100", "sha256": "0" * 64}
                    ]
                },
                "schema_version": 1,
                "selection": {"minimum_support": 1, "parent_strength": 1},
                "surfaces": {},
            }
        ),
        encoding="utf-8",
    )


def _best(text: str, profile=None) -> str:
    lattice = resolve_lattice(list(detect(text, [])), source_text=text)
    units = verbalize_lattice(lattice, profile=profile).best_path.units
    return "".join(unit.best.text for unit in units)


def test_profile_off_is_byte_identical_and_profile_on_respects_support(tmp_path, monkeypatch):
    from frend import verbalize

    acronym = tmp_path / "acronym_surfaces.json"
    britishisms = tmp_path / "britishisms.json"
    _acronym_profile(acronym)
    _profile(
        britishisms,
        pairs={"theatre": {"target": "theater", "converted": 4, "left": 0}},
        support=5,
    )
    monkeypatch.setenv("FREND_GOOGLE_TN_PROFILE_PATH", str(acronym))
    monkeypatch.setenv("FREND_GOOGLE_TN_BRITISHISMS_PATH", str(britishisms))
    with monkeypatch.context() as guard:
        guard.setattr(
            verbalize,
            "load_britishisms",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("profile table consulted")),
        )
        assert _best("Theatre").encode() == b"Theatre"
    assert _best("Theatre", "google-tn") == "Theatre"
    _profile(
        britishisms,
        pairs={"theatre": {"target": "theater", "converted": 5, "left": 0}},
        support=5,
    )
    assert _best("Theatre", "google-tn") == "Theater"


def test_profile_rule_generalizes_only_for_a_learned_stem(tmp_path, monkeypatch):
    acronym = tmp_path / "acronym_surfaces.json"
    britishisms = tmp_path / "britishisms.json"
    _acronym_profile(acronym)
    _profile(
        britishisms,
        pairs={},
        rules={
            "our-to-or": {
                "converted": 3,
                "eligible": 4,
                "rate": 0.75,
                "enabled": True,
                "stems": ["col"],
            }
        },
    )
    monkeypatch.setenv("FREND_GOOGLE_TN_PROFILE_PATH", str(acronym))
    monkeypatch.setenv("FREND_GOOGLE_TN_BRITISHISMS_PATH", str(britishisms))
    assert _best("colours", "google-tn") == "colors"
    assert _best("flour", "google-tn") == "flour"


def test_builder_refuses_held_out_input(tmp_path):
    from corpus_inputs import VerifiedInput

    shard = tmp_path / "output-00095-of-00100"
    shard.write_text("PLAIN\tcolour\tcolor\n", encoding="utf-8")
    item = VerifiedInput("source", shard.name, "unused", "license", shard)
    with pytest.raises(ValueError, match="training shards 00-89.*output-00095"):
        _builder().build_document(tmp_path, inputs=[item])


def test_profile_output_must_be_outside_repository(tmp_path, monkeypatch):
    builder = _builder()
    destination = Path(__file__).resolve().parents[1] / "frend" / "data" / "britishisms.json"
    monkeypatch.setattr(
        builder,
        "_default_corpus_dir",
        lambda: (_ for _ in ()).throw(AssertionError("corpus must not be opened")),
    )
    with pytest.raises(ValueError, match="--profile-out must be outside"):
        builder.main(
            [
                "--locale",
                "en_US",
                "--source-id",
                "google/tn-en_with_types",
                "--pool",
                "training",
                "--receipt",
                str(tmp_path / "receipt.json"),
                "--profile-out",
                str(destination),
            ]
        )
