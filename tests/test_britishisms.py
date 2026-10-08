from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
from icukit.detectors import detect

from frend import resolve_lattice, verbalize_lattice
from frend.britishisms import _file_key, _load_britishisms_for, apply_edit_rule

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
        ("doubled-consonant", "travelled", "traveled"),
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
        "PLAIN\tcolour\tcolor\nPLAIN\tcolours\tcolors\nPLAIN\tflour\t<self>\n",
        encoding="utf-8",
    )
    document = _builder().build_document(tmp_path)
    assert document["pairs"]["colour"]["lower"] == {
        "class": "respelling",
        "target": "color",
        "converted": 4,
        "left": 1,
    }
    assert document["rules"]["our-to-or"]["lower"]["converted"] == 5
    assert document["rules"]["our-to-or"]["lower"]["eligible"] == 6
    assert document["rules"]["our-to-or"]["lower"]["stems"] == ["col"]
    assert document["rules"]["our-to-or"]["lower"]["enabled"] is True
    assert document["selection"]["classes"]["respelling"]["enabled"] is True
    assert document["selection"]["development_shards"][0] == "output-00080-of-00100"


def test_builder_crlf_rows_match_lf_counts_and_predictions(tmp_path, monkeypatch):
    builder = _builder()
    for index in range(90):
        (tmp_path / f"output-{index:05d}-of-00100").touch()

    def build(newline):
        monkeypatch.setattr(
            builder,
            "_open",
            lambda _item: io.StringIO(f"PLAIN\tcolour\tcolor{newline}"),
        )
        return builder.build_document(tmp_path)

    lf = build("\n")
    crlf = build("\r\n")

    assert builder._render(crlf).encode() == builder._render(lf).encode()
    assert crlf["pairs"]["colour"]["lower"] == {
        "class": "respelling",
        "target": "color",
        "converted": 90,
        "left": 0,
    }


def test_builder_abstains_when_top_exact_rewrites_tie(tmp_path):
    builder = _builder()
    counts = builder.Counts()
    counts.add("colour", "color")
    counts.add("colour", "colur")
    assert (
        builder._exact_prediction(
            counts,
            ("colour", "lower"),
            {"respelling": 1},
        )
        is None
    )

    for index in range(90):
        (tmp_path / f"output-{index:05d}-of-00100").write_text("", encoding="utf-8")
    (tmp_path / "output-00000-of-00100").write_text(
        "PLAIN\tcolour\tcolor\nPLAIN\tcolour\tcolur\nPLAIN\ttheatre\ttheater\n",
        encoding="utf-8",
    )
    (tmp_path / "output-00080-of-00100").write_text(
        "PLAIN\ttheatre\ttheater\n",
        encoding="utf-8",
    )
    document = builder.build_document(tmp_path)
    assert document["selection"]["admitted_classes"] == ["respelling"]
    assert "colour" not in document["pairs"]


def _shards(digest="0" * 64):
    return [
        {"relative_path": f"output-{index:05d}-of-00100", "sha256": digest} for index in range(90)
    ]


def _profile(path: Path, *, pairs, rules=None, support=1, fold=False, digest="0" * 64):
    path.write_text(
        json.dumps(
            {
                "locale": "en",
                "pairs": pairs,
                "profile": "google-tn",
                "provenance": {"source_shards": _shards(digest)},
                "rules": rules or {},
                "schema_version": 1,
                "selection": {
                    "admitted_classes": ["respelling"],
                    "case_folding": {"enabled": fold},
                    "class_minimum_support": {"respelling": support},
                    "rule_rate_threshold": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("target", ["", " ", "-"])
def test_profile_rejects_non_alphabetic_rewrite_targets(tmp_path, target):
    britishisms = tmp_path / "britishisms.json"
    _profile(
        britishisms,
        pairs={
            "colour": {
                "lower": {
                    "class": "respelling",
                    "target": target,
                    "converted": 1,
                    "left": 0,
                }
            }
        },
    )

    with pytest.raises(ValueError, match="bad pair row"):
        _load_britishisms_for(*_file_key(britishisms))


def test_profile_accepts_alphabetic_rewrite_target(tmp_path):
    britishisms = tmp_path / "britishisms.json"
    _profile(
        britishisms,
        pairs={
            "colour": {
                "lower": {
                    "class": "respelling",
                    "target": "color",
                    "converted": 1,
                    "left": 0,
                }
            }
        },
    )

    table = _load_britishisms_for(*_file_key(britishisms))

    assert table.pairs["colour"]["lower"]["target"] == "color"


@pytest.mark.parametrize("support", [1.5, "1", True])
def test_profile_rejects_non_integer_minimum_support(tmp_path, support):
    britishisms = tmp_path / "britishisms.json"
    _profile(britishisms, pairs={}, support=support)

    with pytest.raises(ValueError, match="bad selection"):
        _load_britishisms_for(*_file_key(britishisms))


def test_profile_accepts_positive_integer_minimum_support(tmp_path):
    britishisms = tmp_path / "britishisms.json"
    _profile(britishisms, pairs={}, support=1)

    table = _load_britishisms_for(*_file_key(britishisms))

    assert table.minimum_support == {"respelling": 1}


def _acronym_profile(path: Path, *, digest="0" * 64):
    path.write_text(
        json.dumps(
            {
                "locale": "en",
                "profile": "google-tn",
                "provenance": {"source_shards": _shards(digest)},
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
        pairs={
            "theatre": {
                "lower": {
                    "class": "respelling",
                    "target": "theater",
                    "converted": 4,
                    "left": 0,
                }
            }
        },
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
        pairs={
            "theatre": {
                "lower": {
                    "class": "respelling",
                    "target": "theater",
                    "converted": 5,
                    "left": 0,
                }
            }
        },
        support=5,
    )
    assert _best("theatre", "google-tn") == "theater"
    assert _best("Theatre", "google-tn") == "Theatre"


def test_case_shapes_require_their_own_evidence_unless_selected_folding_allows_it(
    tmp_path, monkeypatch
):
    acronym = tmp_path / "acronym_surfaces.json"
    britishisms = tmp_path / "britishisms.json"
    _acronym_profile(acronym)
    pairs = {
        "colour": {
            "lower": {
                "class": "respelling",
                "target": "color",
                "converted": 5,
                "left": 0,
            },
            "title": {
                "class": "respelling",
                "target": "color",
                "converted": 5,
                "left": 0,
            },
        }
    }
    _profile(britishisms, pairs=pairs)
    monkeypatch.setenv("FREND_GOOGLE_TN_PROFILE_PATH", str(acronym))
    monkeypatch.setenv("FREND_GOOGLE_TN_BRITISHISMS_PATH", str(britishisms))
    assert _best("colour", "google-tn") == "color"
    assert _best("Colour", "google-tn") == "Color"
    assert _best("COLOUR", "google-tn") == "COLOUR"

    _profile(britishisms, pairs=pairs, fold=True)
    assert _best("COLOUR", "google-tn") == "COLOR"


def test_builder_classifies_pairs_and_only_admits_dev_positive_classes(tmp_path):
    for index in range(90):
        (tmp_path / f"output-{index:05d}-of-00100").write_text("", encoding="utf-8")
    training = tmp_path / "output-00000-of-00100"
    development = tmp_path / "output-00080-of-00100"
    training.write_text(
        "PLAIN\tcolour\tcolor\nPLAIN\tcafé\tcafe\nPLAIN\tbrdg\tbridge\nPLAIN\tbarbecue\tbarbeque\n",
        encoding="utf-8",
    )
    development.write_text(
        "PLAIN\tcolour\tcolor\nPLAIN\tcafé\t<self>\nPLAIN\tbrdg\t<self>\nPLAIN\tbarbecue\t<self>\n",
        encoding="utf-8",
    )
    document = _builder().build_document(tmp_path)
    classes = document["selection"]["classes"]
    assert classes["respelling"]["enabled"] is True
    assert classes["diacritic"]["enabled"] is False
    assert classes["expansion/abbreviation"]["enabled"] is False
    assert classes["other"]["enabled"] is False
    assert list(document["pairs"]) == ["colour"]
    assert document["pairs"]["colour"]["lower"]["class"] == "respelling"


def test_profile_rule_generalizes_only_for_a_learned_stem(tmp_path, monkeypatch):
    acronym = tmp_path / "acronym_surfaces.json"
    britishisms = tmp_path / "britishisms.json"
    _acronym_profile(acronym)
    _profile(
        britishisms,
        pairs={},
        rules={
            "our-to-or": {
                "lower": {
                    "converted": 3,
                    "eligible": 4,
                    "rate": 0.75,
                    "enabled": True,
                    "stems": ["col"],
                }
            }
        },
    )
    monkeypatch.setenv("FREND_GOOGLE_TN_PROFILE_PATH", str(acronym))
    monkeypatch.setenv("FREND_GOOGLE_TN_BRITISHISMS_PATH", str(britishisms))
    assert _best("colours", "google-tn") == "colors"
    assert _best("flour", "google-tn") == "flour"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("converted", None),
        ("converted", "3"),
        ("converted", True),
        ("eligible", None),
        ("eligible", 4.0),
        ("eligible", False),
        ("rate", None),
        ("rate", "0.75"),
        ("rate", 1),
        ("stems", None),
        ("stems", "col"),
    ],
)
def test_profile_rejects_enabled_rule_with_missing_or_mistyped_audit_field(
    tmp_path, field, bad_value
):
    britishisms = tmp_path / "britishisms.json"
    row = {
        "converted": 3,
        "eligible": 4,
        "rate": 0.75,
        "enabled": True,
        "stems": ["col"],
    }
    if bad_value is None:
        del row[field]
    else:
        row[field] = bad_value
    _profile(
        britishisms,
        pairs={},
        rules={"our-to-or": {"lower": row}},
    )

    with pytest.raises(ValueError, match="bad rule"):
        _load_britishisms_for(*_file_key(britishisms))


def test_profile_tables_must_name_identical_source_shard_digests(tmp_path, monkeypatch):
    acronym = tmp_path / "acronym_surfaces.json"
    britishisms = tmp_path / "britishisms.json"
    _acronym_profile(acronym, digest="0" * 64)
    _profile(britishisms, pairs={}, digest="1" * 64)
    monkeypatch.setenv("FREND_GOOGLE_TN_PROFILE_PATH", str(acronym))
    monkeypatch.setenv("FREND_GOOGLE_TN_BRITISHISMS_PATH", str(britishisms))
    with pytest.raises(ValueError, match="source-shard digests do not match"):
        _best("colour", "google-tn")


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
