"""Behavior schema v1 regression and characterization contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

import frend
from frend import BehaviorLoadError, SpokenAlternative
from frend.behavior import resolve_behavior, shipped_behaviors
from frend.profiles import validate_groups
from frend.verbalize import verbalize_lattice

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("tests/data/behaviors")
GOLDEN = ROOT / "tests/data/normalize_golden.json"
GOLDEN_PROVENANCE = ROOT / "tests/data/normalize_golden.provenance.json"


def _golden_generator():
    path = ROOT / "tests/generate_normalize_golden.py"
    spec = importlib.util.spec_from_file_location("generate_normalize_golden", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _golden_texts() -> list[str]:
    return [row["text"] for row in json.loads(GOLDEN.read_text(encoding="utf-8"))["rows"]]


def _names(resolved) -> list[str]:
    return [member.name for member in resolved.members]


def _error(refs, *, search=()) -> BehaviorLoadError:
    with pytest.raises(BehaviorLoadError) as caught:
        resolve_behavior(refs, search=search)
    return caught.value


def _document(name: str, **updates) -> dict[str, object]:
    value = {
        "schema_version": 1,
        "kind": "behavior-schema",
        "name": name,
        "version": 1,
        "extends": [],
        "provenance": {"source": "frend tests"},
        "sections": {},
    }
    value.update(updates)
    return value


def test_g4_no_schema_normalization_matches_the_baseline_byte_for_byte():
    generator = _golden_generator()
    assert len(generator.input_rows()) == 58
    assert generator.encoded_document() == GOLDEN.read_bytes()
    assert json.loads(GOLDEN_PROVENANCE.read_text(encoding="utf-8")) == {
        "frend_commit": "45badcdf5ccdc5f677914bc41c3a9a31134412d4",
        "icukit_commit": "79852b5f69f7fa3a9547a53ec992955d132eb841",
        "command": (
            "BASE=$(mktemp -d) && git -C $FREND_CHECKOUT archive "
            "45badcdf5ccdc5f677914bc41c3a9a31134412d4 | tar -x -C $BASE && "
            "cd $BASE && PYTHONPATH=. $FREND_BEHAVIOR_WORKTREE/.venv/bin/python -B "
            "$FREND_BEHAVIOR_WORKTREE/tests/generate_normalize_golden.py "
            "$FREND_BEHAVIOR_WORKTREE/tests/data/normalize_golden.json"
        ),
    }


def test_c1_empty_resolution_is_neutral_for_every_golden_input():
    resolved = resolve_behavior([])
    assert resolved.kwargs == {}
    assert resolved.icukit_sections == ()
    for text in _golden_texts():
        assert frend.normalize(text, **resolved.kwargs) == frend.normalize(text)
        assert repr(frend.normalize(text, offsets=True, **resolved.kwargs)) == repr(
            frend.normalize(text, offsets=True)
        )


def test_c2_groups_kwarg_selects_the_named_symbol_run_reading():
    groups = {"tts-sanity": ["named", "described", "silent"]}
    assert frend.normalize("****", fold=None, groups=groups).strip() == (
        "asterisk asterisk asterisk asterisk"
    )


def test_c3_groups_reorder_all_symbol_run_alternatives_without_dropping_any():
    lattice = frend.resolve_lattice(source_text="****", fold=None)
    result = verbalize_lattice(lattice, groups={"tts-sanity": ["named", "silent", "described"]})
    alternatives = result.best_path.units[0].alternatives
    assert [item.text for item in alternatives] == [
        "asterisk asterisk asterisk asterisk",
        "",
        "line of asterisk",
    ]
    assert len(alternatives) == 3
    assert {item.group for item in alternatives} == {"tts-sanity"}


def test_c4_preset_order_controls_group_winner(monkeypatch):
    import frend.verbalize as verbalize_module

    monkeypatch.setattr(verbalize_module, "_google_tn_britishisms", lambda **_kwargs: None)
    monkeypatch.setattr(verbalize_module, "_acronym_surface_priors", lambda **_kwargs: None)
    search = [DATA / "user"]
    named = resolve_behavior(["google-tn", "screen-reader-symbols"], search=search)
    silent = resolve_behavior(["screen-reader-symbols", "google-tn"], search=search)

    assert named.kwargs == {
        "profile": "google-tn",
        "groups": {"tts-sanity": ("named", "described", "silent")},
    }
    assert silent.kwargs == {
        "groups": {"tts-sanity": ("silent", "described", "named")},
        "profile": "google-tn",
    }
    assert named.setters["groups.tts-sanity"] == "screen-reader-symbols"
    assert silent.setters["groups.tts-sanity"] == "google-tn"
    assert frend.normalize("****", fold=None, **named.kwargs).strip() == (
        "asterisk asterisk asterisk asterisk"
    )
    assert frend.normalize("****", fold=None, **silent.kwargs).strip() == ""


@pytest.mark.parametrize(
    ("refs", "search", "message"),
    [
        (
            [DATA / "invalid/version-2.json"],
            (),
            "INVALID_SCHEMA_VERSION: tests/data/behaviors/invalid/version-2.json: "
            "top-level field 'schema_version' must be integer 1, got 2",
        ),
        (
            [DATA / "invalid/typo-key.json"],
            (),
            "INVALID_KEY: tests/data/behaviors/invalid/typo-key.json: sections.frend: "
            "unknown key 'grups'; known keys: case_variant_lookup, fold, groups, "
            "max_input_chars, max_unit_chars, profile, symbol_run_threshold",
        ),
        (
            [DATA / "invalid/typo-section.json"],
            (),
            "INVALID_KEY: tests/data/behaviors/invalid/typo-section.json: sections: "
            "unknown section 'icukt'; known sections: frend, icukit",
        ),
        (
            [DATA / "invalid/google-tn.json"],
            (),
            "NAME_COLLISION: behavior 'google-tn' is shipped with frend and also found "
            "at tests/data/behaviors/invalid/google-tn.json; give your file another name "
            'and use "extends": ["google-tn"]',
        ),
        (
            [DATA / "invalid/bad-profile.json"],
            (),
            "INVALID_VALUE: tests/data/behaviors/invalid/bad-profile.json: "
            "sections.frend.profile: unknown conformance profile 'googl-tn'; known "
            "profiles: 'google-tn'",
        ),
        (
            [DATA / "invalid/big.json"],
            (),
            "TOO_LARGE: tests/data/behaviors/invalid/big.json exceeds 65536 bytes; "
            "behavior files are capped at 65536 bytes",
        ),
        (
            [DATA / "invalid/loose.json"],
            (),
            "BOUND_LOOSENED: tests/data/behaviors/invalid/loose.json: "
            "sections.frend.max_unit_chars=100000 exceeds the default 8192; a behavior "
            "file may only tighten bounds; pass max_unit_chars= explicitly to raise it",
        ),
        (
            ["cycle-a"],
            [DATA / "invalid"],
            "EXTENDS_CYCLE: cycle-a -> cycle-b -> cycle-a",
        ),
        (
            [DATA / "invalid/mine.json"],
            (),
            "NAME_MISMATCH: tests/data/behaviors/invalid/mine.json declares name 'ours'; "
            "the file must be named ours.json",
        ),
        (
            ["z"],
            [DATA / "inconsistent"],
            "EXTENDS_INCONSISTENT: 'x' lists 'a' before 'b' and 'y' lists 'b' before "
            "'a'; no single order satisfies both",
        ),
        (
            [DATA / "p/a.json", DATA / "q/a.json"],
            (),
            "NAME_COLLISION: behavior 'a' is declared by both "
            "tests/data/behaviors/p/a.json and tests/data/behaviors/q/a.json with "
            "different contents; one name denotes one document",
        ),
        (
            [DATA / "invalid/double-parent.json"],
            (),
            "INVALID_EXTENDS: tests/data/behaviors/invalid/double-parent.json: extends "
            "lists 'a' twice; name each parent once",
        ),
    ],
)
def test_c6_r1_through_r12_are_exact(refs, search, message):
    assert str(_error(refs, search=search)) == message


def test_c6_one_load_phase_collects_all_three_refusals():
    error = _error([DATA / "invalid/three-faults.json"])
    assert str(error) == (
        "INVALID_KEY: tests/data/behaviors/invalid/three-faults.json: unknown top-level "
        "key 'unexpected'; INVALID_SCHEMA_VERSION: "
        "tests/data/behaviors/invalid/three-faults.json: top-level field "
        "'schema_version' must be integer 1, got 2; INVALID_PROVENANCE: "
        "tests/data/behaviors/invalid/three-faults.json: provenance.source must be a "
        "non-empty string"
    )
    assert len(error.refusals) == 3


def test_c7_strict_parser_and_envelope_edges(tmp_path):
    invalid = DATA / "conformance/envelope/invalid"
    for name in ["duplicate-key.json", "nan.json", "depth.json", "values.json", "string.json"]:
        assert _error([invalid / name]).refusals[0].code == "INVALID_JSON"

    bool_path = tmp_path / "bool-version.json"
    bool_path.write_text(json.dumps(_document("bool-version", schema_version=True)))
    assert _error([bool_path]).refusals[0].code == "INVALID_SCHEMA_VERSION"
    assert _error([DATA / "invalid/mine.json"]).refusals[0].code == "NAME_MISMATCH"
    assert _error([DATA / "invalid/double-parent.json"]).refusals[0].code == "INVALID_EXTENDS"

    collision = _error(["search-collision"], search=[DATA / "p", DATA / "q"])
    assert collision.refusals[0].code == "NAME_COLLISION"
    declared_twice = _error([DATA / "p/a.json", DATA / "q/a.json"])
    assert declared_twice.refusals[0].code == "NAME_COLLISION"

    same = resolve_behavior([DATA / "p/same.json", DATA / "p/same.json"])
    assert _names(same) == ["same"]
    neutral = resolve_behavior([DATA / "conformance/envelope/valid/empty.json"])
    assert neutral.kwargs == {}


@pytest.mark.parametrize("groups", [None, []])
def test_c7_non_mapping_groups_are_invalid_values(tmp_path, groups):
    path = tmp_path / "bad-groups.json"
    path.write_text(
        json.dumps(
            _document(
                "bad-groups",
                sections={"frend": {"schema_version": 1, "groups": groups}},
            )
        )
    )

    assert _error([path]).refusals[0].code == "INVALID_VALUE"


def test_c7_exponent_overflow_is_invalid_json_inside_opaque_sections(tmp_path):
    path = tmp_path / "overflow.json"
    document = json.dumps(
        _document(
            "overflow",
            sections={"icukit": {"nested": {"values": [0.0]}}},
        )
    ).replace("0.0", "1e999")
    path.write_text(document)

    assert _error([path]).refusals[0].code == "INVALID_JSON"


@pytest.mark.parametrize("user_first", [False, True])
def test_c7_shipped_name_collision_precedes_identical_digest_deduplication(tmp_path, user_first):
    user = tmp_path / "google-tn.json"
    user.write_bytes((ROOT / "frend/behaviors/google-tn.json").read_bytes())
    refs = [user, "google-tn"] if user_first else ["google-tn", user]

    assert _error(refs).refusals[0].code == "NAME_COLLISION"


def test_c8_linearization_examples_and_canonical_digest(tmp_path):
    chain = [DATA / "chain"]
    graph = [DATA / "graph"]
    assert _names(resolve_behavior(["c", "a"], search=chain)) == ["b", "c", "a"]
    diamond = resolve_behavior(["d"], search=graph)
    assert _names(diamond) == ["a", "b", "c", "d"]
    assert diamond.kwargs == {"profile": "google-tn", "fold": None}
    assert _names(resolve_behavior(["b", "a"], search=graph)) == ["b", "a"]
    assert _names(resolve_behavior(["a", "b"], search=graph)) == ["a", "b"]
    layered = resolve_behavior(["d", "c"], search=graph)
    assert _names(layered) == ["b", "d", "a", "c"]
    assert layered.kwargs == {"profile": None, "fold": None}
    assert _error(["z"], search=[DATA / "inconsistent"]).refusals[0].code == (
        "EXTENDS_INCONSISTENT"
    )
    assert _error([DATA / "invalid/double-parent.json"]).refusals[0].code == ("INVALID_EXTENDS")

    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first = _document("canonical", sections={"frend": {"schema_version": 1}})
    second = dict(reversed(first.items()))
    (left / "canonical.json").write_text(json.dumps(first, indent=4))
    (right / "canonical.json").write_text(json.dumps(second, separators=(",", ":")))
    one = resolve_behavior([left / "canonical.json"])
    two = resolve_behavior([right / "canonical.json"])
    together = resolve_behavior([left / "canonical.json", right / "canonical.json"])
    assert one.members[0].digest == two.members[0].digest
    assert one.digest == two.digest == together.digest
    assert _names(together) == ["canonical"]


def test_c9_only_google_tn_ships_and_resolves_exactly():
    assert shipped_behaviors() == ("google-tn",)
    resolved = resolve_behavior(["google-tn"])
    assert resolved.kwargs == {
        "profile": "google-tn",
        "groups": {"tts-sanity": ("silent", "described", "named")},
    }


@pytest.mark.parametrize("owner", ["envelope", "frend"])
def test_c10_conformance_manifests(owner):
    root = DATA / "conformance" / owner
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    for relative, contract in manifest.items():
        path = root / relative
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == contract["sha256"], relative
        if contract["expected"] == "valid":
            resolved = resolve_behavior([path])
            raw = json.loads(path.read_text(encoding="utf-8"))
            if "icukit" in raw["sections"]:
                assert dict(resolved.icukit_sections[0].mapping) == raw["sections"]["icukit"]
        else:
            assert _error([path]).refusals[0].code == contract["expected"]


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (
            {"tts-santy": ["described", "named", "silent"]},
            "unknown verbalization group 'tts-santy'; known groups: 'tts-sanity'",
        ),
        (
            {"tts-sanity": ["described", "named", "loud"]},
            "group 'tts-sanity' order must list each of 'described', 'named', 'silent' "
            "exactly once",
        ),
        (
            {"tts-sanity": ["named"]},
            "group 'tts-sanity' order must list each of 'described', 'named', 'silent' "
            "exactly once",
        ),
    ],
)
def test_c11_direct_groups_errors_are_exact(groups, message):
    with pytest.raises(ValueError, match="^" + __import__("re").escape(message) + "$"):
        validate_groups(groups)


def test_c12_weighted_supplement_stays_first_and_replaces_the_grouped_duplicate():
    lattice = frend.resolve_lattice(source_text="****", fold=None)
    supplement = SpokenAlternative("line of asterisk", "curated:t", Decimal(1))
    supplements = {("symbol:run", "****"): (supplement,)}
    for groups in [None, {"tts-sanity": ["named", "silent", "described"]}]:
        result = verbalize_lattice(lattice, supplements=supplements, groups=groups)
        alternatives = result.best_path.units[0].alternatives
        assert alternatives[0] == supplement
        assert [item.text for item in alternatives].count("line of asterisk") == 1
        assert alternatives[0].group is None


def test_c13_icukit_section_is_opaque_and_inert_for_every_golden_input():
    resolved = resolve_behavior([DATA / "user/with-icukit.json"])
    assert resolved.kwargs == {}
    assert len(resolved.icukit_sections) == 1
    assert resolved.icukit_sections[0].mapping == {
        "schema_version": 1,
        "sentence": {"base": "en-tn@1"},
    }
    for text in _golden_texts():
        assert frend.normalize(text, **resolved.kwargs) == frend.normalize(text)
