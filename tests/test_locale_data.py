"""Locale data: where each table lives, the order a locale looks, and what a locale
with no table gets."""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from importlib.resources import files
from pathlib import Path, PurePosixPath

import pytest

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "frend" / "data"
_TOOLS = _REPO / "tools"
_MEASURED = (
    "acronym_priors",
    "electronic_priors",
    "spoken_priors",
    "type_priors",
    "zero_priors",
)
_CORPUS = "google-tn:en_with_types"


def test_chain_walks_en_US_to_en_to_root():
    from frend.locale_data import locale_chain

    assert locale_chain("en_US") == ("en_US", "en", "root")
    assert locale_chain("en") == ("en", "root")
    assert locale_chain("root") == ("root",)


def test_a_locale_with_no_measured_table_gets_none():
    """No en counts stand in for a locale that has none: every measured loader says
    so, and English keeps its tables."""
    from frend.electronic import load_electronic_priors
    from frend.spoken_priors import load_spoken_prior_table, source_prior
    from frend.type_priors import load_prior_table
    from frend.verbalize import _acronym_priors, _zero_priors

    assert source_prior("cardinal", "icu-rbnf:%spellout-numbering", locale="ru_RU") is None
    assert load_spoken_prior_table(locale="ru_RU") is None
    assert load_prior_table(locale="ru_RU") is None
    assert load_electronic_priors(locale="ru_RU") is None
    assert _zero_priors(locale="ru_RU") == {}
    assert _acronym_priors(locale="ru_RU") == {}

    assert source_prior("cardinal", "icu-rbnf:%spellout-numbering", locale="en_US") is not None
    assert load_prior_table(locale="en_US") is load_prior_table()
    assert load_spoken_prior_table(locale="en_US") is load_spoken_prior_table()


def test_measured_tables_never_resolve_from_root():
    """A measured table is one language's counts, so a locale with none gets None even
    where root holds a file of that name (the cross-locale ICU backfill)."""
    from frend.locale_data import measured_table, root_table

    assert (_DATA / "root" / "icu_shape_backfill.json").is_file()
    assert root_table("icu_shape_backfill")["provenance"]["source"] == "icu-reflective-generation"
    assert measured_table("icu_shape_backfill", "xx") is None
    assert measured_table("icu_shape_backfill", "en_US") is None
    for name in _MEASURED:
        assert measured_table(name, "xx") is None, name
        assert measured_table(name, "en_US") is not None, name


def test_every_table_names_locale_and_corpus():
    """Each measured table names the locale its counts are for (the directory it sits
    in) and the corpus it was counted from; type_priors keeps its 98 shards and its
    build date through the move (the date tracks counts, and the move changes none)."""
    tables = {
        path.stem: path for path in sorted(_DATA.rglob("*_priors.json")) if "root" not in path.parts
    }
    assert set(tables) == set(_MEASURED)
    for name, path in tables.items():
        provenance = json.loads(path.read_text(encoding="utf-8"))["provenance"]
        assert provenance.get("locale") is not None, f"{name} names no locale"
        assert provenance["locale"] == path.parent.name == "en", name
        assert provenance.get("corpus") == _CORPUS, name
    type_provenance = json.loads(tables["type_priors"].read_text(encoding="utf-8"))["provenance"]
    assert len(type_provenance["shards"]) == 98
    assert type_provenance["generated"] == "2026-09-28"
    spoken = json.loads(tables["spoken_priors"].read_text(encoding="utf-8"))["provenance"]
    assert spoken["sample_rule"]["shards"]


def test_every_table_loads_from_package_resources():
    """Every shipped table loads as a package resource at its locale path, and the
    package-data globs ship every file under frend/data."""
    data = files("frend").joinpath("data")
    for name in _MEASURED:
        document = json.loads(data.joinpath("en", f"{name}.json").read_text(encoding="utf-8"))
        assert document["provenance"], name
    assert json.loads(data.joinpath("en", "exceptions.json").read_text(encoding="utf-8"))["rules"]
    backfill = json.loads(data.joinpath("root", "icu_shape_backfill.json").read_text("utf-8"))
    assert backfill["counts_by_class"]
    assert "COM" in data.joinpath("root", "tlds-alpha-by-domain.txt").read_text("ascii")

    from frend.electronic import load_electronic_priors, top_level_domains
    from frend.exceptions import english_break_exceptions
    from frend.type_priors import load_icu_backfill_table, load_prior_table
    from frend.verbalize import _acronym_priors, _zero_priors

    assert load_prior_table().n("N") > 0
    assert load_icu_backfill_table().provenance["status"] == "generated-estimate"
    assert load_electronic_priors()["letters"]
    assert _zero_priors()["date"] and _acronym_priors()["*"]
    assert "com" in top_level_domains()
    assert english_break_exceptions()

    globs = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
        "setuptools"
    ]["package-data"]["frend"]
    for path in sorted(_DATA.rglob("*")):
        if path.is_file():
            relative = PurePosixPath(path.relative_to(_REPO / "frend").as_posix())
            assert any(relative.match(glob) for glob in globs), relative


def _evaluator():
    sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(
        "evaluate_google_tn", _TOOLS / "evaluate_google_tn.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("written", "first"),
    [
        ("2008-09-30", "the thirtieth of september two thousand eight"),
        ("10:30", "ten thirty"),
        ("St. Paul", "saint paul"),
        ("Mr. Smith", "mister smith"),
        ("J.R.R.", "j r r"),
        ("ATM", "a t m"),
        ("UFOs", "u f o s"),
        ("S.", "s"),
        ("IN", "in"),
    ],
)
def test_english_first_readings_are_unchanged(written, first):
    """Moving the tables and giving each loader a locale moves no English first reading,
    read through the evaluator's own profile."""
    evaluate = _evaluator()
    assert evaluate._score_text(written, first)[0], written
