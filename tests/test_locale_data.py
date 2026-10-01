"""Locale data: where each table lives, the order a locale looks, and what a locale
with no table gets."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tomllib
from importlib.resources import files
from pathlib import Path, PurePosixPath

import pytest

from frend import data_sources, locale_data

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "frend" / "data"
_TOOLS = _REPO / "tools"
_MEASURED = (
    "abbreviation_priors",
    "acronym_priors",
    "electronic_priors",
    "number_priors",
    "range_priors",
    "spelled_token_priors",
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


@pytest.mark.parametrize("spelling", ["en_US", "en-US", "EN-US", "en-us", "en_us", "EN_us"])
def test_every_spelling_of_a_locale_walks_one_chain(spelling):
    """A tag is canonicalized through ICU before it names a directory, so its case and
    its separator never matter (a case-sensitive installed package would otherwise
    look for ``EN_US`` and ``EN``)."""
    from frend.locale_data import locale_chain

    assert locale_chain(spelling) == ("en_US", "en", "root")


def test_the_canonical_spelling_is_icus_base_name():
    from frend.locale_data import canonical_locale

    for spelling in ("en_US", "en-US", "EN-US", "en-us", "EN_us"):
        assert canonical_locale(spelling) == "en_US", spelling
    assert canonical_locale("SR-latn-rs") == "sr_Latn_RS"
    assert canonical_locale("ROOT") == "root"


def test_a_script_is_title_cased_and_walks_its_chain():
    from frend.locale_data import locale_chain

    assert locale_chain("zh-hant-tw") == ("zh_Hant_TW", "zh_Hant", "root")


@pytest.mark.parametrize("spelling", ["root", "ROOT", "Root", "rOoT"])
def test_root_in_any_case_is_root_and_serves_no_measured_table(spelling):
    from frend.locale_data import locale_chain, measured_table

    assert locale_chain(spelling) == ("root",)
    assert measured_table("icu_shape_backfill", spelling) is None
    for name in _MEASURED:
        assert measured_table(name, spelling) is None, name


@pytest.mark.parametrize(
    "tag",
    [
        "en/../root",
        "../root",
        "en/root",
        "en\\..\\root",
        "..",
        "",
        "en__US",
        "en_",
        "_en",
        "en-",
        "en US",
        "x",
        "123",
        "en.US",
    ],
)
def test_a_malformed_tag_is_refused_before_any_path(tag):
    """Anything that is not a well-formed tag never reaches a path join: a path-shaped
    tag would otherwise walk out of its locale directory (``en/../root`` read the
    root-only ICU backfill as though it were measured)."""
    from frend.electronic import load_electronic_priors
    from frend.locale_data import locale_chain, measured_table
    from frend.spoken_priors import load_spoken_prior_table
    from frend.type_priors import load_prior_table

    with pytest.raises(ValueError):
        locale_chain(tag)
    with pytest.raises(ValueError):
        measured_table("icu_shape_backfill", tag)
    for load in (load_prior_table, load_spoken_prior_table, load_electronic_priors):
        with pytest.raises(ValueError):
            load(locale=tag)


def test_the_most_specific_locale_wins(tmp_path, monkeypatch):
    """With both ``en_US`` and ``en`` holding a table, ``en_US`` is read; a locale with
    only the parent's falls back to it."""
    import frend.locale_data as locale_data

    for tag, value in (("en_US", "us"), ("en", "en"), ("root", "root")):
        (tmp_path / tag).mkdir()
        (tmp_path / tag / "probe.json").write_text(json.dumps({"from": value}), "utf-8")
    monkeypatch.setattr(locale_data, "_data", lambda: tmp_path)

    assert locale_data.measured_table("probe", "en_US") == {"from": "us"}
    assert locale_data.measured_table("probe", "EN-us") == {"from": "us"}
    assert locale_data.measured_table("probe", "en_GB") == {"from": "en"}
    assert locale_data.measured_table("probe", "en") == {"from": "en"}
    assert locale_data.measured_table("probe", "fr_FR") is None
    assert locale_data.measured_table("probe", "root") is None


def test_every_loader_caches_on_the_canonical_locale():
    """Equivalent spellings share one cached table: the electronic loader hands out
    mutable dictionaries, so two copies could drift apart."""
    from frend.abbreviation_variants import abbreviation_priors
    from frend.electronic import load_electronic_priors
    from frend.spoken_priors import load_spoken_prior_table
    from frend.type_priors import load_prior_table
    from frend.verbalize import _acronym_priors, _zero_priors

    loaders = (
        abbreviation_priors,
        load_electronic_priors,
        load_prior_table,
        load_spoken_prior_table,
        _zero_priors,
        _acronym_priors,
    )
    for load in loaders:
        default = load()
        assert default, load.__name__
        for spelling in ("en_US", "en-US", "EN-US", "en_us"):
            assert load(locale=spelling) is default, (load.__name__, spelling)


def test_a_missing_locale_table_passed_on_gives_no_english_counts():
    """``load_prior_table`` answers ``None`` for a locale with no table; handing that
    to a consumer means no measured prior, not the default English table. Only an
    omitted argument means the default."""
    from frend.type_priors import BlendedPrior, CorpusPrior, load_prior_table

    missing = load_prior_table(locale="ru_RU")
    assert missing is None
    english = load_prior_table()
    detection = {"type": "number:cardinal", "text": "12", "start": 0, "end": 2}
    assert english.reading_prior(detection).tier == "measured"

    corpus = CorpusPrior(missing)
    assert corpus._table is None
    assert corpus.reading_prior(detection) is None
    assert corpus.features(detection, None) == ()

    blended = BlendedPrior(missing)
    assert blended.measured is None
    prior = blended.reading_prior(detection)
    assert prior is None or prior.tier != "measured"

    assert CorpusPrior()._table is english
    assert BlendedPrior().measured is english


def _builder(name: str):
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name",
    [
        "build_abbreviation_priors",
        "build_acronym_priors",
        "build_electronic_priors",
        "build_spoken_priors",
        "build_zero_priors",
    ],
)
def test_a_builder_records_the_corpus_it_read(name):
    """A table built from another shard directory names that directory as its corpus,
    not the shipped corpus's (the type-priors builder: test_build_google_tn)."""
    fixture = _REPO / "tests" / "data" / "google_tn"
    provenance = _builder(name).build_document(fixture)["provenance"]
    assert provenance["corpus"] == "google-tn:google_tn", name


def test_a_locale_with_no_measured_table_gets_none():
    """No en counts stand in for a locale that has none: every measured loader says
    so, and English keeps its tables."""
    from frend.abbreviation_variants import abbreviation_priors
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
    assert abbreviation_priors(locale="ru_RU") == {}

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
    in) and the corpus it was counted from; type_priors counts the corpus README's 90
    training shards (00-89)."""
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
    assert len(type_provenance["shards"]) == 90
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


def test_shipped_spelled_token_priors_have_attested_exceptions():
    document = locale_data.measured_table("spelled_token_priors", "en_US")
    assert document is not None
    assert document["tokens"]
    assert document["tokens"]["by"]["counts"] == {"say": 4_335_164, "spell": 0}
    assert document["tokens"]["fMRI"]["counts"] == {"say": 0, "spell": 678}
    assert document["provenance"]["acronym_cases"]["NASA"]["say"] == 19_509


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


@pytest.mark.parametrize("tag", ["und", "und-Latn", "und_US", "UND"])
def test_und_is_cldrs_root(tag):
    assert locale_data.canonical_locale(tag) == "root"
    assert locale_data.locale_chain(tag) == ("root",)


@pytest.mark.parametrize("tag", ["en_US_POSIX_POSIX", "en-US-posix-POSIX"])
def test_a_repeated_variant_is_refused(tag):
    with pytest.raises(ValueError):
        locale_data.canonical_locale(tag)


def test_a_locale_cache_holds_a_bounded_number_of_locales():
    from frend.type_priors import _locale_prior_table, load_prior_table

    for index in range(3 * locale_data.LOCALE_CACHE):
        load_prior_table(locale=f"xx_{chr(65 + index % 26)}{chr(65 + index // 26)}")
    assert _locale_prior_table.cache_info().currsize <= locale_data.LOCALE_CACHE


# What an LDC corpus looks like when a provenance names it: a store id under ``ldc/``,
# a catalog number ("LDC93S6A"), or a label such as "ldc:wsj0".
_LDC = re.compile(r"(?<![a-z0-9])ldc(?:[/:_-]|\d{2}[a-z]\d)", re.IGNORECASE)


def _mentions_ldc(value: object) -> bool:
    if isinstance(value, str):
        return _LDC.search(value) is not None
    if isinstance(value, dict):
        return any(_mentions_ldc(k) or _mentions_ldc(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_mentions_ldc(item) for item in value)
    return False


def _refusals(name: str, document: object) -> list[str]:
    """Why a shipped file may not ship, by what it names as its source; empty when it may.

    A JSON table names its source in ``corpus`` or ``source``, at its top level or in its
    ``provenance``; each must resolve (``data_sources.source_id``) to a declared source
    of a class frend ships. A table naming none, an undeclared one or an ``ldc/`` one is
    refused, and so is any document that mentions an LDC corpus anywhere. A file that
    is not JSON ships only as a declared source's vendored copy.
    """
    if not isinstance(document, dict):
        vendored = {
            str(file)
            for entry in data_sources.SHIPPABLE_SOURCES.values()
            for file in entry["vendored"]  # type: ignore[union-attr]
        }
        return [] if name in vendored else [f"{name}: not a table and no source vendors it"]
    provenance = document.get("provenance")
    records = [document] + ([provenance] if isinstance(provenance, dict) else [])
    labels = [
        (record, record[field])
        for record in records
        for field in ("corpus", "source")
        if field in record
    ]
    if not labels:
        return [f"{name}: names no source"]
    refusals = []
    for record, label in labels:
        if not isinstance(label, str):
            refusals.append(f"{name}: source {label!r} is not an id")
            continue
        source = data_sources.source_id(label, record)
        class_ = data_sources.source_class(source)
        if class_ not in data_sources.SHIPPABLE_CLASSES:
            refusals.append(f"{name}: source {source!r} is {class_ or 'undeclared'}")
    # The package boundary also rejects identity hidden behind an innocent source label:
    # known internal digests, internal ancestry, and processed-LDC paths.
    refusals.extend(
        f"{name}: {reason}" for reason in data_sources.shipping_refusals(document, name)
    )
    return refusals


def _document(path: Path) -> object:
    if path.suffix == ".cart":
        # A binary cartlet model names its source in its embedded metadata.
        from cartlet.runner import read_cart_metadata

        return read_cart_metadata(path.read_bytes())
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix == ".json" else text


def test_every_shipped_table_names_a_shippable_source():
    """frend ships nothing derived from an LDC corpus (kal, 2026-09-28: every ``ldc/*``
    store id is internal-only for frend), so every file under ``frend/data/`` names its
    source by an id in the one declared list (``frend.data_sources``), of a class frend
    ships; the README documents them and is not a table."""
    files = [path for path in sorted(_DATA.rglob("*")) if path.is_file()]
    assert files
    refusals = [
        refusal
        for path in files
        if path.name != "README.md"
        for refusal in _refusals(str(PurePosixPath(path.relative_to(_DATA))), _document(path))
    ]
    assert refusals == []


@pytest.mark.parametrize(
    "document",
    [
        {"provenance": {"corpus": "ldc/LDC93S6A"}},
        {"provenance": {"corpus": "LDC93S6A"}},
        {"provenance": {"corpus": "ldc:wsj0"}},
        {"corpus": "ldc/wsj0"},
        {"provenance": {"source": "nist/timit"}},
        {"provenance": {"corpus": "google-tn:en_with_types", "shards": ["LDC/wsj0/si_tr_s"]}},
        {"provenance": {"corpus": "google-tn:en_with_types", "note": "counts from LDC93S6A"}},
        # A mention nested in top-level metadata no named field holds.
        {"source": "google/tn-en_with_types", "inputs": {"ldc/LDC93S6A": "checksum"}},
        {"source": "google/tn-en_with_types", "derivation": {"from": ["ldc:wsj0"]}},
        {"provenance": {"shards": ["output-00000-of-00100"]}},
        {"counts": {"x": 1}},
        {"provenance": {"corpus": "icu/"}},
        {"provenance": {"source": "icu-reflective-generation"}},
        {"provenance": {"corpus": ["google/tn-en_with_types"]}},
    ],
)
def test_a_table_naming_no_shippable_source_is_refused(document):
    """Planted tables the guard must refuse: an LDC id, a bare catalog number, an LDC
    label, an undeclared id, an LDC mention beside a declared source (in its provenance
    or nested in other metadata), no source at all,
    a bare namespace, an ICU label missing its version, a source that is not an id."""
    assert _refusals("en/planted.json", document)


@pytest.mark.parametrize(
    "document",
    [
        {"provenance": {"corpus": "google-tn:en_with_types"}},
        {"provenance": {"corpus": "google/tn-en_with_types"}},
        {"provenance": {"source": "icu-reflective-generation", "icu_version": "78.3"}},
        {"corpus": "break-exceptions-en-curated"},
        {"source": "frend/curated"},
    ],
)
def test_a_table_naming_a_declared_source_ships(document):
    assert _refusals("en/planted.json", document) == []


def test_a_declared_ldc_source_is_still_internal_only():
    assert data_sources.source_class("ldc/wsj0") == data_sources.INTERNAL_ONLY
    assert data_sources.source_class("LDC/LDC93S6A") == data_sources.INTERNAL_ONLY


def test_every_declared_source_has_a_license_and_a_class():
    for source, entry in data_sources.SHIPPABLE_SOURCES.items():
        assert entry["class"] in data_sources.LICENSE_CLASSES, source
        assert entry["class"] != data_sources.INTERNAL_ONLY, source
        assert entry["license"], source
        for file in entry["vendored"]:
            assert (_DATA / file).is_file(), (source, file)


def test_a_text_file_ships_only_as_a_vendored_source():
    assert _refusals("root/tlds-alpha-by-domain.txt", "# Version 1\nCOM\n") == []
    assert _refusals("root/other.txt", "# source: google/tn-en_with_types\n")
