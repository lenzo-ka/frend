"""P8a REG/CHAR contract from dualplan-ranges/23-p8-r4.md section 2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))


def test_connector_accepts_locale_id_and_source(monkeypatch):
    from frend import verbalize

    forms = {
        "range.connector": {"range": [{"id": "do", "pattern": "{0} до {1}"}]},
        "range.separator": {"range": ["-", "—"]},
    }
    monkeypatch.setattr(verbalize, "_lexical", lambda key, locale: forms.get(key))
    connector = verbalize.range_connector("ru-RU")
    assert connector.id == "do"
    assert connector.words == "до"
    assert connector.provenance == "lexical:ru_RU"
    assert verbalize.range_separators("ru_RU") == frozenset({"-", "–", "—"})


def test_range_candidate_inventory_reaches_frozen_categories():
    from google_tn_rows import range_candidate_denominators, range_candidates

    build_context_trees = _load("build_context_trees")
    build_range_priors = _load("build_range_priors")
    evaluate_google_tn = _load("evaluate_google_tn")

    expected = {
        ("—", "PUNCT", "sil"): 340,
        ("-", "VERBATIM", "sil"): 10,
        ("—", "PLAIN", "до"): 136,
        ("-", "PLAIN", "до"): 133,
    }
    sentences = []
    for (separator, middle_class, middle_spoken), count in expected.items():
        sentences.extend(
            [
                [
                    ("CARDINAL", "5", "пяти"),
                    (middle_class, separator, middle_spoken),
                    ("CARDINAL", "10", "десяти"),
                ]
            ]
            * count
        )
    found = Counter(
        (item.separator, item.middle[0], item.middle[2]) for item in range_candidates(sentences)
    )
    assert found == expected
    assert range_candidate_denominators(sentences) == expected
    assert build_range_priors._range_denominators(sentences) == expected
    assert build_context_trees._range_denominators(sentences) == expected
    assert evaluate_google_tn._range_denominators(sentences) == expected


def test_resolve_choices_binds_locale():
    from frend.lattice import resolve_choices

    class Neutral:
        locale = "*"

        def features(self, detection, context):
            return ()

    assert resolve_choices([], locale="RU-ru").locale == "ru_RU"
    with pytest.raises(ValueError, match="does not match"):
        resolve_choices([], locale="ru_RU", feature_sources=(SimpleNamespace(locale="en_US"),))
    with pytest.raises(ValueError, match="locale None"):
        resolve_choices([], locale="ru_RU", feature_sources=(SimpleNamespace(),))
    assert resolve_choices([], locale="ru_RU", feature_sources=(Neutral(),)).locale == "ru_RU"


def test_ranked_resolvers_bind_locale():
    from frend.fold_resolve import resolve, resolve_cover
    from frend.lattice import resolve_lattice
    from frend.locale_data import locale_chain
    from frend.type_priors import BlendedPrior

    assert resolve_lattice([], locale="es-MX").locale == "es_MX"
    assert resolve([], locale="ru_RU").best == ()
    assert resolve_cover([], locale="ru_RU").best == ()
    assert BlendedPrior(locale="ru-RU").locale == "ru_RU"
    assert locale_chain("ru_RU") == ("ru_RU", "ru", "root")
    assert locale_chain("es_ES") == ("es_ES", "es", "root")
    assert locale_chain("es_MX") == ("es_MX", "es_419", "es", "root")


def test_compose_uses_lattice_locale_and_rejects_mismatch():
    from frend.lattice import compose_choices, resolve_choices

    lattice = resolve_choices([], locale="es_ES", source_text="")
    assert compose_choices(lattice).lattice.locale == "es_ES"
    with pytest.raises(ValueError, match="does not match lattice locale"):
        compose_choices(lattice, locale="en_US")


def test_non_english_letters_require_locale_names():
    from frend.letters import LettersValue, letter_names
    from frend.verbalize import _acronym_priors, _spoken_letters, _spoken_runs

    assert letter_names("D", "es_ES") is None
    assert letter_names("НАТО", "ru_RU") is None
    for locale in ("es_ES", "ru_RU"):
        with pytest.raises(NotImplementedError):
            _spoken_runs(SimpleNamespace(runs=(("digits", "3"), ("letters", "D"))), locale)
        with pytest.raises(NotImplementedError):
            _spoken_letters(LettersValue("NATO", "NATO", ""), locale)
        assert _acronym_priors(locale=locale) == {}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_verified_inputs_reject_wrong_identity(tmp_path, monkeypatch):
    corpus_inputs = _load("corpus_inputs")
    root = tmp_path / "store"
    root.mkdir()
    good = b"PLAIN\tword\t<self>\n"
    digest = hashlib.sha256(good).hexdigest()
    name = "output-00000-of-00100"
    (root / name).write_bytes(good)
    monkeypatch.setattr(corpus_inputs, "store_root", lambda source_id: root.resolve())
    monkeypatch.setattr(corpus_inputs, "_entry", lambda source_id: {"shards": {name: digest}})
    source = "google/tn-en_with_types"
    verified = corpus_inputs.verified_inputs(
        source, [root / name], locale="en_US", pools=("training",)
    )
    assert verified[0].sha256 == digest
    (root / name).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="catalog pins"):
        corpus_inputs.verified_inputs(source, [root / name], locale="en_US", pools=("training",))
    renamed = root / "output-00095-of-00100"
    renamed.write_bytes(good)
    with pytest.raises(ValueError, match="requested pools"):
        corpus_inputs.verified_inputs(source, [renamed], locale="en_US", pools=("training",))
    wrong_root = tmp_path / "wrong-store" / name
    wrong_root.parent.mkdir()
    wrong_root.write_bytes(good)
    with pytest.raises(ValueError, match="outside corpus store root"):
        corpus_inputs.verified_inputs(source, [wrong_root], locale="en_US", pools=("training",))
    link = root / name
    link.unlink()
    target = tmp_path / "target"
    target.write_bytes(good)
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        corpus_inputs.verified_inputs(source, [link], locale="en_US", pools=("training",))


def test_corpus_store_root_uses_environment_or_refuses(tmp_path, monkeypatch):
    corpus_inputs = _load("corpus_inputs")
    monkeypatch.setattr(corpus_inputs, "_entry", lambda _source_id: {"dest": "en_with_types"})
    monkeypatch.setenv("FREND_CORPORA", str(tmp_path / "corpora"))
    assert (
        corpus_inputs.store_root("google/tn-en_with_types")
        == (tmp_path / "corpora" / "google" / "tn-en_with_types" / "en_with_types").resolve()
    )

    monkeypatch.delenv("FREND_CORPORA")
    with pytest.raises(ValueError, match="--corpus-dir.*FREND_CORPORA"):
        corpus_inputs.store_root("google/tn-en_with_types")


def test_internal_taint_is_transitive():
    from corpus_inputs import VerifiedInput
    from corpus_receipts import derive_receipt

    from frend.data_sources import shipping_refusals

    digest = "5d1b84c18b57c62e2c369ce93b3fdc607a2487de17acb74ec72503e242300034"
    masquerade = VerifiedInput("icu/78.3/cldr", "probe", digest, "shippable")
    assert shipping_refusals({"source": "icu/78.3/cldr", "input_sha256": digest})
    first = derive_receipt(
        inputs=(masquerade,), producer_commit="a", command=("one",), artifacts={}, shipping=False
    )
    second = derive_receipt(
        inputs=(first,), producer_commit="b", command=("two",), artifacts={}, shipping=False
    )
    assert second.license_class == "internal-only"
    with pytest.raises(ValueError, match="cannot produce a shipping receipt"):
        derive_receipt(
            inputs=(second,), producer_commit="c", command=("three",), artifacts={}, shipping=True
        )


def test_no_russian_abbreviation_prior_without_lexicon():
    from frend.abbreviation_variants import abbreviation_priors

    assert not (_REPO / "frend/data/ru/abbreviation_priors.json").exists()
    assert abbreviation_priors(locale="ru_RU") == {}


def test_russian_abbreviation_lookup_never_requests_english(monkeypatch):
    import frend.abbreviation_variants as module

    seen = []
    module._priors_for.cache_clear()
    monkeypatch.setattr(
        module,
        "measured_table",
        lambda name, locale: (seen.append((name, locale)), None)[1],
    )
    assert module.abbreviation_priors(locale="ru_RU") == {}
    assert seen and all(locale in {"ru_RU", "ru", "root"} for _, locale in seen)
    module._priors_for.cache_clear()


def test_latency_harness_schema_and_boundaries(tmp_path):
    output = tmp_path / "latency.json"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    subprocess.run(
        [
            sys.executable,
            str(_TOOLS / "benchmark_latency.py"),
            "--subject-root",
            str(_REPO),
            "--expected-head",
            head,
            "--manifest",
            str(_TOOLS / "latency_inputs-v1.json"),
            "--locales",
            "en_US,ru_RU,es_ES",
            "--warmup",
            "1",
            "--runs",
            "3",
            "--seed",
            "20260929",
            "--output",
            str(output),
        ],
        check=True,
        env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["manifest_sha256"]
    assert receipt["gc"]["enabled"]
    assert receipt["denominators"] == {"en_US": 3, "es_ES": 3, "ru_RU": 3}
    for locale in receipt["denominators"]:
        assert receipt["vectors"][locale]["end_to_end_ns"]
        assert receipt["vectors"][locale]["resolve_only_ns"]
