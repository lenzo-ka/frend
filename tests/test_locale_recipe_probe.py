from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOL = _REPO / "tools" / "locale_recipe.py"


def _probe(locale: str, out: Path) -> tuple[bytes, dict]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_REPO)
    subprocess.run(
        [sys.executable, "-B", str(_TOOL), "probe", "--locale", locale, "--out", str(out)],
        cwd=_REPO,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    raw = out.read_bytes()
    return raw, json.loads(raw)


@pytest.mark.parametrize(
    ("locale", "own_rbnf", "rbnf_source", "script", "plurals"),
    [
        ("en_US", True, "en", "Latn", {"one", "other"}),
        ("ru_RU", True, "ru", "Cyrl", {"one", "few", "many", "other"}),
        ("rw_RW", False, "en", "Latn", {"other"}),
    ],
)
def test_probe_reports_locale_capabilities(
    tmp_path: Path,
    locale: str,
    own_rbnf: bool,
    rbnf_source: str,
    script: str,
    plurals: set[str],
) -> None:
    _raw, document = _probe(locale, tmp_path / locale / "probe.json")

    assert document["schema"] == "frend-locale-capability-probe/1"
    assert document["locale"] == locale
    capabilities = document["capabilities"]
    assert capabilities["rbnf"]["own_language"] is own_rbnf
    assert capabilities["rbnf"]["source_locale"] == rbnf_source
    assert capabilities["rbnf"]["fallback_source"] == (None if own_rbnf else rbnf_source)
    assert capabilities["script"] == script
    assert set(capabilities["plural_categories"]) == plurals
    assert all(item["available"] for item in capabilities["cldr"].values())
    assert set(capabilities["cldr"]) == {"currency", "date", "number", "unit"}
    assert set(document["versions"]) == {"icu", "icukit", "pyicu", "unicode"}
    assert all(len(value) == 64 for value in document["hashes"].values())


def test_russian_probe_lists_gendered_rbnf_variants(tmp_path: Path) -> None:
    _raw, document = _probe("ru_RU", tmp_path / "probe.json")

    rbnf = document["capabilities"]["rbnf"]
    assert "%spellout-cardinal-feminine" in rbnf["rule_sets"]
    assert "%spellout-cardinal-masculine" in rbnf["gendered_rule_sets"]
    assert "%spellout-cardinal-neuter" in rbnf["gendered_rule_sets"]
    assert "%spellout-cardinal-feminine" in rbnf["gendered_rule_sets"]


def test_probe_reports_actual_cldr_fallback_source(tmp_path: Path) -> None:
    _raw, document = _probe("rw_RW", tmp_path / "probe.json")

    assert document["capabilities"]["cldr"]["date"]["source_locale"] == "root"


def test_probe_output_is_byte_deterministic(tmp_path: Path) -> None:
    first, _document = _probe("en_US", tmp_path / "first.json")
    second, _document = _probe("en_US", tmp_path / "second.json")

    assert first == second


def test_only_probe_subcommand_is_exposed() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_REPO)
    result = subprocess.run(
        [sys.executable, "-B", str(_TOOL), "build"],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "invalid choice: 'build'" in result.stderr
