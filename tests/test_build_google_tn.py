"""Focused tests for the Google-TN corpus reader and the build tool.

These use tiny checked-in shard fixtures under ``tests/data/google_tn`` (a couple
of ``output-*-of-*`` files with a handful of TAB lines), never the real ~20 GB
corpus. They pin the reader's contract -- 3-column parse, ``<eos>`` skip, class
mapping with dropped classes excluded from both numerator and per-shape totals,
the Unicode-Nd digit filter, and the unmapped classes (DIGIT/MEASURE/TELEPHONE/
ADDRESS) still swelling each shape's total -- plus the provenance profile, the
serial/parallel byte-identity, the default corpus selection, and that the
documented ``python tools/build_type_priors.py --check`` runs in a clean env with
no repo on PYTHONPATH.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from frend.type_priors import PriorTable, corpus_classes

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "tools" / "build_type_priors.py"
_FIX = Path(__file__).parent / "data" / "google_tn"

# The exact canonical table the fixtures must yield. Dropped classes (PLAIN, PUNCT,
# LETTERS) and the non-digit MONEY surface "dollars" contribute nothing; the
# unmapped classes digit/measure/telephone/address are counted into their shapes.
_EXPECTED = {
    "N": {"cardinal": 2, "date": 1, "digit": 1},  # 12, 99 | 2006 | 007
    "N A": {"address": 1, "measure": 1},  # "10 Downing" | "5 km"
    "N-N": {"telephone": 1},  # "555-1234"
    "N.N": {"decimal": 1},  # "3.14"
    "N/N": {"fraction": 1},  # "1/2"
    "¤N": {"money": 1},  # "$5" ("dollars" dropped: no digit)
}


def _load_build_tool():
    """Load tools/build_type_priors.py by path (tools is not an installed package)."""
    spec = importlib.util.spec_from_file_location("build_type_priors", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clean_env() -> dict[str, str]:
    """A subprocess environment with no repo on PYTHONPATH (and no corpus override),
    so the script must put the repo root on sys.path itself."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FREND_TN_CORPUS_DIR")}
    return env


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
    )


# --------------------------------------------------------------------------- reader


def test_reader_parses_maps_filters_and_totals():
    """3-column parse, <eos> skip, class mapping, digit filter, and per-shape totals
    all in one exact-table assertion over the fixtures."""
    build = _load_build_tool()
    counts = build.build_counts(build.google_tn_pairs(_FIX))
    assert counts == _EXPECTED


def test_dropped_classes_and_nondigit_surfaces_are_excluded():
    """Dropped corpus classes and any surface without a Unicode-Nd digit never reach
    the pairs -- excluded from both the numerator and the per-shape totals."""
    build = _load_build_tool()
    pairs = list(build.google_tn_pairs(_FIX))
    surfaces = {s for s, _ in pairs}
    classes = {c for _, c in pairs}
    # Dropped classes contribute nothing.
    assert classes.isdisjoint({"plain", "punct", "letters", "verbatim", "electronic"})
    assert "Hello" not in surfaces and "IUCN" not in surfaces  # PLAIN / LETTERS surfaces
    # The non-digit MONEY surface "dollars" is filtered out, so ¤N carries only "$5".
    assert "dollars" not in surfaces
    assert all(build._has_digit(s) for s, _ in pairs)


def test_unmapped_classes_count_into_totals_but_have_no_reading_type():
    """DIGIT/TELEPHONE/ADDRESS are counted into each shape's total (swelling
    n and therefore the denominator of every base rate) even though no reading type
    maps to them, so a lookup for such a type withholds."""
    build = _load_build_tool()
    counts = build.build_counts(build.google_tn_pairs(_FIX))
    table = PriorTable(counts, {"source": "fixture"})

    # Counted into each shape's total: the unmapped classes sit in the counts.
    assert counts["N"] == {"cardinal": 2, "date": 1, "digit": 1}  # digit counted
    assert counts["N A"] == {"address": 1, "measure": 1}  # measure + address counted
    assert counts["N-N"] == {"telephone": 1}  # telephone counted
    # n includes the unmapped digit class (4 = cardinal 2 + date 1 + digit 1).
    assert table.n("N") == 4

    # The digit count swells the denominator of a mapped reading on the same shape.
    card = table.reading_prior({"type": "number:cardinal", "text": "12"})
    assert card is not None and card.supported is True
    assert card.p == Decimal(2) / Decimal(4)  # 2 cardinals of n=4 (digit included)

    # Measure is now mapped for ICU backfill; the other classes still withhold.
    assert corpus_classes("measure:unit") == ("measure",)
    for type_, text in [
        ("telephone:us", "555-1234"),
        ("address:street", "10 Downing"),
        ("digit:string", "007"),
    ]:
        assert corpus_classes(type_) == ()
        assert table.reading_prior({"type": type_, "text": text}) is None


def test_google_provenance_profile():
    """The google-tn profile records source, CC BY-SA 4.0 license, attribution to
    Sproat & Jaitly, the fixed build date, and the derivation note."""
    build = _load_build_tool()
    prov = build._build("google-tn", _FIX, 1)["provenance"]
    assert prov["source"] == "google-tn-en_with_types"
    assert prov["license"] == "CC BY-SA 4.0"
    assert prov["attribution"].startswith("derived from Sproat & Jaitly")
    assert prov["generated"] == "2026-08-23"
    assert "en_with_types" in prov["note"]
    assert "LETTERS and PLAIN" in prov["note"]


def test_single_uppercase_letters_lift_numeric_filters_and_record_each_letter(tmp_path):
    build = _load_build_tool()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "output-00000-of-00001").write_text(
        "LETTERS\tI\ti\nPLAIN\tI\ti\nCARDINAL\tI\tone\nLETTERS\tΩ\tomega\nPLAIN\thello\t<self>\n",
        encoding="utf-8",
    )

    document = build._build("google-tn", corpus, 1)

    assert document["counts"]["<Lu>"] == {"cardinal": 1, "letters": 2, "plain": 1}
    assert document["single_uppercase_letters"] == {
        "I": {"cardinal": 1, "letters": 1, "plain": 1},
        "Ω": {"letters": 1},
    }
    assert "A" not in document["counts"]


# --------------------------------------------------------------------- build tool


def test_serial_and_parallel_builds_are_byte_identical(tmp_path):
    """--jobs 1 (serial) and --jobs 2 (multiprocessing) emit byte-identical JSON;
    run via the real script so the workers pickle the module-level worker fn."""
    out1, out2 = tmp_path / "j1.json", tmp_path / "j2.json"
    a = _run(["--corpus-dir", str(_FIX), "--jobs", "1", "--out", str(out1)])
    b = _run(["--corpus-dir", str(_FIX), "--jobs", "2", "--out", str(out2)])
    assert a.returncode == 0 and b.returncode == 0, (a.stderr, b.stderr)
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


def test_default_corpus_is_google_tn(tmp_path):
    """With no --corpus flag the build selects the google-tn seam."""
    out = tmp_path / "default.json"
    result = _run(["--corpus-dir", str(_FIX), "--out", str(out)])
    assert result.returncode == 0, result.stderr
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["provenance"]["source"] == "google-tn-en_with_types"
    assert document["counts"] == _EXPECTED


def test_check_runs_in_a_clean_environment(tmp_path):
    """Regression: ``python tools/build_type_priors.py`` (build and --check) must
    work with NO repo on PYTHONPATH -- the script puts the repo root on sys.path
    itself, in the main process and in the spawned workers. Build, verify byte-match,
    then confirm a mutation is detected as drift."""
    out = tmp_path / "clean.json"
    env = _clean_env()

    built = _run(["--corpus-dir", str(_FIX), "--out", str(out)], env=env)
    assert built.returncode == 0, built.stderr

    ok = _run(["--check", "--corpus-dir", str(_FIX), "--out", str(out)], env=env)
    assert ok.returncode == 0, ok.stderr
    assert "up to date" in ok.stdout

    out.write_text(out.read_text(encoding="utf-8").replace("}", "} ", 1), encoding="utf-8")
    drift = _run(["--check", "--corpus-dir", str(_FIX), "--out", str(out)], env=env)
    assert drift.returncode == 1
