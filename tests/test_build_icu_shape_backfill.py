from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

from icukit.locale import format_number

from frend.shape import shape


def _load_builder():
    """Load tools/build_icu_shape_backfill.py by path (tools is not an installed package)."""
    spec = importlib.util.spec_from_file_location(
        "build_icu_shape_backfill",
        Path(__file__).resolve().parent.parent / "tools" / "build_icu_shape_backfill.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _small_manifests() -> dict:
    return {
        "cardinal": (12,),
        "decimal": (1.25,),
        "percent": (0.125,),
        "date": (datetime(2024, 2, 29, 13, 5, 9, tzinfo=UTC),),
        "date_skeletons": ("yMd",),
        "time": (datetime(2024, 2, 29, 13, 5, 9, tzinfo=UTC),),
        "time_skeletons": ("Hms",),
        "money": ((12.5, None),),
        "fraction": ((1, 1, 2),),
        "ordinal": (2,),
        "measure": (2,),
        "measure_units": ("meter",),
        "measure_widths": ("SHORT",),
    }


def test_small_panel_builds_every_v1_class() -> None:
    document = builder.build_document(("en_US", "ar_EG"), _small_manifests())

    assert tuple(document["counts_by_class"]) == builder.CLASSES
    assert all(document["counts_by_class"][name] for name in builder.CLASSES)
    assert all(document["totals_by_class"][name] > 0 for name in builder.CLASSES)
    assert document["provenance"]["status"] == "generated-estimate"


def test_non_latin_decimal_digits_share_ascii_shape() -> None:
    arabic = format_number(12, "ar_EG")
    devanagari = format_number(12, "hi_IN@numbers=deva")

    assert arabic != "12"
    assert devanagari != "12"
    assert shape(arabic) == shape(devanagari) == shape("12") == "N"


def test_check_passes_then_fails_on_mutation(tmp_path, monkeypatch, capsys) -> None:
    document = builder.build_document(("en_US",), _small_manifests())
    out = tmp_path / "backfill.json"
    out.write_text(builder.serialize(document), encoding="utf-8")
    monkeypatch.setattr(builder, "build_document", lambda: document)

    assert builder.main(["--check", "--out", str(out)]) == 0
    mutated = json.loads(out.read_text(encoding="utf-8"))
    mutated["totals_by_class"]["cardinal"] += 1
    out.write_text(builder.serialize(mutated), encoding="utf-8")
    assert builder.main(["--check", "--out", str(out)]) == 1
    assert "drift:" in capsys.readouterr().err


def test_build_is_byte_deterministic() -> None:
    first = builder.serialize(builder.build_document(("en_US", "de_DE"), _small_manifests()))
    second = builder.serialize(builder.build_document(("en_US", "de_DE"), _small_manifests()))

    assert first.encode() == second.encode()
