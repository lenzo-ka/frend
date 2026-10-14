from pathlib import Path

import pytest

from tools.recount_electronic import (
    REPO,
    _canonical_bytes,
    _refuse_shard_99,
    _validate_subjects,
    build_recount,
)


def _write_checked(path: Path) -> None:
    path.write_text("written\tspoken\tclass\tsource\n", encoding="utf-8")


def test_electronic_recount_is_canonical_utf8_json():
    assert _canonical_bytes({"schema": "x", "value": "앳"}) == (
        '{\n  "schema": "x",\n  "value": "앳"\n}\n'.encode()
    )


@pytest.mark.parametrize("name", ("output-00099-of-00100", "shard-99", "shard_99"))
def test_electronic_recount_refuses_shard_99(name):
    with pytest.raises(ValueError, match="refusing shard 99"):
        _refuse_shard_99(Path("fixtures") / name)


def test_build_recount_classifies_complete_electronic_miss(tmp_path):
    fixture = tmp_path / "en" / "data_text_normalization" / "test_cases_electronic.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("example@example.com~intentionally unmatched\n", encoding="utf-8")
    checked = tmp_path / "checked.tsv"
    _write_checked(checked)

    recount = build_recount(tmp_path, checked)

    assert recount["locales"]["en_US"] == {
        "fixture_sources": [
            {
                "path": "en/data_text_normalization/test_cases_electronic.txt",
                "sha256": ("f9d50f1c70b9f3438141054c7e7f196bcfce203ac2aceaf03c0c98f089b3de75"),
            }
        ],
        "physical_rows": 1,
        "deduplicated_logical_rows": 1,
        "complete_strict_misses": 1,
        "partial_strict_misses": 0,
        "no_electronic_detection_strict_misses": 0,
        "strict_passes": 0,
        "absent_population": False,
    }
    assert recount["totals"] == {
        "complete_strict_misses_physical": 1,
        "complete_strict_misses_deduplicated_logical": 1,
    }


def test_build_recount_refuses_fixture_symlink_to_shard_99(tmp_path):
    forbidden = tmp_path / "output-00099-of-00100"
    forbidden.write_text("example@example.com~spoken\n", encoding="utf-8")
    fixture = tmp_path / "en" / "data_text_normalization" / "test_cases_electronic.txt"
    fixture.parent.mkdir(parents=True)
    fixture.symlink_to(forbidden)
    checked = tmp_path / "checked.tsv"
    _write_checked(checked)

    with pytest.raises(ValueError, match="refusing shard 99"):
        build_recount(tmp_path, checked)


def test_build_recount_records_custom_checked_source_path(tmp_path):
    checked = tmp_path / "custom-checked.tsv"
    _write_checked(checked)

    recount = build_recount(tmp_path, checked)

    assert recount["locales"]["pt_PT"]["fixture_sources"][0]["path"] == str(checked.resolve())


def test_electronic_recount_refuses_missing_fixture_sentinels(tmp_path, monkeypatch):
    fixture_root = tmp_path / "tests" / "nemo_text_processing"
    fixture_root.mkdir(parents=True)
    monkeypatch.setattr(
        "tools.recount_electronic._git_head",
        lambda path: "frend" if path == REPO else "nemo",
    )
    monkeypatch.setattr("tools.recount_electronic._git_toplevel", lambda _path: tmp_path)

    with pytest.raises(ValueError, match="missing sentinels"):
        _validate_subjects(fixture_root, expect_base="frend", expect_nemo_commit="nemo")


def test_electronic_recount_refuses_nested_fixture_copy(tmp_path, monkeypatch):
    fixture_root = tmp_path / "nested"
    for sentinel in (
        "en/data_text_normalization/test_cases_electronic.txt",
        "es/data_text_normalization/test_cases_electronic.txt",
    ):
        path = fixture_root / sentinel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("written~spoken\n", encoding="utf-8")
    monkeypatch.setattr(
        "tools.recount_electronic._git_head",
        lambda path: "frend" if path == REPO else "nemo",
    )
    monkeypatch.setattr("tools.recount_electronic._git_toplevel", lambda _path: tmp_path)

    with pytest.raises(ValueError, match="wrong NeMo fixture root"):
        _validate_subjects(fixture_root, expect_base="frend", expect_nemo_commit="nemo")


def test_electronic_recount_refuses_frend_commit_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tools.recount_electronic._git_head",
        lambda path: "observed" if path == REPO else "nemo",
    )

    with pytest.raises(ValueError, match="wrong frend base"):
        _validate_subjects(tmp_path, expect_base="frend", expect_nemo_commit="nemo")


def test_electronic_recount_refuses_nemo_commit_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tools.recount_electronic._git_head",
        lambda path: "frend" if path == REPO else "observed",
    )

    with pytest.raises(ValueError, match="wrong NeMo commit"):
        _validate_subjects(tmp_path, expect_base="frend", expect_nemo_commit="nemo")
