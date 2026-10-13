from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

import pytest

from frend.external_profile import _KEYS, verified_profile_key
from tools.evaluation_metadata import distribution_version
from tools.literal_spell_say import literal_spell_say


def _missing(path: Path) -> FileNotFoundError:
    return FileNotFoundError(f"missing {path}")


def _changed(path: Path) -> ValueError:
    return ValueError(f"changed {path}")


def test_external_profile_missing_cache_and_same_metadata_replacement(tmp_path, monkeypatch):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="missing"):
        verified_profile_key(missing, missing=_missing, changed=_changed)

    path = tmp_path / "profile.json"
    path.write_bytes(b"first")
    _KEYS.clear()
    reads = 0
    original = Path.read_bytes

    def counted(candidate):
        nonlocal reads
        if candidate == path:
            reads += 1
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", counted)
    first = verified_profile_key(path, missing=_missing, changed=_changed)
    assert verified_profile_key(path, missing=_missing, changed=_changed) == first
    assert reads == 1

    timestamp = path.stat().st_mtime_ns
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b"other")
    os.utime(replacement, ns=(timestamp, timestamp))
    replacement.replace(path)
    second = verified_profile_key(path, missing=_missing, changed=_changed)
    assert second[:3] == first[:3]
    assert second[3] != first[3]


def test_external_profile_rejects_change_during_read(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    path.write_bytes(b"first")
    _KEYS.clear()
    original = Path.read_bytes

    def changing(candidate):
        content = original(candidate)
        if candidate == path:
            candidate.write_bytes(b"second")
        return content

    monkeypatch.setattr(Path, "read_bytes", changing)
    with pytest.raises(ValueError, match="changed"):
        verified_profile_key(path, missing=_missing, changed=_changed)


def test_literal_classifier_preserves_caller_eligibility_populations():
    def allowed(_written, _locale):
        return True

    def refused(_written, _locale):
        return False

    assert (
        literal_spell_say(
            "PLAIN", "cafe\N{COMBINING ACUTE ACCENT}", "café", "en_US", eligible=allowed
        )
        == "say"
    )
    assert literal_spell_say("PLAIN", "cat", "cat", "en_US", eligible=refused) is None
    assert literal_spell_say("LETTERS", "FBI", "f b i", "en_US", eligible=allowed) == "spell"


def test_distribution_version_reports_unavailable_package(monkeypatch):
    def unavailable(_distribution):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", unavailable)
    assert distribution_version("not-installed") == "unknown"
