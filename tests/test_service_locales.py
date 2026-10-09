"""Service-startup and resolver-cost contracts for the eleven-locale profile."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from icukit.detectors import detect
from icukit.recognize import FlexibleNumberDetector

import frend
import frend.ranges as ranges
import frend.runtime
from frend.fold_resolve import resolve
from frend.normalize import _reading_detectors
from frend.ranges import _SPOKEN_INTERVAL_FIELDS, date_interval_readers

REPO = Path(__file__).parents[1]
LOCALES = (
    "en_US",
    "es_MX",
    "es_ES",
    "fr_FR",
    "de_DE",
    "pt_BR",
    "it_IT",
    "zh_CN",
    "ko_KR",
    "ja_JP",
    "pt_PT",
)
SPOKEN_DATE_INTERVALS = getattr(ranges, "SPOKEN_DATE_INTERVALS", None)


def test_reading_profile_has_one_number_detector():
    for locale in LOCALES:
        numbers = [
            detector
            for detector in _reading_detectors(locale)
            if isinstance(detector, FlexibleNumberDetector)
        ]
        assert len(numbers) == 1, locale


def test_detect_yields_no_duplicate_detections():
    found = detect("나는 123 개를 샀다", _reading_detectors("ko_KR"))
    decimals = [
        (item["type"], item["start"], item["end"])
        for item in found
        if item["type"] == "number:decimal" and item["start"] == 3 and item["end"] == 6
    ]
    assert decimals == [("number:decimal", 3, 6)]


@pytest.mark.parametrize("locale", LOCALES)
def test_interval_readers_identical_to_full_family_filtered(locale, monkeypatch):
    assert SPOKEN_DATE_INTERVALS is not None
    monkeypatch.setattr(ranges, "has_icu_ranges", lambda: True)
    from icukit.engine import DATE_INTERVAL_FAMILY, generated_detectors

    expected = {
        str(reader.type)
        for reader in generated_detectors(locale, [DATE_INTERVAL_FAMILY]).detectors
        if set(str(reader.type).removeprefix("date-interval:")) <= _SPOKEN_INTERVAL_FIELDS
    }
    actual = {str(reader.type) for reader in date_interval_readers(locale)}
    assert expected, f"test did not exercise interval readers for {locale}"
    assert actual == expected


def test_interval_family_is_one_module_level_instance():
    assert SPOKEN_DATE_INTERVALS is not None
    from icukit import engine

    seen = []
    original = engine.generated_detectors

    def record(locale, families):
        seen.extend(families)
        return original(locale, families)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(ranges, "has_icu_ranges", lambda: True)
        monkeypatch.setattr(engine, "generated_detectors", record)
        date_interval_readers("en_US")
        date_interval_readers("fr_FR")

    assert len(seen) == 2
    assert all(family is SPOKEN_DATE_INTERVALS for family in seen)


def test_lazy_first_hit_builds_only_that_locale():
    code = """
import json
from frend import normalize
from frend.normalize import _reading_detectors
normalize('123', locale='ko_KR')
print(json.dumps(_reading_detectors.cache_info()._asdict()))
"""
    environment = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=REPO,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["currsize"] == 1


@pytest.mark.parametrize(
    "locales",
    (
        ("en-US", "fr_FR", "en_US", "FR-fr"),
        (locale for locale in ("en-US", "fr_FR", "en_US", "FR-fr")),
    ),
    ids=("tuple", "generator"),
)
def test_prewarm_loads_each_requested_locale_once(monkeypatch, locales):
    calls = []

    def record(text, *, locale):
        calls.append((text, locale))

    normalize_module = importlib.import_module("frend.normalize")
    monkeypatch.setattr(normalize_module, "normalize", record)
    loaded = frend.runtime.prewarm(locales)
    assert loaded == ("en_US", "fr_FR")
    assert calls == [
        ("123", "en_US"),
        ("March 3, 2020", "en_US"),
        ("123", "fr_FR"),
        ("02.03.2003", "fr_FR"),
    ]


@pytest.mark.parametrize("locales", ("en_US", b"en_US"), ids=("str", "bytes"))
def test_prewarm_rejects_string_like_container_before_loading(locales, monkeypatch):
    calls = []

    def record(text, *, locale):
        calls.append((text, locale))

    normalize_module = importlib.import_module("frend.normalize")
    monkeypatch.setattr(normalize_module, "normalize", record)

    with pytest.raises(
        TypeError,
        match=r"locales must be an iterable of locale tags, not str or bytes",
    ):
        frend.runtime.prewarm(locales)

    assert calls == []


def test_prewarm_is_exported():
    assert "prewarm" in frend.__all__
    assert frend.prewarm is frend.runtime.prewarm


@pytest.mark.parametrize(
    "detections",
    (
        (),
        ({"type": "number:decimal", "text": "1", "start": 0, "end": 1, "value": 1},),
        (
            {"type": "x", "text": "12", "start": 0, "end": 2, "value": "a"},
            {"type": "y", "text": "12", "start": 0, "end": 2, "value": "b"},
            {"type": "z", "text": "23", "start": 1, "end": 3, "value": "c"},
        ),
    ),
)
def test_cover_resolution_identical_on_golden_inputs(detections):
    selection = resolve(detections, locale="en_US", n=64)
    assert (
        tuple(tuple(item["type"] for item in cover) for cover in selection.covers)
        == {
            0: ((),),
            1: (("number:decimal",), ()),
            3: (("x",), ("y",), ("z",), ()),
        }[len(detections)]
    )
