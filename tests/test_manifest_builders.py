from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.manifest_builder import ManifestBuilder


def test_family_builders_delegate_to_the_shared_builder(monkeypatch):
    import tools.build_fraction_manifests as fraction
    import tools.build_time_manifests as time

    calls = []

    def fake_build(self, base_path, head_path):
        calls.append((self.output.family, base_path, head_path))
        return ({"family": self.output.family}, {"family": self.output.family})

    monkeypatch.setattr(ManifestBuilder, "build", fake_build)
    base_path = Path("base.json")
    head_path = Path("head.json")

    assert time.build(base_path, head_path)[0]["family"] == "time"
    assert fraction.build(base_path, head_path)[0]["family"] == "fraction"
    assert calls == [
        ("time", base_path, head_path),
        ("fraction", base_path, head_path),
    ]


def _write_coverage(path: Path, cases: list[dict]) -> None:
    path.write_text(json.dumps({"frend_head": path.stem, "cases": cases}), encoding="utf-8")


def test_family_builders_preserve_head_time_and_base_fraction_routing(tmp_path):
    import tools.build_fraction_manifests as fraction
    import tools.build_time_manifests as time

    base_path = tmp_path / "base.json"
    head_path = tmp_path / "head.json"
    _write_coverage(
        base_path,
        [
            {
                "id": "base-fraction",
                "locale": "fr_FR",
                "offer_signature": [{"type": "fraction:cardinal"}],
            },
            {"id": "head-time", "locale": "fr_FR", "offer_signature": []},
        ],
    )
    _write_coverage(
        head_path,
        [
            {"id": "base-fraction", "locale": "fr_FR", "offer_signature": []},
            {
                "id": "head-time",
                "locale": "fr_FR",
                "offer_signature": [{"alternatives": [{"provenance": "time-rule:clock"}]}],
            },
        ],
    )

    time_changes, _ = time.build(base_path, head_path)
    fraction_changes, _ = fraction.build(base_path, head_path)

    assert [row["id"] for row in time_changes["rows"]] == ["head-time"]
    assert [row["id"] for row in fraction_changes["rows"]] == ["base-fraction"]


@pytest.mark.parametrize(
    ("module_name", "expected_message"),
    [
        ("tools.build_time_manifests", "coverage row 'recovered' has no sourced recovered route"),
        (
            "tools.build_fraction_manifests",
            "coverage row 'recovered' has no sourced recovered target route",
        ),
    ],
)
def test_family_builders_preserve_recovery_failure_diagnostics(
    tmp_path, monkeypatch, module_name, expected_message
):
    import importlib

    import tools.locale_gate as locale_gate

    module = importlib.import_module(module_name)
    edge = {
        "type": "fraction:cardinal",
        "alternatives": [{"provenance": "time-rule:clock+normalization-record:source"}],
    }
    base_path = tmp_path / "base.json"
    head_path = tmp_path / "head.json"
    _write_coverage(
        base_path,
        [
            {
                "id": "recovered",
                "locale": "fr_FR",
                "strict": False,
                "offer_signature": [edge],
            }
        ],
    )
    _write_coverage(
        head_path,
        [
            {
                "id": "recovered",
                "locale": "fr_FR",
                "strict": True,
                "targets": ["speech"],
                "target_hits": {"strict": [True]},
                "offer_signature": [edge],
            }
        ],
    )
    monkeypatch.setattr(locale_gate, "_source_record_offers", lambda *args, **kwargs: False)

    with pytest.raises(ValueError, match=f"^{expected_message}$"):
        module.build(base_path, head_path)
