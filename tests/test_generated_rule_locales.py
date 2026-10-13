from pathlib import Path

from frend.generated_rule_locales import GENERATED_RULE_CAPABILITIES, locales_for

REPO = Path(__file__).resolve().parents[1]


def test_every_consumer_projects_the_one_ordered_capability_declaration():
    expected = {
        "date": (
            "es_MX",
            "es_ES",
            "fr_FR",
            "de_DE",
            "pt_BR",
            "pt_PT",
            "it_IT",
            "zh_CN",
            "ko_KR",
            "ja_JP",
        ),
        "time": (
            "es_MX",
            "es_ES",
            "fr_FR",
            "de_DE",
            "pt_BR",
            "pt_PT",
            "it_IT",
            "zh_CN",
            "ko_KR",
            "ja_JP",
        ),
        "fraction": (
            "es_MX",
            "es_ES",
            "fr_FR",
            "de_DE",
            "pt_BR",
            "pt_PT",
            "it_IT",
            "zh_CN",
            "ko_KR",
            "ja_JP",
        ),
        "money": ("es_MX", "fr_FR", "pt_BR", "it_IT", "ko_KR"),
    }
    assert tuple(GENERATED_RULE_CAPABILITIES) == expected["date"]
    for family, locales in expected.items():
        assert locales_for(family) == locales
        assert (
            tuple(
                path.stem
                for path in sorted(
                    (REPO / "frend" / "data" / f"{family}_rules").glob("*.json"),
                    key=lambda path: locales.index(path.stem),
                )
            )
            == locales
        )


def test_locale_gate_keeps_its_independent_coverage_population():
    source = (REPO / "tools" / "locale_gate.py").read_text(encoding="utf-8")
    assert "generated_rule_locales" not in source
