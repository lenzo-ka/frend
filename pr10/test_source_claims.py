from __future__ import annotations

import csv
from pathlib import Path

CLAIMS_PATH = Path(__file__).with_name("source-claims.tsv")


def _claims() -> dict[str, dict[str, str]]:
    with CLAIMS_PATH.open(encoding="utf-8", newline="") as source:
        return {row["claim_id"]: row for row in csv.DictReader(source, delimiter="\t")}


def test_supported_speech_claims_link_to_the_evidence_pages() -> None:
    claims = _claims()
    expected_pages = {
        "pr10-002": ("fr-FR", (15, 11)),
        "pr10-004": ("de-DE", (17,)),
        "pr10-008": ("pt-BR", (16,)),
        "pr10-010": ("pt-PT", (15,)),
        "pr10-012": ("zh-CN", (17,)),
        "pr10-014": ("ja-JP", (14, 10)),
        "pr10-020": ("de-DE", (18,)),
    }

    for claim_id, (manual, pages) in expected_pages.items():
        locators = claims[claim_id]["stable_locator"].split(" ; ")
        assert locators == [
            "https://download.mytobiidynavox.com/Voices/AcapelaDocumentation/"
            f"AcapelaLanguageManual_{manual}.pdf#page={page}"
            for page in pages
        ]


def test_japanese_connector_quote_preserves_the_sourced_word() -> None:
    passage = _claims()["pr10-014"]["quoted_passage"]
    assert "with の (no) between groups" in passage
