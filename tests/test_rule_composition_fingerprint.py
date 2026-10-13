import json
from pathlib import Path

from tools.fingerprint_rule_composition import fingerprint

REPO = Path(__file__).resolve().parents[1]


def test_generated_rule_composition_matches_the_reviewed_full_matrix():
    expected = json.loads(
        (REPO / "tests/data/rule_composition_fingerprint.json").read_text(encoding="utf-8")
    )
    assert fingerprint() == expected
