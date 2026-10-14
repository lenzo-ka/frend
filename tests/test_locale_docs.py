from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "locales.md"


def test_locale_guide_is_linked_from_the_readme():
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    assert "[The locale contribution guide](docs/locales.md)" in readme


def test_locale_guide_references_only_existing_repository_paths():
    text = DOC.read_text(encoding="utf-8")
    references = set(
        re.findall(
            r"(?<![A-Za-z0-9_./-])((?:frend|pr10|tests|tools)/[A-Za-z0-9_.-]+"
            r"(?:/[A-Za-z0-9_.-]+)*)",
            text,
        )
    )

    assert references
    missing = sorted(
        path
        for path in references
        if not (REPO / path).exists() and not (path.endswith("_") and (REPO / path).parent.is_dir())
    )
    assert missing == []


def test_every_documented_python_script_exists():
    text = DOC.read_text(encoding="utf-8")
    scripts = set(re.findall(r"python(?: -B)? ((?:tests|tools)/[A-Za-z0-9_./-]+\.py)", text))

    assert scripts
    missing = sorted(script for script in scripts if not (REPO / script).is_file())
    assert missing == []
