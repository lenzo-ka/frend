"""Tracked text must not encode a contributor's machine layout or name."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SELF = Path(__file__).relative_to(_REPO).as_posix()
_LOCAL = re.compile(r"/Users/|/home/|/mnt/|\b[A-Za-z]:\\[A-Za-z]|~/dev|/Volumes/|kalman|\bk02\b")


def _tracked_text() -> list[tuple[str, str]]:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=_REPO).split(b"\0")
    files = []
    for raw_name in names:
        if not raw_name:
            continue
        name = os.fsdecode(raw_name)
        if name == _SELF:
            continue
        data = (_REPO / name).read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        files.append((name, text))
    return files


def test_tracked_text_has_no_local_machine_names_or_paths():
    matches = [
        f"{name}:{line_number}:{line}"
        for name, text in _tracked_text()
        for line_number, line in enumerate(text.splitlines(), 1)
        if _LOCAL.search(line)
    ]
    assert matches == []
