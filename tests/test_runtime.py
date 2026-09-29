"""frend.freeze_after_setup: the collector is the caller's to change, after setup."""

from __future__ import annotations

import gc
import subprocess
import sys

import frend


def test_importing_frend_freezes_nothing():
    """Importing frend (and resolving with it) leaves the collector as it was."""
    code = (
        "import gc\n"
        "before = gc.get_freeze_count()\n"
        "import frend\n"
        "frend.resolve([{'start': 0, 'end': 1, 'type': 'number', 'text': '1'}])\n"
        "print(gc.get_freeze_count() - before, gc.isenabled())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["0", "True"]


def test_freeze_after_setup_freezes_what_is_alive_and_leaves_collection_on():
    before = gc.get_freeze_count()
    try:
        frozen = frend.freeze_after_setup()
        assert frozen == gc.get_freeze_count() > before
        assert gc.isenabled()
    finally:
        gc.unfreeze()
