"""frend.freeze_after_setup: the collector is the caller's to change, after setup."""

from __future__ import annotations

import gc
import subprocess
import sys

import frend  # noqa: F401 - imported so the process state below includes it
from frend.normalize import _reading_detectors

_FROZEN_AT_IMPORT = gc.get_freeze_count()


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
    """Run in its own interpreter: freezing, and the unfreeze that undoes it, must not
    touch the test process, whose interpreter may have frozen objects of its own."""
    code = (
        "import gc, frend\n"
        "before = gc.get_freeze_count()\n"
        "frozen = frend.freeze_after_setup()\n"
        "print(frozen == gc.get_freeze_count() > before, gc.isenabled())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["True", "True"]


def test_this_process_keeps_its_frozen_objects():
    """The tests above leave the test process's frozen generation as they found it."""
    assert gc.get_freeze_count() == _FROZEN_AT_IMPORT


def test_normalize_retains_one_compiled_reading_gang_per_locale():
    threshold = 987_654
    registry = _reading_detectors("en_US", threshold)

    assert registry.gang._compiled is None
    frend.normalize("123", symbol_run_threshold=threshold)
    compiled = registry.gang._compiled
    frend.normalize("456", symbol_run_threshold=threshold)

    assert compiled is not None
    assert registry.gang._compiled is compiled
    assert _reading_detectors("en_US", threshold).gang is registry.gang
