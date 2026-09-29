"""Process-level helpers for callers that keep frend loaded and resolve many texts.

Nothing here runs on import: changing the interpreter's garbage collector is the
caller's decision, made once its own setup is done.
"""

from __future__ import annotations

import gc


def freeze_after_setup() -> int:
    """Move every object alive now into the collector's permanent generation.

    Call it once, after setup: after the readers are built (``reading_detectors``),
    the priors and tables are loaded, and a first text has been resolved and
    verbalized so every lazy cache is filled. Those objects live for the process, yet
    each full collection walks all of them; frozen, they are never scanned again, so
    a later collection pays only for what a resolve allocates. In a server that
    forks workers, call it in the parent before forking: frozen objects are also left
    untouched by the collector in the children, so their pages stay shared.

    It collects first, so garbage made during setup is freed rather than frozen, and
    returns how many objects are now frozen (``gc.get_freeze_count()``). Objects
    made later are collected as usual. ``gc.unfreeze()`` undoes it.
    """
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()
