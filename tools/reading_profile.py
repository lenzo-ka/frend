"""The evaluator's reading profile: every reader frend reads a written token with, by locale.

It is the spoken-priors recognition profile (``build_spoken_priors._detectors``),
flattened in its kinds' order with each reader once, followed by the readers that sit
outside the corpus kinds: icukit's abbreviation lexicon, frend's letters reader, and
frend's reader of the lexicon's written variants ("Mr", "st", "Vol"), and last frend's
reader of ranges written in running text ("5-10"), whose ends every reader before it
reads.
Every reader is built for the locale asked for; none is built with a literal locale.
The list is cached per locale (canonicalized first, so "en-US" is "en_US"), so a caller
holds the very list the evaluator reads.

This is the evaluator's profile. ARCTIC keeps its own detector set (its
``build_graphs.py``) and does not read this one.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
for _path in (_REPO, _TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from frend.locale_data import LOCALE_CACHE, canonical_locale  # noqa: E402

__all__ = ["reading_detectors"]


def reading_detectors(locale: str = "en_US") -> list[object]:
    """The readers for ``locale``: the spoken profile's, then the abbreviation, letters,
    abbreviation-variant and written-range readers. The same list object for every call
    with the same locale."""
    return _profile(canonical_locale(locale))


@lru_cache(maxsize=LOCALE_CACHE)
def _profile(locale: str) -> list[object]:
    from build_spoken_priors import _detectors
    from icukit import FlexibleNumericDurationDetector
    from icukit.abbreviation_recognize import AbbreviationDetector

    from frend.abbreviation_variants import AbbreviationVariantDetector
    from frend.durations import NumericDurationDetector
    from frend.letters import LettersDetector
    from frend.ranges import RangeDetector

    seen, detectors = set(), []
    for group in _detectors(locale).values():
        for detector in group:
            if isinstance(detector, FlexibleNumericDurationDetector):
                detector = NumericDurationDetector(locale)
            if id(detector) not in seen:
                seen.add(id(detector))
                detectors.append(detector)
    detectors.append(AbbreviationDetector(locale))
    detectors.append(LettersDetector(locale))
    detectors.append(AbbreviationVariantDetector(locale))
    # A range written in running text ("5-10", "16:79", "3x4"), its ends read by every
    # reader above (``frend.ranges.RangeDetector``; none where the locale has no range
    # table).
    detectors.append(RangeDetector(locale, endpoints=tuple(detectors)))
    return detectors
