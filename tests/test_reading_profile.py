"""The evaluator's shared reading profile (``tools/reading_profile.py``), by locale."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))


def _evaluator():
    spec = importlib.util.spec_from_file_location(
        "evaluate_google_tn", _TOOLS / "evaluate_google_tn.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _signature(detector) -> tuple:
    """A reader's class and its plain settings: two builds of one profile compare equal."""
    return (
        type(detector).__name__,
        tuple(
            sorted(
                (key, repr(value))
                for key, value in vars(detector).items()
                if isinstance(value, (str, int, float, bool, type(None)))
            )
        ),
    )


def test_evaluator_reads_the_shared_profile():
    import reading_profile

    assert _evaluator()._detectors() is reading_profile.reading_detectors("en_US")


def test_shared_profile_is_the_spoken_profile_plus_the_outside_readers():
    import build_spoken_priors
    import reading_profile
    from icukit.abbreviation_recognize import AbbreviationDetector

    from frend.letters import LettersDetector

    seen, spoken = set(), []
    for group in build_spoken_priors._detectors("en_US").values():
        for detector in group:
            if id(detector) not in seen:
                seen.add(id(detector))
                spoken.append(detector)
    shared = reading_profile.reading_detectors("en_US")
    assert [_signature(d) for d in shared[: len(spoken)]] == [_signature(d) for d in spoken]
    assert len(shared) == len(spoken) + 2
    assert isinstance(shared[-2], AbbreviationDetector)
    assert isinstance(shared[-1], LettersDetector)


def _names_an_en_locale(value: object) -> bool:
    return isinstance(value, str) and (value == "en" or value.startswith(("en_", "en-")))


def test_reading_detectors_ru_constructs_no_en_US_detector():
    """Every reader asked for Russian is built for Russian: none keeps English as its
    locale or holds another attribute naming an English locale (``FlexibleTimeDetector``
    also keeps ``_language``). This does not say the readers read Russian well; only
    that none of them is built for English when Russian is asked for."""
    import reading_profile

    detectors = reading_profile.reading_detectors("ru_RU")
    assert detectors
    assert {getattr(detector, "locale", None) for detector in detectors} == {"ru_RU"}
    english = [
        (type(detector).__name__, key, value)
        for detector in detectors
        for key, value in vars(detector).items()
        if _names_an_en_locale(value)
    ]
    assert english == []
