"""The evaluator's shared reading profile (``tools/reading_profile.py``), by locale."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

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
    """A reader's class and its plain public settings: two builds of one profile compare
    equal. Private attributes are left out: a reader may fill a lazy cache once it has
    read something (icukit's range readers set ``_bare`` on first use), which says
    nothing about how it was built."""
    return (
        type(detector).__name__,
        tuple(
            sorted(
                (key, repr(value))
                for key, value in vars(detector).items()
                if not key.startswith("_")
                and isinstance(value, (str, int, float, bool, type(None)))
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
    from frend.abbreviation_variants import AbbreviationVariantDetector

    shared = reading_profile.reading_detectors("en_US")
    assert [_signature(d) for d in shared[: len(spoken)]] == [_signature(d) for d in spoken]
    assert len(shared) == len(spoken) + 3
    assert isinstance(shared[-3], AbbreviationDetector)
    assert isinstance(shared[-2], LettersDetector)
    assert isinstance(shared[-1], AbbreviationVariantDetector)


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


def _paths(graph) -> list[tuple[str, ...]]:
    """Every root-to-sink token sequence of an alignment graph, read from its wire JSON
    (``frend:next`` relations, ``frend:token`` attributes), as ARCTIC's measure reads it."""
    import json
    from collections import defaultdict

    from tiergraph import wire

    wired = json.loads(wire.dumps(graph.graph))["graph"]
    items = wired["tiers"][0]["items"]
    token = [
        next((a["lexical"] for a in item["attributes"] if a["name"] == "frend:token"), None)
        for item in items
    ]
    ids = [item["durable_id"] for item in items]
    following = defaultdict(list)
    for relation in wired["relations"]:
        following[relation["left"]["index"]].append(relation["right"]["index"])
    root = ids.index("B0")
    sink = max((i for i, d in enumerate(ids) if d.startswith("B")), key=lambda i: int(ids[i][1:]))
    found, stack = [], [(root, ())]
    while stack:
        at, said = stack.pop()
        said = said + ((token[at],) if token[at] is not None else ())
        if at == sink:
            found.append(said)
        stack.extend((after, said) for after in following[at])
    return found


@pytest.mark.parametrize(("text", "said"), [("Mr McVeigh", "Mister"), ("Mrs Hall", "Missus")])
def test_keep_all_path_reads_mister(text, said):
    """ARCTIC's keep-all path (detect, resolve_choices, compose_choices, build_align_graph)
    with the variant reader alone, as ``build_graphs-P4.patch`` adds it: the variant is a
    branch beside the text as written. The graph counts 4 paths: the passthrough route,
    the variant's spoken form, its as-written form (which says what the passthrough route
    says), and its letters (the corpus spells "mr" 135 times of 8,756, "mrs" 12 of
    3,309)."""
    from icukit.detectors import detect

    from frend import compose_choices, resolve_choices
    from frend.abbreviation_variants import AbbreviationVariantDetector
    from frend.align_graph import build_align_graph
    from frend.spoken_priors import spoken_tokens

    choices = compose_choices(
        resolve_choices(detect(text, [AbbreviationVariantDetector("en_US")]), source_text=text)
    )
    assert said in [item.text for unit in choices.units for item in unit.alternatives]
    graph = build_align_graph(choices)
    assert graph.count_plan().evaluate().value == 4
    rest = spoken_tokens(text)[1:]
    letters = spoken_tokens(" ".join(text.split()[0].lower())) + rest
    assert sorted(_paths(graph)) == sorted(
        [spoken_tokens(said) + rest, spoken_tokens(text), spoken_tokens(text), letters]
    )
