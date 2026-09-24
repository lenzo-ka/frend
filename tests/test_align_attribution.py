"""Alignment-attribution falsifiers and brute-force oracles."""

from __future__ import annotations

import ast
import importlib.metadata
import math
from collections import defaultdict
from dataclasses import replace
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from frend.align_attribution import attribute
from frend.align_export import to_fsg_text
from frend.align_graph import build_align_graph
from frend.fold_resolve import CoverScore
from frend.lattice import (
    ChoiceGraph,
    ReadingEdge,
    ReadingRank,
    SemanticRank,
    resolve_choices,
)
from frend.type_priors import ReadingPrior
from frend.verbalize import SpokenAlternative, VerbalizedUnit


def prior(p=".2", *, tier="measured", supported=True, n=100):
    return ReadingPrior(
        "a",
        "A",
        Decimal(p) if p is not None else None,
        n,
        supported,
        tier,
    )


def choices(text, specs=()):
    base = resolve_choices([], source_text=text)
    edges, units = [], []
    for index, (start, end, forms, measurement) in enumerate(specs):
        item_id = f"c{index}"
        geometry = CoverScore(end - start, 1, 0)
        semantic = SemanticRank("unsupported", None, None)
        edges.append(
            ReadingEdge(
                item_id,
                start,
                end,
                None,
                "reading",
                geometry,
                measurement,
                ReadingRank(end - start, 1, 0, semantic),
            )
        )
        units.append(
            VerbalizedUnit(
                item_id,
                tuple(
                    SpokenAlternative(form, f"source:{form_index}", Decimal(".7"))
                    for form_index, form in enumerate(forms)
                ),
                None,
                None,
                True,
            )
        )
    return ChoiceGraph(replace(base, edges=tuple(edges) + base.edges), tuple(units))


def paths(alignment):
    plan = alignment.log_plan()

    def walk(index, path):
        path = (*path, index)
        if not plan.children[index]:
            yield path
        else:
            for child in plan.children[index]:
                yield from walk(child, path)

    return list(walk(plan.roots[0], ()))


def _path_tokens(alignment, path):
    plan = alignment.log_plan()
    return tuple(token for index in path for token in alignment.items[plan.labels[index]].tokens)


def _path_mass(alignment, path, *, prior_scale=1.0, prior_enabled=True):
    plan = alignment.log_plan()
    total = 0.0
    for index in path:
        item = alignment.items[plan.labels[index]]
        if item.weight.scored:
            total += plan.values[index] * prior_scale if prior_enabled else 0.0
        else:
            total += plan.values[index]
    return math.exp(total)


def _attribute_accepted(alignment, tokens):
    try:
        return attribute(alignment, tokens)
    except ValueError as error:
        raise AssertionError(
            f"attribute() refused an aligned output the graph accepts: {error}"
        ) from error


def _expected_inventory(alignment, all_paths):
    plan = alignment.log_plan()
    readings = set()
    classes = defaultdict(lambda: defaultdict(set))
    for path in all_paths:
        path_items = [alignment.items[plan.labels[index]] for index in path]
        for item in path_items:
            if item.role == "reading":
                readings.add((item.span, item.reading_id))
            elif item.role == "form-exit":
                tokens = tuple(
                    token
                    for path_item in path_items
                    if path_item.reading_id == item.reading_id
                    for token in path_item.tokens
                )
                member = (item.reading_id, item.id, item.reading_id, "reading")
                classes[item.span, tokens][member].add(item.id)
            elif item.role == "token" and item.span is not None:
                member = (item.id, item.id, None, "passthrough")
                classes[item.span, item.tokens][member].add(item.id)
    return readings, classes


@pytest.mark.parametrize(
    "carrier",
    [
        pytest.param(choices("x", [(0, 1, ["x"], prior(".7"))]), id="one-scored-member"),
        pytest.param(
            choices(
                "x",
                [
                    (0, 1, ["same"], prior(".8")),
                    (0, 1, ["same"], prior(".2")),
                ],
            ),
            id="two-scored-members-one-span",
        ),
        pytest.param(choices("x"), id="unscored-only-class"),
        pytest.param(choices("x", [(0, 1, [""], prior(".7"))]), id="empty-form"),
        pytest.param(
            choices("x", [(0, 1, ["a b", "a c"], prior(".7"))]),
            id="within-reading-shared-token-prefix",
        ),
        pytest.param(
            choices(
                "x",
                [
                    (0, 1, ["same"], prior(".7")),
                    (0, 1, ["same"], prior(".4", tier="icu-backfill")),
                ],
            ),
            id="mixed-scored-and-unscored-members",
        ),
        pytest.param(
            choices(
                "ab",
                [
                    (0, 1, ["shared", "alpha"], prior(".8")),
                    (0, 1, ["shared"], prior(".2")),
                    (1, 2, ["bee", "b"], prior(".4")),
                    (0, 2, ["shared b"], prior(".1")),
                ],
            ),
            id="overlapping-spans",
        ),
    ],
)
def test_attribution_matches_brute_force_path_sums(carrier):
    """Compare complete inventories and masses with enumerated exported paths."""
    alignment = build_align_graph(carrier)
    all_paths = paths(alignment)
    expected_readings, expected_classes = _expected_inventory(alignment, all_paths)
    candidates = sorted({_path_tokens(alignment, path) for path in all_paths})
    for tokens in candidates:
        result = _attribute_accepted(alignment, tokens)
        assert result.meaning == "acoustics plus measured prior"
        accepted = [path for path in all_paths if _path_tokens(alignment, path) == tokens]
        total = math.fsum(_path_mass(alignment, path) for path in accepted)
        assert not result.zero_mass

        actual_readings = {
            (span.span, reading.reading_id): reading
            for span in result.spans
            for reading in span.readings
        }
        actual_classes = {
            (span.span, output_class.tokens): output_class
            for span in result.spans
            for output_class in span.output_classes
        }
        assert set(actual_readings) == expected_readings
        assert set(actual_classes) == set(expected_classes)
        for key, expected_members in expected_classes.items():
            actual_members = [
                (member.id, member.item_id, member.reading_id, member.role)
                for member in actual_classes[key].members
            ]
            assert sorted(actual_members) == sorted(expected_members)

        assert result.total_mass == pytest.approx(total)
        by_label = defaultdict(float)
        for path in accepted:
            labels = {alignment.log_plan().labels[index] for index in path}
            mass = _path_mass(alignment, path)
            for label in labels:
                by_label[label] += mass
        for (_span, reading_id), reading in actual_readings.items():
            reading_mass = by_label[reading_id]
            reading_log_mass = math.log(reading_mass) if reading_mass else -math.inf
            assert reading.mass == pytest.approx(reading_mass)
            assert reading.log_mass == pytest.approx(reading_log_mass)
            assert reading.posterior == pytest.approx(reading_mass / total)
        for key, expected_members in expected_classes.items():
            output_class = actual_classes[key]
            member_labels = {label for labels in expected_members.values() for label in labels}
            expected = math.fsum(by_label[label] for label in member_labels)
            assert output_class.mass == pytest.approx(expected)
            assert output_class.posterior == pytest.approx(expected / total)
            assert output_class.posterior == pytest.approx(
                math.fsum(member.posterior for member in output_class.members)
            )
            actual_members = {
                (member.id, member.item_id, member.reading_id, member.role): member
                for member in output_class.members
            }
            for member_record, labels in expected_members.items():
                member_mass = math.fsum(by_label[label] for label in labels)
                member_log_mass = math.log(member_mass) if member_mass else -math.inf
                assert actual_members[member_record].mass == pytest.approx(member_mass)
                assert actual_members[member_record].log_mass == pytest.approx(member_log_mass)


def test_isolated_i_pools_audio_class_and_splits_only_scored_members():
    """Kill per-reading acoustic labels and shares assigned to passthrough."""
    alignment = build_align_graph(
        choices(
            "I",
            [
                (0, 1, ["I"], prior(".6")),
                (0, 1, ["I"], prior(".3")),
                (0, 1, ["one"], prior(".1")),
            ],
        )
    )
    result = attribute(alignment, ("i",))
    output_class = next(
        output_class
        for output_class in result.spans[0].output_classes
        if output_class.tokens == ("i",)
    )
    assert output_class.posterior == pytest.approx(1.0)
    assert {member.role for member in output_class.members} == {"reading", "passthrough"}
    assert output_class.no_evidence == ("tok0_1",)
    assert output_class.prior_split.meaning == "prior, not audio evidence"
    assert [(share.member, share.p, share.share) for share in output_class.prior_split.shares] == [
        ("c0", Decimal(".6"), Fraction(2, 3)),
        ("c1", Decimal(".3"), Fraction(1, 3)),
    ]
    assert output_class.prior_split.tied == ("c0",)


def test_one_scored_and_one_unscored_reading_has_no_invented_share():
    """Kill treating an unmeasured member's unit path factor as evidence."""
    alignment = build_align_graph(
        choices(
            "x",
            [
                (0, 1, ["same"], prior(".7")),
                (0, 1, ["same"], prior(None, supported=False)),
            ],
        )
    )
    items = dict(alignment.items)
    items["c1"] = replace(
        items["c1"],
        weight=replace(items["c1"].weight, p=Decimal(".4")),
    )
    alignment = replace(alignment, items=items)
    output_class = next(
        output_class
        for output_class in attribute(alignment, ("same",)).spans[0].output_classes
        if output_class.tokens == ("same",)
    )
    assert [(share.member, share.share) for share in output_class.prior_split.shares] == [
        ("c0", Fraction(1))
    ]
    assert output_class.no_evidence == ("c1",)


def test_prior_split_is_exact_and_independent_of_decimal_context():
    """Keep exact normalization independent of the caller's Decimal context."""
    alignment = build_align_graph(
        choices(
            "x",
            [
                (0, 1, ["same"], prior(".1")),
                (0, 1, ["same"], prior(".1")),
                (0, 1, ["same"], prior(".1")),
            ],
        )
    )
    observed = []
    for precision in (1, 2, getcontext().prec):
        with localcontext() as context:
            context.prec = precision
            output_class = next(
                output_class
                for output_class in attribute(alignment, ("same",)).spans[0].output_classes
                if output_class.tokens == ("same",)
            )
        shares = tuple(share.share for share in output_class.prior_split.shares)
        assert shares == (Fraction(1, 3),) * 3
        assert sum(shares, Fraction()) == Fraction(1)
        observed.append(shares)
    assert observed[0] == observed[1] == observed[2]

    unscored_class = attribute(build_align_graph(choices("x")), ("x",)).spans[0].output_classes[0]
    assert unscored_class.prior_split.meaning == "prior, not audio evidence"
    assert unscored_class.prior_split.shares == ()


def test_prior_split_ties_use_exact_decimal_measurements():
    """Kill deciding within-class prior ties from rounded float values."""
    high = ".50000000000000000000000000000000000000000000000001"
    low = ".50000000000000000000000000000000000000000000000000"
    alignment = build_align_graph(
        choices(
            "x",
            [
                (0, 1, ["same"], prior(high)),
                (0, 1, ["same"], prior(low)),
            ],
        )
    )
    output_class = next(
        output_class
        for output_class in attribute(alignment, ("same",)).spans[0].output_classes
        if output_class.tokens == ("same",)
    )
    assert output_class.prior_split.tied == ("c0",)


def test_foreign_output_refuses_at_first_divergent_token():
    """Kill fabricated distributions and imprecise foreign-output diagnostics."""
    alignment = build_align_graph(choices("I", [(0, 1, ["I"], prior("1"))]))
    with pytest.raises(
        ValueError,
        match="decoder output is not a path of the exported graph.*'foreign'",
    ):
        attribute(alignment, ("i", "foreign"))
    multi = build_align_graph(choices("x", [(0, 1, ["one two"], prior("1"))]))
    with pytest.raises(ValueError, match="first divergent token: 'foreign'"):
        attribute(multi, ("one", "foreign"))


def test_all_negative_infinity_returns_zero_mass_without_values():
    """Kill uniform fallback distributions on a zero-mass conditioned plan."""
    alignment = build_align_graph(choices("x", [(0, 1, ["said"], prior("1"))]))
    plan = alignment.log_plan()
    alignment.__dict__["_log_plan"] = replace(plan, values=(-math.inf,) * len(plan.values))
    result = attribute(alignment, ("said",))
    assert result.zero_mass
    assert result.total_log_mass is None
    assert result.total_mass is None
    assert result.spans == ()


def test_prior_switch_and_scale_change_only_measured_factors():
    """Kill ignored scaling, absolute replacement, and a distinct scale-zero path."""
    alignment = build_align_graph(
        choices(
            "x",
            [
                (0, 1, ["same"], prior(".8")),
                (0, 1, ["same"], prior(".2")),
            ],
        )
    )

    def readings(result):
        return [reading.posterior for reading in result.spans[0].readings]

    enabled = readings(attribute(alignment, ("same",)))
    doubled = readings(attribute(alignment, ("same",), prior_scale=2))
    disabled = attribute(alignment, ("same",), prior=False)
    zero = attribute(alignment, ("same",), prior_scale=0)
    assert enabled != pytest.approx(doubled)
    assert readings(disabled) == pytest.approx([1 / 2, 1 / 2])
    assert readings(zero) == pytest.approx(readings(disabled))
    assert zero.spans == disabled.spans


def test_path_removed_by_pronounceable_filter_refuses():
    """Kill attribution through alternatives excluded from the exported graph."""
    alignment = build_align_graph(
        choices("x", [(0, 1, ["said"], prior("1"))]),
        pronounceable=lambda token: token != "said",
    )
    with pytest.raises(ValueError, match="first divergent token: 'said'"):
        attribute(alignment, ("said",))


def test_one_best_modules_do_not_import_attribution():
    """Kill coupling the exact 1-best resolver to pass-2 attribution."""
    root = Path(__file__).resolve().parents[1]
    for filename in ("fold_resolve.py", "lattice.py"):
        tree = ast.parse((root / "frend" / filename).read_text(encoding="utf-8"))
        names = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        names.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert "frend.align_attribution" not in names


def test_real_fsg_reader_sequence_can_be_attributed(tmp_path):
    """Exercise an accepted real-reader sequence without inventing acoustics."""
    pocketsphinx = pytest.importorskip("pocketsphinx")
    alignment = build_align_graph(
        choices("x", [(0, 1, ["one"], prior(".8")), (0, 1, ["two"], prior(".2"))])
    )
    text, _manifest = to_fsg_text(alignment, "attribution-reader")
    source = tmp_path / "attribution-reader.fsg"
    source.write_text(text, encoding="utf-8")
    model = pocketsphinx.FsgModel.readfile(str(source), pocketsphinx.LogMath(), 1.0)
    assert model is not None
    assert model.accept("one")
    result = attribute(alignment, ("one",))
    assert not result.zero_mass
    assert result.aligned == ("one",)
    assert importlib.metadata.version("pocketsphinx") == "5.1.1"
