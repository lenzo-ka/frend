"""Run source mutations against alignment falsifiers, restoring every named path.

Run with the same Python used for pytest. Each mutation is compiled first and
must fail a collected test through an assertion or runtime behavior; collection,
syntax, and import failures do not count as kills. Results go to the named JSON
file. This script is independent of the normal pytest gate.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = "tests/test_align_graph.py::"
EXPORT_TEST = "tests/test_align_export.py::"
ATTRIBUTION_TEST = "tests/test_align_attribution.py::"
TEST_TIMEOUT_SECONDS = 30
# These exceptions, and their subclasses (UnboundLocalError, ModuleNotFoundError),
# show that a mutation removed a symbol or import, not that a behavioral assertion
# or runtime contract detected the mutated implementation.
NON_BEHAVIORAL_EXCEPTION_BASES = (NameError, AttributeError, ImportError)


def _is_non_behavioral(exception_type: str | None) -> bool:
    candidate = getattr(builtins, exception_type or "", None)
    return isinstance(candidate, type) and issubclass(candidate, NON_BEHAVIORAL_EXCEPTION_BASES)


@dataclass(frozen=True)
class Mutation:
    name: str
    test: str
    path: str
    before: str
    after: str


def graph(name, test, before, after):
    return Mutation(name, TEST + test, "frend/align_graph.py", before, after)


def normalization(name, before, after):
    return Mutation(
        name,
        TEST + "test_normalization_contract_generated",
        "frend/spoken_priors.py",
        before,
        after,
    )


def export(name, test, before, after):
    return Mutation(name, EXPORT_TEST + test, "frend/align_export.py", before, after)


def attribution(name, test, before, after):
    return Mutation(name, ATTRIBUTION_TEST + test, "frend/align_attribution.py", before, after)


MUTATIONS = [
    graph(
        "Bypass pre-reading nodes",
        "test_routes_match_brute_force_enumeration",
        "link(id, pre)",
        'link(id, f"B{j}")',
    ),
    graph(
        "Drop mid-word passthrough pairs",
        "test_routes_match_brute_force_enumeration",
        "if i >= j:",
        "if i >= j or j != b:",
    ),
    graph(
        "Drop separator reading junctions",
        "test_routes_match_brute_force_enumeration",
        "junctions = sorted({a, b} | ({p for p in kept if a <= p <= b}))",
        "junctions = [a, b]",
    ),
    graph(
        "Shift word-run end by one",
        "test_routes_match_brute_force_enumeration",
        "tokens = spoken_tokens(text[i:j])",
        "tokens = spoken_tokens(text[i:j-1])",
    ),
    graph(
        "Share a reading trie across readings",
        "test_routes_match_brute_force_enumeration",
        "prefixes = {(): edge.id}",
        'prefixes = {**locals().get("prefixes", {}), (): edge.id}',
    ),
    graph(
        "Merge distinct token forms",
        "test_routes_match_brute_force_enumeration",
        "forms[tokens].append(alternative)",
        "forms[tokens[:1]].append(alternative)",
    ),
    graph(
        "Duplicate relation instances",
        "test_routes_match_brute_force_enumeration",
        "links.append((left, right))",
        "links.extend([(left, right), (left, right)])",
    ),
    graph(
        "Double the reading factor",
        "test_posteriors_match_enumeration",
        "float((prior.p / maximum).ln())",
        "2 * float((prior.p / maximum).ln())",
    ),
    graph(
        "Place extra factors on word items",
        "test_posteriors_match_enumeration",
        'weight or ArcWeight(False, reason.get(role, "structure"))',
        'weight or ArcWeight(False, reason.get(role, "structure"), '
        '-1.0 if role == "word" else 0.0)',
    ),
    graph(
        "Place extra factors on junctions",
        "test_posteriors_match_enumeration",
        'weight or ArcWeight(False, reason.get(role, "structure"))',
        'weight or ArcWeight(False, reason.get(role, "structure"), '
        '-1.0 if role == "position" else 0.0)',
    ),
    graph(
        "Score backfill p",
        "test_weight_rule_table",
        'reason = "icu-backfill"',
        'return ArcWeight(True, "measured", float(prior.p.ln()), prior.p, prior.n)',
    ),
    graph(
        "Score backfill generated_p",
        "test_weight_rule_table",
        'reason = "icu-backfill"',
        'return ArcWeight(True, "measured", float(prior.generated_p.ln()), '
        "prior.generated_p, prior.n)",
    ),
    graph(
        "Turn attested zero into impossibility",
        "test_arc_weight_rule_table_directly",
        'reason = "attested-zero"',
        'return ArcWeight(False, "attested-zero", -float("inf"))',
    ),
    graph(
        "Score sparse shapes",
        "test_arc_weight_rule_table_directly",
        'reason = "sparse-shape"',
        'return ArcWeight(True, "sparse-shape")',
    ),
    graph(
        "Score unsupported readings",
        "test_arc_weight_rule_table_directly",
        'reason = "unsupported"',
        'return ArcWeight(True, "unsupported")',
    ),
    graph(
        "Score missing corpus sources",
        "test_arc_weight_rule_table_directly",
        'reason = "no-corpus-source"',
        'return ArcWeight(True, "no-corpus-source")',
    ),
    graph(
        "Score passthrough",
        "test_weight_rule_table",
        'weight or ArcWeight(False, reason.get(role, "structure"))',
        'weight or ArcWeight(role == "token", reason.get(role, "structure"))',
    ),
    graph(
        "Score spoken-source shares",
        "test_weight_rule_table",
        "alternatives=tuple(alternatives),",
        'alternatives=tuple(alternatives), weight=ArcWeight(True, "measured", '
        "float(alternatives[0].weight.ln()), alternatives[0].weight, 100),",
    ),
    graph(
        "Use corpus-group maxima",
        "test_overlapping_groups",
        "and _scored(mate)\n",
        "and _scored(mate) and mate.prior.group == prior.group\n",
    ),
    graph(
        "Use absolute log p",
        "test_best_scored_ties_passthrough",
        "float((prior.p / maximum).ln())",
        "float(prior.p.ln())",
    ),
    graph(
        "Move reading factor to exclusive exits",
        "test_factor_is_on_reading_item",
        "AttributeValue(_LOG_WEIGHT, XsdType.DOUBLE, repr(weight.log_weight)),",
        "AttributeValue(_LOG_WEIGHT, XsdType.DOUBLE, repr("
        '0.0 if item.role == "reading" else items[item.reading_id].weight.log_weight '
        'if item.role == "form-exit" else weight.log_weight)), ',
    ),
    graph(
        "Drop graph evidence marker",
        "test_factor_is_on_reading_item",
        'AttributeValue(_PRIOR_KIND, XsdType.STRING, "scored" if weight.scored else "unscored"),',
        "",
    ),
    graph(
        "Underflow via float before log",
        "test_arc_weight_survives_an_underflowing_ratio_directly",
        "float((prior.p / maximum).ln())",
        '__import__("math").log(float(prior.p / maximum))',
    ),
    graph(
        "Include backfill in measured maximum",
        "test_backfill_mate_does_not_change_measured_weights",
        "and _scored(mate)\n",
        "and mate.prior is not None and mate.prior.p is not None\n",
    ),
    graph(
        "Disable graph item bound",
        "test_item_cap_refuses_whole",
        "if len(items) >= item_cap:",
        "if False:",
    ),
    graph(
        "Allow reachable childless items",
        "test_missing_source_and_dead_end_refuse_before_graph",
        "if current != sink and not children[current]:",
        "if False:",
    ),
    graph(
        "Fabricate missing source text",
        "test_missing_source_and_dead_end_refuse_before_graph",
        'raise ValueError("source_text is required for alignment")',
        'text = "x" * lattice.text_length',
    ),
    graph(
        "Drop form-exit successor",
        "test_plans_accept_graph_and_only_final_sink",
        'link(exit_id, f"B{edge.end}")',
        "pass",
    ),
    graph(
        "Declare AND instead of OR",
        "test_plans_accept_graph_and_only_final_sink",
        "FoldTransition(_NEXT, ChildCombination.OR)",
        "FoldTransition(_NEXT, ChildCombination.AND)",
    ),
    graph(
        "Value COUNTING through double attribute",
        "test_plans_accept_graph_and_only_final_sink",
        "return self._prepare(COUNTING, _COUNT)",
        "return self._prepare(COUNTING, _LOG_WEIGHT)",
    ),
    graph(
        "Recompile cached log plan",
        "test_plans_accept_graph_and_only_final_sink",
        "@cached_property\n    def _log_plan",
        "@property\n    def _log_plan",
    ),
    graph(
        "Ignore replacement value vectors",
        "test_all_negative_infinity_has_zero_mass",
        "return self._log_plan\n",
        """plan = self._log_plan
        class StaleValues:
            def __getattr__(self, name):
                return getattr(plan, name)
            def marginals(self, values=None):
                return plan.marginals()
        return StaleValues()
""",
    ),
    graph(
        "Skip lexicon trimming",
        "test_lexicon_filter_trims_and_records",
        "if pronounceable is not None:\n        parents",
        "if False:\n        parents",
    ),
    export(
        "Sum instead of max when merging",
        "test_exports_equal_graph_language_and_best_products",
        """                if (
                    previous is None
                    or product > previous[0]
                    or (product == previous[0] and here < previous[1])
                ):
                    best[child] = (product, here)
""",
        """                if previous is None:
                    best[child] = (product, here)
                else:
                    best[child] = (previous[0] + product, min(previous[1], here))
""",
    ),
    export(
        "Charge the reading factor per token",
        "test_exports_equal_graph_language_and_best_products",
        "_Candidate(item_id, child, item.tokens[0], factors[item_id], (item_id,))",
        "_Candidate(\n"
        "                    item_id, child, item.tokens[0],\n"
        "                    factors[item_id] * factors.get(item.reading_id, Fraction(1)),\n"
        "                    (item_id,),\n"
        "                )",
    ),
    export(
        "Charge the reading factor per alternative",
        "test_exports_equal_graph_language_and_best_products",
        """            product = _bounded_fraction(
                probability * factors[current], f"null-chain product at item {current!r}"
            )
""",
        """            product = _bounded_fraction(
                probability
                * factors[current]
                * (
                    factors[item.reading_id]
                    if item.role == "form-exit" and item.reading_id
                    else Fraction(1)
                ),
                f"null-chain product at item {current!r}",
            )
""",
    ),
    export(
        "Multiply rounded Decimals during chain merge",
        "test_random_whole_routes_obey_fsg_text_tolerance_and_exact_manifest",
        """            product = _bounded_fraction(
                probability * factors[current], f"null-chain product at item {current!r}"
            )
""",
        """            product = _bounded_fraction(
                Fraction(
                    context.multiply(
                        _round_fraction(probability, context),
                        _round_fraction(factors[current], context),
                    )
                ),
                f"null-chain product at item {current!r}",
            )
""",
    ),
    # Moving the one reading factor from its entry to every mutually exclusive
    # alternative exit is output-equivalent, so it is deliberately not a mutation.
    export(
        "Drop every word transition",
        "test_exports_equal_graph_language_and_best_products",
        "for child in children[item_id]:\n            budget.charge",
        "for child in ():\n            budget.charge",
    ),
    export(
        "Merge across a kept junction",
        "test_exports_equal_graph_language_and_best_products",
        """            for child in children[current]:
                budget.charge("pairs_examined")
                previous = best.get(child)
""",
        """            successors = (
                children[current][:1] if item.role == "position" else children[current]
            )
            for child in successors:
                budget.charge("pairs_examined")
                previous = best.get(child)
""",
    ),
    export(
        "Remove the float32 underflow check",
        "test_structural_refusals_leave_no_writer_files",
        "if binary32 == 0.0:",
        "if False:",
    ),
    export(
        "Disable export budget charging",
        "test_structural_refusals_leave_no_writer_files",
        """    def charge(self, field: str, units: int = 1) -> None:
        setattr(self, field, getattr(self, field) + units)
""",
        """    def charge(self, field: str, units: int = 1) -> None:
        return
""",
    ),
    export(
        "Disable empty-only refusal",
        "test_structural_refusals_leave_no_writer_files",
        "if accepts_empty and not accepts_nonempty:",
        "if False:",
    ),
    export(
        "Force accepts_empty false",
        "test_accepts_empty_is_exact",
        "        accepts_empty,\n        precision,",
        "        False,\n        precision,",
    ),
    export(
        "Round written probabilities to six places",
        "test_nonterminating_probability_text_and_log_tolerance",
        'format(written_probability, "f"),',
        'format(round(written_probability, 6), "f"),',
    ),
    export(
        "Ignore weighted path length in FSG tolerance",
        "test_two_visible_two_thirds_obeys_fsg_route_tolerance",
        "derived_bound = min(Fraction(1), longest_weighted_path * per_transition_limit)",
        "derived_bound = min(Fraction(1), per_transition_limit)",
    ),
    export(
        "Write logs at reduced precision",
        "test_att_precision_contract_bounds_tolerance_and_written_digits",
        '''f"{format(cost, 'f')}"''',
        '''f"{format(round(cost, 6), 'f')}"''',
    ),
    export(
        "Round log weights before computing tolerance",
        "test_att_precision_contract_bounds_tolerance_and_written_digits",
        "costs = tuple(_fraction_cost(probability, context, budget) "
        "for probability in probabilities)",
        "costs = tuple(\n"
        "        round(_fraction_cost(probability, context, budget), 6)\n"
        "        for probability in probabilities\n"
        "    )",
    ),
    export(
        "Fix log working precision at three guard digits",
        "test_att_near_tie_logs_are_independently_correctly_rounded",
        """        if lower == upper:
            return lower
        work_precision *= 2
""",
        """        with localcontext(context):
            return +approximation
""",
    ),
    normalization(
        "Lower whole string before splitting",
        "for character in text\n",
        "for character in text.lower()\n",
    ),
    normalization(
        "Keep Unicode punctuation", 'unicodedata.category(character).startswith("P")', "False"
    ),
    normalization("Split on ASCII spaces only", "separated.split()", 'separated.split(" ")'),
    normalization(
        "Change normalization join contract",
        'return " ".join(spoken_tokens(text))',
        'return "".join(spoken_tokens(text))',
    ),
    Mutation(
        "Link resolver to alignment",
        TEST + "test_one_best_has_no_alignment_import",
        "frend/fold_resolve.py",
        "from frend.shape import shape",
        "from frend.shape import shape\ndef alignment_dependency():\n"
        "    from frend import align_graph\n    return align_graph",
    ),
    Mutation(
        "Change resolved geometry bytes",
        TEST + "test_one_best_untouched",
        "frend/fold_resolve.py",
        "return CoverScore(\n        coverage=",
        "return CoverScore(\n        coverage=1 + ",
    ),
    Mutation(
        "Round exact ranking axis to float",
        TEST + "test_exact_ranking_below_double_resolution",
        "frend/fold_resolve.py",
        "supported, strength = True, prior.p",
        "supported, strength = True, Decimal(float(prior.p))",
    ),
    Mutation(
        "Truncate upstream reading carrier",
        TEST + "test_upstream_caps_still_refuse",
        "frend/lattice.py",
        "if len(candidates) > reading_cap:",
        "candidates = candidates[:reading_cap]\n    if False:",
    ),
    Mutation(
        "Disable upstream spoken bound",
        TEST + "test_upstream_caps_still_refuse",
        "frend/lattice.py",
        "if spoken_total > _COMPOSED_SPOKEN_CAP:",
        "if False:",
    ),
    Mutation(
        "Restore date month-first slice",
        TEST + "test_partial_date_alternatives_are_never_cut",
        "frend/verbalize.py",
        "return month_first",
        "return month_first[:8]",
    ),
    Mutation(
        "Restore date ordering half slices",
        "tests/test_verbalize.py::test_date_alternatives_are_never_cut_to_a_prefix",
        "frend/verbalize.py",
        "return _ranked([*month_first, *day_first])",
        "return _ranked([*month_first[:4], *day_first[:4]])",
    ),
    Mutation(
        "Bypass shared builder normalization",
        "tests/test_spoken_priors.py::test_fixture_matching_normalization_sentinels_and_unmatched",
        "tools/build_spoken_priors.py",
        "if normalize_spoken(alternative) == target",
        "if alternative.lower() == target",
    ),
    Mutation(
        "Accept an out-of-date spoken artifact",
        "tests/test_spoken_priors.py::test_fixture_check_rederives_byte_identical_document",
        "tools/build_spoken_priors.py",
        'if not args.out.exists() or args.out.read_text(encoding="utf-8") != rendered:',
        "if False:",
    ),
    attribution(
        "Change attribution meaning",
        "test_attribution_matches_brute_force_path_sums",
        'MEANING = "acoustics plus measured prior"',
        'MEANING = "wrong attribution semantics"',
    ),
    attribution(
        "Drop attribution emissions",
        "test_attribution_matches_brute_force_path_sums",
        "{label: graph.items[label].tokens for label in plan.labels if graph.items[label].tokens}",
        "{}",
    ),
    attribution(
        "Use Viterbi class posterior",
        "test_attribution_matches_brute_force_path_sums",
        "math.fsum(member.posterior for member in members)",
        "max(member.posterior for member in members)",
    ),
    attribution(
        "Drop passthrough class members",
        "test_attribution_matches_brute_force_path_sums",
        'elif item.role == "token" and item.span is not None:',
        'elif item.role == "token" and item.span is not None and False:',
    ),
    attribution(
        "Drop reading form-exit class members",
        "test_attribution_matches_brute_force_path_sums",
        'elif item.role == "form-exit":',
        'elif item.role == "form-exit" and False:',
    ),
    attribution(
        "Drop a whole output class",
        "test_attribution_matches_brute_force_path_sums",
        "for (class_span, class_tokens), members in by_class.items():",
        "for (class_span, class_tokens), members in list(by_class.items())[1:]:",
    ),
    attribution(
        "Replace class member item IDs",
        "test_attribution_matches_brute_force_path_sums",
        "        item.id,\n        item.reading_id,",
        '        "wrong-item-id",\n        item.reading_id,',
    ),
    attribution(
        "Invert class member roles",
        "test_attribution_matches_brute_force_path_sums",
        '        "reading" if reading is not None else "passthrough",',
        '        "passthrough" if reading is not None else "reading",',
    ),
    attribution(
        "Replace reading posterior log masses",
        "test_attribution_matches_brute_force_path_sums",
        "                    item.reading_id,\n                    log_mass,",
        "                    item.reading_id,\n                    0.0,",
    ),
    attribution(
        "Replace class member masses",
        "test_attribution_matches_brute_force_path_sums",
        "        log_mass,\n        _exp(log_mass),\n        _posterior(log_mass, total),",
        "        log_mass,\n        0.0,\n        _posterior(log_mass, total),",
    ),
    attribution(
        "Replace class member log masses",
        "test_attribution_matches_brute_force_path_sums",
        "                    tuple(members),\n                    log_mass,",
        "                    tuple(\n"
        "                        ClassMember(\n"
        "                            member.id,\n"
        "                            member.item_id,\n"
        "                            member.reading_id,\n"
        "                            member.role,\n"
        "                            0.0,\n"
        "                            member.mass,\n"
        "                            member.posterior,\n"
        "                        )\n"
        "                        for member in members\n"
        "                    ),\n"
        "                    log_mass,",
    ),
    attribution(
        "Replace published class member reading IDs",
        "test_attribution_matches_brute_force_path_sums",
        "                    tuple(members),\n                    log_mass,",
        "                    tuple(\n"
        "                        ClassMember(\n"
        "                            member.id,\n"
        "                            member.item_id,\n"
        '                            "wrong-reading-id",\n'
        "                            member.role,\n"
        "                            member.log_mass,\n"
        "                            member.mass,\n"
        "                            member.posterior,\n"
        "                        )\n"
        "                        for member in members\n"
        "                    ),\n"
        "                    log_mass,",
    ),
    attribution(
        "Replace measured prior values",
        "test_isolated_i_pools_audio_class_and_splits_only_scored_members",
        "PriorShare(member_id, p, Fraction(p) / denominator)",
        "PriorShare(member_id, Decimal(0), Fraction(p) / denominator)",
    ),
    attribution(
        "Change unscored prior split meaning",
        "test_prior_split_is_exact_and_independent_of_decimal_context",
        "return PriorSplit(PRIOR_SPLIT_MEANING, shares, tied), tuple(no_evidence)",
        'return PriorSplit(\n        "wrong unscored split semantics" if not scored else '
        "PRIOR_SPLIT_MEANING, shares, tied\n    ), tuple(no_evidence)",
    ),
    attribution(
        "Split measured prior uniformly",
        "test_isolated_i_pools_audio_class_and_splits_only_scored_members",
        "Fraction(p) / denominator",
        "Fraction(1, len(scored))",
    ),
    attribution(
        "Give unscored members prior evidence",
        "test_one_scored_and_one_unscored_reading_has_no_invented_share",
        "if item is None or not item.weight.scored:",
        "if item is None:",
    ),
    attribution(
        "Decide prior ties through float",
        "test_prior_split_ties_use_exact_decimal_measurements",
        "if p == maximum)",
        "if float(p) == float(maximum))",
    ),
    attribution(
        "Ignore attribution prior scale",
        "test_prior_switch_and_scale_change_only_measured_factors",
        "plan.values[index] * prior_scale if prior else 0.0",
        "plan.values[index] if prior else 0.0",
    ),
    attribution(
        "Ignore attribution prior switch",
        "test_prior_switch_and_scale_change_only_measured_factors",
        "plan.values[index] * prior_scale if prior else 0.0",
        "plan.values[index] * prior_scale",
    ),
    attribution(
        "Continue after foreign decoder output",
        "test_foreign_output_refuses_at_first_divergent_token",
        "if not output.accepted[0]:",
        "if False:",
    ),
    attribution(
        "Fabricate zero-mass attribution values",
        "test_all_negative_infinity_returns_zero_mass_without_values",
        "return Attribution(tokens, MEANING, prior, scale, True, None, None, ())",
        "return Attribution(tokens, MEANING, prior, scale, False, 0.0, 1.0, ())",
    ),
]


def clear_cache(path):
    for cached in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
        cached.unlink()


def _exception_type(element):
    declared = element.get("type", "")
    if declared:
        return declared.rsplit(".", 1)[-1]
    message = element.get("message", "")
    first_line = message.splitlines()[0] if message else ""
    if first_line == "assert" or first_line.startswith("assert "):
        return "AssertionError"
    return first_line.split(":", 1)[0].rsplit(".", 1)[-1]


def _failure_details(cases):
    details = []
    for case in cases:
        for outcome in ("failure", "error"):
            element = case.find(outcome)
            if element is None:
                continue
            message = element.get("message", "")
            details.append(
                {
                    "testcase": case.get("name", ""),
                    "outcome": outcome,
                    "exception_type": _exception_type(element),
                    "message": message.splitlines()[0] if message else "",
                }
            )
    return details


def _pytest_result(test, root, timeout):
    with tempfile.TemporaryDirectory(prefix="frend-align-mutation-") as temporary:
        report = Path(temporary) / "pytest.xml"
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--color=no",
                    "--junitxml",
                    str(report),
                    test,
                ],
                cwd=root,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PY_COLORS": "0"},
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "exit_code": None,
                "timed_out": True,
                "tests": 0,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "failure_details": [],
            }

        counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        details = []
        if report.exists():
            tree = ET.parse(report)
            cases = tree.findall(".//testcase")
            counts = {
                "tests": len(cases),
                "failures": sum(case.find("failure") is not None for case in cases),
                "errors": sum(case.find("error") is not None for case in cases),
                "skipped": sum(case.find("skipped") is not None for case in cases),
            }
            details = _failure_details(cases)
        return {
            "exit_code": result.returncode,
            "timed_out": False,
            **counts,
            "failure_details": details,
        }


def _summary(result):
    if result["timed_out"]:
        return "timed out"
    singular = {"failures": "failure", "errors": "error", "passed": "passed", "skipped": "skipped"}
    return (
        ", ".join(
            f"{result[name]} {singular[name] if result[name] == 1 else name}"
            for name in ("failures", "errors", "passed", "skipped")
            if result[name]
        )
        or "no test results"
    )


def run_mutations(mutations, root=ROOT, output_path=None, timeout=TEST_TIMEOUT_SECONDS):
    rows = []
    originals = {m.path: (root / m.path).read_bytes() for m in mutations}
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for mutation in mutations:
        original = originals[mutation.path].decode()
        if original.count(mutation.before) != 1:
            raise ValueError(f"mutation target is not unique: {mutation.name}")
        compile(original.replace(mutation.before, mutation.after, 1), mutation.path, "exec")
    try:
        for mutation in mutations:
            path = root / mutation.path
            original = originals[mutation.path].decode()
            baseline = _pytest_result(mutation.test, root, timeout)
            baseline["passed"] = (
                baseline["tests"] - baseline["failures"] - baseline["errors"] - baseline["skipped"]
            )
            baseline_valid = (
                not baseline["timed_out"]
                and baseline["exit_code"] == 0
                and baseline["tests"] > 0
                and baseline["failures"] == 0
                and baseline["errors"] == 0
            )
            source_hash = hashlib.sha256(originals[mutation.path]).hexdigest()
            test_path = root / mutation.test.split("::", 1)[0]
            test_hash = hashlib.sha256(test_path.read_bytes()).hexdigest()
            before_hash = hashlib.sha256(mutation.before.encode()).hexdigest()
            after_hash = hashlib.sha256(mutation.after.encode()).hexdigest()
            if not baseline_valid:
                rows.append(
                    {
                        "mutation": mutation.name,
                        "test": mutation.test,
                        "status": "invalid",
                        "killed": False,
                        "behavioral": False,
                        "exit_code": None,
                        "baseline": baseline,
                        "mutation_result": None,
                        "source_sha256": source_hash,
                        "test_sha256": test_hash,
                        "before_sha256": before_hash,
                        "after_sha256": after_hash,
                        "runner_sha256": runner_hash,
                        "reason": f"baseline invalid: {_summary(baseline)}",
                        "summary": f"baseline invalid: {_summary(baseline)}",
                    }
                )
                print(f"{mutation.name}: INVALID BASELINE", flush=True)
                continue
            changed = original.replace(mutation.before, mutation.after, 1)
            path.write_text(changed)
            clear_cache(path)
            try:
                mutated = _pytest_result(mutation.test, root, timeout)
                mutated["passed"] = (
                    mutated["tests"] - mutated["failures"] - mutated["errors"] - mutated["skipped"]
                )
                behavioral_failures = [
                    detail
                    for detail in mutated["failure_details"]
                    if detail["outcome"] == "failure"
                    and not _is_non_behavioral(detail["exception_type"])
                ]
                symbol_or_import_failures = [
                    detail
                    for detail in mutated["failure_details"]
                    if detail["outcome"] == "failure"
                    and _is_non_behavioral(detail["exception_type"])
                ]
                behavioral = bool(behavioral_failures)
                only_symbol_or_import_failures = (
                    mutated["failures"] > 0
                    and not behavioral_failures
                    and len(symbol_or_import_failures) == mutated["failures"]
                )
                valid_test_failure = (
                    not mutated["timed_out"]
                    and mutated["exit_code"] == 1
                    and mutated["failures"] > 0
                    and mutated["errors"] == 0
                )
                invalid = (
                    mutated["timed_out"]
                    or mutated["exit_code"] in {2, 3, 4, 5}
                    or mutated["errors"] > 0
                    or mutated["tests"] == 0
                    or only_symbol_or_import_failures
                )
                killed = valid_test_failure and behavioral and not invalid
                status = "killed" if killed else "invalid" if invalid else "survived"
                reason = None
                if only_symbol_or_import_failures:
                    kinds = sorted(
                        {detail["exception_type"] for detail in symbol_or_import_failures}
                    )
                    reason = "only non-behavioral symbol/import failures: " + ", ".join(kinds)
                elif invalid:
                    reason = _summary(mutated)
                row = {
                    "mutation": mutation.name,
                    "test": mutation.test,
                    "status": status,
                    "killed": killed,
                    "behavioral": behavioral,
                    "exit_code": mutated["exit_code"],
                    "baseline": baseline,
                    "mutation_result": mutated,
                    "source_sha256": source_hash,
                    "test_sha256": test_hash,
                    "before_sha256": before_hash,
                    "after_sha256": after_hash,
                    "runner_sha256": runner_hash,
                    "reason": reason,
                    "summary": _summary(mutated),
                }
                rows.append(row)
                print(f"{mutation.name}: {status.upper()}", flush=True)
            finally:
                path.write_bytes(originals[mutation.path])
                clear_cache(path)
    finally:
        for name, original in originals.items():
            (root / name).write_bytes(original)
            clear_cache(root / name)
    if output_path is not None:
        output_path = Path(output_path)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent, delete=False
        ) as temporary:
            temporary.write(json.dumps(rows, indent=2) + "\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(output_path)
    return rows


def main():
    rows = run_mutations(MUTATIONS, output_path=ROOT / "tests/data/align_mutations.json")
    return 0 if rows and all(row["killed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
