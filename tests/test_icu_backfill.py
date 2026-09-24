from decimal import Decimal

from frend.fold_resolve import resolve, resolve_cover
from frend.type_priors import BlendedPrior, IcuBackfillTable, PriorTable, load_icu_backfill_table


def _det(type_: str, text: str = "x") -> dict:
    return {
        "text": text,
        "start": 0,
        "end": len(text),
        "type": type_,
        "value": None,
        "captures": (),
    }


def _blend(measured, counts, totals, *, class_prior=None, source: str | None = "flat-policy"):
    return BlendedPrior(
        PriorTable(measured, {"source": "test-corpus"}),
        IcuBackfillTable(counts, totals, {"status": "generated-estimate"}),
        class_prior=class_prior,
        class_prior_source=source,
    )


def test_loader_exposes_exact_per_class_likelihood_and_provenance():
    table = load_icu_backfill_table()
    assert table.has("date", "N/N")
    assert table.p_shape_given_class("date", "N/N") == Decimal(44) / Decimal(300)
    assert table.provenance["status"] == "generated-estimate"


def test_measured_rows_block_generated_backfill_including_zero_and_below_floor():
    generated = {"date": {"A": 100}, "fraction": {"A": 1}}
    totals = {"date": 100, "fraction": 100}

    positive = _blend({"A": {"date": 3, "fraction": 0}}, generated, totals)
    result = resolve([_det("date", "x"), _det("fraction", "x")], feature_sources=(positive,))
    assert result.best[0]["type"] == "date"
    assert [reading.prior.tier for reading in result.spans[0].readings] == [
        "measured",
        "measured",
    ]
    assert result.spans[0].readings[1].prior.p is None  # measured attested-zero

    sparse = _blend({"A": {"date": 1}}, generated, totals)
    prior = sparse.reading_prior(_det("date", "x"))
    assert prior is not None
    assert prior.tier == "measured" and prior.p is None and prior.n is None


def test_absent_shape_uses_normalized_icu_posterior_and_generated_honesty():
    blend = _blend(
        {"N": {"date": 3}},
        {"date": {"A": 1}, "fraction": {"A": 3}},
        {"date": 10, "fraction": 10},
    )
    result = resolve([_det("date"), _det("fraction")], feature_sources=(blend,))
    assert result.best[0]["type"] == "fraction"
    by_type = {reading.detection["type"]: reading.prior for reading in result.spans[0].readings}
    assert by_type["date"].p == Decimal(1) / Decimal(4)
    assert by_type["fraction"].p == Decimal(3) / Decimal(4)
    for prior in by_type.values():
        assert prior.tier == "icu-backfill"
        assert prior.provenance == "icu-backfill:flat-policy"
        assert prior.n is None and prior.generated_p is not None


def test_equal_backfill_posteriors_are_honestly_ambiguous():
    blend = _blend(
        {"N": {"date": 3}},
        {"date": {"A": 1}, "fraction": {"A": 2}},
        {"date": 10, "fraction": 20},
    )
    result = resolve([_det("date"), _det("fraction")], feature_sources=(blend,))
    assert result.semantic_ambiguous is True
    assert {reading.prior.p for reading in result.spans[0].readings} == {Decimal("0.5")}


def test_same_class_subtypes_do_not_duplicate_posterior_mass():
    blend = _blend(
        {"N": {"date": 3}},
        {"date": {"A": 1}, "fraction": {"A": 1}},
        {"date": 10, "fraction": 10},
    )
    result = resolve(
        [_det("date:Md"), _det("date:yMd"), _det("fraction")], feature_sources=(blend,)
    )
    by_type = {reading.detection["type"]: reading.prior.p for reading in result.spans[0].readings}
    assert by_type == {
        "date:Md": Decimal("0.5"),
        "date:yMd": Decimal("0.5"),
        "fraction": Decimal("0.5"),
    }


def test_supplied_class_prior_changes_backfill_ranking_and_provenance():
    counts = {"date": {"A": 2}, "fraction": {"A": 1}}
    totals = {"date": 10, "fraction": 10}
    flat = _blend({"N": {"date": 3}}, counts, totals)
    supplied = _blend(
        {"N": {"date": 3}},
        counts,
        totals,
        class_prior={"date": Decimal(1), "fraction": Decimal(3)},
        source="measured-class-marginal",
    )
    detections = [_det("date"), _det("fraction")]
    assert resolve(detections, feature_sources=(flat,)).best[0]["type"] == "date"
    result = resolve(detections, feature_sources=(supplied,))
    assert result.best[0]["type"] == "fraction"
    assert all(
        reading.prior.provenance == "icu-backfill:measured-class-marginal"
        for reading in result.spans[0].readings
    )


def test_cover_outputs_reuse_candidate_normalized_span_priors():
    blend = _blend(
        {"N": {"date": 3}},
        {"date": {"A": 3}, "fraction": {"A": 1}},
        {"date": 10, "fraction": 10},
    )
    detections = [_det("date"), _det("fraction")]
    result = resolve(detections, feature_sources=(blend,))
    span_priors = {reading.detection["type"]: reading.prior for reading in result.spans[0].readings}
    for cover, priors in zip(result.covers, result.priors, strict=True):
        if not cover:
            assert priors == ()
            continue
        assert priors[0] == span_priors[cover[0]["type"]]
    assert result.priors[0][0].p == Decimal(3) / Decimal(4)
    assert result.priors[0][0].generated_p == Decimal(3) / Decimal(10)

    cover = resolve_cover(detections, feature_sources=(blend,))
    assert cover.priors[0] == span_priors[cover.best[0]["type"]]
    assert cover.priors[0].p == Decimal(3) / Decimal(4)


def test_capture_filtered_lower_cover_reuses_all_candidate_span_posterior():
    blend = _blend(
        {"N": {"date": 3}},
        {"date": {"A": 3}},
        {"date": 10},
    )
    higher_capture = _det("date:Md")
    higher_capture["captures"] = ({"name": "month"},)
    lower_capture = _det("date:yMd")

    result = resolve([higher_capture, lower_capture], feature_sources=(blend,))
    populated = [
        (cover, priors) for cover, priors in zip(result.covers, result.priors, strict=True) if cover
    ]
    assert {cover[0]["type"] for cover, _ in populated} == {"date:Md", "date:yMd"}
    assert all(priors[0] is not None for _, priors in populated)
    assert all(priors[0].p == Decimal(1) for _, priors in populated)
    assert all(priors[0].generated_p == Decimal(3) / Decimal(10) for _, priors in populated)
    assert all(prior is not None for priors in result.priors for prior in priors)


def test_unsourced_supplied_prior_is_honestly_labeled_and_changes_ranking():
    blend = _blend(
        {"N": {"date": 3}},
        {"date": {"A": 2}, "fraction": {"A": 1}},
        {"date": 10, "fraction": 10},
        class_prior={"date": Decimal(1), "fraction": Decimal(3)},
        source=None,
    )
    result = resolve([_det("date"), _det("fraction")], feature_sources=(blend,))
    assert result.best[0]["type"] == "fraction"
    assert all(
        reading.prior.provenance == "icu-backfill:supplied" for reading in result.spans[0].readings
    )


def test_all_zero_class_prior_mass_withholds_backfill():
    blend = _blend(
        {"N": {"date": 3}},
        {"date": {"A": 2}, "fraction": {"A": 1}},
        {"date": 10, "fraction": 10},
        class_prior={"date": Decimal(0), "fraction": Decimal(0)},
        source="zero-policy",
    )
    result = resolve([_det("date"), _det("fraction")], feature_sources=(blend,))
    assert result.semantic_ambiguous is True
    for reading in result.spans[0].readings:
        prior = reading.prior
        assert prior.tier == "unsupported"
        assert prior.supported is False and prior.p is None
        assert prior.n is None and prior.generated_p is not None
        assert prior.provenance == "icu-backfill:zero-policy"


def test_default_runtime_backfills_a_non_ascii_decimal_digit_surface():
    surface = "-١\N{NO-BREAK SPACE}٢"
    result = resolve([_det("number:cardinal", surface), _det("script:arabic", surface)])
    assert result.best[0]["type"] == "number:cardinal"
    prior = result.spans[0].readings[0].prior
    assert prior.tier == "icu-backfill"
    assert prior.provenance == "icu-backfill:flat-policy" and prior.n is None


def test_resolve_class_prior_override_is_wired_to_default_blend():
    surface = "-١\N{NO-BREAK SPACE}٢"
    result = resolve(
        [_det("number:cardinal", surface)],
        class_prior={"cardinal": Decimal(2)},
    )
    assert result.spans[0].winner["type"] == "number:cardinal"
    assert result.spans[0].readings[0].prior.provenance == "icu-backfill:supplied"
