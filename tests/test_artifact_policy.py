import pytest

from frend.artifact_policy import (
    ELECTRONIC_SPAN_POLICY,
    TELEPHONE_CONTEXT_POLICY,
    policy_document,
    require_policy,
    strict_majority,
)


def test_strict_majority_requires_three_selected_rows_and_rejects_ties():
    assert not strict_majority({"chosen": 2}, "chosen", minimum_support=3)
    assert strict_majority({"chosen": 3}, "chosen", minimum_support=3)
    assert not strict_majority({"chosen": 3, "other": 3}, "chosen", minimum_support=3)
    assert strict_majority({"chosen": 4, "other": 3}, "chosen", minimum_support=3)


def test_telephone_context_exclusions_are_declared_policy_parameters():
    assert TELEPHONE_CONTEXT_POLICY["excluded_predictions"] == (
        "PLAIN",
        "PUNCT",
        "TELEPHONE",
    )


@pytest.mark.parametrize(
    ("key", "value"), (("id", "wrong"), ("version", 2), ("minimum_support", 4))
)
def test_policy_identity_version_and_parameters_are_exact(key, value):
    expected = policy_document(ELECTRONIC_SPAN_POLICY)
    require_policy(expected, ELECTRONIC_SPAN_POLICY, "test")
    with pytest.raises(ValueError, match="invalid test policy"):
        require_policy({**expected, key: value}, ELECTRONIC_SPAN_POLICY, "test")


def test_electronic_minority_remains_raw_but_is_not_admitted():
    row = {"ELECTRONIC": 3, "PLAIN": 4}
    artifact = {"feature": {"classes": row}}
    assert artifact["feature"]["classes"] == row
    assert not strict_majority(
        row,
        str(ELECTRONIC_SPAN_POLICY["label"]),
        minimum_support=int(ELECTRONIC_SPAN_POLICY["minimum_support"]),
    )
