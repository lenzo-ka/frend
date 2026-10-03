"""Compare text-free per-sentence evaluation records with a paired bootstrap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from bootstrap_intervals import (  # noqa: E402
    DEFAULT_INTERVAL_SEED,
    MIN_DEFINED_FRACTION,
    paired_delta_intervals,
)
from seen_strata import SENTENCE_STRATA, STRATA  # noqa: E402


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("unit") != "sentence":
        raise ValueError(f"{path}: unsupported per-sentence record schema")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: records must be a list")
    return payload


def _align(a_records: list[dict], b_records: list[dict]) -> tuple[list[dict], list[dict]]:
    a_by_id = {row["id"]: row for row in a_records}
    b_by_id = {row["id"]: row for row in b_records}
    if len(a_by_id) != len(a_records) or len(b_by_id) != len(b_records):
        raise ValueError("each arm must have unique sentence ids")
    if a_by_id.keys() != b_by_id.keys():
        raise ValueError("arms do not contain the same sentence ids")
    a = list(a_records)
    b = [b_by_id[row["id"]] for row in a]
    for a_row, b_row in zip(a, b, strict=True):
        if a_row["tokens"] != b_row["tokens"]:
            raise ValueError(f"sentence {a_row['id']}: token denominator differs between arms")
        if a_row.get("strata") != b_row.get("strata"):
            _validate_strata_denominators(a_row, b_row)
    return a, b


def _validate_strata_denominators(a_row: dict, b_row: dict) -> None:
    if ("strata" in a_row) != ("strata" in b_row):
        raise ValueError("both arms must either contain strata or omit them")
    if "strata" not in a_row:
        return
    if a_row["strata"]["sentences"] != b_row["strata"]["sentences"]:
        raise ValueError(f"sentence {a_row['id']}: sentence stratum differs between arms")
    for name in STRATA:
        if a_row["strata"]["tokens"][name]["tokens"] != b_row["strata"]["tokens"][name]["tokens"]:
            raise ValueError(f"sentence {a_row['id']}: {name} token denominator differs")


def _metrics(a: list[dict], b: list[dict]) -> dict:
    ones = [1] * len(a)
    metrics = {
        "overall:first_choice_tokens": _vectors(a, b, "first", "tokens"),
        "overall:any_reading_tokens": _vectors(a, b, "any", "tokens"),
        "overall:first_choice_sentences": (
            [row["sentence_first"] for row in a],
            ones,
            [row["sentence_first"] for row in b],
            ones,
        ),
    }
    if a and "strata" in a[0]:
        _add_strata_metrics(metrics, a, b)
    return metrics


def _vectors(a, b, numerator: str, denominator: str) -> tuple[list, list, list, list]:
    return (
        [row[numerator] for row in a],
        [row[denominator] for row in a],
        [row[numerator] for row in b],
        [row[denominator] for row in b],
    )


def _add_strata_metrics(metrics: dict, a: list[dict], b: list[dict]) -> None:
    for stratum in STRATA:
        a_rows = [row["strata"]["tokens"][stratum] for row in a]
        b_rows = [row["strata"]["tokens"][stratum] for row in b]
        metrics[f"token_stratum:{stratum}:first_choice"] = _vectors(
            a_rows, b_rows, "first", "tokens"
        )
        metrics[f"token_stratum:{stratum}:any_reading"] = _vectors(a_rows, b_rows, "any", "tokens")
    for stratum in SENTENCE_STRATA:
        _add_sentence_stratum_metric(metrics, a, b, stratum)


def _add_sentence_stratum_metric(metrics: dict, a: list[dict], b: list[dict], stratum: str) -> None:
    a_membership = [stratum in row["strata"]["sentences"] for row in a]
    b_membership = [stratum in row["strata"]["sentences"] for row in b]
    metrics[f"sentence_stratum:{stratum}:first_choice"] = (
        [member and row["sentence_first"] for member, row in zip(a_membership, a, strict=True)],
        a_membership,
        [member and row["sentence_first"] for member, row in zip(b_membership, b, strict=True)],
        b_membership,
    )


def compare(a_payload: dict, b_payload: dict, replicates: int, seed: int) -> dict:
    a_fingerprint = _sample_fingerprint(a_payload, "a")
    b_fingerprint = _sample_fingerprint(b_payload, "b")
    if a_fingerprint != b_fingerprint:
        raise ValueError("arms do not have the same sample fingerprint")
    a, b = _align(a_payload["records"], b_payload["records"])
    metrics = _metrics(a, b)
    intervals = paired_delta_intervals(metrics, replicates, seed)
    rows = {name: _comparison_row(vectors, intervals[name]) for name, vectors in metrics.items()}
    return _comparison_report(rows, len(a), replicates, seed, a_fingerprint)


def _sample_fingerprint(payload: dict, arm: str) -> str:
    fingerprint = payload.get("sample_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"arm {arm} has no sample fingerprint")
    return fingerprint


def _ratio(numerators, denominators) -> float:
    denominator = sum(denominators)
    return sum(numerators) / denominator if denominator else 0.0


def _comparison_row(vectors, interval) -> dict[str, object]:
    a_value = _ratio(vectors[0], vectors[1])
    b_value = _ratio(vectors[2], vectors[3])
    bounds = interval["ci95"]
    scaled = None if bounds is None else [100.0 * value for value in bounds]
    return {
        "a_percent": 100.0 * a_value,
        "b_percent": 100.0 * b_value,
        "delta_pp": 100.0 * (b_value - a_value),
        "delta_ci95_pp": scaled,
        "delta_ci95_defined_replicates": interval["defined_replicates"],
        "delta_ci_excludes_zero": (None if scaled is None else scaled[0] > 0.0 or scaled[1] < 0.0),
    }


def _comparison_report(
    rows, sentence_count, replicates, seed, sample_fingerprint
) -> dict[str, object]:
    report = {
        "schema_version": 1,
        "comparison": "b_minus_a",
        "sentences": sentence_count,
        "sample_fingerprint": sample_fingerprint,
    }
    report["metrics"] = rows
    report["bootstrap"] = _bootstrap_receipt(replicates, seed)
    return report


def _bootstrap_receipt(replicates, seed) -> dict[str, object]:
    return {
        "method": "paired sentence-cluster percentile bootstrap",
        "confidence": 0.95,
        "replicates": replicates,
        "seed": seed,
        "minimum_defined_fraction": MIN_DEFINED_FRACTION,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--intervals", type=int, default=1000, metavar="N")
    parser.add_argument("--interval-seed", type=int, default=DEFAULT_INTERVAL_SEED)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.intervals < 1:
        parser.error("--intervals must be positive")
    report = compare(_load(args.a), _load(args.b), args.intervals, args.interval_seed)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
