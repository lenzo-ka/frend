"""Sentence-cluster percentile bootstrap helpers for evaluation tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

DEFAULT_INTERVAL_SEED = 20261002
CI_LEVEL = 0.95


def _numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "interval calculation requires the development dependency numpy"
        ) from error
    return np


def percentile_ratio_intervals(
    metrics: Mapping[str, tuple[Sequence[int | bool], Sequence[int | bool]]],
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    """Bootstrap ratio intervals, drawing sentence indices once for every metric."""
    np = _numpy()
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not metrics:
        return {}
    lengths = {len(numerators) for numerators, _denominators in metrics.values()}
    lengths.update(len(denominators) for _numerators, denominators in metrics.values())
    if len(lengths) != 1:
        raise ValueError("all bootstrap metric vectors must have the same length")
    sentence_count = lengths.pop()
    if sentence_count == 0:
        return {name: [0.0, 0.0] for name in metrics}

    names = tuple(metrics)
    columns = []
    for name in names:
        numerator, denominator = metrics[name]
        columns.extend((numerator, denominator))
    values = np.asarray(columns, dtype=np.float64).T
    estimates = np.empty((replicates, len(names)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    probabilities = np.full(sentence_count, 1.0 / sentence_count)

    # Bound peak allocation while keeping aggregation in NumPy. A multinomial row is
    # exactly the count vector produced by sampling n sentence indices with replacement.
    batch_size = max(1, min(replicates, 64, 32_000_000 // sentence_count))
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        weights = rng.multinomial(sentence_count, probabilities, size=stop - start)
        totals = weights @ values
        numerators = totals[:, 0::2]
        denominators = totals[:, 1::2]
        np.divide(
            numerators,
            denominators,
            out=estimates[start:stop],
            where=denominators != 0,
        )
        estimates[start:stop][denominators == 0] = 0.0

    bounds = np.percentile(estimates, [2.5, 97.5], axis=0)
    return {name: [float(bounds[0, at]), float(bounds[1, at])] for at, name in enumerate(names)}


def paired_delta_intervals(
    metrics: Mapping[
        str,
        tuple[
            Sequence[int | bool],
            Sequence[int | bool],
            Sequence[int | bool],
            Sequence[int | bool],
        ],
    ],
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    """Paired B-minus-A ratio intervals using the same sentence draw for both arms."""
    np = _numpy()
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not metrics:
        return {}
    lengths = {len(vector) for vectors in metrics.values() for vector in vectors}
    if len(lengths) != 1:
        raise ValueError("all paired bootstrap metric vectors must have the same length")
    sentence_count = lengths.pop()
    if sentence_count == 0:
        return {name: [0.0, 0.0] for name in metrics}

    names = tuple(metrics)
    columns = []
    for name in names:
        columns.extend(metrics[name])
    values = np.asarray(columns, dtype=np.float64).T
    estimates = np.empty((replicates, len(names)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    probabilities = np.full(sentence_count, 1.0 / sentence_count)
    batch_size = max(1, min(replicates, 64, 32_000_000 // sentence_count))
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        weights = rng.multinomial(sentence_count, probabilities, size=stop - start)
        totals = weights @ values
        a_denominators = totals[:, 1::4]
        b_denominators = totals[:, 3::4]
        a = np.zeros_like(a_denominators)
        b = np.zeros_like(b_denominators)
        np.divide(totals[:, 0::4], a_denominators, out=a, where=a_denominators != 0)
        np.divide(totals[:, 2::4], b_denominators, out=b, where=b_denominators != 0)
        estimates[start:stop] = b - a

    bounds = np.percentile(estimates, [2.5, 97.5], axis=0)
    return {name: [float(bounds[0, at]), float(bounds[1, at])] for at, name in enumerate(names)}
