#!/usr/bin/env python3
"""Benchmark consolidated native spans, range offers, and width-shape helpers."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from frend.context import TextContext, neighbor_words
from frend.grouped_ids import grouped_id_shape
from frend.telephone import telephone_shape
from frend.verbalize import _range_to

_WORD_CASES = (
    ("Latin words before 42 and after", 19, 21, "en_US"),
    ("中文词语 42 后面", 5, 7, "zh_CN"),
    ("한국어 낱말 42 다음", 7, 9, "ko_KR"),
    ("combine e\u0301lan 42 déjà", 14, 16, "fr_FR"),
    ("astral 𝟜𝟚 😀 42 tail", 12, 14, "en_US"),
    ("emoji 👩\u200d🔬 works 42 now", 17, 19, "en_US"),
    ("punctuation...word 42—next", 19, 21, "en_US"),
    ("white\tspace\nnear 42 end", 17, 19, "en_US"),
    ("x" * 92 + " word 42 " + "y" * 92, 98, 100, "en_US"),
    ("a" * 95 + " 42 " + "中" * 95, 96, 98, "zh_CN"),
)
_RANGE_CASES = (
    ("5-10", 1, 2, "en_US"),
    ("5 – 10", 2, 3, "en_US"),
    ("5 : 10", 2, 3, "en_US"),
    ("A-5", 1, 2, "en_US"),
    ("978-1-2", 3, 4, "en_US"),
)
_SHAPES = tuple(
    f"{'7' * width}-{('3' * width)}{terminal}"
    for width in range(1, 33)
    for terminal in ("", "x", "X")
)


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def _measure(call: Callable[[int], object], iterations: int) -> dict[str, int]:
    samples = []
    for index in range(iterations):
        started = time.perf_counter_ns()
        call(index)
        samples.append(time.perf_counter_ns() - started)
    return {
        "p50_ns": int(statistics.median(samples)),
        "p95_ns": _percentile(samples, 0.95),
    }


def benchmark(iterations: int) -> dict[str, object]:
    def words(index: int):
        text, start, end, locale = _WORD_CASES[index % len(_WORD_CASES)]
        return neighbor_words(text, start, end, locale=locale)

    def ranges(index: int):
        text, start, end, locale = _RANGE_CASES[index % len(_RANGE_CASES)]
        return _range_to(TextContext(text), start, end, locale)

    return {
        "iterations": iterations,
        "neighbor_words": _measure(words, iterations),
        "range_offers": _measure(ranges, iterations),
        "telephone_shape": _measure(
            lambda index: telephone_shape(_SHAPES[index % len(_SHAPES)]), iterations
        ),
        "grouped_id_shape": _measure(
            lambda index: grouped_id_shape(_SHAPES[index % len(_SHAPES)]), iterations
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    document = benchmark(args.iterations)
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
