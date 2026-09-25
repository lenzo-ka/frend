"""Build or verify sampled spoken-alternative source measurements.

The table measures every supported kind at kind level. Fractions additionally
declare and measure the captured denominator as a sub-key, and measures the
reading's ICU unit identifier; no other kind has a declared sub-key. Each matched
row is credited to the source and sub-key of the specific reading that produced
its spoken text. Ranking selects a sub-key row
for every alternative on an edge when available, or the kind row for every
alternative otherwise, so a comparison never mixes conditioning levels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "frend" / "data" / "spoken_priors.json"
_BUILD_DATE = "2026-09-24"
_SHARD_STEP = 10
_ROWS_PER_CLASS_PER_SHARD = 200
_TOP_UNMATCHED = 20
_UNMATCHED_REASONS = ("unrecognized", "unverbalized", "no_alternative_matched")
_DATE_SKELETONS = ("yMd", "Md", "y")

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from frend.electronic import ElectronicDetector, decode_letter_notation  # noqa: E402
from frend.spoken_priors import normalize_spoken  # noqa: E402
from frend.written_forms import WrittenFormsDetector  # noqa: E402

CLASS_TO_KIND = {
    "CARDINAL": "cardinal",
    "DATE": "date",
    "DECIMAL": "decimal",
    "ELECTRONIC": "electronic",
    "FRACTION": "fraction",
    "MEASURE": "measure",
    "MONEY": "money",
    "ORDINAL": "ordinal",
    "TIME": "time",
}
OUTSIDE_CLASSES = {
    "DIGIT": "digit strings read digit by digit; not a measured kind",
    "ADDRESS": "no verbalized value family",
    "LETTERS": "no verbalized value family",
    "PLAIN": "not a structured value",
    "PUNCT": "not a structured value",
    "TELEPHONE": "no verbalized value family",
    "VERBATIM": "not a structured value",
}
_CURRENCIES = (
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CNY",
    "INR",
    "CAD",
    "AUD",
    "KRW",
    "RUB",
    "XCD",
)

# The measure units the profile recognizes: the corpus's most frequent MEASURE units,
# as ICU unit identifiers. icukit derives every written form of each from ICU.
_MEASURE_UNITS = (
    "kilometer",
    "meter",
    "millimeter",
    "centimeter",
    "micrometer",
    "nanometer",
    "foot",
    "inch",
    "mile",
    "yard",
    "kilometer-per-hour",
    "mile-per-hour",
    "kilogram",
    "gram",
    "milligram",
    "pound",
    "ounce",
    "ton",
    "hectare",
    "acre",
    "square-kilometer",
    "square-mile",
    "square-meter",
    "kilowatt",
    "megawatt",
    "watt",
    "horsepower",
    "volt",
    "hertz",
    "kilohertz",
    "megahertz",
    "gigahertz",
    "kilobyte",
    "megabyte",
    "gigabyte",
    "terabyte",
    "liter",
    "milliliter",
    "celsius",
    "fahrenheit",
    "second",
    "minute",
    "hour",
    "gigawatt",
    "meter-per-second",
    "square-foot",
    "millibar",
    "cubic-meter",
)
_MIXED_MEASURES = ("foot-and-inch", "pound-and-ounce")


def _default_corpus_dir() -> Path:
    env = os.environ.get("FREND_TN_CORPUS_DIR")
    if env:
        return Path(env)
    for base in (_REPO, *_REPO.parents):
        candidate = base / "tn-corpus" / "en_with_types"
        if candidate.is_dir():
            return candidate
    return _REPO.parent / "tn-corpus" / "en_with_types"


def _detectors():
    from icukit.detectors import date_detectors
    from icukit.recognize import (
        AlphanumericRunsDetector,
        FlexibleCompactDetector,
        FlexibleCurrencyDetector,
        FlexibleCurrencyNameDetector,
        FlexibleDateDetector,
        FlexibleFractionDetector,
        FlexibleMeasureDetector,
        FlexibleMixedMeasureDetector,
        FlexibleNumberDetector,
        FlexibleNumericDurationDetector,
        FlexibleOrdinalDetector,
        FlexiblePercentDetector,
        FlexibleTextDateDetector,
        FlexibleTimeDetector,
        LetterNameDetector,
        PluralNumeralDetector,
        SingleLetterWordDetector,
    )

    # A letter-digit token read as its runs competes inside the classes it occurs in
    # (TIME "5pm", ORDINAL "4th", DATE "1830s"); it has no corpus class of its own.
    runs = AlphanumericRunsDetector("en_US")
    # Forms ICU writes nowhere, read by frend itself (kal's ruling): "2: 13", "339 U.S.",
    # "6 3", "500 B.C.".
    written = WrittenFormsDetector("en_US")
    dates = date_detectors("en_US", _DATE_SKELETONS).with_(
        FlexibleDateDetector("en_US"),
        FlexibleTextDateDetector("en_US"),
        PluralNumeralDetector("en_US"),
        runs,
    )
    numbers = (
        FlexibleNumberDetector("en_US"),
        FlexibleCompactDetector("en_US", "long"),
        FlexibleCompactDetector("en_US", "short"),
    )
    cardinals = numbers + (LetterNameDetector("en_US"), SingleLetterWordDetector("en_US"))
    return {
        "cardinal": (*cardinals, written),
        "decimal": numbers,
        "date": (*dates.detectors, written),
        "fraction": (FlexibleFractionDetector("en_US"),),
        "ordinal": (FlexibleOrdinalDetector("en_US"), FlexibleNumberDetector("en_US"), runs),
        "time": (
            FlexibleTimeDetector("en_US"),
            FlexibleNumericDurationDetector("en_US"),
            runs,
            written,
        ),
        "electronic": (ElectronicDetector("en_US"),),
        "measure": (
            FlexiblePercentDetector("en_US"),
            *(FlexibleMeasureDetector("en_US", unit) for unit in _MEASURE_UNITS),
            *(FlexibleMixedMeasureDetector("en_US", mixed) for mixed in _MIXED_MEASURES),
        ),
        "money": tuple(
            detector
            for code in _CURRENCIES
            for detector in (
                FlexibleCurrencyDetector("en_US", code),
                FlexibleCurrencyNameDetector("en_US", code),
            )
        ),
    }


def _alternatives(
    written: str, kind: str, detectors: dict
) -> tuple[bool, list[tuple[str, str, str | None]]]:
    from frend.lattice import resolve_lattice
    from frend.spoken_priors import measurement_sub_key
    from frend.verbalize import verbalize_edge

    alternatives: list[tuple[str, str, str | None]] = []
    recognized = False
    seen_detections: set[tuple[str, str]] = set()
    for detector in detectors[kind]:
        for detection in detector.detect(written):
            if int(detection["start"]) != 0 or int(detection["end"]) != len(written):
                continue
            recognized = True
            sub_key = measurement_sub_key(kind, detection)
            key = (str(detection["type"]), repr(detection["value"]), sub_key)
            if key in seen_detections:
                continue
            seen_detections.add(key)
            lattice = resolve_lattice([detection], source_text=written)
            edge = next(edge for edge in lattice.edges if edge.kind == "reading")
            unit = verbalize_edge(edge, source_text=written, apply_source_priors=False)
            if unit.verbalized:
                alternatives.extend(
                    (item.text, item.provenance, sub_key) for item in unit.alternatives
                )
    return recognized, alternatives


def _files(corpus_dir: Path) -> list[Path]:
    available = sorted(corpus_dir.glob("output-*-of-*"))
    if not available:
        raise FileNotFoundError(f"no corpus shards under {corpus_dir}")
    return available[::_SHARD_STEP]


def build_document(corpus_dir: Path) -> dict:
    files = _files(corpus_dir)
    detectors = _detectors()
    aggregates: dict[str, dict] = {
        kind: {
            "total": 0,
            "matched": 0,
            "unmatched": 0,
            "unmatched_by_reason": Counter({reason: 0 for reason in _UNMATCHED_REASONS}),
            "source_matched": Counter(),
            "sub_keys": defaultdict(Counter),
            "unmatched_forms": defaultdict(
                lambda: {
                    "count": 0,
                    "reasons": Counter({reason: 0 for reason in _UNMATCHED_REASONS}),
                    "written_forms": Counter(),
                }
            ),
        }
        for kind in sorted(set(CLASS_TO_KIND.values()))
    }
    rows_by_class: Counter[str] = Counter()
    skipped_sentinels = 0
    for path in files:
        per_class: Counter[str] = Counter()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3 or parts[0] not in CLASS_TO_KIND:
                    continue
                corpus_class, written, spoken = parts[:3]
                if spoken in {"<self>", "sil"}:
                    skipped_sentinels += 1
                    continue
                if per_class[corpus_class] >= _ROWS_PER_CLASS_PER_SHARD:
                    continue
                per_class[corpus_class] += 1
                rows_by_class[corpus_class] += 1
                kind = CLASS_TO_KIND[corpus_class]
                aggregate = aggregates[kind]
                aggregate["total"] += 1
                if corpus_class == "ELECTRONIC":
                    spoken = decode_letter_notation(spoken)
                target = normalize_spoken(spoken)
                recognition_written = written.rstrip(" ,")
                recognized, alternatives = _alternatives(recognition_written, kind, detectors)
                matched = next(
                    (
                        (source, sub_key)
                        for alternative, source, sub_key in alternatives
                        if normalize_spoken(alternative) == target
                    ),
                    None,
                )
                if matched is not None:
                    matched_source, matched_sub_key = matched
                    aggregate["matched"] += 1
                    aggregate["source_matched"][matched_source] += 1
                    if matched_sub_key is not None:
                        aggregate["sub_keys"][matched_sub_key][matched_source] += 1
                else:
                    if not recognized:
                        reason = "unrecognized"
                    elif not alternatives:
                        reason = "unverbalized"
                    else:
                        reason = "no_alternative_matched"
                    aggregate["unmatched"] += 1
                    aggregate["unmatched_by_reason"][reason] += 1
                    item = aggregate["unmatched_forms"][target]
                    item["count"] += 1
                    item["reasons"][reason] += 1
                    item["written_forms"][written] += 1

    output = {}
    for kind, aggregate in aggregates.items():
        unmatched = sorted(
            aggregate.pop("unmatched_forms").items(), key=lambda item: (-item[1]["count"], item[0])
        )[:_TOP_UNMATCHED]
        output[kind] = {
            **aggregate,
            "source_matched": dict(sorted(aggregate["source_matched"].items())),
            "sub_keys": {
                sub_key: {
                    "matched": sum(sources.values()),
                    "source_matched": dict(sorted(sources.items())),
                }
                for sub_key, sources in sorted(
                    aggregate["sub_keys"].items(),
                    key=lambda item: (
                        (0, int(item[0]), "") if item[0].isdigit() else (1, 0, item[0])
                    ),
                )
            },
            "unmatched_by_reason": dict(sorted(aggregate["unmatched_by_reason"].items())),
            # No unmatched examples for electronic rows: they would copy URLs from the corpus.
            "top_unmatched": []
            if kind == "electronic"
            else [
                {
                    "spoken": spoken,
                    "count": detail["count"],
                    "reasons": dict(sorted(detail["reasons"].items())),
                    "written_forms": dict(sorted(detail["written_forms"].items())),
                }
                for spoken, detail in unmatched
            ],
        }
    return {
        "provenance": {
            "source": "google-tn-en_with_types",
            "license": "CC BY-SA 4.0",
            "attribution": "derived from Sproat & Jaitly (2016) Google TN corpus",
            "generated": _BUILD_DATE,
            "sample_rule": {
                "shards": [path.name for path in files],
                "first_eligible_rows_per_class_per_shard": _ROWS_PER_CLASS_PER_SHARD,
            },
            "row_counts": dict(sorted(rows_by_class.items())),
            "skipped_sentinel_rows": skipped_sentinels,
            "normalization": (
                "Written tokens: remove trailing spaces and commas before recognition. "
                "Spoken forms and alternatives: split at characters whose Unicode general "
                "category starts with P and at whitespace recognized by str.split(), then "
                "lowercase each token independently."
            ),
            "class_to_kind": CLASS_TO_KIND,
            "sub_key_rules": {
                kind: {
                    "fraction": "the decimal integer value of the reading's denominator capture",
                    "measure": "the reading's ICU unit identifier; percent for a percent",
                    "date": "the written order of month and day: month-first, day-first, or other",
                }.get(kind)
                for kind in sorted(aggregates)
            },
            "outside_classes": OUTSIDE_CLASSES,
            "recognition_profile": {
                "locale": "en_US",
                "detector_classes_by_kind": {
                    kind: [type(detector).__name__ for detector in kind_detectors]
                    for kind, kind_detectors in detectors.items()
                },
                "date_skeletons": list(_DATE_SKELETONS),
                "compact_styles": ["long", "short"],
                "currency_codes": list(_CURRENCIES),
                "measure_units": list(_MEASURE_UNITS),
                "mixed_measures": list(_MIXED_MEASURES),
                "full_span_only": True,
                "detection_handling": "resolve and verbalize each detection separately",
            },
            "source_measurement": (
                "The verbalizer deduplicates equal text in rule-set order, so a matched row "
                "is credited to the first retained source that produces its text and to the "
                "sub-key of that same reading. Source priors are disabled while measuring so "
                "the table is not measured through the ranking it drives. Shares are raw "
                "matched-source counts divided by matched rows in the selected scope, with no "
                "smoothing, confidence threshold, or other small-sample correction; each "
                "source record retains its matched count."
            ),
        },
        "kinds": output,
    }


def _render(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", nargs="?", type=Path, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    corpus_dir = args.corpus or args.corpus_dir or _default_corpus_dir()
    rendered = _render(build_document(corpus_dir))
    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != rendered:
            print(f"out of date: {args.out}", file=sys.stderr)
            return 1
        print(f"up to date: {args.out}")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
