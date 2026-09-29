"""Benchmark fixed short-input resolution boundaries and compare exact receipts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = Path(__file__).with_name("latency_inputs-v1.json")


def _canonical(document: object) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _percentile(samples: list[int], percentile: float) -> int:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * percentile))]


def _summary(samples: list[int]) -> dict[str, int]:
    return {
        "p50": _percentile(samples, 0.50),
        "p90": _percentile(samples, 0.90),
        "p99": _percentile(samples, 0.99),
    }


def _environment() -> dict[str, object]:
    import icu
    import icukit
    import tiergraph

    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "machine_model": platform.uname().machine,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "icu": icu.ICU_VERSION,
        "pyicu": getattr(icu, "VERSION", "unknown"),
        "icukit": getattr(icukit, "__version__", "unknown"),
        "tiergraph": getattr(tiergraph, "__version__", "unknown"),
        "power_mode": None,
    }


def _worker(args: argparse.Namespace) -> int:
    setup_started = time.perf_counter_ns()
    from icukit.detectors import detect
    from reading_profile import reading_detectors

    from frend.lattice import resolve_lattice
    from frend.verbalize import verbalize_lattice

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    locales = tuple(part for part in args.locales.split(",") if part)
    inputs = {
        locale: [
            text for text in manifest["locales"][locale] if len(text.split()) <= args.max_words
        ]
        for locale in locales
    }
    if any(not values for values in inputs.values()):
        raise SystemExit("every requested locale bucket must be nonempty after --max-words")
    gangs = {locale: reading_detectors(locale) for locale in locales}
    prepared = {
        (locale, text): list(detect(text, gangs[locale]))
        for locale, values in inputs.items()
        for text in values
    }
    order = [(locale, text) for locale, values in inputs.items() for text in values]
    rng = random.Random(args.seed)
    rng.shuffle(order)
    setup_ns = time.perf_counter_ns() - setup_started
    gc_events: list[dict[str, int]] = []
    starts: dict[int, int] = {}

    def callback(phase, info):
        generation = int(info["generation"])
        if phase == "start":
            starts[generation] = time.perf_counter_ns()
        elif generation in starts:
            gc_events.append(
                {
                    "generation": generation,
                    "pause_ns": time.perf_counter_ns() - starts.pop(generation),
                }
            )

    gc.callbacks.append(callback)
    started = time.time_ns()
    try:
        for _ in range(args.warmup):
            locale, text = order[_ % len(order)]
            detections = list(detect(text, gangs[locale]))
            verbalize_lattice(resolve_lattice(detections, source_text=text, locale=locale))
        vectors = {locale: {"end_to_end_ns": [], "resolve_only_ns": []} for locale in locales}
        counts = {locale: 0 for locale in locales}
        index = 0
        while any(count < args.runs for count in counts.values()):
            locale, text = order[index % len(order)]
            index += 1
            if counts[locale] >= args.runs:
                continue
            counts[locale] += 1
            before = time.perf_counter_ns()
            detections = list(detect(text, gangs[locale]))
            verbalize_lattice(resolve_lattice(detections, source_text=text, locale=locale))
            vectors[locale]["end_to_end_ns"].append(time.perf_counter_ns() - before)
            before = time.perf_counter_ns()
            resolve_lattice(prepared[(locale, text)], source_text=text, locale=locale)
            vectors[locale]["resolve_only_ns"].append(time.perf_counter_ns() - before)
    finally:
        gc.callbacks.remove(callback)
    payload = {
        "environment": _environment(),
        "denominators": {locale: len(values) for locale, values in inputs.items()},
        "vectors": vectors,
        "summaries": {
            locale: {boundary: _summary(samples) for boundary, samples in by_boundary.items()}
            for locale, by_boundary in vectors.items()
        },
        "gc": {
            "enabled": gc.isenabled(),
            "events": gc_events,
            "by_generation": {
                str(generation): sum(1 for item in gc_events if item["generation"] == generation)
                for generation in range(3)
            },
        },
        "errors": [],
        "setup_ns": setup_ns,
        "started_ns": started,
        "ended_ns": time.time_ns(),
    }
    args.output.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return 0


def _git(root: Path, *argv: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *argv], text=True).strip()


def _run(args: argparse.Namespace) -> int:
    subject = args.subject_root.resolve()
    head = _git(subject, "rev-parse", "HEAD")
    if head != args.expected_head:
        raise SystemExit(f"subject head {head} does not match --expected-head {args.expected_head}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(_canonical(manifest)).hexdigest()
    temporary = args.output.with_suffix(args.output.suffix + ".worker")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--manifest",
        str(args.manifest.resolve()),
        "--locales",
        args.locales,
        "--max-words",
        str(args.max_words),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--seed",
        str(args.seed),
        "--output",
        str(temporary),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(subject), str(subject / "tools")))
    subprocess.run(command, cwd=subject, env=environment, check=True)
    worker = json.loads(temporary.read_text(encoding="utf-8"))
    temporary.unlink()
    receipt = {
        "schema_version": 1,
        "subject": {
            "root": str(subject),
            "commit": head,
            "dirty": bool(_git(subject, "status", "--porcelain")),
        },
        "manifest_sha256": manifest_hash,
        "argv": sys.argv,
        "warmup": args.warmup,
        "runs": args.runs,
        "seed": args.seed,
        "max_words": args.max_words,
        **worker,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _compare(args: argparse.Namespace) -> int:
    baseline = json.loads(args.compare[0].read_text(encoding="utf-8"))
    candidate = json.loads(args.compare[1].read_text(encoding="utf-8"))
    for key in ("manifest_sha256",):
        if baseline[key] != candidate[key]:
            raise SystemExit(f"comparison refuses different {key}")
    identity_fields = (
        "hostname",
        "machine",
        "machine_model",
        "os",
        "cpu",
        "python",
        "icu",
        "pyicu",
        "icukit",
        "tiergraph",
        "power_mode",
    )
    for key in identity_fields:
        if baseline["environment"][key] != candidate["environment"][key]:
            raise SystemExit(f"comparison refuses different environment field {key}")
    for key in ("warmup", "runs", "seed", "max_words"):
        if baseline[key] != candidate[key]:
            raise SystemExit(f"comparison refuses different {key}")
    if baseline["denominators"].get("en_US") != candidate["denominators"].get("en_US"):
        raise SystemExit("comparison refuses different English input denominator")
    failures = []
    for boundary in ("end_to_end_ns", "resolve_only_ns"):
        for percentile in ("p50", "p90"):
            old = baseline["summaries"]["en_US"][boundary][percentile]
            new = candidate["summaries"]["en_US"][boundary][percentile]
            if new > old * (1 + args.english_max_regression):
                failures.append(f"English {boundary} {percentile} regressed {(new / old - 1):.1%}")
    for locale in ("ru_RU", "es_ES"):
        if locale not in candidate["summaries"]:
            continue
        summary = candidate["summaries"][locale]
        if summary["end_to_end_ns"]["p50"] > 3_000_000:
            failures.append(f"{locale} end-to-end p50 exceeds 3.0 ms")
        if summary["resolve_only_ns"]["p50"] > 500_000:
            failures.append(f"{locale} resolve-only p50 exceeds 0.5 ms")
        if summary["end_to_end_ns"]["p90"] > 5_000_000:
            failures.append(f"{locale} end-to-end p90 exceeds 5.0 ms")
    print(json.dumps({"ok": not failures, "failures": failures}, indent=2))
    return int(bool(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--english-max-regression", type=float, default=0.05)
    parser.add_argument("--subject-root", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--locales", default="en_US")
    parser.add_argument("--max-words", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.compare:
        return _compare(args)
    if args.output is None:
        parser.error("--output is required")
    if args._worker:
        return _worker(args)
    if args.subject_root is None or args.expected_head is None:
        parser.error("--subject-root and --expected-head are required")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
