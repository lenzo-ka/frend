"""Benchmark fixed resolution boundaries and the optional Google-TN profile A/B."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
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


def _external_output(path: Path) -> Path:
    """Resolve an output path and refuse benchmark writes inside the repository."""
    resolved = path.resolve()
    if resolved.is_relative_to(_REPO.resolve()):
        raise SystemExit(f"--output and its temporary siblings must be outside {_REPO}")
    return resolved


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
    import reading_profile
    from icukit.detectors import detect
    from reading_profile import reading_detectors

    import frend
    from frend.lattice import resolve_lattice
    from frend.verbalize import verbalize_lattice

    resolve_accepts_locale = "locale" in inspect.signature(resolve_lattice).parameters

    def resolve(detections, text):
        kwargs = {"source_text": text}
        if resolve_accepts_locale:
            kwargs["locale"] = locale
        return resolve_lattice(detections, **kwargs)

    subject = args.subject_root.resolve()
    imports = {
        "reading_profile": str(Path(reading_profile.__file__).resolve()),
        "frend": str(Path(frend.__file__).resolve()),
    }
    for name, imported in imports.items():
        if not Path(imported).is_relative_to(subject):
            raise SystemExit(f"{name} imported from {imported}, outside subject root {subject}")
    if getattr(args, "imports_only", False):
        args.output.write_text(json.dumps({"imports": imports}), encoding="utf-8")
        return 0

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
            verbalize_lattice(resolve(detections, text))
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
            verbalize_lattice(resolve(detections, text))
            vectors[locale]["end_to_end_ns"].append(time.perf_counter_ns() - before)
            before = time.perf_counter_ns()
            resolve(prepared[(locale, text)], text)
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
        "imports": imports,
    }
    args.output.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return 0


_GOOGLE_TN_CORPUS = Path("/Volumes/k02/corpora/unpacked/google/tn-en_with_types/en_with_types")
_PROFILE_BUCKETS = (("le5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-40", 21, 40))


def _profile_measure(text: str, detectors: list[object], profile: str | None) -> dict:
    from icukit.detectors import detect

    from frend import resolve_lattice
    from frend.verbalize import verbalize_lattice

    started = time.perf_counter_ns()
    detections = list(detect(text, detectors))
    detected = time.perf_counter_ns()
    lattice = resolve_lattice(detections, source_text=text, locale="en_US")
    resolved = time.perf_counter_ns()
    verbalize_lattice(lattice, profile=profile)
    ended = time.perf_counter_ns()
    return {
        "stages_ns": {
            "detect": detected - started,
            "resolve_lattice": resolved - detected,
            "verbalize": ended - resolved,
            "pipeline": ended - started,
        },
        "candidates": len(detections),
    }


def _profile_worker(args: argparse.Namespace) -> int:
    in_process_setup_started = time.perf_counter_ns()
    import reading_profile
    from reading_profile import reading_detectors

    import frend

    subject = args.subject_root.resolve()
    imports = {
        "reading_profile": str(Path(reading_profile.__file__).resolve()),
        "frend": str(Path(frend.__file__).resolve()),
    }
    for name, imported in imports.items():
        if not Path(imported).is_relative_to(subject):
            raise SystemExit(f"{name} imported from {imported}, outside subject root {subject}")
    detectors = reading_detectors("en_US")
    in_process_setup_ns = time.perf_counter_ns() - in_process_setup_started
    profile = None if args.profile_condition == "off" else args.profile
    request = json.loads(args.profile_input.read_text(encoding="utf-8"))
    if args.profile_cold:
        measured = _profile_measure(request["text"], detectors, profile)
        measured.update(
            {
                "bucket": request["bucket"],
                "text": request["text"],
                "in_process_setup_ns": in_process_setup_ns,
            }
        )
        args.output.write_text(json.dumps(measured, sort_keys=True), encoding="utf-8")
        return 0

    inputs = request["inputs"]
    flat = [(bucket, text) for bucket, texts in inputs.items() for text in texts]
    for index in range(args.profile_warmup):
        _bucket, text = flat[index % len(flat)]
        _profile_measure(text, detectors, profile)
    rows = []
    for bucket, text in flat:
        measured = _profile_measure(text, detectors, profile)
        measured.update({"bucket": bucket, "text": text})
        rows.append(measured)
    args.output.write_text(
        json.dumps(
            {"rows": rows, "in_process_setup_ns": in_process_setup_ns, "imports": imports},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


def _google_tn_sentences(corpus_dir: Path):
    sentence = []
    for shard in range(90, 95):
        path = corpus_dir / f"output-000{shard}-of-00100"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if fields[0] == "<eos>":
                    if sentence:
                        yield " ".join(sentence)
                    sentence = []
                elif len(fields) >= 3:
                    sentence.append(fields[1])
    if sentence:
        yield " ".join(sentence)


def _profile_inputs(args: argparse.Namespace) -> dict[str, list[str]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    inputs = {"existing": list(manifest["locales"]["en_US"])}
    reservoirs = {name: [] for name, _low, _high in _PROFILE_BUCKETS}
    seen = {name: 0 for name in reservoirs}
    rng = random.Random(args.seed)
    for text in _google_tn_sentences(args.corpus_dir):
        words = len(text.split())
        for name, low, high in _PROFILE_BUCKETS:
            if not low <= words <= high:
                continue
            seen[name] += 1
            values = reservoirs[name]
            if len(values) < args.profile_sample_per_bucket:
                values.append(text)
            else:
                replacement = rng.randrange(seen[name])
                if replacement < len(values):
                    values[replacement] = text
            break
    missing = [name for name, values in reservoirs.items() if not values]
    if missing:
        raise SystemExit(f"empty Google TN length buckets: {', '.join(missing)}")
    inputs.update(reservoirs)
    return inputs


def _profile_summary(rows: list[dict], *, cold: bool) -> dict:
    by_bucket: dict[str, list[dict]] = {}
    for row in rows:
        by_bucket.setdefault(row["bucket"], []).append(row)
    summaries = {}
    for bucket, values in by_bucket.items():
        stage_names = tuple(values[0]["stages_ns"])
        stages = {
            stage: _summary([row["stages_ns"][stage] for row in values]) for stage in stage_names
        }
        if cold:
            stages["in_process_setup"] = _summary([row["in_process_setup_ns"] for row in values])
            stages["process_wall"] = _summary([row["process_wall_ns"] for row in values])
        summaries[bucket] = {
            "utterances": len(values),
            "stages_ns": stages,
            "candidates": _summary([row["candidates"] for row in values]),
        }
    return summaries


def _profile_command(args: argparse.Namespace, temporary: Path) -> list[str]:
    command = [
        sys.executable,
        "-c",
        f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')",
        "--_profile-worker",
        "--profile",
        args.profile,
        "--subject-root",
        str(args.subject_root.resolve()),
        "--profile-input",
        str(args.profile_input_file),
        "--profile-condition",
        args.profile_condition,
        "--profile-warmup",
        str(args.profile_warmup),
        "--output",
        str(temporary),
    ]
    if args.profile_cold:
        command.append("--_profile-cold")
    return command


def _run_profile_worker(args: argparse.Namespace, temporary: Path) -> dict:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subject = args.subject_root.resolve()
    environment["PYTHONPATH"] = os.pathsep.join((str(subject), str(subject / "tools")))
    process_started = time.perf_counter_ns()
    subprocess.run(_profile_command(args, temporary), cwd=subject, env=environment, check=True)
    process_wall_ns = time.perf_counter_ns() - process_started
    payload = json.loads(temporary.read_text(encoding="utf-8"))
    temporary.unlink()
    if args.profile_cold:
        payload["process_wall_ns"] = process_wall_ns
    return payload


def _profile_run(args: argparse.Namespace) -> int:
    if args.profile_sample_per_bucket < 1 or args.profile_cold_runs < 1:
        raise SystemExit("profile sample and cold-run counts must be positive")
    if args.profile_warmup < 0:
        raise SystemExit("profile warmup count must be nonnegative")
    args.output = _external_output(args.output)
    subject = args.subject_root.resolve()
    head = _git(subject, "rev-parse", "HEAD")
    if head != args.expected_head:
        raise SystemExit(f"subject head {head} does not match --expected-head {args.expected_head}")
    inputs = _profile_inputs(args)
    input_path = args.output.with_suffix(args.output.suffix + ".profile-inputs")
    worker_path = args.output.with_suffix(args.output.suffix + ".profile-worker")
    args.profile_input_file = input_path
    conditions = {}
    try:
        input_path.write_text(json.dumps({"inputs": inputs}), encoding="utf-8")
        for condition in ("off", "on"):
            args.profile_condition = condition
            args.profile_cold = False
            input_path.write_text(json.dumps({"inputs": inputs}), encoding="utf-8")
            warm = _run_profile_worker(args, worker_path)
            cold_rows = []
            args.profile_cold = True
            for bucket, texts in inputs.items():
                for text in texts[: args.profile_cold_runs]:
                    input_path.write_text(
                        json.dumps({"bucket": bucket, "text": text}), encoding="utf-8"
                    )
                    cold_rows.append(_run_profile_worker(args, worker_path))
            conditions[condition] = {
                "warm": _profile_summary(warm["rows"], cold=False),
                "cold": _profile_summary(cold_rows, cold=True),
                "warm_rows": warm["rows"],
                "cold_rows": cold_rows,
                "warm_in_process_setup_ns": warm["in_process_setup_ns"],
                "imports": warm["imports"],
            }
    finally:
        input_path.unlink(missing_ok=True)
        worker_path.unlink(missing_ok=True)
    receipt = {
        "schema_version": 2,
        "benchmark": "google-tn-profile-ab",
        "profile": args.profile,
        "subject": {
            "root": str(subject),
            "commit": head,
            "dirty": bool(_git(subject, "status", "--porcelain")),
        },
        "environment": _environment(),
    }
    receipt.update(
        {
            "corpus_dir": str(args.corpus_dir.resolve()),
            "corpus_shards": [f"output-000{shard}-of-00100" for shard in range(90, 95)],
            "sample_sha256": hashlib.sha256(_canonical(inputs)).hexdigest(),
            "sample_sizes": {bucket: len(values) for bucket, values in inputs.items()},
            "profile_warmup": args.profile_warmup,
            "profile_cold_runs": args.profile_cold_runs,
            "seed": args.seed,
            "conditions": conditions,
        }
    )
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _git(root: Path, *argv: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *argv], text=True).strip()


def _run(args: argparse.Namespace) -> int:
    args.output = _external_output(args.output)
    subject = args.subject_root.resolve()
    head = _git(subject, "rev-parse", "HEAD")
    if head != args.expected_head:
        raise SystemExit(f"subject head {head} does not match --expected-head {args.expected_head}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(_canonical(manifest)).hexdigest()
    temporary = args.output.with_suffix(args.output.suffix + ".worker")
    command = [
        sys.executable,
        "-c",
        (f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')"),
        "--_worker",
        "--subject-root",
        str(subject),
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
    if getattr(args, "imports_only", False):
        command.append("--_imports-only")
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
    required_boundaries = ("end_to_end_ns", "resolve_only_ns")
    required_percentiles = ("p50", "p90")
    for label, receipt, locales in (
        ("baseline", baseline, ("en_US",)),
        ("candidate", candidate, ("en_US", "ru_RU", "es_ES")),
    ):
        for locale in locales:
            if not receipt.get("denominators", {}).get(locale):
                raise SystemExit(f"comparison refuses {label} without a nonempty {locale} input")
            summary = receipt.get("summaries", {}).get(locale)
            if not isinstance(summary, dict):
                raise SystemExit(f"comparison refuses {label} without a {locale} summary")
            for boundary in required_boundaries:
                values = summary.get(boundary)
                if not isinstance(values, dict):
                    raise SystemExit(
                        f"comparison refuses {label} without {locale} {boundary} summary"
                    )
                for percentile in required_percentiles:
                    if percentile not in values:
                        raise SystemExit(
                            f"comparison refuses {label} without {locale} {boundary} {percentile}"
                        )
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
    parser.add_argument("--profile", choices=("google-tn",), default=None)
    parser.add_argument("--corpus-dir", type=Path, default=_GOOGLE_TN_CORPUS)
    parser.add_argument("--profile-sample-per-bucket", type=int, default=100)
    parser.add_argument("--profile-cold-runs", type=int, default=20)
    parser.add_argument("--profile-warmup", type=int, default=10)
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
    parser.add_argument("--_profile-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_profile-cold", dest="profile_cold", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--profile-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--profile-condition", choices=("off", "on"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--_imports-only", dest="imports_only", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.compare:
        return _compare(args)
    if args.output is None:
        parser.error("--output is required")
    args.output = _external_output(args.output)
    if args._profile_worker:
        return _profile_worker(args)
    if args._worker:
        return _worker(args)
    if args.subject_root is None or args.expected_head is None:
        parser.error("--subject-root and --expected-head are required")
    if args.profile is not None:
        return _profile_run(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
