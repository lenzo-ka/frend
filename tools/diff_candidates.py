"""Compare isolated base/head public unit signatures for one recognition family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from icukit.detectors import detect

from frend import compose_choices, normalize, resolve_choices
from frend.normalize import _reading_detectors, _sentence_ranges

SHARDS = tuple(f"output-{number:05d}-of-00100" for number in range(90, 95))


def _graph_signature(text: str, locale: str) -> list[dict]:
    detectors = _reading_detectors(locale)
    detections = list(detect(text, detectors))
    graph = compose_choices(resolve_choices(detections, source_text=text, locale=locale))
    signature = []
    for edge, unit in zip(graph.lattice.edges, graph.units, strict=True):
        signature.append(
            {
                "span": [edge.start, edge.end],
                "kind": edge.kind,
                "type": None if edge.detection is None else edge.detection.get("type"),
                "alternatives": [
                    {
                        "text": item.text,
                        "provenance": item.provenance,
                        "prior_provenance": item.prior_provenance,
                        "weight": None if item.weight is None else str(item.weight),
                    }
                    for item in unit.alternatives
                ],
            }
        )
    return signature


def _offer_signature(text: str, locale: str) -> tuple[bool, dict]:
    detectors = _reading_detectors(locale)
    candidate_detectors = tuple(
        detector for detector in detectors if str(getattr(detector, "type", "")).startswith("date:")
    )
    if not list(detect(text, candidate_detectors)):
        return False, {}
    try:
        public = normalize(text, locale=locale, offsets=True)
        sentences = [
            {
                "span": [start, end],
                "offers": _graph_signature(text[start:end], locale),
            }
            for start, end in _sentence_ranges(text, locale)
        ]
    except Exception as error:  # Public failures are part of the identity signature.
        return True, {
            "error": {
                "module": type(error).__module__,
                "type": type(error).__qualname__,
                "message": str(error),
            }
        }
    return True, {
        "text": public.text,
        "fold": public.fold,
        "units": [
            {
                "output_span": list(unit.output_span),
                "source_span": list(unit.source_span),
                "reader": unit.reader,
                "provenance": unit.provenance,
            }
            for unit in public.units
        ],
        "sentences": sentences,
    }


def _rows(corpus_dir: Path, sentence_cap: int | None, shards: tuple[str, ...] = SHARDS):
    for shard in shards:
        path = corpus_dir / shard
        sentence = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                columns = line.rstrip("\n").split("\t")
                if columns[0] == "<eos>":
                    sentence += 1
                    if sentence_cap is not None and sentence >= sentence_cap:
                        break
                    continue
                if len(columns) < 3:
                    continue
                class_, written, _spoken = columns[:3]
                yield shard, line_number, written


def _signature_digest(signature: dict) -> str:
    payload = json.dumps(
        signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _child(corpus_dir: Path, out: Path, sentence_cap: int | None, shard: str | None = None) -> int:
    cases = {}
    scanned = 0
    selected_shards = SHARDS if shard is None else (shard,)
    for shard_name, line_number, written in _rows(corpus_dir, sentence_cap, selected_shards):
        scanned += 1
        found, signature = _offer_signature(written, "en_US")
        if not found:
            continue
        identifier = hashlib.sha256(f"{shard_name}:{line_number}:{written}".encode()).hexdigest()
        cases[identifier] = _signature_digest(signature)
    out.write_text(
        json.dumps(
            {
                "schema": 1,
                "shards": list(selected_shards),
                "scanned": scanned,
                "signature_encoding": "sha256-canonical-json",
                "cases": cases,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def compare(
    base_repo: Path,
    head_repo: Path,
    corpus_dir: Path,
    out_dir: Path,
    sentence_cap: int | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    destinations: dict[str, dict[str, Path]] = {"base": {}, "head": {}}
    for side, repo in (("base", base_repo), ("head", head_repo)):
        side_dir = out_dir / side
        side_dir.mkdir(exist_ok=True)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repo)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        processes = []
        for shard in SHARDS:
            destination = side_dir / f"{shard}.json"
            destinations[side][shard] = destination
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--child-out",
                str(destination),
                "--corpus-dir",
                str(corpus_dir),
                "--shard",
                shard,
            ]
            if sentence_cap is not None:
                command.extend(("--sentences-per-shard", str(sentence_cap)))
            processes.append(subprocess.Popen(command, cwd=repo, env=environment))
        for process in processes:
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args)

    scanned = {"base": 0, "head": 0}
    candidate_union = 0
    changed = []
    for shard in SHARDS:
        documents = {
            side: json.loads(destinations[side][shard].read_text(encoding="utf-8"))
            for side in ("base", "head")
        }
        for side in scanned:
            scanned[side] += documents[side]["scanned"]
        base_cases = documents["base"]["cases"]
        head_cases = documents["head"]["cases"]
        identifiers = sorted(set(base_cases) | set(head_cases))
        candidate_union += len(identifiers)
        changed.extend(
            identifier
            for identifier in identifiers
            if base_cases.get(identifier) != head_cases.get(identifier)
        )
    report = {
        "schema": 1,
        "family": "date",
        "scanned": scanned,
        "candidate_union": candidate_union,
        "signature_encoding": "sha256-canonical-json",
        "changed": changed,
        "empty_diff": not changed,
    }
    (out_dir / "diff.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("date",), default="date")
    parser.add_argument("--base-repo", type=Path)
    parser.add_argument("--head-repo", type=Path)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--sentences-per-shard", type=int)
    parser.add_argument("--child-out", type=Path)
    parser.add_argument("--shard", choices=SHARDS)
    args = parser.parse_args()
    if args.child_out is not None:
        return _child(args.corpus_dir, args.child_out, args.sentences_per_shard, args.shard)
    if args.base_repo is None or args.head_repo is None or args.out_dir is None:
        parser.error("--base-repo, --head-repo, and --out-dir are required")
    report = compare(
        args.base_repo,
        args.head_repo,
        args.corpus_dir,
        args.out_dir,
        args.sentences_per_shard,
    )
    return int(not report["empty_diff"])


if __name__ == "__main__":
    raise SystemExit(main())
