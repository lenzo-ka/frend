#!/usr/bin/env python3
"""Measure the opaque base-frozen PR9 email witness without exposing its rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

from frend.electronic import digit_forms
from tools.locale_gate import FixtureCase, _evaluate

LETTERS = {
    "a": "a",
    "b": "be",
    "c": "ce",
    "d": "de",
    "e": "e",
    "f": "ef",
    "g": "ge",
    "h": "ha",
    "i": "i",
    "j": "jot",
    "k": "ka",
    "l": "el",
    "m": "em",
    "n": "en",
    "o": "o",
    "p": "pe",
    "q": "ku",
    "r": "er",
    "s": "es",
    "t": "te",
    "u": "u",
    "v": "vau",
    "w": "we",
    "x": "ix",
    "y": "Ypsilon",
    "z": "zett",
}
PUBLIC_FIELDS = (
    "strict",
    "presentation",
    "insensitive",
    "first_strict",
    "first_presentation",
    "first",
    "error",
)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _rows(seed: int, count: int):
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    separators = (".", "-", "_")
    for index in range(count):
        left = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 4)))
        if index % 2:
            left += rng.choice(separators) + "".join(
                rng.choice(alphabet) for _ in range(rng.randint(1, 3))
            )
        if index % 3 == 0:
            left += str(rng.randrange(10))
        host = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 4)))
        if index % 5 == 0:
            host += "-" + "".join(rng.choice(alphabet) for _ in range(2))
        text = f"{left}@{host}.de"
        words = []
        records = []
        for character in text:
            if character.isalpha():
                words.append(LETTERS[character])
                records.append(f"letter-{character.upper()}")
            elif character.isdigit():
                words.append(digit_forms(character, "de_DE")["digits"][0][0])
            else:
                words.append({"@": "at", ".": "Punkt", "-": "Minus", "_": "Unterstrich"}[character])
                records.append(f"separator-{ord(character):04x}")
        yield index, text, " ".join(words), records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    witness = json.loads(args.witness.read_text(encoding="utf-8"))
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != witness["manifest_sha256"]:
        raise SystemExit("frozen electronic manifest hash mismatch")
    frozen_rows = witness.get("rows")
    if not isinstance(frozen_rows, list) or _canonical_hash(frozen_rows) != witness.get(
        "canonical_rows_sha256"
    ):
        raise SystemExit("frozen electronic witness hash mismatch")
    generated = list(_rows(witness["generator"]["seed"], witness["generator"]["rows"]))
    if len(generated) != len(frozen_rows):
        raise SystemExit("frozen electronic witness count mismatch")
    observations = []
    counts = Counter()
    for frozen, (ordinal, text, target, _records) in zip(frozen_rows, generated, strict=True):
        identifier = hashlib.sha256(
            f"{witness['generator']['seed']}:{ordinal}:{text}".encode()
        ).hexdigest()[:24]
        if frozen["id"] != identifier or frozen["ordinal"] != ordinal:
            raise SystemExit("frozen electronic witness identity mismatch")
        if frozen["expected_speech_sha256"] != hashlib.sha256(target.encode()).hexdigest():
            raise SystemExit("frozen electronic expected-speech hash mismatch")
        case = FixtureCase(
            "de",
            "de_DE",
            "electronic",
            "opaque-pr9",
            ordinal + 1,
            text,
            (target,),
            False,
        )
        observed = _evaluate(case, {})
        head = {field: observed[field] for field in PUBLIC_FIELDS}
        head["offer_signature_sha256"] = _canonical_hash(observed["offer_signature"])
        base = frozen["base"]
        if not base["strict"] and head["strict"]:
            status = "recovered"
        elif base["strict"] and not head["strict"]:
            status = "negative_flip"
        else:
            status = "unchanged"
        counts[status] += 1
        observations.append({"id": identifier, "status": status, "base": base, "head": head})
    head_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
    ).strip()
    report = {
        "schema": 1,
        "witness_rows_sha256": witness["canonical_rows_sha256"],
        "base_head": witness["base_head"],
        "measured_head": head_commit,
        "counts": {name: counts[name] for name in ("recovered", "negative_flip", "unchanged")},
        "observations": observations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["counts"], sort_keys=True))
    return int(bool(counts["negative_flip"]))


if __name__ == "__main__":
    raise SystemExit(main())
