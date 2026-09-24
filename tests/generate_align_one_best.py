"""Regenerate the 1-best golden using the frend imported from PYTHONPATH."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from icukit.detectors import detect
from icukit.recognize import (
    FlexibleCurrencyDetector,
    FlexibleDateDetector,
    FlexibleFractionDetector,
    FlexibleNumberDetector,
    LetterNameDetector,
    SingleLetterWordDetector,
)

from frend.fold_resolve import resolve, resolve_cover
from frend.lattice import resolve_lattice


def main(path):
    artifact = Path(path)
    rows = json.loads(artifact.read_text())
    detectors = [
        FlexibleDateDetector("en_US"),
        FlexibleNumberDetector("en_US"),
        LetterNameDetector("en_US"),
        SingleLetterWordDetector("en_US"),
        FlexibleCurrencyDetector("en_US", "USD"),
        FlexibleFractionDetector("en_US"),
    ]
    for row in rows:
        text = row["text"]
        detections = detect(text, detectors)
        outputs = [
            repr(resolve(detections, source_text=text)),
            repr(resolve_lattice(detections, source_text=text)),
            repr(resolve_cover(detections, source_text=text)),
        ]
        row["outputs"] = [hashlib.sha256(value.encode()).hexdigest() for value in outputs]
    artifact.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
