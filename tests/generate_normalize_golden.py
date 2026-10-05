"""Regenerate the no-behavior-schema normalization golden."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import frend

DATA = Path(__file__).parent / "data"


def input_rows() -> list[dict[str, str]]:
    """Return the 58 plan-selected inputs with stable source-local names."""
    align = json.loads((DATA / "align_one_best.json").read_text(encoding="utf-8"))
    typographic = json.loads(
        (DATA / "typographic_fold_synthetic.json").read_text(encoding="utf-8")
    )["items"]
    authored = json.loads((DATA / "normalize_golden_inputs.json").read_text(encoding="utf-8"))[
        "items"
    ]
    rows = [
        {"source": "align_one_best", "name": f"row-{index:02d}", "text": row["text"]}
        for index, row in enumerate(align, 1)
    ]
    rows.extend(
        {
            "source": "typographic_fold_synthetic",
            "name": row["name"],
            "text": row["written"],
        }
        for row in typographic
    )
    rows.extend(
        {"source": "normalize_golden_inputs", "name": row["name"], "text": row["text"]}
        for row in authored
    )
    if len(rows) != 58:
        raise ValueError(f"normalization golden needs 58 inputs, got {len(rows)}")
    return rows


def document() -> dict[str, object]:
    rows = []
    for row in input_rows():
        text = row["text"]
        rows.append(
            {
                **row,
                "normalized": frend.normalize(text),
                "offsets_repr": repr(frend.normalize(text, offsets=True)),
            }
        )
    return {"schema_version": 1, "rows": rows}


def encoded_document() -> bytes:
    return (json.dumps(document(), indent=2, ensure_ascii=False) + "\n").encode()


def main(path: str) -> None:
    Path(path).write_bytes(encoded_document())


if __name__ == "__main__":
    main(sys.argv[1])
