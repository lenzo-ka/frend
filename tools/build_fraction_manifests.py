"""Build exact PR6 fraction change and recovery manifests from paired coverage reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from tools.manifest_builder import ManifestBuilder, OutputIdentity

REPO = Path(__file__).resolve().parents[1]


def _is_routed(row: Mapping[str, object]) -> bool:
    if row.get("locale") == "en_US":
        return False
    return any(
        str(edge.get("type", "")).startswith("fraction:")
        or str(edge.get("type", "")).startswith("number:fraction")
        or edge.get("type") == "number:percent"
        for edge in row.get("offer_signature", ())
        if isinstance(edge, Mapping)
    )


BUILDER = ManifestBuilder(
    routing_predicate=lambda before, _after: _is_routed(before),
    output=OutputIdentity(
        family="fraction",
        changes=REPO / "tools/fraction_expected_changes.json",
        recoveries=REPO / "tools/fraction_expected_recoveries.json",
    ),
)


def build(base_path: Path, head_path: Path) -> tuple[dict, dict]:
    return BUILDER.build(base_path, head_path)


def main() -> int:
    return BUILDER.main()


if __name__ == "__main__":
    raise SystemExit(main())
