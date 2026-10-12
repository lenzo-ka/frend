"""Build exact PR7 development change/recovery manifests from paired coverage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from tools.manifest_builder import ManifestBuilder, OutputIdentity

REPO = Path(__file__).resolve().parents[1]


def _is_routed(row: Mapping[str, object]) -> bool:
    if row.get("locale") == "en_US":
        return False
    return any(
        isinstance(alternative, Mapping) and "time-rule:" in str(alternative.get("provenance", ""))
        for edge in row.get("offer_signature", ())
        if isinstance(edge, Mapping)
        for alternative in edge.get("alternatives", ())
    )


BUILDER = ManifestBuilder(
    routing_predicate=lambda _before, after: _is_routed(after),
    output=OutputIdentity(
        family="time",
        changes=REPO / "tools/time_expected_changes.json",
        recoveries=REPO / "tools/time_expected_recoveries.json",
    ),
    recovery_route_description="route",
)


def build(base_path: Path, head_path: Path) -> tuple[dict, dict]:
    return BUILDER.build(base_path, head_path)


def main() -> int:
    return BUILDER.main()


if __name__ == "__main__":
    raise SystemExit(main())
