#!/usr/bin/env python3
"""Derive the notice file index from provenance in shipped package data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from frend import data_sources

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "frend" / "data"
NOTICE = REPO / "THIRD_PARTY_NOTICES.md"
BEGIN = "<!-- BEGIN GENERATED THIRD-PARTY FILE INDEX -->"
END = "<!-- END GENERATED THIRD-PARTY FILE INDEX -->"
PROJECT_SOURCES = frozenset({"frend/curated", "lenzo/break_exceptions"})
_REFERENCE = re.compile(r"^(?:normalization-record:)?(.+?)(?:#.*)?$")


def _canonical_source(value: str, record: Mapping[str, object] | None = None) -> str | None:
    source = data_sources.source_id(value, record)
    return source if data_sources.source_class(source) is not None else None


def _reference_source(value: str) -> str | None:
    match = _REFERENCE.fullmatch(value)
    if match is None:
        return None
    candidate = match.group(1)
    source = _canonical_source(candidate)
    if source is not None:
        return source
    # Compiled telephone provenance prefixes a rule id before the source id.
    parts = candidate.split(":")
    for index in range(1, len(parts)):
        source = _canonical_source(":".join(parts[index:]))
        if source is not None:
            return source
    return None


def _sources_in_document(document: object) -> set[str]:
    found: set[str] = set()

    def visit(value: object, *, field: str | None = None) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"source", "corpus"} and isinstance(child, str):
                    source = _canonical_source(child, value)
                    if source is not None:
                        found.add(source)
                elif key == "sources" and isinstance(child, list):
                    for item in child:
                        if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                            source = _canonical_source(item["id"], item)
                            if source is not None:
                                found.add(source)
                visit(child, field=str(key))
        elif isinstance(value, list):
            for child in value:
                visit(child, field=field)
        elif isinstance(value, str) and field in {"derived_from", "provenance"}:
            source = _reference_source(value)
            if source is not None:
                found.add(source)

    visit(document)
    return found


def _document(path: Path) -> object:
    if path.suffix == ".cart":
        from cartlet.runner import read_cart_metadata

        return read_cart_metadata(path.read_bytes())
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _canonical_sha256(document: object) -> str:
    encoded = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_file_index(data_root: Path = DATA) -> dict[str, tuple[str, ...]]:
    """Return each third-party source and all shipped files that name its ancestry."""
    indexed: dict[str, set[str]] = defaultdict(set)
    documents = {
        path: _document(path)
        for path in sorted(item for item in data_root.rglob("*") if item.is_file())
        if path.name != "README.md"
    }
    normalization_sources = {
        _canonical_sha256(document): _sources_in_document(document)
        for path, document in documents.items()
        if path.parent.name == "normalization" and isinstance(document, Mapping)
    }
    for path, document in documents.items():
        if path.name == "README.md":
            continue
        relative = path.relative_to(data_root).as_posix()
        sources = _sources_in_document(document)
        if isinstance(document, Mapping) and isinstance(document.get("normalization_sha256"), str):
            sources.update(normalization_sources.get(document["normalization_sha256"], ()))
        sources.update(
            source
            for source, entry in data_sources.SHIPPABLE_SOURCES.items()
            if relative in entry["vendored"]
        )
        package_path = f"frend/data/{relative}"
        for source in sources - PROJECT_SOURCES:
            indexed[source].add(package_path)
    return {source: tuple(sorted(paths)) for source, paths in sorted(indexed.items())}


def _wrapped_paths(paths: Iterable[str]) -> str:
    return "\n".join(f"- `{path}`" for path in paths)


def render_index(index: Mapping[str, Iterable[str]] | None = None) -> str:
    """Render the authoritative source-to-derived-file notice index."""
    index = source_file_index() if index is None else index
    lines = [
        BEGIN,
        "## Transitive covered-file index",
        "",
        "This index is generated from the shipped files' own provenance fields by",
        "`python -B tools/build_third_party_notices.py`. It is authoritative for the",
        "files covered by each source notice, including compiled derivatives.",
    ]
    for source, paths in index.items():
        lines.extend(("", f"### `{source}`", "", _wrapped_paths(paths)))
    lines.extend(("", END))
    return "\n".join(lines)


def update_notice(text: str, index: Mapping[str, Iterable[str]] | None = None) -> str:
    generated = render_index(index)
    if BEGIN not in text or END not in text:
        return text.rstrip() + "\n\n" + generated + "\n"
    prefix, remainder = text.split(BEGIN, 1)
    _, suffix = remainder.split(END, 1)
    return prefix.rstrip() + "\n\n" + generated + suffix.rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    current = NOTICE.read_text(encoding="utf-8")
    expected = update_notice(current)
    if args.check:
        if current != expected:
            print(f"drift: {NOTICE} does not match shipped provenance", file=sys.stderr)
            return 1
        print(f"--check: {NOTICE} matches shipped provenance")
        return 0
    NOTICE.write_text(expected, encoding="utf-8")
    print(f"wrote {NOTICE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
