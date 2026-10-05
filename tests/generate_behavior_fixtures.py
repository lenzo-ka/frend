"""Regenerate project-authored behavior-schema fixtures and manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent / "data" / "behaviors"


def base(name, *, extends=None, sections=None):
    return {
        "schema_version": 1,
        "kind": "behavior-schema",
        "name": name,
        "version": 1,
        "extends": extends or [],
        "provenance": {"source": "frend tests"},
        "sections": sections or {},
    }


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def write(relative, value):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else encoded(value)
    path.write_bytes(raw)
    return raw


def manifest(where, entries):
    files = {
        name: {
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "expected": expected,
        }
        for name, (raw, expected) in sorted(entries.items())
    }
    write(f"{where}/MANIFEST.json", {"files": files})


def main():
    write(
        "user/screen-reader-symbols.json",
        {
            **base("screen-reader-symbols"),
            "provenance": {
                "source": "frend tests",
                "note": "symbol runs read character by character; not a screen-reader mode",
            },
            "sections": {
                "frend": {
                    "schema_version": 1,
                    "groups": {"tts-sanity": {"order": ["named", "described", "silent"]}},
                }
            },
        },
    )
    write(
        "user/with-icukit.json",
        {
            **base("with-icukit"),
            "sections": {
                "frend": {"schema_version": 1},
                "icukit": {"schema_version": 1, "sentence": {"base": "en-tn@1"}},
            },
        },
    )
    invalid = {
        "version-2.json": {**base("version-2"), "schema_version": 2},
        "typo-key.json": {
            **base("typo-key"),
            "sections": {"frend": {"schema_version": 1, "grups": {}}},
        },
        "typo-section.json": {**base("typo-section"), "sections": {"icukt": {}}},
        "google-tn.json": base("google-tn"),
        "bad-profile.json": {
            **base("bad-profile"),
            "sections": {"frend": {"schema_version": 1, "profile": "googl-tn"}},
        },
        "loose.json": {
            **base("loose"),
            "sections": {"frend": {"schema_version": 1, "max_unit_chars": 100000}},
        },
        "cycle-a.json": base("cycle-a", extends=["cycle-b"]),
        "cycle-b.json": base("cycle-b", extends=["cycle-a"]),
        "mine.json": base("ours"),
        "double-parent.json": base("double-parent", extends=["a", "a"]),
        "three-faults.json": {
            **base("three-faults"),
            "schema_version": 2,
            "unexpected": True,
            "provenance": {},
        },
    }
    for name, value in invalid.items():
        write(f"invalid/{name}", value)
    write("invalid/big.json", encoded(base("big"))[:-1] + b" " * 65536 + b"\n")
    graph = {
        "a.json": base("a", sections={"frend": {"schema_version": 1, "profile": None}}),
        "b.json": base(
            "b", extends=["a"], sections={"frend": {"schema_version": 1, "profile": "google-tn"}}
        ),
        "c.json": base("c", extends=["a"], sections={"frend": {"schema_version": 1, "fold": None}}),
        "d.json": base("d", extends=["b", "c"]),
        "x.json": base("x", extends=["a", "b"]),
        "y.json": base("y", extends=["b", "a"]),
        "z.json": base("z", extends=["x", "y"]),
    }
    for name, value in graph.items():
        write(f"graph/{name}", value)
    chain = {
        "a.json": base("a", sections={"frend": {"schema_version": 1, "profile": None}}),
        "b.json": base(
            "b", extends=["a"], sections={"frend": {"schema_version": 1, "profile": "google-tn"}}
        ),
        "c.json": base("c", extends=["b"]),
    }
    for name, value in chain.items():
        write(f"chain/{name}", value)
    inconsistent = {
        "a.json": base("a"),
        "b.json": base("b"),
        "x.json": base("x", extends=["a", "b"]),
        "y.json": base("y", extends=["b", "a"]),
        "z.json": base("z", extends=["x", "y"]),
    }
    for name, value in inconsistent.items():
        write(f"inconsistent/{name}", value)
    write("p/a.json", base("a", sections={"frend": {"schema_version": 1}}))
    write("q/a.json", base("a", sections={"frend": {"schema_version": 1, "fold": None}}))
    same = base("same", sections={"frend": {"schema_version": 1}})
    write("p/same.json", same)
    write("q/same.json", same)
    collision = base("search-collision")
    write("p/search-collision.json", collision)
    write("q/search-collision.json", collision)
    envelope = {}

    def env(name, value, expected):
        envelope[name] = (write(f"conformance/envelope/{name}", value), expected)

    env("valid/empty.json", base("empty"), "valid")
    env(
        "valid/frend.json",
        base(
            "frend",
            sections={
                "frend": {
                    "schema_version": 1,
                    "groups": {"tts-sanity": {"order": ["described", "named", "silent"]}},
                }
            },
        ),
        "valid",
    )
    env(
        "invalid/schema-version.json",
        {**base("schema-version"), "schema_version": 2},
        "INVALID_SCHEMA_VERSION",
    )
    env("invalid/kind.json", {**base("kind"), "kind": "other"}, "INVALID_KIND")
    env("invalid/top-key.json", {**base("top-key"), "extra": 1}, "INVALID_KEY")
    env(
        "invalid/section-name.json",
        {**base("section-name"), "sections": {"other": {}}},
        "INVALID_KEY",
    )
    env("invalid/provenance.json", {**base("provenance"), "provenance": {}}, "INVALID_PROVENANCE")
    env(
        "invalid/repeated-extends.json",
        base("repeated-extends", extends=["a", "a"]),
        "INVALID_EXTENDS",
    )
    duplicate = encoded(base("duplicate")).replace(
        b'"version": 1,', b'"version": 1,\n  "version": 1,'
    )
    env("invalid/duplicate-key.json", duplicate, "INVALID_JSON")
    nan = encoded(base("nan")).replace(b'"version": 1', b'"version": NaN')
    env("invalid/nan.json", nan, "INVALID_JSON")
    nested = "leaf"
    for _ in range(17):
        nested = [nested]
    env(
        "invalid/depth.json",
        {**base("depth"), "sections": {"icukit": {"nested": nested}}},
        "INVALID_JSON",
    )
    env(
        "invalid/values.json",
        {**base("values"), "sections": {"icukit": {"values": [0] * 4097}}},
        "INVALID_JSON",
    )
    env(
        "invalid/string.json",
        {**base("string"), "provenance": {"source": "x" * 257}},
        "INVALID_JSON",
    )
    huge = encoded(base("too-large"))[:-1] + b" " * 65536 + b"\n"
    env("invalid/too-large.json", huge, "TOO_LARGE")
    manifest("conformance/envelope", envelope)

    owned = {}

    def own(name, value, expected):
        owned[name] = (write(f"conformance/frend/{name}", value), expected)

    own(
        "valid/all-options.json",
        base(
            "all-options",
            sections={
                "frend": {
                    "schema_version": 1,
                    "case_variant_lookup": True,
                    "fold": None,
                    "groups": {"tts-sanity": {"order": ["named", "described", "silent"]}},
                    "max_input_chars": 1000,
                    "max_unit_chars": 100,
                    "profile": None,
                    "symbol_run_threshold": 3,
                }
            },
        ),
        "valid",
    )
    own(
        "invalid/typo.json",
        {
            **base("typo"),
            "sections": {"frend": {"schema_version": 1, "grups": {}}},
        },
        "INVALID_KEY",
    )
    own(
        "invalid/groups.json",
        base(
            "groups",
            sections={
                "frend": {
                    "schema_version": 1,
                    "groups": {"tts-sanity": {"order": ["named"]}},
                }
            },
        ),
        "INVALID_VALUE",
    )
    own("invalid/loose.json", invalid["loose.json"], "BOUND_LOOSENED")
    manifest("conformance/frend", owned)


if __name__ == "__main__":
    main()
