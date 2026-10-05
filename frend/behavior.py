"""Strict loading and composition of named behavior schemas."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple, cast

from frend.input_folds import apply_input_fold
from frend.input_limits import (
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_UNIT_CHARS,
    _validate_limit,
)
from frend.profiles import GROUP_OPTIONS, GROUP_ROLES, validate_groups, validate_profile
from frend.symbols import SymbolDetector

__all__ = [
    "BehaviorLoadError",
    "BehaviorMember",
    "BehaviorRefusal",
    "FREND_KEYS",
    "IcukitSection",
    "KIND",
    "ResolvedBehavior",
    "SCHEMA_VERSION",
    "SECTIONS",
    "resolve_behavior",
    "shipped_behaviors",
]

SCHEMA_VERSION = 1
KIND = "behavior-schema"
SECTIONS = ("frend", "icukit")
FREND_KEYS = (
    "case_variant_lookup",
    "fold",
    "groups",
    "max_input_chars",
    "max_unit_chars",
    "profile",
    "symbol_run_threshold",
)

_TOP_KEYS = frozenset(
    {"schema_version", "kind", "name", "version", "extends", "provenance", "sections"}
)
_PROVENANCE_KEYS = frozenset({"source", "note"})
_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_MAX_BYTES = 65_536
_MAX_DEPTH = 16
_MAX_VALUES = 4_096
_MAX_STRING = 256
_MAX_EXTENDS_DEPTH = 8
_MAX_DOCUMENTS = 32
_SHIPPED = Path(__file__).with_name("behaviors")


class BehaviorRefusal(NamedTuple):
    """One reason a behavior schema was refused."""

    code: str
    detail: str


class BehaviorLoadError(ValueError):
    """Every refusal found in one transactional validation phase."""

    def __init__(self, refusals: Sequence[BehaviorRefusal]) -> None:
        self.refusals = tuple(refusals)
        super().__init__("; ".join(f"{item.code}: {item.detail}" for item in self.refusals))


@dataclass(frozen=True)
class BehaviorMember:
    """One loaded behavior document in effective composition order."""

    name: str
    version: int
    digest: str
    origin: str


@dataclass(frozen=True)
class IcukitSection:
    """An uninterpreted icukit behavior section carried for its owner."""

    member: str
    digest: str
    mapping: Mapping[str, object]


@dataclass(frozen=True)
class ResolvedBehavior:
    """Immutable effective behavior kwargs and their composition provenance."""

    members: tuple[BehaviorMember, ...]
    digest: str
    kwargs: Mapping[str, object]
    setters: Mapping[str, str]
    icukit_sections: tuple[IcukitSection, ...]


@dataclass(eq=False)
class _Document:
    name: str
    version: int
    digest: str
    origin: str
    extends: tuple[str, ...]
    sections: Mapping[str, object]
    parents: tuple[_Document, ...] = field(default=())


class _InvalidJSON(ValueError):
    pass


class _FrozenList(Sequence[object]):
    """An immutable JSON array that retains JSON list equality semantics."""

    __slots__ = ("_items",)

    def __init__(self, items: Sequence[object]) -> None:
        self._items = tuple(items)

    def __getitem__(self, index: int | slice) -> object:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenList):
            return self._items == other._items
        if isinstance(other, list):
            return list(self._items) == other
        return NotImplemented

    def __repr__(self) -> str:
        return repr(list(self._items))


def _refuse(code: str, detail: str) -> BehaviorRefusal:
    return BehaviorRefusal(code, detail)


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJSON(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise _InvalidJSON(f"non-finite number {value!r}")


def _float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _InvalidJSON(f"non-finite number {value!r}")
    return parsed


def _bounded(value: object, *, depth: int = 1, budget: list[int] | None = None) -> None:
    if budget is None:
        budget = [_MAX_VALUES]
    if depth > _MAX_DEPTH:
        raise _InvalidJSON(f"JSON depth exceeds {_MAX_DEPTH}")
    budget[0] -= 1
    if budget[0] < 0:
        raise _InvalidJSON(f"JSON contains more than {_MAX_VALUES} values")
    if isinstance(value, str) and len(value) > _MAX_STRING:
        raise _InvalidJSON(f"JSON string exceeds {_MAX_STRING} code points")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if len(key) > _MAX_STRING:
                raise _InvalidJSON(f"JSON string exceeds {_MAX_STRING} code points")
            _bounded(item, depth=depth + 1, budget=budget)
    elif isinstance(value, list):
        for item in value:
            _bounded(item, depth=depth + 1, budget=budget)


def _canonical(data: Mapping[str, object]) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList([_freeze(item) for item in value])
    return value


def _read(path: Path, shown: str) -> Mapping[str, object]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_BYTES + 1)
    except OSError as error:
        raise BehaviorLoadError([_refuse("INVALID_JSON", f"{shown}: {error}")]) from error
    if len(raw) > _MAX_BYTES:
        raise BehaviorLoadError(
            [
                _refuse(
                    "TOO_LARGE",
                    f"{shown} exceeds {_MAX_BYTES} bytes; behavior files are capped at "
                    f"{_MAX_BYTES} bytes",
                )
            ]
        )
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object,
            parse_constant=_constant,
            parse_float=_float,
        )
        if not isinstance(parsed, Mapping):
            raise _InvalidJSON("top level is not an object")
        _bounded(parsed)
    except (UnicodeError, json.JSONDecodeError, _InvalidJSON, RecursionError) as error:
        raise BehaviorLoadError([_refuse("INVALID_JSON", f"{shown}: {error}")]) from error
    return cast(Mapping[str, object], parsed)


def _envelope(data: Mapping[str, object], shown: str) -> list[BehaviorRefusal]:
    errors: list[BehaviorRefusal] = []
    keys = set(data)
    for key in sorted(_TOP_KEYS - keys):
        errors.append(_refuse("INVALID_KEY", f"{shown}: missing top-level key {key!r}"))
    for key in sorted(keys - _TOP_KEYS):
        errors.append(_refuse("INVALID_KEY", f"{shown}: unknown top-level key {key!r}"))
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != SCHEMA_VERSION:
        errors.append(
            _refuse(
                "INVALID_SCHEMA_VERSION",
                f"{shown}: top-level field 'schema_version' must be integer 1, got {version!r}",
            )
        )
    kind = data.get("kind")
    if kind != KIND:
        errors.append(
            _refuse(
                "INVALID_KIND",
                f"{shown}: top-level field 'kind' must be {KIND!r}, got {kind!r}",
            )
        )
    name = data.get("name")
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
        errors.append(
            _refuse(
                "INVALID_VALUE",
                f"{shown}: top-level field 'name' must match [a-z0-9][a-z0-9-]{{0,63}}",
            )
        )
    doc_version = data.get("version")
    if not isinstance(doc_version, int) or isinstance(doc_version, bool) or doc_version <= 0:
        errors.append(
            _refuse(
                "INVALID_VALUE",
                f"{shown}: top-level field 'version' must be a positive integer, got "
                f"{doc_version!r}",
            )
        )
    extends = data.get("extends")
    if not isinstance(extends, list) or any(
        not isinstance(parent, str) or _NAME_RE.fullmatch(parent) is None for parent in extends
    ):
        errors.append(
            _refuse(
                "INVALID_EXTENDS",
                f"{shown}: extends must be a list of behavior names",
            )
        )
    else:
        seen: set[str] = set()
        for parent in extends:
            if parent in seen:
                errors.append(
                    _refuse(
                        "INVALID_EXTENDS",
                        f"{shown}: extends lists {parent!r} twice; name each parent once",
                    )
                )
            seen.add(parent)
    provenance = data.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append(_refuse("INVALID_PROVENANCE", f"{shown}: provenance must be an object"))
    else:
        for key in sorted(set(provenance) - _PROVENANCE_KEYS):
            errors.append(
                _refuse("INVALID_PROVENANCE", f"{shown}: provenance has unknown key {key!r}")
            )
        source = provenance.get("source")
        if not isinstance(source, str) or not source:
            errors.append(
                _refuse(
                    "INVALID_PROVENANCE",
                    f"{shown}: provenance.source must be a non-empty string",
                )
            )
        note = provenance.get("note")
        if "note" in provenance and not isinstance(note, str):
            errors.append(
                _refuse("INVALID_PROVENANCE", f"{shown}: provenance.note must be a string")
            )
    sections = data.get("sections")
    if not isinstance(sections, Mapping):
        errors.append(_refuse("INVALID_VALUE", f"{shown}: sections must be an object"))
    else:
        for section in sorted(set(sections) - set(SECTIONS)):
            known = ", ".join(SECTIONS)
            errors.append(
                _refuse(
                    "INVALID_KEY",
                    f"{shown}: sections: unknown section {section!r}; known sections: {known}",
                )
            )
        for section, value in sections.items():
            if section in SECTIONS and not isinstance(value, Mapping):
                errors.append(
                    _refuse(
                        "INVALID_VALUE",
                        f"{shown}: sections.{section} must be an object",
                    )
                )
    return errors


def _frend_section(data: Mapping[str, object], shown: str) -> list[BehaviorRefusal]:
    errors: list[BehaviorRefusal] = []
    allowed = {"schema_version", *FREND_KEYS}
    for key in sorted(set(data) - allowed):
        known = ", ".join(FREND_KEYS)
        errors.append(
            _refuse(
                "INVALID_KEY",
                f"{shown}: sections.frend: unknown key {key!r}; known keys: {known}",
            )
        )
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != SCHEMA_VERSION:
        errors.append(
            _refuse(
                "INVALID_SCHEMA_VERSION",
                f"{shown}: sections.frend.schema_version must be integer 1, got {version!r}",
            )
        )
    groups = data.get("groups")
    group_orders: dict[str, object] | None = None
    if "groups" in data:
        if not isinstance(groups, Mapping):
            errors.append(
                _refuse(
                    "INVALID_VALUE",
                    f"{shown}: sections.frend.groups: groups must be a mapping, got "
                    f"{type(groups).__name__}",
                )
            )
        else:
            group_orders = dict(groups)
    validators = {
        "profile": lambda value: validate_profile(cast(str | None, value)),
        "fold": lambda value: apply_input_fold("", value),
        "case_variant_lookup": lambda value: (
            None
            if isinstance(value, bool)
            else (_ for _ in ()).throw(ValueError(f"must be a bool, got {value!r}"))
        ),
        "symbol_run_threshold": lambda value: SymbolDetector(
            "root", run_threshold=cast(int, value)
        ),
        "max_input_chars": lambda value: _validate_limit(
            "max_input_chars", cast(int | None, value)
        ),
        "max_unit_chars": lambda value: _validate_limit("max_unit_chars", cast(int | None, value)),
    }
    for key, validator in validators.items():
        if key not in data:
            continue
        value = data[key]
        if key in {"max_input_chars", "max_unit_chars"}:
            default = (
                DEFAULT_MAX_INPUT_CHARS if key == "max_input_chars" else DEFAULT_MAX_UNIT_CHARS
            )
            if value is None or (
                isinstance(value, int) and not isinstance(value, bool) and value > default
            ):
                errors.append(
                    _refuse(
                        "BOUND_LOOSENED",
                        f"{shown}: sections.frend.{key}={value} exceeds the default {default}; "
                        f"a behavior file may only tighten bounds; pass {key}= explicitly "
                        "to raise it",
                    )
                )
                continue
        try:
            validator(value)
        except (TypeError, ValueError) as error:
            errors.append(
                _refuse(
                    "INVALID_VALUE",
                    f"{shown}: sections.frend.{key}: {error}",
                )
            )
    if group_orders is not None and not any(
        refusal.detail.startswith(f"{shown}: sections.frend.groups:") for refusal in errors
    ):
        try:
            validate_groups(cast(Mapping, group_orders))
        except (TypeError, ValueError) as error:
            errors.append(_refuse("INVALID_VALUE", f"{shown}: sections.frend.groups: {error}"))
    return errors


def shipped_behaviors() -> tuple[str, ...]:
    """Return the names of behavior schemas shipped with frend."""
    return tuple(sorted(path.stem for path in _SHIPPED.glob("*.json")))


def _located(
    ref: str | os.PathLike[str], search: Sequence[str | os.PathLike[str]]
) -> tuple[Path, str, bool]:
    raw = os.fspath(ref)
    if os.sep in raw or raw.endswith(".json"):
        path = Path(raw)
        return path, raw, False
    matches: list[tuple[Path, str, bool]] = []
    for directory in search:
        shown = os.path.join(os.fspath(directory), f"{raw}.json")
        path = Path(shown)
        if path.is_file():
            matches.append((path, shown, False))
    shipped = _SHIPPED / f"{raw}.json"
    if shipped.is_file():
        matches.append((shipped, "shipped", True))
    if not matches:
        shown = os.path.join(os.fspath(search[0]), f"{raw}.json") if search else raw
        raise BehaviorLoadError(
            [_refuse("INVALID_JSON", f"{shown}: behavior {raw!r} was not found")]
        )
    if len(matches) > 1:
        user = next((match for match in matches if not match[2]), None)
        if user is not None and any(match[2] for match in matches):
            raise BehaviorLoadError(
                [
                    _refuse(
                        "NAME_COLLISION",
                        f"behavior {raw!r} is shipped with frend and also found at {user[1]}; "
                        f'give your file another name and use "extends": ["{raw}"]',
                    )
                ]
            )
        origins = " and ".join(match[1] for match in matches)
        raise BehaviorLoadError(
            [_refuse("NAME_COLLISION", f"behavior {raw!r} is found at both {origins}")]
        )
    return matches[0]


def _load_document(path: Path, shown: str, shipped: bool) -> _Document:
    data = _read(path, shown)
    errors = _envelope(data, shown)
    if errors:
        raise BehaviorLoadError(errors)
    name = cast(str, data["name"])
    if not shipped and path.name != f"{name}.json":
        raise BehaviorLoadError(
            [
                _refuse(
                    "NAME_MISMATCH",
                    f"{shown} declares name {name!r}; the file must be named {name}.json",
                )
            ]
        )
    sections = cast(Mapping[str, object], data["sections"])
    frend = sections.get("frend")
    if isinstance(frend, Mapping):
        errors = _frend_section(frend, shown)
        if errors:
            raise BehaviorLoadError(errors)
    digest = "sha256:" + sha256(_canonical(data)).hexdigest()
    return _Document(
        name,
        cast(int, data["version"]),
        digest,
        "shipped" if shipped else shown,
        tuple(cast(list[str], data["extends"])),
        sections,
    )


def _inconsistent(doc: _Document, linear: Mapping[_Document, tuple[_Document, ...]]) -> str:
    parents = doc.parents
    for left_at, left in enumerate(parents):
        left_order = linear[left]
        for right in parents[left_at + 1 :]:
            right_order = linear[right]
            common = [item for item in left_order if item in right_order]
            for first_at, first in enumerate(common):
                for second in common[first_at + 1 :]:
                    if right_order.index(first) > right_order.index(second):
                        return (
                            f"{left.name!r} lists {first.name!r} before {second.name!r} and "
                            f"{right.name!r} lists {second.name!r} before {first.name!r}; "
                            "no single order satisfies both"
                        )
    return f"behavior {doc.name!r} has inconsistent parent orders"


def resolve_behavior(
    refs: Sequence[str | os.PathLike[str]],
    /,
    *,
    search: Sequence[str | os.PathLike[str]] = (),
) -> ResolvedBehavior:
    """Resolve behavior presets in caller order, with later values winning."""
    docs: dict[str, _Document] = {}

    def load(ref: str | os.PathLike[str], stack: tuple[str, ...]) -> _Document:
        if len(stack) > _MAX_EXTENDS_DEPTH:
            raise BehaviorLoadError(
                [_refuse("EXTENDS_TOO_DEEP", f"extends depth exceeds {_MAX_EXTENDS_DEPTH}")]
            )
        path, shown, is_shipped = _located(ref, search)
        doc = _load_document(path, shown, is_shipped)
        if doc.name in stack:
            cycle = " -> ".join((*stack, doc.name))
            raise BehaviorLoadError([_refuse("EXTENDS_CYCLE", cycle)])
        previous = docs.get(doc.name)
        if doc.name in shipped_behaviors() and not is_shipped:
            raise BehaviorLoadError(
                [
                    _refuse(
                        "NAME_COLLISION",
                        f"behavior {doc.name!r} is shipped with frend and also found at {shown}; "
                        f'give your file another name and use "extends": ["{doc.name}"]',
                    )
                ]
            )
        if previous is not None:
            if previous.digest != doc.digest:
                raise BehaviorLoadError(
                    [
                        _refuse(
                            "NAME_COLLISION",
                            f"behavior {doc.name!r} is declared by both {previous.origin} and "
                            f"{doc.origin} with different contents; one name denotes one document",
                        )
                    ]
                )
            return previous
        docs[doc.name] = doc
        doc.parents = tuple(load(parent, (*stack, doc.name)) for parent in doc.extends)
        return doc

    roots = tuple(load(ref, ()) for ref in refs)
    if len(docs) > _MAX_DOCUMENTS:
        raise BehaviorLoadError(
            [
                _refuse(
                    "TOO_MANY_DOCUMENTS",
                    f"behavior composition contains {len(docs)} documents; maximum is "
                    f"{_MAX_DOCUMENTS}",
                )
            ]
        )
    memo: dict[_Document, tuple[_Document, ...]] = {}

    def linearize(doc: _Document) -> tuple[_Document, ...]:
        if doc in memo:
            return memo[doc]
        sequences = [list(linearize(parent)) for parent in doc.parents]
        sequences.append(list(doc.parents))
        merged: list[_Document] = []
        while any(sequences):
            sequences = [sequence for sequence in sequences if sequence]
            candidate = next(
                (
                    sequence[0]
                    for sequence in sequences
                    if not any(sequence[0] in other[1:] for other in sequences)
                ),
                None,
            )
            if candidate is None:
                raise BehaviorLoadError([_refuse("EXTENDS_INCONSISTENT", _inconsistent(doc, memo))])
            merged.append(candidate)
            for sequence in sequences:
                while candidate in sequence:
                    sequence.remove(candidate)
        result = (*merged, doc)
        memo[doc] = result
        return result

    sequence = tuple(item for root in roots for item in linearize(root))
    order = tuple(item for index, item in enumerate(sequence) if item not in sequence[index + 1 :])
    kwargs: dict[str, object] = {}
    setters: dict[str, str] = {}
    icukit: list[IcukitSection] = []
    for doc in order:
        frend = doc.sections.get("frend")
        if isinstance(frend, Mapping):
            for key, value in frend.items():
                if key == "schema_version":
                    continue
                if key == "groups":
                    groups = kwargs.setdefault("groups", {})
                    assert isinstance(groups, dict)
                    assert isinstance(value, Mapping)
                    settings = dict(validate_groups(cast(Mapping, value)) or ())
                    for group in GROUP_ROLES:
                        if group not in settings:
                            continue
                        spec = value[group]
                        setting = settings[group]
                        previous = (
                            dict(validate_groups({group: groups[group]}) or ())[group]
                            if group in groups
                            else None
                        )
                        options = dict(
                            previous.options if previous is not None else setting.options
                        )
                        if isinstance(spec, Mapping):
                            for option in GROUP_OPTIONS.get(group, {}):
                                if option in spec:
                                    options[option] = cast(str, spec[option])
                                    setters[f"groups.{group}.{option}"] = doc.name
                        groups.pop(group, None)
                        defaults = {
                            name: choices[0]
                            for name, choices in GROUP_OPTIONS.get(group, {}).items()
                        }
                        groups[group] = (
                            setting.order
                            if options == defaults
                            else MappingProxyType({"order": setting.order, **options})
                        )
                        setters[f"groups.{group}"] = doc.name
                else:
                    kwargs[key] = value
                    setters[key] = doc.name
        raw_icukit = doc.sections.get("icukit")
        if isinstance(raw_icukit, Mapping):
            icukit.append(
                IcukitSection(
                    doc.name,
                    doc.digest,
                    cast(Mapping[str, object], _freeze(raw_icukit)),
                )
            )
    digest_source = "\n".join(doc.digest for doc in order).encode()
    return ResolvedBehavior(
        tuple(BehaviorMember(doc.name, doc.version, doc.digest, doc.origin) for doc in order),
        "sha256:" + sha256(digest_source).hexdigest(),
        cast(Mapping[str, object], _freeze(kwargs)),
        cast(Mapping[str, str], _freeze(setters)),
        tuple(icukit),
    )
