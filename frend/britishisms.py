"""Corpus-measured Kestrel UK-to-US respelling for the ``google-tn`` profile."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from frend.locale_data import LOCALE_CACHE, canonical_locale
from frend.profiles import GOOGLE_TN, google_tn_britishisms_path

__all__ = [
    "CASE_SHAPES",
    "EDIT_RULES",
    "PAIR_CLASSES",
    "apply_edit_rule",
    "case_shape",
    "load_britishisms",
    "respell",
    "respell_from_table",
]


@dataclass(frozen=True)
class EditRule:
    name: str
    pattern: re.Pattern[str]
    replacement: str

    def apply(self, word: str) -> str:
        return self.pattern.sub(self.replacement, word, count=1)

    def stem(self, word: str) -> str | None:
        match = self.pattern.search(word)
        return None if match is None else word[: match.start()]


# Deliberately small and auditable. The builder measures every rule on 00--79 and
# selects its common rate threshold on 80--89; a rule below that threshold abstains.
EDIT_RULES = (
    EditRule("isation-to-ization", re.compile(r"isation(?=s?$)"), "ization"),
    EditRule("ise-to-ize", re.compile(r"s(?=(?:e(?:s|d|r|rs)?|ing|ation(?:s)?)$)"), "z"),
    EditRule("our-to-or", re.compile(r"our"), "or"),
    EditRule("re-to-er", re.compile(r"re(?=(?:s|d)?$)"), "er"),
    EditRule("ogue-to-og", re.compile(r"ogue(?=(?:s|d)?$)"), "og"),
    EditRule("mme-to-m", re.compile(r"mme(?=(?:s|d)?$)"), "m"),
    EditRule(
        "doubled-consonant",
        re.compile(r"([bcdfghjklmnpqrstvwxyz])\1(?=(?:ed|ing|er|ers|or|ors)?$)"),
        r"\1",
    ),
    EditRule("yse-to-yze", re.compile(r"yse(?=(?:s|d|r|rs|ing)?$)"), "yze"),
    EditRule("ae-to-e", re.compile(r"ae"), "e"),
    EditRule("oe-to-e", re.compile(r"oe"), "e"),
)
_RULES = {rule.name: rule for rule in EDIT_RULES}
PAIR_CLASSES = ("respelling", "diacritic", "expansion/abbreviation", "other")
CASE_SHAPES = ("lower", "title", "upper")


def apply_edit_rule(name: str, word: str) -> str:
    """Apply one named edit rule to a case-normalized word."""
    return _RULES[name].apply(word)


@dataclass(frozen=True)
class BritishismTable:
    locale: str
    source_shards: tuple[tuple[str, str], ...]
    minimum_support: dict[str, int]
    admitted_classes: frozenset[str]
    case_folding: bool
    rule_rate_threshold: float
    pairs: dict[str, dict[str, dict[str, object]]]
    rules: dict[str, dict[str, frozenset[str]]]


_FILE_KEYS: dict[str, tuple[tuple[int, int, int, int, int], tuple[str, int, int, str]]] = {}


def _missing(path: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"{GOOGLE_TN!r} Britishism profile data is missing at {path}; build it from your "
        "licensed Google TN corpus with tools/build_britishisms.py --profile-out PATH"
    )


def _file_key(path: Path) -> tuple[str, int, int, str]:
    resolved = path.expanduser().resolve()
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        raise _missing(resolved) from None
    freshness = (stat.st_dev, stat.st_ino, stat.st_ctime_ns, stat.st_mtime_ns, stat.st_size)
    cached = _FILE_KEYS.get(str(resolved))
    if cached is not None and cached[0] == freshness:
        return cached[1]
    content = resolved.read_bytes()
    after = resolved.stat()
    after_freshness = (
        after.st_dev,
        after.st_ino,
        after.st_ctime_ns,
        after.st_mtime_ns,
        after.st_size,
    )
    if freshness != after_freshness:
        raise ValueError(f"{GOOGLE_TN!r} Britishism profile data changed while being read")
    key = (str(resolved), stat.st_mtime_ns, stat.st_size, hashlib.sha256(content).hexdigest())
    _FILE_KEYS[str(resolved)] = (freshness, key)
    return key


def load_britishisms(*, locale: str = "en_US") -> BritishismTable:
    """Load and validate the external table before a profile-scored token is read."""
    table = _load_britishisms_for(*_file_key(google_tn_britishisms_path()))
    effective = canonical_locale(locale)
    if effective.split("_", 1)[0] != table.locale.split("_", 1)[0]:
        raise ValueError(f"{GOOGLE_TN!r} Britishism data is for {table.locale}, not {effective}")
    return table


@lru_cache(maxsize=LOCALE_CACHE)
def _load_britishisms_for(path_text: str, mtime_ns: int, size: int, sha256: str) -> BritishismTable:
    path = Path(path_text)
    try:
        content = path.read_bytes()
        stat = path.stat()
    except FileNotFoundError:
        raise _missing(path) from None
    if (
        stat.st_mtime_ns != mtime_ns
        or stat.st_size != size
        or hashlib.sha256(content).hexdigest() != sha256
    ):
        raise ValueError(f"{GOOGLE_TN!r} Britishism profile data changed while being loaded")
    data = json.loads(content)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: wrong schema version")
    if data.get("profile") != GOOGLE_TN:
        raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: wrong profile name")
    locale = data.get("locale")
    provenance = data.get("provenance")
    source_shards = provenance.get("source_shards") if isinstance(provenance, dict) else None
    if not isinstance(locale, str) or not locale:
        raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: no locale")
    if not isinstance(source_shards, list) or len(source_shards) != 90:
        raise ValueError(
            f"invalid {GOOGLE_TN!r} Britishism data at {path}: "
            "expected all 90 verified training shards"
        )
    expected_names = {f"output-{index:05d}-of-00100" for index in range(90)}
    if {
        item.get("relative_path") for item in source_shards if isinstance(item, dict)
    } != expected_names or any(
        not isinstance(item, dict)
        or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None
        for item in source_shards
    ):
        raise ValueError(
            f"invalid {GOOGLE_TN!r} Britishism data at {path}: invalid source-shard digests"
        )
    selection = data.get("selection")
    pairs = data.get("pairs")
    rules = data.get("rules")
    if (
        not isinstance(selection, dict)
        or not isinstance(pairs, dict)
        or not isinstance(rules, dict)
    ):
        raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: missing tables")
    try:
        admitted_classes = frozenset(selection["admitted_classes"])
        minimum_support = {
            name: int(value) for name, value in selection["class_minimum_support"].items()
        }
        case_folding = selection["case_folding"]["enabled"]
        threshold = float(selection["rule_rate_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad selection") from exc
    if any(not isinstance(name, str) or not isinstance(row, dict) for name, row in rules.items()):
        raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad rules table")
    enabled: dict[str, dict[str, frozenset[str]]] = {}
    for name, shapes in rules.items():
        if not isinstance(shapes, dict) or not set(shapes) <= set(CASE_SHAPES):
            raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad rule shapes")
        selected_shapes = {}
        for shape, row in shapes.items():
            if not isinstance(row, dict) or not isinstance(row.get("stems", []), list):
                raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad rule stems")
            if row.get("enabled") is True:
                if (
                    not isinstance(row.get("converted"), int)
                    or isinstance(row.get("converted"), bool)
                    or not isinstance(row.get("eligible"), int)
                    or isinstance(row.get("eligible"), bool)
                    or not isinstance(row.get("rate"), float)
                    or not isinstance(row.get("stems"), list)
                ):
                    raise ValueError(
                        f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad rule audit fields"
                    )
                selected_shapes[shape] = frozenset(row["stems"])
        if selected_shapes:
            enabled[name] = selected_shapes
    if (
        not admitted_classes <= set(PAIR_CLASSES)
        or set(minimum_support) != set(admitted_classes)
        or any(value <= 0 for value in minimum_support.values())
        or not isinstance(case_folding, bool)
        or not 0 <= threshold <= 1
        or not set(enabled) <= _RULES.keys()
        or any(
            not isinstance(stem, str) or not stem
            for shapes in enabled.values()
            for stems in shapes.values()
            for stem in stems
        )
    ):
        raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad selection")
    for word, shapes in pairs.items():
        if not isinstance(word, str) or word != word.casefold() or not isinstance(shapes, dict):
            raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad pair row")
        if not shapes or not set(shapes) <= set(CASE_SHAPES):
            raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad pair shapes")
        for row in shapes.values():
            if (
                not isinstance(row, dict)
                or row.get("class") not in admitted_classes
                or not isinstance(row.get("target"), str)
                or not row["target"].isalpha()
                or row["target"] != row["target"].casefold()
                or not isinstance(row.get("converted"), int)
                or isinstance(row.get("converted"), bool)
                or not isinstance(row.get("left"), int)
                or isinstance(row.get("left"), bool)
                or row["converted"] < 0
                or row["left"] < 0
            ):
                raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad pair row")
    shards = tuple(sorted((item["relative_path"], item["sha256"]) for item in source_shards))
    return BritishismTable(
        canonical_locale(locale),
        shards,
        minimum_support,
        admitted_classes,
        case_folding,
        threshold,
        pairs,
        enabled,
    )


def _restore_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source.istitle():
        return target.title()
    return target


def case_shape(word: str) -> str | None:
    """Return the evidence bucket used by the profile for an alphabetic surface."""
    if word.islower():
        return "lower"
    if word.istitle():
        return "title"
    if word.isupper():
        return "upper"
    return None


def respell(word: str, *, locale: str = "en_US") -> str:
    """Return Kestrel's selected spelling, or ``word`` when the profile abstains."""
    return respell_from_table(word, load_britishisms(locale=locale))


def respell_from_table(word: str, table: BritishismTable) -> str:
    """Apply one already-validated table without re-reading its filesystem key."""
    key = word.casefold()
    shape = case_shape(word)
    if shape is None:
        return word
    shapes = table.pairs.get(key, {})
    row = shapes.get(shape)
    if row is None and shape != "lower" and table.case_folding:
        row = shapes.get("lower")
    if (
        row is not None
        and row["class"] in table.admitted_classes
        and row["converted"] >= table.minimum_support[row["class"]]
        and row["converted"] > row["left"]
    ):
        return _restore_case(word, str(row["target"]))
    for name, by_shape in table.rules.items():
        stems = by_shape.get(shape)
        if stems is None and shape != "lower" and table.case_folding:
            stems = by_shape.get("lower")
        if stems is None:
            continue
        rule = _RULES[name]
        candidate = rule.apply(key)
        if candidate != key and rule.stem(key) in stems:
            return _restore_case(word, candidate)
    return word
