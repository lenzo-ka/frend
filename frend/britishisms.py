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
    "EDIT_RULES",
    "apply_edit_rule",
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
    EditRule("our-to-or", re.compile(r"our(?=(?:s|ed|ing|er|ers|ful|less|able|ism|ist)?$)"), "or"),
    EditRule("re-to-er", re.compile(r"re(?=(?:s|d)?$)"), "er"),
    EditRule("ogue-to-og", re.compile(r"ogue(?=(?:s|d)?$)"), "og"),
    EditRule("mme-to-m", re.compile(r"mme(?=(?:s|d)?$)"), "m"),
    EditRule("doubled-l", re.compile(r"ll(?=(?:ed|ing|er|ers|or|ors)?$)"), "l"),
    EditRule("yse-to-yze", re.compile(r"yse(?=(?:s|d|r|rs|ing)?$)"), "yze"),
    EditRule("ae-to-e", re.compile(r"ae"), "e"),
    EditRule("oe-to-e", re.compile(r"oe"), "e"),
)
_RULES = {rule.name: rule for rule in EDIT_RULES}


def apply_edit_rule(name: str, word: str) -> str:
    """Apply one named edit rule to a case-normalized word."""
    return _RULES[name].apply(word)


@dataclass(frozen=True)
class BritishismTable:
    locale: str
    minimum_support: int
    rule_rate_threshold: float
    pairs: dict[str, dict[str, object]]
    rules: dict[str, frozenset[str]]


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
        minimum_support = int(selection["minimum_support"])
        threshold = float(selection["rule_rate_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad selection") from exc
    if any(not isinstance(name, str) or not isinstance(row, dict) for name, row in rules.items()):
        raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad rules table")
    if any(not isinstance(row.get("stems", []), list) for row in rules.values()):
        raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad rule stems")
    enabled = {
        name: frozenset(row.get("stems", ()))
        for name, row in rules.items()
        if row.get("enabled") is True
    }
    if (
        minimum_support <= 0
        or not 0 <= threshold <= 1
        or not set(enabled) <= _RULES.keys()
        or any(
            not isinstance(stem, str) or not stem for stems in enabled.values() for stem in stems
        )
    ):
        raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad selection")
    for word, row in pairs.items():
        if (
            not isinstance(word, str)
            or word != word.casefold()
            or not isinstance(row, dict)
            or not isinstance(row.get("target"), str)
            or not isinstance(row.get("converted"), int)
            or not isinstance(row.get("left"), int)
            or row["converted"] < 0
            or row["left"] < 0
        ):
            raise ValueError(f"invalid {GOOGLE_TN!r} Britishism data at {path}: bad pair row")
    return BritishismTable(canonical_locale(locale), minimum_support, threshold, pairs, enabled)


def _restore_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source.istitle():
        return target.title()
    return target


def respell(word: str, *, locale: str = "en_US") -> str:
    """Return Kestrel's selected spelling, or ``word`` when the profile abstains."""
    return respell_from_table(word, load_britishisms(locale=locale))


def respell_from_table(word: str, table: BritishismTable) -> str:
    """Apply one already-validated table without re-reading its filesystem key."""
    key = word.casefold()
    row = table.pairs.get(key)
    if (
        row is not None
        and row["converted"] >= table.minimum_support
        and row["converted"] > row["left"]
    ):
        return _restore_case(word, str(row["target"]))
    for name, stems in table.rules.items():
        rule = _RULES[name]
        candidate = rule.apply(key)
        if candidate != key and rule.stem(key) in stems:
            return _restore_case(word, candidate)
    return word
