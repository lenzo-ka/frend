"""Recognize and read URLs, email addresses and bare domains, which icukit leaves to frend.

ICU has no link recognition, so the span shape is frend's own: a scheme URL
("http://..."), a "www." address, an email address, or a bare domain whose last label
is a top-level domain in IANA's list (vendored as ``data/tlds-alpha-by-domain.txt``).
Every host must pass ICU's IDNA processing (UTS #46 with STD3 rules). Only maximal
spans are emitted: a domain inside a URL or an email address is not a reading of its
own.

A reading's value is the token split into runs of letters, digits and single other
characters. How each run is said -- a letter run as a word or spelled, a digit run as
a cardinal, a year or digit by digit, a separator by its name -- is measured from the
corpus (``data/electronic_priors.json``, built by ``tools/build_electronic_priors.py``);
nothing here decides it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from importlib.resources import files

import icu
from icukit.detectors import Capture

__all__ = [
    "ElectronicDetector",
    "ElectronicValue",
    "decode_letter_notation",
    "letter_key",
    "load_electronic_priors",
    "runs",
    "top_level_domains",
]

_DATA = files("frend").joinpath("data")

# Blending strength for a sparse key toward its parent, as for spoken-prior sub-keys.
PRIOR_STRENGTH = 5

_SCHEME = re.compile(r"(?i)(?<![\w.+-])[a-z][a-z0-9+.-]*://[^\s<>\"]+")
_WWW = re.compile(r"(?i)(?<![\w./@-])www\.[^\s<>\"]+")
_EMAIL = re.compile(r"(?<![\w.%+-])[\w.%+-]+@(?:[\w-]+\.)+[^\W\d_]{2,}(?![\w-])")
_DOMAIN = re.compile(r"(?<![\w@./:-])(?:[\w-]+\.)+([^\W\d_]{2,})(?:/[^\s<>\"]*)?(?![\w-])")
_RUN = re.compile(r"[^\W\d_]+|\d+|.", re.DOTALL)
_TRAILING = ".,;:!?'\""
_CLOSERS = {")": "(", "]": "[", "}": "{"}


@cache
def _tld_lines() -> tuple[str, ...]:
    text = _DATA.joinpath("tlds-alpha-by-domain.txt").read_text(encoding="ascii")
    return tuple(text.splitlines())


def top_level_domains() -> frozenset[str]:
    """IANA's top-level domains, lowercased (internationalized ones in their A-label)."""
    return frozenset(line.strip().lower() for line in _tld_lines() if line and line[0] != "#")


def tld_version() -> str:
    """The version line IANA puts at the head of the vendored list."""
    return _tld_lines()[0].lstrip("# ").strip()


@cache
def _idna() -> icu.IDNA:
    return icu.IDNA(
        icu.IDNA.DEFAULT
        | icu.IDNA.USE_STD3_RULES
        | icu.IDNA.CHECK_BIDI
        | icu.IDNA.CHECK_CONTEXTJ
        | icu.IDNA.CHECK_NONTRANSITIONAL_TO_ASCII
    )


def _valid_host(host: str) -> str | None:
    """The host's ASCII form if ICU's IDNA accepts it and its last label is a TLD."""
    info = icu.IDNAInfo()
    ascii_host = str(_idna().nameToASCII(host, info))
    if info.errors() or "." not in ascii_host:
        return None
    if ascii_host.rsplit(".", 1)[1] not in top_level_domains():
        return None
    return ascii_host


@dataclass(frozen=True)
class ElectronicValue:
    """A URL, email address or bare domain as its written runs.

    ``kind`` is ``url``, ``email`` or ``domain``; ``parts`` pairs each run's kind
    (``letters``, ``digits`` or ``separator``) with its text, in written order, and
    ``host`` is the host's ASCII form as ICU's IDNA gives it.
    """

    kind: str
    host: str
    parts: tuple[tuple[str, str], ...]


def runs(text: str) -> tuple[tuple[str, str], ...]:
    """Split a token into letter runs, digit runs and single other characters."""
    parts = []
    for match in _RUN.finditer(text):
        run = match.group()
        if run[0].isdigit():
            parts.append(("digits", run))
        elif run[0].isalpha():
            parts.append(("letters", run))
        else:
            parts.append(("separator", run))
    return tuple(parts)


def _trim(text: str, start: int, end: int) -> int:
    """Drop sentence punctuation that follows a span, and an unmatched closer."""
    while end > start:
        last = text[end - 1]
        if last in _TRAILING:
            end -= 1
        elif last in _CLOSERS and text[start:end].count(last) > text[start:end].count(
            _CLOSERS[last]
        ):
            end -= 1
        else:
            break
    return end


def _host(kind: str, span: str) -> str:
    if kind == "email":
        return span.rsplit("@", 1)[1]
    rest = span.split("://", 1)[1] if kind == "url" and "://" in span else span
    host = re.split(r"[/?#]", rest, maxsplit=1)[0]
    host = host.rsplit("@", 1)[-1]
    return host.split(":", 1)[0]


class ElectronicDetector:
    """Detect URLs, email addresses and bare domains as frend readings."""

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = locale

    def detect(self, text: str) -> list[dict]:
        found: list[tuple[int, int, str]] = []
        for kind, pattern in (("url", _SCHEME), ("url", _WWW), ("email", _EMAIL)):
            for match in pattern.finditer(text):
                found.append((match.start(), _trim(text, match.start(), match.end()), kind))
        for match in _DOMAIN.finditer(text):
            found.append((match.start(), _trim(text, match.start(), match.end()), "domain"))
        detections = []
        for start, end, kind in found:
            if any(
                other_start <= start
                and end <= other_end
                and (other_start, other_end) != (start, end)
                for other_start, other_end, _ in found
            ):
                continue
            span = text[start:end]
            host = _valid_host(_host(kind, span))
            if host is None or any(d["start"] == start and d["end"] == end for d in detections):
                continue
            parts = runs(span)
            captures = []
            offset = start
            for part_kind, part in parts:
                captures.append(Capture(part_kind, offset, offset + len(part), part, part))
                offset += len(part)
            detections.append(
                {
                    "text": span,
                    "start": start,
                    "end": end,
                    "type": f"electronic:{kind}",
                    "value": ElectronicValue(kind, host, parts),
                    "captures": tuple(captures),
                }
            )
        return sorted(detections, key=lambda d: (d["start"], d["end"]))


def decode_letter_notation(spoken: str) -> str:
    """Join the corpus's per-letter tokens into words.

    The corpus writes an ELECTRONIC spoken form letter by letter ("b_letter o_letter"),
    with a bare ``_letter`` marking a word break; other tokens ("dot") are words.
    """
    words: list[str] = []
    current = ""
    for token in spoken.split(" "):
        if not token:
            continue
        if token == "_letter":
            if current:
                words.append(current)
                current = ""
        elif token.endswith("_letter"):
            current += token[: -len("_letter")]
        else:
            if current:
                words.append(current)
                current = ""
            words.append(token)
    if current:
        words.append(current)
    return " ".join(words)


def letter_key(run: str) -> str:
    """The shape a letter run's word-or-spelled measurement is keyed by.

    Case pattern, length (7 and more pooled) and whether it has a vowel letter; the
    keys are features, the probabilities under them are measured.
    """
    if run.islower():
        case = "lower"
    elif run.isupper():
        case = "upper"
    elif run[0].isupper() and run[1:].islower():
        case = "title"
    else:
        case = "mixed"
    vowel = "v" if any(ch in "aeiouyAEIOUY" for ch in run) else "nv"
    return f"{case}:{min(len(run), 7)}:{vowel}"


def digit_key(run: str) -> str:
    return f"{min(len(run), 5)}:{'z' if len(run) > 1 and run[0] == '0' else 'n'}"


@cache
def load_electronic_priors() -> dict:
    return json.loads(_DATA.joinpath("electronic_priors.json").read_text(encoding="utf-8"))


def _blend(counts: dict[str, int], parent: dict[str, Decimal]) -> dict[str, Decimal]:
    total = sum(counts.values())
    keys = set(counts) | set(parent)
    return {
        key: (Decimal(counts.get(key, 0)) + PRIOR_STRENGTH * parent.get(key, Decimal(0)))
        / (total + PRIOR_STRENGTH)
        for key in keys
    }


def _shares(counts: dict[str, int]) -> dict[str, Decimal]:
    total = sum(counts.values())
    return {key: Decimal(value) / total for key, value in counts.items()} if total else {}


def letter_probabilities(run: str, *, tld: bool) -> dict[str, Decimal]:
    """P(word) and P(spelled) for a letter run: its shape blended toward all runs,
    and a top-level domain's own evidence blended toward its shape."""
    table = load_electronic_priors()["letters"]
    overall = _shares(table["*"])
    shaped = _blend(table.get(letter_key(run), {}), overall)
    if tld and f"tld:{run.lower()}" in table:
        return _blend(table[f"tld:{run.lower()}"], shaped)
    return shaped


def digit_probabilities(run: str) -> dict[str, Decimal]:
    table = load_electronic_priors()["digits"]
    return _blend(table.get(digit_key(run), {}), _shares(table["*"]))


def separator_names(character: str) -> dict[str, Decimal]:
    """The corpus's names for a separator character, with their shares."""
    return _shares(load_electronic_priors()["separators"].get(character, {}))


def tld_positions(parts: tuple[tuple[str, str], ...]) -> frozenset[int]:
    """Indexes of letter runs that end a host: a top-level domain after a dot."""
    tlds = top_level_domains()
    found = set()
    for index, (kind, text) in enumerate(parts):
        if kind != "letters" or text.lower() not in tlds or index == 0:
            continue
        if parts[index - 1] != ("separator", "."):
            continue
        following = parts[index + 1][1] if index + 1 < len(parts) else None
        if following is None or following in "/:?#":
            found.add(index)
    return frozenset(found)


def digit_forms(run: str, locale: str = "en_US") -> dict[str, tuple[tuple[str, str], ...]]:
    """A digit run's candidate readings, each as (text, provenance) pairs.

    ``cardinal`` and ``year`` are ICU's rule sets for the run's value; ``digits`` reads
    each digit by ICU's cardinal, with zero also as the corpus's "o" (lexical: ICU has
    no digit rule that calls zero "o").
    """
    from frend.verbalize import LEXICAL_SOURCE, _number_leaf

    value = Decimal(run)
    words = [_number_leaf(Decimal(digit), "cardinal", locale)[0].text for digit in run]
    forms = {
        "cardinal": tuple(
            (item.text, item.provenance) for item in _number_leaf(value, "cardinal", locale)
        ),
        "year": tuple((item.text, item.provenance) for item in _number_leaf(value, "year", locale)),
        "digits": ((" ".join(words), "icu-rbnf:%spellout-cardinal"),),
    }
    if "0" in run:
        spoken = " ".join("o" if d == "0" else w for d, w in zip(run, words, strict=True))
        forms["digits_o"] = ((spoken, LEXICAL_SOURCE),)
    return forms
