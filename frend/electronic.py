"""Recognize and read URLs, email addresses and bare domains, which icukit leaves to frend.

ICU has no link recognition, so the span shape is frend's own: a scheme URL
("http://..."), a "www." address, an email address, or a bare domain whose last label
is a top-level domain in IANA's list (vendored as ``data/root/tlds-alpha-by-domain.txt``).
Every host must pass ICU's IDNA processing (UTS #46 with STD3 rules). Only maximal
spans are emitted: a domain inside a URL or an email address is not a reading of its
own.

A reading's value is the token split into runs of letters, digits and single other
characters. How each run is said -- a letter run as a word or spelled, a digit run as
a cardinal, a year or digit by digit, a separator by its name -- is measured from the
corpus (``data/en/electronic_priors.json``, built by ``tools/build_electronic_priors.py``);
nothing here decides it.
"""

from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass
from decimal import Decimal
from functools import cache, lru_cache
from importlib.resources import files

import icu
from icukit.detectors import Capture

from frend.input_limits import DEFAULT_MAX_INPUT_CHARS, validate_input
from frend.letters import letter_vowels
from frend.locale_data import LOCALE_CACHE, canonical_locale, measured_table

__all__ = [
    "DEFAULT_MAX_EMAIL_CHARS",
    "DEFAULT_MAX_URL_CHARS",
    "ElectronicDetector",
    "ElectronicValue",
    "decode_letter_notation",
    "letter_key",
    "load_electronic_priors",
    "load_electronic_span_priors",
    "runs",
    "top_level_domains",
]

_ROOT_DATA = files("frend").joinpath("data", "root")
_MISSING_ROW = object()

# Blending strength for a sparse key toward its parent, as for spoken-prior sub-keys.
PRIOR_STRENGTH = 5

# These are recognition work bounds, not claims about protocol validity. Oversized
# contiguous candidates are not emitted; returning a prefix would invent a different
# address while pretending to preserve source offsets.
DEFAULT_MAX_URL_CHARS = 8 * 1024
DEFAULT_MAX_EMAIL_CHARS = 254

_SCHEME = re.compile(r"(?i)(?<![\w.+-])[a-z][a-z0-9+.-]*://[^\s<>\"]+")
_WWW = re.compile(r"(?i)(?<![\w./@-])www\.[^\s<>\"]+")
_EMAIL = re.compile(r"(?<![\w.%+-])[\w.%+-]+@(?:[\w-]+\.)+[^\W\d_]{2,}(?![\w-])")
_DOMAIN = re.compile(r"(?<![\w@./:-])(?:[\w-]+\.)+([^\W\d_]{2,})(?:/[^\s<>\"]*)?(?![\w-])")
_HASHTAG = re.compile(r"#[A-Za-z][A-Za-z0-9_]*\Z")
_PATH_DOCUMENT = re.compile(r"[A-Za-z0-9._~/-]*/[A-Za-z0-9_-]+\.(?P<extension>[A-Za-z0-9]+)\Z")
_KNOWN_FILE_EXTENSIONS = frozenset(
    suffix.removeprefix(".").lower() for suffix in mimetypes.types_map
)
_RUN = re.compile(r"[^\W\d_]+|\d+|.", re.DOTALL)
_TRAILING = ".,;:!?'\""
_CLOSERS = {")": "(", "]": "[", "}": "{"}


@cache
def _tld_lines() -> tuple[str, ...]:
    text = _ROOT_DATA.joinpath("tlds-alpha-by-domain.txt").read_text(encoding="ascii")
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


def _camel_tld_host(text: str) -> str | None:
    """A valid host prefix whose TLD is immediately followed by an uppercase suffix."""
    for boundary in re.finditer(r"(?<=[a-z])(?=[A-Z])", text):
        end = boundary.start()
        start = max(text.rfind("/", 0, end), text.rfind(":", 0, end)) + 1
        host = _valid_host(text[start:end])
        if host is not None:
            return host
    return None


def _path_document(text: str) -> bool:
    """Whether ``text`` is a path ending in a standard-library MIME extension."""
    match = _PATH_DOCUMENT.fullmatch(text)
    if match is None:
        return False
    return match.group("extension").lower() in _KNOWN_FILE_EXTENSIONS


@dataclass(frozen=True)
class ElectronicValue:
    """An electronic span as its written runs.

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


def _positive_cap(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _span_features(text: str) -> tuple[str, ...]:
    """Corpus-counted whole-token shape features, containing no lexical identities."""
    if "." not in text and not text.startswith("#"):
        return ()
    features = []
    scheme = _SCHEME.search(text)
    if scheme is not None and scheme.start() == 1 and text.startswith("/"):
        features.append("leading-slash-scheme")
    if scheme is not None and scheme.start() == 0 and text.endswith(";"):
        features.append("trailing-semicolon-scheme")
    if scheme is not None and scheme.start() == 0 and text.endswith(")"):
        features.append("trailing-parenthesis-scheme")
    if _HASHTAG.fullmatch(text):
        features.append("hashtag")
    if text.startswith("//") and _valid_host(_host("domain", text[2:])) is not None:
        features.append("scheme-relative-url")
    if _camel_tld_host(text) is not None:
        features.append("tld-uppercase-suffix")
    if _path_document(text):
        features.append("path-document")
    return tuple(features)


def _bounded_segments(text: str, max_chars: int):
    """Yield electronic-token segments without copying any segment over ``max_chars``."""
    start = 0
    for end, char in enumerate(text):
        if char.isspace() or char in '<>"':
            if start < end and end - start <= max_chars:
                yield start, text[start:end]
            start = end + 1
    if start < len(text) and len(text) - start <= max_chars:
        yield start, text[start:]


def _supported_span_features(text: str, locale: str) -> tuple[str, ...]:
    document = load_electronic_span_priors(locale=locale)
    if document is None:
        return ()
    supported = []
    for feature in _span_features(text):
        if feature not in document["features"]:
            continue
        row = document["features"][feature]
        classes = row.get("classes") if isinstance(row, dict) else None
        _validate_count_row(classes, f"features.{feature}.classes")
        electronic = classes.get("ELECTRONIC", 0)
        if electronic >= 3 and electronic > sum(classes.values()) - electronic:
            supported.append(feature)
    return tuple(supported)


def _special_kind_host(text: str, locale: str, *, whole_input: bool):
    features = set(_supported_span_features(text, locale))
    boundary = {"leading-slash-scheme", "trailing-semicolon-scheme"}
    boundary.add("trailing-parenthesis-scheme")
    if not whole_input:
        features -= boundary
    if not features:
        return None
    if "hashtag" in features:
        return "hashtag", ""
    if "path-document" in features:
        return "document", ""
    camel_host = _camel_tld_host(text)
    if "tld-uppercase-suffix" in features and camel_host is not None:
        return "domain", camel_host
    if "scheme-relative-url" in features:
        return "url", _valid_host(_host("domain", text[2:]))
    host_text = text[:-1] if text.endswith((";", ")")) else text
    host = _valid_host(_host("url", host_text))
    return ("url", host) if host is not None else None


class ElectronicDetector:
    """Detect corpus-supported electronic spans as frend readings."""

    def __init__(
        self,
        locale: str = "en_US",
        *,
        max_url_chars: int = DEFAULT_MAX_URL_CHARS,
        max_email_chars: int = DEFAULT_MAX_EMAIL_CHARS,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    ) -> None:
        self.locale = locale
        self.max_url_chars = _positive_cap("max_url_chars", max_url_chars)
        self.max_email_chars = _positive_cap("max_email_chars", max_email_chars)
        self.max_input_chars = _positive_cap("max_input_chars", max_input_chars)

    def detect(self, text: str) -> list[dict]:
        validate_input(text, max_input_chars=self.max_input_chars)
        return self._detect_validated(text)

    def _detect_validated(self, text: str) -> list[dict]:
        """Detect in text already checked by an enclosing document API."""
        found: list[tuple[int, int, str, str | None]] = []
        for base, segment in _bounded_segments(text, max(self.max_url_chars, self.max_email_chars)):
            if len(segment) <= self.max_url_chars:
                for kind, pattern in (("url", _SCHEME), ("url", _WWW), ("domain", _DOMAIN)):
                    for match in pattern.finditer(segment):
                        end = _trim(segment, match.start(), match.end())
                        found.append((base + match.start(), base + end, kind, None))
            if len(segment) <= self.max_email_chars:
                for match in _EMAIL.finditer(segment):
                    end = _trim(segment, match.start(), match.end())
                    found.append((base + match.start(), base + end, "email", None))
            if len(segment) <= self.max_url_chars:
                trimmed_end = _trim(segment, 0, len(segment))
                candidates = [(segment[:trimmed_end], trimmed_end)]
                if trimmed_end != len(segment):
                    candidates.append((segment, len(segment)))
                for candidate, end in candidates:
                    special = _special_kind_host(
                        candidate,
                        self.locale,
                        whole_input=base == 0 and len(segment) == len(text),
                    )
                    if special is not None:
                        kind, host = special
                        found.append((base, base + end, kind, host))
                        break
        detections = []
        for start, end, kind, explicit_host in found:
            if any(
                other_start <= start
                and end <= other_end
                and (other_start, other_end) != (start, end)
                for other_start, other_end, _, _ in found
            ):
                continue
            span = text[start:end]
            host = explicit_host
            if host is None:
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


class _ValidatedElectronicDetector(ElectronicDetector):
    """Electronic detector for an enclosing path that already validated its input."""

    def detect(self, text: str) -> list[dict]:
        return self._detect_validated(text)


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


def letter_key(run: str, locale: str = "en_US") -> str:
    """The shape a letter run's word-or-spelled measurement is keyed by.

    Case pattern, length (7 and more pooled) and whether it has a vowel letter, the
    locale's vowels (``lexical.json``'s ``letter.vowels``); a locale with none keys by
    case and length alone. The keys are features, the probabilities under them are
    measured.
    """

    if run.islower():
        case = "lower"
    elif run.isupper():
        case = "upper"
    elif run[0].isupper() and run[1:].islower():
        case = "title"
    else:
        case = "mixed"
    shape = f"{case}:{min(len(run), 7)}"
    vowels = letter_vowels(locale)
    if vowels is None:
        return shape
    return f"{shape}:{'v' if any(ch in vowels for ch in run) else 'nv'}"


def digit_key(run: str) -> str:
    return f"{min(len(run), 5)}:{'z' if len(run) > 1 and run[0] == '0' else 'n'}"


def load_electronic_priors(*, locale: str = "en_US") -> dict | None:
    """The locale's measured electronic table (``data/<locale>/electronic_priors.json``),
    or ``None`` when the locale has none; cached on the canonical locale, so every
    spelling of a locale gets the same table."""
    return _electronic_priors(canonical_locale(locale))


def load_electronic_span_priors(*, locale: str = "en_US") -> dict | None:
    """Aggregate training support for whole-token electronic span features."""
    return _electronic_span_priors(canonical_locale(locale))


def _validate_count_row(row: object, path: str) -> None:
    if (
        not isinstance(row, dict)
        or not row
        or any(type(count) is not int or count < 0 for count in row.values())
        or sum(row.values()) == 0
    ):
        raise ValueError(f"invalid electronic prior count row at {path}")


def _validate_electronic_priors(document: object) -> dict:
    if not isinstance(document, dict):
        raise ValueError("invalid electronic prior table")
    for section in ("letters", "digits", "separators"):
        rows = document.get(section)
        if not isinstance(rows, dict):
            raise ValueError(f"invalid electronic prior table: {section} must be an object")
        for key, row in rows.items():
            _validate_count_row(row, f"{section}.{key}")
    return document


def _validate_electronic_span_priors(document: object) -> dict:
    if not isinstance(document, dict) or not isinstance(document.get("features"), dict):
        raise ValueError("invalid electronic span prior table: features must be an object")
    for feature, row in document["features"].items():
        classes = row.get("classes") if isinstance(row, dict) else None
        _validate_count_row(classes, f"features.{feature}.classes")
    return document


@lru_cache(maxsize=LOCALE_CACHE)
def _electronic_priors(locale: str) -> dict | None:
    document = measured_table("electronic_priors", locale)
    return None if document is None else _validate_electronic_priors(document)


@lru_cache(maxsize=LOCALE_CACHE)
def _electronic_span_priors(locale: str) -> dict | None:
    document = measured_table("electronic_span_priors", locale)
    return None if document is None else _validate_electronic_span_priors(document)


def _blend(counts: object, parent: dict[str, Decimal], *, path: str) -> dict[str, Decimal]:
    if counts is _MISSING_ROW:
        counts = {}
    else:
        _validate_count_row(counts, path)
    assert isinstance(counts, dict)
    total = sum(counts.values())
    keys = set(counts) | set(parent)
    return {
        key: (Decimal(counts.get(key, 0)) + PRIOR_STRENGTH * parent.get(key, Decimal(0)))
        / (total + PRIOR_STRENGTH)
        for key in keys
    }


def _shares(counts: object, *, path: str) -> dict[str, Decimal]:
    if counts is _MISSING_ROW:
        return {}
    _validate_count_row(counts, path)
    assert isinstance(counts, dict)
    total = sum(counts.values())
    return {key: Decimal(value) / total for key, value in counts.items()}


def letter_probabilities(run: str, *, tld: bool, locale: str = "en_US") -> dict[str, Decimal]:
    """P(word) and P(spelled) for a letter run: its shape blended toward all runs,
    and a top-level domain's own evidence blended toward its shape."""
    document = load_electronic_priors(locale=locale)
    if document is None:
        return {}
    table = document["letters"]
    overall = _shares(table["*"], path="letters.*")
    shape = letter_key(run, locale)
    shaped = _blend(table.get(shape, _MISSING_ROW), overall, path=f"letters.{shape}")
    if tld and f"tld:{run.lower()}" in table:
        key = f"tld:{run.lower()}"
        return _blend(table[key], shaped, path=f"letters.{key}")
    return shaped


def digit_probabilities(run: str, *, locale: str = "en_US") -> dict[str, Decimal]:
    document = load_electronic_priors(locale=locale)
    if document is None:
        return {}
    table = document["digits"]
    key = digit_key(run)
    return _blend(
        table.get(key, _MISSING_ROW),
        _shares(table["*"], path="digits.*"),
        path=f"digits.{key}",
    )


def separator_names(character: str, *, locale: str = "en_US") -> dict[str, Decimal]:
    """The corpus's names for a separator character, with their shares."""
    document = load_electronic_priors(locale=locale)
    if document is None:
        return {}
    return _shares(
        document["separators"].get(character, _MISSING_ROW),
        path=f"separators.{character}",
    )


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
    no digit rule that calls zero "o"; ``lexical.json`` ``zero.digit``, so a locale with
    none has no ``digits_o``).
    """
    from frend.verbalize import _lexical, _number_leaf, _rbnf_provenance, lexical_source

    value = Decimal(run)
    digit_leaves = [_number_leaf(Decimal(digit), "cardinal", locale)[0] for digit in run]
    words = [item.text for item in digit_leaves]
    forms = {
        "cardinal": tuple(
            (item.text, item.provenance) for item in _number_leaf(value, "cardinal", locale)
        ),
        "year": tuple((item.text, item.provenance) for item in _number_leaf(value, "year", locale)),
        "digits": ((" ".join(words), _rbnf_provenance("%spellout-cardinal", locale)),),
    }
    zero = _lexical("zero.digit", locale)
    if "0" in run and zero is not None:
        spoken = " ".join(zero if d == "0" else w for d, w in zip(run, words, strict=True))
        forms["digits_o"] = ((spoken, lexical_source(locale)),)
    return forms
