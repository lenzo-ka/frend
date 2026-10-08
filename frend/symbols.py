"""Recognize a standalone symbol or letter of another script, which frend speaks by name.

The names come from icukit: CLDR's per-locale spoken names for symbols
(``icu_abbreviations(locale, kinds=["symbol"])``: "&" ampersand, and; "#" hash sign,
number) and ICU's formal character names for everything else ("GREEK SMALL LETTER
ALPHA" gives "alpha"). A character reads only when it stands alone -- between spaces,
punctuation, or the ends of the text -- so "R&D" and "AT&T" are left as written. Every
reading also offers silence: the corpus says most punctuation, a lone dash and a
character of a script it does not read ("風") as nothing, and which name or silence
comes first is measured, per character for a symbol and per script for a letter.

A letter is "of another script" when its ICU script is none of the locale's: the script
ICU's likely subtags give the locale (``en_US`` -> ``en_Latn_US``), the scripts of its
exemplar characters (icukit's ``get_locale_scripts``), and Common and Inherited. Single
out-of-script letters set apart by single spaces ("Т О Д Н") are one span,
:class:`ScriptRunValue`, read as a unit: by ICU's
transliteration into the locale's script (a transform ICU lists for the two scripts,
else ``Any-<script>``), read as the locale reads its own letters and words; by the
letters' names; as written; or as nothing. One reading of the run replaces a free choice
per letter, which multiplied a sentence of spaced letters into 2^n covers. A lone
out-of-script letter reads as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

import icu
from icukit import break_grapheme_spans
from icukit.detectors import Capture

__all__ = [
    "ScriptRunValue",
    "SymbolDetector",
    "SymbolRunValue",
    "SymbolValue",
    "VariationValue",
    "DEFAULT_SYMBOL_RUN_THRESHOLD",
    "locale_scripts",
    "run_readings",
    "silent_property_class",
    "symbol_names",
    "transform_id",
]

# ICU's Common and Inherited scripts belong to every locale.
_NEUTRAL = frozenset(
    icu.Script(code).getShortName() for code in (icu.UScriptCode.COMMON, icu.UScriptCode.INHERITED)
)
_MARKS = icu.UnicodeSet("[:M:]")
_MARKS.freeze()
_SYMBOLS = icu.UnicodeSet("[[:P:][:S:]]")
_SYMBOLS.freeze()
_VARIATION_SELECTORS = icu.UnicodeSet("[:Variation_Selector:]")
_VARIATION_SELECTORS.freeze()
_SENTENCE_TERMINAL = icu.UnicodeSet("[:Sentence_Terminal:]")
_SENTENCE_TERMINAL.freeze()
_MIXED_RUN_SYMBOL = icu.UnicodeSet("[[:So:][:Sm:][:Sk:][:Extended_Pictographic:]]")
_MIXED_RUN_SYMBOL.freeze()
_EXTENDED_PICTOGRAPHIC = icu.UnicodeSet("[:Extended_Pictographic:]")
_EXTENDED_PICTOGRAPHIC.freeze()
_EMOJI_PRESENTATION = icu.UnicodeSet("[:Emoji_Presentation:]")
_EMOJI_PRESENTATION.freeze()
_NFC = icu.Normalizer2.getNFCInstance()
TRANSLITERATION_SOURCE = "icu-transliteration"
LETTER_NAME_SOURCE = "icu-name:letter"
PROPERTY_NAME_SOURCE = "icu-name:property"
SYMBOL_NAME_SOURCE = "icu-name:symbol"
DEFAULT_SYMBOL_RUN_THRESHOLD = 3

# Training shards 00--89: these ICU (General_Category, Script) classes each have at
# least 100 occurrences and at least 99% of their occurrences are in wholly silent
# tokens. Lower-support classes and mixed classes are report-only.
_SILENT_PROPERTY_CLASSES = frozenset(
    {
        (icu.UCharCategory.MODIFIER_LETTER, "Latn"),
        (icu.UCharCategory.MODIFIER_LETTER, "Hani"),
    }
)


@dataclass(frozen=True)
class SymbolValue:
    """One standalone character: its script's short name and the names it can be read by."""

    char: str
    script: str
    names: tuple[tuple[str, str], ...]
    silent_first: bool = False


@dataclass(frozen=True)
class ScriptRunValue:
    """A run of letters outside the locale's scripts, read as one unit.

    ``script`` is the run's ICU short script name ("Cyrl"), ``transform`` the ICU
    transform id that writes it in the locale's script (``None`` when ICU has none), and
    ``names`` the (reading, source) pairs it can be read by, silence aside.
    """

    text: str
    script: str
    transform: str | None
    names: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SymbolRunValue:
    """More than the configured number of symbol graphemes, read as one unit."""

    text: str
    symbols: tuple[str, ...]
    names: tuple[tuple[str, str], ...]
    repeated: bool
    emoji: bool


@dataclass(frozen=True)
class VariationValue:
    """A grapheme whose variation selectors are attached to, and removed with, its base."""

    text: str
    base: str


def _script(char: str) -> str:
    return icu.Script.getScript(ord(char)).getShortName()


def silent_property_class(char: str) -> tuple[int, str] | None:
    """The measured ICU property class that may make a whole token silent-first."""
    key = (icu.Char.charType(char), _script(char))
    return key if key in _SILENT_PROPERTY_CLASSES else None


def _short(name: str) -> str | None:
    """ICU's short code for a script named either way ("Latin" or "Latn")."""
    try:
        codes = icu.Script.getCode(name)
    except icu.ICUError:
        return None
    for code in codes:
        script = icu.Script(code)
        if name in (script.getShortName(), script.getName()):
            return script.getShortName()
    return None


def _other_writings(locale: str) -> tuple[str, ...]:
    """Where the locale names no script, the scripts of ICU's locales for its language
    that do ("sr" -> sr_Cyrl, sr_Latn): the language is written in each."""
    from icukit.locale import list_locales

    parsed = icu.Locale(locale)
    if parsed.getScript():
        return ()
    language = parsed.getLanguage()
    return tuple(
        sorted(
            {
                icu.Locale(name).getScript()
                for name in list_locales()
                if icu.Locale(name).getLanguage() == language and icu.Locale(name).getScript()
            }
        )
    )


@cache
def locale_scripts(locale: str) -> tuple[str, ...]:
    """The locale's scripts as ICU short names, its likely script first: likely subtags'
    script, then its exemplar characters' (icukit), then, where the locale names no
    script, the scripts its language's other ICU locales name (``sr`` -> Latn too), then
    Common and Inherited."""
    from icukit.locale import add_likely_subtags, get_locale_scripts

    scripts: list[str] = []
    likely = icu.Locale(add_likely_subtags(locale)).getScript()
    for name in (likely, *get_locale_scripts(locale), *_other_writings(locale)):
        code = _short(name) if name else None
        if code and code not in scripts:
            scripts.append(code)
    return (*scripts, *sorted(_NEUTRAL - set(scripts)))


def _lettered(char: str) -> bool:
    """A letter of a script of its own (not a Common modifier such as "ʹ")."""
    return icu.Char.isalpha(char) and _script(char) not in _NEUTRAL


def _foreign(char: str, locale: str) -> bool:
    """A letter of a script the locale does not write."""
    return icu.Char.isalpha(char) and _script(char) not in locale_scripts(locale)


_LATIN = icu.Script(icu.UScriptCode.LATIN).getShortName()


def _lone_foreign(char: str, locale: str) -> bool:
    """A lone letter read by name: out of the locale's scripts and, as before runs were
    gated by locale, not Latin (a lone Latin letter reads as written in every locale)."""
    return _foreign(char, locale) and _script(char) != _LATIN


@cache
def _available() -> frozenset[str]:
    from icukit.transliterator import list_transliterators

    return frozenset(list_transliterators())


@cache
def _transforms() -> dict[tuple[str, str], tuple[str, ...]]:
    """ICU's available transform ids without a variant, by (source, target) short script."""
    table: dict[tuple[str, str], list[str]] = {}
    for transform in sorted(_available()):
        if "/" in transform or transform.count("-") != 1:
            continue
        source, target = transform.split("-")
        target_code = _short(target)
        source_code = "Any" if source == "Any" else _short(source)
        if target_code and source_code:
            table.setdefault((source_code, target_code), []).append(transform)
    return {key: tuple(sorted(ids)) for key, ids in table.items()}


@cache
def transform_id(script: str, locale: str) -> str | None:
    """The ICU transform writing ``script`` in the locale's script: one ICU lists from that
    script to the locale's (likely script first), else ICU's ``Any-<script>``."""
    targets = [code for code in locale_scripts(locale) if code not in _NEUTRAL]
    for source in (script, "Any"):
        for target in targets:
            ids = _transforms().get((source, target))
            if ids:
                # ICU's Latin keeps its diacritics ("ž", "ē"); Latin-ASCII folds them to
                # the letters a Latin locale spells with.
                if target == _LATIN and "Latin-ASCII" in _available():
                    return f"{ids[0]}; Latin-ASCII"
                return ids[0]
    return None


@cache
def _transliterator(transform: str):
    from icukit.transliterator import Transliterator

    return Transliterator(transform)


@cache
def _cldr_names(locale: str) -> dict[str, tuple[str, ...]]:
    """CLDR's names per symbol, from icukit's symbol rows."""
    from icukit import icu_abbreviations

    rows = icu_abbreviations(locale, kinds=["symbol"], locales=())
    names: dict[str, list[str]] = {}
    for row in rows:
        for expansion in row.expansions:
            if expansion not in names.setdefault(row.surface, []):
                names[row.surface].append(expansion)
    return {surface: tuple(items) for surface, items in names.items()}


def _without_variation_selectors(text: str) -> str:
    return "".join(char for char in text if not _VARIATION_SELECTORS.contains(char))


def _symbol_grapheme(text: str) -> bool:
    """Whether a grapheme is made from a punctuation/symbol base and joiners/marks."""
    base = _without_variation_selectors(text).replace("\u200d", "")
    return bool(base) and all(_SYMBOLS.contains(char) or _MARKS.contains(char) for char in base)


def _formal_symbol_name(text: str) -> str | None:
    names = [
        icu.Char.charName(char) for char in _without_variation_selectors(text) if char != "\u200d"
    ]
    return " ".join(name.lower() for name in names) if names and all(names) else None


def _letter_name(char: str) -> str | None:
    """The letter's own name from ICU's formal name ("GREEK SMALL LETTER ALPHA" -> "alpha")."""
    from icukit import get_char_name

    name = get_char_name(char) or ""
    if " LETTER " not in name:
        return None
    return name.split(" LETTER ", 1)[1].lower()


def symbol_names(char: str, locale: str = "en_US") -> tuple[tuple[str, str], ...]:
    """(name, source) pairs a character can be read by, CLDR's first."""
    base = _without_variation_selectors(char)
    cldr = _cldr_names(locale).get(char, ()) or _cldr_names(locale).get(base, ())
    if cldr:
        return tuple((name, f"cldr-symbol:{name}") for name in cldr)
    letter = _letter_name(base) if len(base) == 1 and _lone_foreign(base, locale) else None
    if letter:
        return ((letter, LETTER_NAME_SOURCE),)
    formal = _formal_symbol_name(char) if _symbol_grapheme(char) else None
    return ((formal, SYMBOL_NAME_SOURCE),) if formal else ()


def _property_name(text: str) -> tuple[tuple[str, str], ...]:
    from icukit import get_char_name

    names = [get_char_name(char) for char in text]
    return ((" ".join(names).lower(), PROPERTY_NAME_SOURCE),) if all(names) else ()


def _speakable(char: str, locale: str) -> bool:
    if char.isspace() or char.isdigit():
        return False
    if char in _cldr_names(locale) or _without_variation_selectors(char) in _cldr_names(locale):
        return True
    base = _without_variation_selectors(char)
    return (len(base) == 1 and _lone_foreign(base, locale)) or _symbol_grapheme(char)


def _standalone(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not (before.isalnum() or after.isalnum())


def _silent_property_tokens(text: str) -> dict[int, tuple[int, tuple[int, str]]]:
    """Standalone alphanumeric tokens made wholly from one measured property class.

    The corpus evidence is about complete silent tokens, not qualifying code points
    embedded in otherwise ordinary words. Punctuation delimits tokens here just as it
    does for the existing standalone-symbol rule.
    """
    tokens: dict[int, tuple[int, tuple[int, str]]] = {}
    index = 0
    while index < len(text):
        if not text[index].isalnum():
            index += 1
            continue
        start = index
        while index < len(text) and text[index].isalnum():
            index += 1
        classes = {silent_property_class(char) for char in text[start:index]}
        if len(classes) == 1 and None not in classes:
            property_class = classes.pop()
            assert property_class is not None
            tokens[start] = (index, property_class)
    return tokens


def _groups(text: str, locale: str) -> list[tuple[int, int]]:
    """Maximal stretches of out-of-script letters, each with the marks written on it."""
    groups: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if not _foreign(text[index], locale):
            index += 1
            continue
        start = index
        while index < len(text) and (_foreign(text[index], locale) or _MARKS.contains(text[index])):
            index += 1
        groups.append((start, index))
    return groups


def _runs(text: str, locale: str) -> list[tuple[int, int]]:
    """(start, end) per run of two or more single out-of-script letters of one script, one
    space (" ") apart ("Т Е С Т"). A word of another script ("αβ", "Москва") is no run: it
    stays as written."""
    runs: list[tuple[int, int]] = []
    chain: list[tuple[int, int]] = []
    for start, end in [*_groups(text, locale), (len(text) + 2, len(text) + 2)]:
        single = sum(not _MARKS.contains(ch) for ch in text[start:end]) == 1
        joins = (
            single
            and chain
            and start == chain[-1][1] + 1
            and text[chain[-1][1]] == " "
            and _script(text[start]) == _script(text[chain[0][0]])
        )
        if joins:
            chain.append((start, end))
            continue
        if len(chain) > 1:
            runs.append((chain[0][0], chain[-1][1]))
        chain = [(start, end)] if single else []
    return runs


def _horizontal_space(text: str) -> bool:
    return bool(text) and all(char.isspace() and char not in "\r\n\v\f" for char in text)


_FENCE_START = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})")


def _inline_code_ranges(line: str, offset: int) -> list[tuple[int, int]]:
    """Paired Markdown backtick spans on one non-fenced line, including delimiters."""
    delimiters = list(re.finditer(r"`+", line))
    ranges = []
    index = 0
    while index < len(delimiters):
        opening = delimiters[index]
        closing = next(
            (
                at
                for at in range(index + 1, len(delimiters))
                if len(delimiters[at].group()) == len(opening.group())
            ),
            None,
        )
        if closing is None:
            index += 1
            continue
        ranges.append((offset + opening.start(), offset + delimiters[closing].end()))
        index = closing + 1
    return ranges


def _code_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Cheap Markdown code contexts: fences, paired backticks, and indented lines."""
    ranges: list[tuple[int, int]] = []
    fence: tuple[str, int] | None = None
    pending_fence: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        marker = _FENCE_START.match(content)
        if fence is not None:
            pending_fence.append((offset, offset + len(line)))
            if marker is not None:
                run = marker.group("marker")
                if (
                    run[0] == fence[0]
                    and len(run) >= fence[1]
                    and not content[marker.end() :].strip()
                ):
                    fence = None
                    ranges.extend(pending_fence)
                    pending_fence = []
        elif marker is not None:
            run = marker.group("marker")
            fence = (run[0], len(run))
            pending_fence = [(offset, offset + len(line))]
        elif content.startswith("\t") or content.startswith("    "):
            ranges.append((offset, offset + len(line)))
        else:
            ranges.extend(_inline_code_ranges(content, offset))
        offset += len(line)
    if fence is not None:
        ranges.extend(pending_fence)
    return tuple(ranges)


def _overlaps(ranges: tuple[tuple[int, int], ...], start: int, end: int) -> bool:
    return any(left < end and start < right for left, right in ranges)


def _url_token(text: str, start: int, end: int) -> bool:
    left = start
    while left and not text[left - 1].isspace():
        left -= 1
    right = end
    while right < len(text) and not text[right].isspace():
        right += 1
    token = text[left:right].casefold()
    return "://" in token or token.startswith("www.")


def _standalone_symbol(text: str, start: int, end: int) -> bool:
    """Judge a whole contiguous symbol delimiter, not one character inside it."""
    base = _without_variation_selectors(text[start:end])
    repeated = base[0] if len(base) == 1 else None
    left = start
    while repeated is not None and left and text[left - 1] == repeated:
        left -= 1
    right = end
    while repeated is not None and right < len(text) and text[right] == repeated:
        right += 1
    return _standalone(text, left, right) and not _url_token(text, left, right)


def _sentence_terminal(grapheme: str) -> bool:
    return any(_SENTENCE_TERMINAL.contains(char) for char in grapheme)


def _mixed_run_symbol(grapheme: str) -> bool:
    base = _without_variation_selectors(grapheme).replace("\u200d", "")
    significant = tuple(char for char in base if not _MARKS.contains(char))
    return bool(significant) and all(_MIXED_RUN_SYMBOL.contains(char) for char in significant)


def _emoji_grapheme(grapheme: str) -> bool:
    return any(
        _EXTENDED_PICTOGRAPHIC.contains(char) or _EMOJI_PRESENTATION.contains(char)
        for char in grapheme
    )


def _append_symbol_chain(text, spans, chain, threshold, code_ranges, runs) -> None:
    keys = [_without_variation_selectors(spans[at]["text"]) for at in chain]
    whole = (
        len(chain) > threshold
        and len(set(keys)) > 1
        and all(
            not _sentence_terminal(spans[at]["text"]) and _mixed_run_symbol(spans[at]["text"])
            for at in chain
        )
    )
    groups: list[list[int]] = [chain] if whole else []
    first = 0
    if not whole:
        for at in range(1, len(chain) + 1):
            if at == len(chain) or keys[at] != keys[first]:
                group = chain[first:at]
                if len(group) > threshold and not _sentence_terminal(spans[group[0]]["text"]):
                    groups.append(group)
                first = at
    for group in groups:
        start = spans[group[0]]["start"]
        end = spans[group[-1]]["end"]
        if (
            _standalone(text, start, end)
            and not _url_token(text, start, end)
            and not _overlaps(code_ranges, start, end)
        ):
            runs.append((start, end, tuple(spans[at]["text"] for at in group)))


def _symbol_runs(text: str, threshold: int) -> list[tuple[int, int, tuple[str, ...]]]:
    """Maximal standalone symbol-grapheme runs, allowing horizontal space between them."""
    spans = break_grapheme_spans(text, "root")
    code_ranges = _code_ranges(text)
    runs: list[tuple[int, int, tuple[str, ...]]] = []
    index = 0
    while index < len(spans):
        if not _symbol_grapheme(spans[index]["text"]):
            index += 1
            continue
        chain = [index]
        index += 1
        while index < len(spans):
            if _symbol_grapheme(spans[index]["text"]):
                chain.append(index)
                index += 1
                continue
            if _horizontal_space(spans[index]["text"]):
                space = index
                while index < len(spans) and _horizontal_space(spans[index]["text"]):
                    index += 1
                if index < len(spans) and _symbol_grapheme(spans[index]["text"]):
                    chain.append(index)
                    index += 1
                    continue
                index = space
            break
        _append_symbol_chain(text, spans, chain, threshold, code_ranges, runs)
    return runs


def run_readings(
    text: str, script: str, locale: str
) -> tuple[str | None, tuple[tuple[str, str], ...]]:
    """The run's transform and its readings besides silence: the locale's reading of its
    transliteration, the letters' names, and the run as written."""
    from frend.abbreviation_variants import AS_WRITTEN_SOURCE
    from frend.letters import spelled

    readings: list[tuple[str, str]] = []
    # The letters without the marks written on them: composed where Unicode composes
    # ("й" stays "й"), and any mark left over ("А́", "й́") dropped.
    base = "".join(ch for ch in _NFC.normalize(text) if not _MARKS.contains(ch))
    transform = transform_id(script, locale)
    if transform is not None:
        written = _transliterator(transform).transliterate(base)
        # Letters set apart are said one by one, each as ICU writes it in the locale's
        # script ("Θ" -> "th"), lower-cased as the locale spells; a word is said whole.
        forms = [spelled(unit, locale) for unit in written.split()]
        units = (
            []
            if any(form is None for form in forms)
            else [form.text.replace(" ", "") for form in forms if form is not None]
        )
        spoken = " ".join(unit for unit in units if any(_lettered(ch) for ch in unit))
        scripts = locale_scripts(locale)
        # A transform that leaves letters outside the locale's scripts ("α" through
        # Any-Hira) gives the locale nothing it reads.
        if spoken.strip() and all(_script(ch) in scripts for ch in spoken if _lettered(ch)):
            readings.append((spoken, f"{TRANSLITERATION_SOURCE}:{transform}"))
    names = [_letter_name(char) for char in base if not char.isspace()]
    if names and all(names):
        readings.append((" ".join(names), LETTER_NAME_SOURCE))  # type: ignore[arg-type]
    readings.append((text, AS_WRITTEN_SOURCE))
    return transform, tuple(readings)


class SymbolDetector:
    """Detect standalone symbols and letters of scripts frend reads by name."""

    def __init__(
        self, locale: str = "en_US", run_threshold: int = DEFAULT_SYMBOL_RUN_THRESHOLD
    ) -> None:
        if (
            isinstance(run_threshold, bool)
            or not isinstance(run_threshold, int)
            or run_threshold < 0
        ):
            raise ValueError(f"run_threshold must be a nonnegative integer, got {run_threshold!r}")
        self.locale = locale
        self.run_threshold = run_threshold

    def detect(self, text: str) -> list[dict]:
        detections = []
        for start, end, symbols in _symbol_runs(text, self.run_threshold):
            names = tuple(symbol_names(symbol, self.locale)[0] for symbol in symbols)
            bases = tuple(_without_variation_selectors(symbol) for symbol in symbols)
            detections.append(
                {
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "type": "symbol:run",
                    "value": SymbolRunValue(
                        text[start:end],
                        symbols,
                        names,
                        len(set(bases)) == 1,
                        all(_emoji_grapheme(symbol) for symbol in symbols),
                    ),
                    "captures": (Capture("symbol", start, end, text[start:end], symbols, None),),
                }
            )
        in_runs: set[int] = set()
        silent_property_tokens = _silent_property_tokens(text)
        in_silent_property_tokens = {
            position
            for start, (end, _property_class) in silent_property_tokens.items()
            for position in range(start, end)
        }
        for start, end in _runs(text, self.locale):
            if not _standalone(text, start, end):
                # A run touching a word is no unit; its letters read one by one, as before.
                continue
            in_runs.update(range(start, end))
            surface = text[start:end]
            script = _script(surface[0])
            transform, names = run_readings(surface, script, self.locale)
            detections.append(
                {
                    "text": surface,
                    "start": start,
                    "end": end,
                    "type": "symbol:script-run",
                    "value": ScriptRunValue(surface, script, transform, names),
                    "captures": (Capture("symbol", start, end, surface, surface, None),),
                }
            )
        for span in break_grapheme_spans(text, "root"):
            index, end, char = span["start"], span["end"], span["text"]
            if any(position in in_runs for position in range(index, end)):
                continue
            if index in in_silent_property_tokens:
                token = silent_property_tokens.get(index)
                if token is None:
                    continue
                token_end, _property_class = token
                surface = text[index:token_end]
                script = _script(char[0])
                detections.append(
                    {
                        "text": surface,
                        "start": index,
                        "end": token_end,
                        "type": "symbol:property",
                        "value": SymbolValue(
                            surface[0],
                            script,
                            _property_name(surface),
                            True,
                        ),
                        "captures": tuple(
                            Capture("symbol", at, at + 1, text[at], text[at], None)
                            for at in range(index, token_end)
                        ),
                    }
                )
                continue
            if any(_VARIATION_SELECTORS.contains(unit) for unit in char):
                base = _without_variation_selectors(char)
                if base and _speakable(char, self.locale) and _standalone_symbol(text, index, end):
                    script = _script(base[0])
                    detections.append(
                        {
                            "text": char,
                            "start": index,
                            "end": end,
                            "type": (
                                "symbol:cldr" if base in _cldr_names(self.locale) else "symbol:icu"
                            ),
                            "value": SymbolValue(char, script, symbol_names(char, self.locale)),
                            "captures": (Capture("symbol", index, end, char, char, None),),
                        }
                    )
                else:
                    detections.append(
                        {
                            "text": char,
                            "start": index,
                            "end": end,
                            "type": "symbol:variation",
                            "value": VariationValue(char, base),
                            "captures": (Capture("symbol", index, end, char, base, None),),
                        }
                    )
                continue
            if not _speakable(char, self.locale) or not _standalone_symbol(text, index, end):
                continue
            base = _without_variation_selectors(char).replace("\u200d", "")
            script = _script(base[0])
            kind = (
                "symbol:cldr"
                if char in _cldr_names(self.locale)
                else (
                    "symbol:letter"
                    if len(base) == 1 and _lone_foreign(base, self.locale)
                    else "symbol:icu"
                )
            )
            detections.append(
                {
                    "text": char,
                    "start": index,
                    "end": end,
                    "type": kind,
                    "value": SymbolValue(char, script, symbol_names(char, self.locale)),
                    "captures": (Capture("symbol", index, end, char, char, None),),
                }
            )
        detections.sort(key=lambda detection: detection["start"])
        return detections
