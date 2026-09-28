"""Lexical forms: every spoken form ICU and CLDR do not give, held per locale in
``data/<locale>/lexical.json`` with the reason it is hand-written.

A locale with no forms has every lexical feature off. The provenance string the forms
carry (``lexical:en_US``) is a measurement key and does not change with the move.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import icu
import pytest
from icukit.detectors import all_detectors, detect
from icukit.recognize import (
    FlexibleCurrencyDetector,
    FlexibleFractionDetector,
    FlexibleMeasureDetector,
    FlexibleNumberDetector,
    FlexibleNumericDurationDetector,
    FlexibleOrdinalDetector,
    FlexibleTimeDetector,
    PluralNumeralDetector,
)

from frend import compose_choices, resolve_choices
from frend import verbalize as verbalize_module
from frend.electronic import ElectronicDetector, digit_forms
from frend.locale_data import lexical_forms
from frend.written_forms import WrittenFormsDetector

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "frend" / "data"
_PACKAGE = _REPO / "frend"
_LEXICAL = "lexical:en_US"

# Forms that sit in the table unread until a later change reads them: the range
# connector and separators (the range readers) and the vowels (the letters reader).
_UNREAD = frozenset({"range.connector", "range.separator", "letter.vowels"})


def _tables() -> dict[str, dict]:
    return {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(_DATA.glob("*/lexical.json"))
    }


def test_every_lexical_form_says_why():
    tables = _tables()
    assert "en" in tables, "frend/data/en/lexical.json is missing"
    assert tables["en"]["forms"], "the English table holds no forms"
    for locale, table in tables.items():
        assert table["locale"] == locale
        assert table["schema_version"] == 1
        for key, entry in table["forms"].items():
            assert set(entry) == {"why", "value"}, (locale, key)
            assert isinstance(entry["why"], str) and entry["why"].strip(), (locale, key)
            assert entry["value"] not in (None, "", [], {}), (locale, key)


# Module-level constants a lexical emitter reads that are not spoken forms.
_NOT_SPOKEN = {
    ("verbalize.py", "_DURATION_UNITS"): "ICU unit identifiers passed to format_measure",
    ("verbalize.py", "ELECTRONIC_SOURCE"): "a provenance label",
}
_LETTER = re.compile(r"[^\W\d_]")
_READERS = {"_lexical", "_lexical_pattern"}


def _names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


class _Text:
    """Where a spoken text's letters come from: the string constants an expression
    writes, following a local name to what it is bound to and a call of one of the
    module's functions to what that function returns. Not a letter of a spoken form:
    a table key (``_lexical("clock.oclock", ...)``, ``rule["add"]``,
    ``rule.get("strip")``) and the arguments of a module function (its parameters,
    not its text: ``_number_leaf(value, "cardinal", locale)``). A field of a value
    already made (``item.text``) and a function from elsewhere (ICU, icukit) write no
    literal here."""

    def __init__(self, tree: ast.Module) -> None:
        self.functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }

    def constants(self, node: ast.AST, scope: ast.AST, seen: set) -> list[str]:
        if id(node) in seen:
            return []
        seen.add(id(node))
        if isinstance(node, ast.Constant):
            text = node.value
            return [text] if isinstance(text, str) and _LETTER.search(text) else []
        if isinstance(node, ast.Subscript):
            return self.constants(node.value, scope, seen)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "SpokenAlternative":
                return self.constants(node.args[0], scope, seen) if node.args else []
            if isinstance(func, ast.Name) and func.id in _READERS:
                return [c for arg in node.args[2:] for c in self.constants(arg, scope, seen)]
            if isinstance(func, ast.Name) and func.id in self.functions:
                target = self.functions[func.id]
                return [
                    c
                    for item in ast.walk(target)
                    if isinstance(item, ast.Return) and item.value is not None
                    for c in self.constants(item.value, target, seen)
                ]
            if isinstance(func, ast.Attribute):
                # A method of a text ("the {}".format(x), " ".join(...)); a key read
                # with .get names a form and is not one.
                parts = (
                    [] if func.attr == "get" else [*node.args, *(k.value for k in node.keywords)]
                )
                return [
                    c for part in (func.value, *parts) for c in self.constants(part, scope, seen)
                ]
            # Any other function (ICU's, icukit's, a builtin) makes its own text.
            return []
        if isinstance(node, ast.Attribute):
            # A field of a value already made (item.text), not a literal written here.
            return []
        if isinstance(node, ast.Name):
            return [
                c
                for value in self._bound(node.id, scope)
                for c in self.constants(value, scope, seen)
            ]
        return [
            c for child in ast.iter_child_nodes(node) for c in self.constants(child, scope, seen)
        ]

    @staticmethod
    def _bound(name: str, scope: ast.AST) -> list[ast.AST]:
        values = []
        for item in ast.walk(scope):
            if isinstance(item, ast.Assign) and name in {
                t.id for target in item.targets for t in ast.walk(target) if isinstance(t, ast.Name)
            }:
                values.append(item.value)
            elif isinstance(item, ast.comprehension | ast.For) and name in _names(item.target):
                values.append(item.iter)
        return values


def _lexical_constant_sites(source: str, name: str) -> list[str]:
    tree = ast.parse(source)
    text = _Text(tree)
    found = []
    # A spoken form written in the code and labeled lexical: the text of a
    # SpokenAlternative, or of a (text, source) pair, whose source is LEXICAL_SOURCE.
    scopes = [tree, *(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))]
    for scope in scopes:
        body = (
            [part for part in tree.body if not isinstance(part, ast.FunctionDef | ast.ClassDef)]
            if scope is tree
            else [scope]
        )
        for node in (item for part in body for item in ast.walk(part)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "SpokenAlternative"
                and node.args
            ):
                spoken = node.args[0]
                provenance = [
                    *node.args[1:2],
                    *(k.value for k in node.keywords if k.arg == "provenance"),
                ]
            elif isinstance(node, ast.Tuple) and len(node.elts) == 2:
                spoken, provenance = node.elts[0], node.elts[1:]
            else:
                continue
            if not any("LEXICAL_SOURCE" in _names(arg) for arg in provenance):
                continue
            letters = text.constants(spoken, scope, set())
            if letters:
                found.append(f"{name}:{node.lineno} {sorted(set(letters))}")
    # A module-level table of words read by a function that labels its forms lexical.
    lexical_reads: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and "LEXICAL_SOURCE" in _names(node):
            lexical_reads |= _names(node)
    for node in tree.body:
        if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id in lexical_reads
                and target.id != "LEXICAL_SOURCE"
                and (name, target.id) not in _NOT_SPOKEN
                and any(
                    isinstance(c, ast.Constant)
                    and isinstance(c.value, str)
                    and _LETTER.search(c.value)
                    for c in ast.walk(node.value)
                )
            ):
                found.append(f"{name}:{node.lineno} {target.id}")
    return sorted(set(found))


def test_no_lexical_constant_outside_the_table():
    """No hand-written spoken form stays in the code: each lexical form is read from
    ``lexical.json``. Checked on the syntax tree of every module in the package."""
    found = []
    for path in sorted(_PACKAGE.glob("*.py")):
        found += _lexical_constant_sites(path.read_text(encoding="utf-8"), path.name)
    assert found == []


def test_the_constant_check_finds_a_form_written_in_the_code():
    """The check above can fail: a lexical form written back into the code is found,
    directly, through a local name, through a module table and through a helper."""
    source = (
        "LEXICAL_SOURCE = 'lexical:en_US'\n"
        "_NAMES = {'@': 'at'}\n"
        "def _plural(word):\n"
        "    return word + 'ies'\n"
        "def speak(text, words, locale):\n"
        "    spoken = ' '.join('o' if w == '0' else w for w in words)\n"
        "    return [\n"
        "        SpokenAlternative(f'the {text}', f'{LEXICAL_SOURCE}+x'),\n"
        "        SpokenAlternative(_NAMES[text], LEXICAL_SOURCE),\n"
        "        (spoken, LEXICAL_SOURCE),\n"
        "        *(SpokenAlternative(f'{z} {text}', LEXICAL_SOURCE) for z in ('oh', 'o')),\n"
        "        SpokenAlternative(_plural(text), LEXICAL_SOURCE),\n"
        "        SpokenAlternative(_lexical_pattern('sign.plus', locale, text), LEXICAL_SOURCE),\n"
        "    ]\n"
    )
    assert _lexical_constant_sites(source, "m.py") == [
        "m.py:10 ['o']",
        "m.py:11 ['o', 'oh']",
        "m.py:12 ['ies']",
        "m.py:2 _NAMES",
        "m.py:8 ['the ']",
    ]


def test_lexical_provenance_string_is_unchanged():
    from frend.verbalize import LEXICAL_SOURCE

    assert LEXICAL_SOURCE == _LEXICAL
    spoken = (_DATA / "en" / "spoken_priors.json").read_text(encoding="utf-8")
    keys = set(re.findall(r'"([^"]*lexical:[^"]*)"', spoken))
    assert keys, "spoken_priors.json names no lexical source"
    assert all(set(re.findall(r"lexical:[^+\"]*", key)) == {_LEXICAL} for key in keys), keys


def _units(text, detectors):
    lattice = resolve_choices(list(detect(text, detectors)), source_text=text)
    return [
        unit
        for unit in compose_choices(lattice).units
        if unit.verbalized and unit.alternatives[0].provenance != "surface:passthrough"
    ]


# One reading per lexical form: the written text, its readers, and a spoken form that
# only the form gives.
_CONSUMERS = [
    ("zero.digit", "3.05", [FlexibleNumberDetector("en_US")], "three point o five"),
    ("zero.minute", "10:05", [FlexibleTimeDetector("en_US")], "ten oh five"),
    ("zero.year", "1908", all_detectors("en_US", ("y",)).detectors, "nineteen o eight"),
    ("clock.oclock", "5pm", [FlexibleTimeDetector("en_US")], "five o'clock p m"),
    ("clock.hundred", "20:00", [FlexibleTimeDetector("en_US")], "twenty hundred"),
    ("possessive.suffix", "II's", [FlexibleNumberDetector("en_US")], "two's"),
    (
        "separator.words",
        "jane@example.org",
        [ElectronicDetector("en_US")],
        "jane at example dot org",
    ),
    (
        "currency.units",
        "$1.50",
        [FlexibleCurrencyDetector("en_US", "USD")],
        "one dollar and fifty cents",
    ),
    (
        "currency.region_names",
        "$2",
        [FlexibleCurrencyDetector("en_US", "USD")],
        "two united states dollars",
    ),
    ("fraction.denominators", "1/2", [FlexibleFractionDetector("en_US")], "one half"),
    ("fraction.one", "3 1/2", [FlexibleFractionDetector("en_US")], "three and a half"),
    ("ordinal.article", "V.", [FlexibleOrdinalDetector("en_US")], "the fifth"),
    ("sign.plus", "+5", [FlexibleNumberDetector("en_US")], "plus five"),
    ("numeral.plural", "1990s", [PluralNumeralDetector("en_US")], "nineteen nineties"),
    ("duration.milliseconds", "1:47.22", [FlexibleNumericDurationDetector("en_US")], None),
    (
        "measure.per_plural",
        "578.3/km2",
        [FlexibleMeasureDetector("en_US", "square-kilometer")],
        None,
    ),
    ("zero.digit", "6 0", [WrittenFormsDetector("en_US")], "six o"),
]


def test_every_read_form_has_a_consumer_here():
    en = set(_tables()["en"]["forms"])
    assert {key for key, *_ in _CONSUMERS} | {"zero.words"} == en - _UNREAD


def _spoken(text, detectors):
    return {
        (item.text, item.provenance)
        for unit in _units(text, detectors)
        for item in unit.alternatives
    }


@pytest.mark.parametrize(("key", "text", "detectors", "form"), _CONSUMERS)
def test_an_empty_locale_turns_each_lexical_feature_off(monkeypatch, key, text, detectors, form):
    """With the Russian table (no forms), each consumer of a lexical form yields no
    lexical form. The readers run on English ICU data, so what goes is exactly what
    the table gave, not what Russian ICU happens to lack."""
    assert (_DATA / "ru" / "lexical.json").is_file(), "frend/data/ru/lexical.json is missing"
    assert lexical_forms("ru_RU") == {}
    assert lexical_forms("ru") == {}

    with_table = _spoken(text, detectors)
    lexical = {spoken for spoken, provenance in with_table if _LEXICAL in provenance}
    assert lexical, (key, sorted(with_table))
    if form is not None:
        assert form in lexical, (key, sorted(lexical))

    ru = dict(lexical_forms("ru_RU"))
    monkeypatch.setattr(verbalize_module, "_lexical_for", lambda locale: ru)
    without = _spoken(text, detectors)
    assert not {spoken for spoken, provenance in without if _LEXICAL in provenance}
    assert not lexical & {spoken for spoken, _ in without}


def test_an_empty_locale_says_no_zero_word_and_no_zero_digit():
    """The zero words (shared weight) and the electronic digit forms read the locale
    itself; Russian's empty table gives neither."""
    assert verbalize_module._zero_words("en_US") == frozenset({"o", "oh", "zero"})
    assert verbalize_module._zero_words("ru_RU") == frozenset()
    assert verbalize_module._zero_key("nineteen o five") == ("nineteen", "0", "five")
    assert verbalize_module._zero_key("nineteen o five", "ru_RU") == ("nineteen", "o", "five")
    assert "digits_o" in digit_forms("105", "en_US")
    assert "digits_o" not in digit_forms("105", "ru_RU")
    assert "digits" in digit_forms("105", "ru_RU")


# The calendar walk. CLDR's calendar data does hold connector words in en ("at" joins
# a date to a time, "of" a week to a month), so the check must find those where they
# are and find no word between a range's two ends (intervalFormats).

_QUOTED = re.compile(r"'((?:[^']|'')*)'")


def _walk(bundle, path: str):
    kind = bundle.getType()
    if kind == icu.UResType.STRING:
        yield path, bundle.getString()
    elif kind in (icu.UResType.TABLE, icu.UResType.ARRAY):
        for index in range(bundle.getSize()):
            child = bundle.get(index)
            key = child.getKey() if kind == icu.UResType.TABLE else str(index)
            yield from _walk(child, f"{path}/{key}")


def _literal_words(pattern: str) -> list[str]:
    """The words a date pattern writes literally: every letter run inside quotes, and
    every non-ASCII letter outside them (ASCII letters outside quotes are fields)."""
    words = []
    for match in _QUOTED.finditer(pattern):
        words += re.findall(r"[^\W\d_]+", match.group(1).replace("''", "'"))
    words += re.findall(r"[^\W\d_A-Za-z]", _QUOTED.sub("", pattern))
    return words


def _calendar_words(locale: str) -> tuple[int, int, dict[str, list[str]]]:
    # Opened as ResourceBundle("", Locale(x)): the construction that does not crash
    # PyICU 2.16.2, and indexed rather than iterated.
    gregorian = icu.ResourceBundle("", icu.Locale(locale)).get("calendar").get("gregorian")
    strings = list(_walk(gregorian, "gregorian"))
    interval = [path for path, _ in strings if "/intervalFormats/" in path]
    words = {path: _literal_words(text) for path, text in strings}
    return len(strings), len(interval), {path: found for path, found in words.items() if found}


_EN_CALENDAR_WORDS = {
    "gregorian/DateTimePatterns%atTime/0": ["at"],
    "gregorian/DateTimePatterns%atTime/1": ["at"],
    "gregorian/DateTimePatterns%relative/0": ["at"],
    "gregorian/DateTimePatterns%relative/1": ["at"],
    "gregorian/availableFormats/MMMMW/one": ["week", "of"],
    "gregorian/availableFormats/MMMMW/other": ["week", "of"],
    "gregorian/availableFormats/yw/one": ["week", "of"],
    "gregorian/availableFormats/yw/other": ["week", "of"],
}
_ROOT_CALENDAR_WORDS = {
    "gregorian/availableFormats/MMMMW/other": ["week", "of"],
    "gregorian/availableFormats/yw/other": ["week", "of"],
}


def test_calendar_walk_finds_no_en_range_connector():
    """calendar/gregorian in root, en and en_US: the walk finds the words CLDR does
    write there ("at", "week", "of"), every one outside intervalFormats, and none
    inside, where a range connector would be. The connector's ``why`` says exactly
    this, and cites the two separator patterns as ICU holds them."""
    expected = {"root": _ROOT_CALENDAR_WORDS, "en": _EN_CALENDAR_WORDS, "en_US": _EN_CALENDAR_WORDS}
    for locale, words in expected.items():
        strings, interval, found = _calendar_words(locale)
        assert strings > interval >= 76, locale
        assert found == words, locale
        assert not any("/intervalFormats/" in path for path in found), locale

    root = icu.ResourceBundle("", icu.Locale("root"))
    number_range = root.get("NumberElements").get("latn").get("miscPatterns").get("range")
    fallback = root.get("calendar").get("gregorian").get("intervalFormats").get("fallback")
    why = _tables()["en"]["forms"]["range.connector"]["why"]
    assert f"miscPatterns/range '{number_range.getString()}'" in why
    assert f"intervalFormats/fallback '{fallback.getString()}'" in why
    assert "its 76 intervalFormats patterns hold no literal word" in why
    assert "'at' in DateTimePatterns%atTime and %relative" in why
    assert "'week' and 'of' in availableFormats MMMMW and yw" in why
    assert "no connector word" not in why


def test_calendar_walk_flags_a_range_connector_where_cldr_writes_one():
    """The check can fail: Chinese intervalFormats write 至 ("to") between the ends."""
    _, _, found = _calendar_words("zh_CN")
    connectors = {
        path: words
        for path, words in found.items()
        if "/intervalFormats/" in path and "至" in words
    }
    assert "gregorian/intervalFormats/Bh/B" in connectors
    assert len(connectors) > 1
