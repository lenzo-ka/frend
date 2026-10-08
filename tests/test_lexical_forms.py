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
    FlexibleDateDetector,
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
from frend.symbols import SymbolDetector
from frend.written_forms import WrittenFormsDetector

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "frend" / "data"
_PACKAGE = _REPO / "frend"
_LEXICAL = "lexical:en_US"

# Forms that sit in the table unread until a later change reads them: none since P7
# read the range connector and the hyphen separator.
_UNREAD = frozenset()


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
    ("verbalize.py", "_DATE_MONTH_FIRST_KEYS"): "field-set names in lexical date data",
    ("verbalize.py", "_PLURAL_ONE"): "an ICU plural-category name",
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
                spoken = _spoken_argument(node)
                return [] if spoken is None else self.constants(spoken, scope, seen)
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
        if isinstance(scope, ast.FunctionDef):
            # A parameter's default is a text the function writes.
            arguments = scope.args
            positional = [*arguments.posonlyargs, *arguments.args]
            pairs = [
                *zip(
                    positional[len(positional) - len(arguments.defaults) :],
                    arguments.defaults,
                    strict=True,
                ),
                *zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True),
            ]
            values += [default for arg, default in pairs if default is not None and arg.arg == name]
        for item in ast.walk(scope):
            if isinstance(item, ast.Assign) and name in {
                t.id for target in item.targets for t in ast.walk(target) if isinstance(t, ast.Name)
            }:
                values.append(item.value)
            elif isinstance(item, ast.comprehension | ast.For) and name in _names(item.target):
                values.append(item.iter)
        return values


def _spoken_argument(node: ast.Call) -> ast.AST | None:
    """The text a ``SpokenAlternative`` is given, positionally or as ``text=``."""
    if node.args:
        return node.args[0]
    return next((k.value for k in node.keywords if k.arg == "text"), None)


def _argument(node: ast.Call, index: int, keyword: str) -> ast.AST | None:
    if len(node.args) > index:
        return node.args[index]
    return next((k.value for k in node.keywords if k.arg == keyword), None)


class _Label:
    """Whether an expression carries the lexical label.

    A provenance is labeled lexical where it names ``LEXICAL_SOURCE``, writes the label
    itself (a string holding ``lexical:``), or names a local or module-level name bound
    to either (an alias). A field of a value already made (``item.provenance``) is
    that value's label, not one given here.

    The parts handed to ``_compose`` carry the label where they are, or are built
    from, a value labeled here: followed through local names, loop and comprehension
    variables, what a list is given by ``append``/``insert``/``extend``, and the
    returns of the module's own functions."""

    def __init__(self, tree: ast.Module, functions: dict[str, ast.FunctionDef]) -> None:
        self.tree = tree
        self.functions = functions

    def _module_bound(self, name: str) -> list[ast.AST]:
        return [
            node.value
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and name
            in {
                t.id for target in node.targets for t in ast.walk(target) if isinstance(t, ast.Name)
            }
        ]

    def _aliases(self, name: str, scope: ast.AST) -> list[ast.AST]:
        local = (
            []
            if scope is self.tree
            else [
                item.value
                for item in ast.walk(scope)
                if isinstance(item, ast.Assign)
                and name
                in {
                    t.id
                    for target in item.targets
                    for t in ast.walk(target)
                    if isinstance(t, ast.Name)
                }
            ]
        )
        return [*local, *self._module_bound(name)]

    def provenance(self, node: ast.AST, scope: ast.AST, seen: set | None = None) -> bool:
        seen = set() if seen is None else seen
        if id(node) in seen:
            return False
        seen.add(id(node))
        if isinstance(node, ast.Constant):
            return isinstance(node.value, str) and "lexical:" in node.value
        if isinstance(node, ast.Attribute):
            return False
        if isinstance(node, ast.Name):
            return node.id == "LEXICAL_SOURCE" or any(
                self.provenance(value, scope, seen) for value in self._aliases(node.id, scope)
            )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "lexical_source":
                return True
            # A label is a string: one built by a string method ("+".join(...),
            # "{}+{}".format(...)) carries what it is built from; any other call makes
            # a value that is not a label.
            if not isinstance(node.func, ast.Attribute):
                return False
            children = [node.func.value, *node.args, *(k.value for k in node.keywords)]
        elif isinstance(node, ast.JoinedStr | ast.FormattedValue | ast.BinOp | ast.IfExp):
            children = list(ast.iter_child_nodes(node))
        else:
            return False
        return any(self.provenance(child, scope, seen) for child in children)

    def parts(self, node: ast.AST, scope: ast.AST, seen: set | None = None) -> bool:
        seen = set() if seen is None else seen
        if id(node) in seen:
            return False
        seen.add(id(node))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "SpokenAlternative":
                label = _argument(node, 1, "provenance")
                if label is not None and self.provenance(label, scope):
                    return True
            elif node.func.id in self.functions:
                target = self.functions[node.func.id]
                if any(
                    self.parts(item.value, target, seen)
                    for item in ast.walk(target)
                    if isinstance(item, ast.Return) and item.value is not None
                ):
                    return True
        if (
            isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and self.provenance(node.elts[1], scope)
        ):
            return True
        if isinstance(node, ast.Name):
            return node.id == "LEXICAL_SOURCE" or any(
                self.parts(value, scope, seen)
                for value in [*_Text._bound(node.id, scope), *self._added(node.id, scope)]
            )
        return any(self.parts(child, scope, seen) for child in ast.iter_child_nodes(node))

    @staticmethod
    def _added(name: str, scope: ast.AST) -> list[ast.AST]:
        """What a list is given after it is made (``name.append(x)``, ``.insert``,
        ``.extend``)."""
        return [
            item
            for call in ast.walk(scope)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in {"append", "insert", "extend"}
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == name
            for item in call.args
        ]


def _template_letters(node: ast.AST, scope: ast.AST, seen: set) -> list[str]:
    """The lettered string literals a template is written from, following a local
    name to what it is bound to; not what a comprehension iterates over or what a
    call returns (``" ".join("{}" for _ in parts)`` writes no letter)."""
    if id(node) in seen:
        return []
    seen.add(id(node))
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) and _LETTER.search(node.value) else []
    if isinstance(node, ast.Name):
        return [
            c
            for value in _Text._bound(node.id, scope)
            for c in _template_letters(value, scope, seen)
        ]
    if isinstance(node, ast.comprehension | ast.Attribute):
        return []
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Attribute):
            return []
        children = [node.func.value, *node.args, *(k.value for k in node.keywords)]
    else:
        children = list(ast.iter_child_nodes(node))
    return [c for child in children for c in _template_letters(child, scope, seen)]


def _lexical_constant_sites(source: str, name: str) -> list[str]:
    tree = ast.parse(source)
    text = _Text(tree)
    label = _Label(tree, text.functions)
    found = []
    # A spoken form written in the code and labeled lexical: the text of a
    # SpokenAlternative (positional or ``text=``), or of a (text, source) pair, whose
    # source is labeled lexical; and a lettered template ``_compose`` fills from parts
    # that carry the label.
    scopes = [tree, *(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))]
    for scope in scopes:
        body = (
            [part for part in tree.body if not isinstance(part, ast.FunctionDef | ast.ClassDef)]
            if scope is tree
            else [scope]
        )
        for node in (item for part in body for item in ast.walk(part)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "SpokenAlternative":
                    spoken = _spoken_argument(node)
                    provenance = _argument(node, 1, "provenance")
                    if (
                        spoken is None
                        or provenance is None
                        or not label.provenance(provenance, scope)
                    ):
                        continue
                elif node.func.id == "_compose":
                    spoken = _argument(node, 1, "template")
                    parts = _argument(node, 0, "parts")
                    if spoken is None or parts is None or not label.parts(parts, scope):
                        continue
                else:
                    continue
            elif (
                isinstance(node, ast.Tuple)
                and len(node.elts) == 2
                and not any(isinstance(elt, ast.Starred) for elt in node.elts)
            ):
                spoken = node.elts[0]
                if not label.provenance(node.elts[1], scope):
                    continue
            else:
                continue
            letters = (
                _template_letters(spoken, scope, set())
                if isinstance(node, ast.Call) and node.func.id == "_compose"
                else text.constants(spoken, scope, set())
            )
            if letters:
                found.append(f"{name}:{node.lineno} {sorted(set(letters))}")
    # A module-level table of words read by a function that labels its forms lexical.
    lexical_reads: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            label.provenance(item, node) for item in ast.walk(node)
        ):
            lexical_reads |= _names(node)
    for node in tree.body:
        if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id in lexical_reads
                and not label.provenance(node.value, tree)
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


# No spoken lexical form may remain written in code.
_OPEN = {}


def _unlined(site: str) -> str:
    return re.sub(r":\d+ ", " ", site, count=1)


def test_no_lexical_constant_outside_the_table():
    """No spoken form labeled lexical is written in the code, outside the open items
    named in ``_OPEN``. Checked on the syntax tree of every module in the package.

    What the check finds (each shown by the tests below): a lettered text given to
    ``SpokenAlternative`` positionally or as ``text=``, or as the first of a (text,
    source) pair, whose source is labeled lexical (``LEXICAL_SOURCE``, a name bound to
    it, or a string holding ``lexical:``); that text followed through local names,
    parameter defaults, module tables and the returns of the module's functions; a
    lettered template given to ``_compose`` with parts that carry the label; and a
    module table of words read by a function that labels its forms lexical.

    What it does not find: English written in the code whose reading is not labeled
    lexical (the fraction's "{} over {}" and ordinal plural "s", filed for P8), a
    label passed in from another module, and a text built by a function from outside
    the module. It shows that no labeled form bypasses the table by these routes, not
    that none can."""
    found = []
    for path in sorted(_PACKAGE.glob("*.py")):
        found += _lexical_constant_sites(path.read_text(encoding="utf-8"), path.name)
    assert sorted({_unlined(site) for site in found}) == sorted(_OPEN), found


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


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            "def speak(x):\n"
            "    return SpokenAlternative(text='the ' + x, provenance=LEXICAL_SOURCE)\n",
            ["m.py:3 ['the ']"],
        ),
        (
            "SRC = LEXICAL_SOURCE\ndef speak(x):\n    return SpokenAlternative('the ' + x, SRC)\n",
            ["m.py:4 ['the ']"],
        ),
        (
            "def speak(x):\n"
            "    source = f'{LEXICAL_SOURCE}+x'\n"
            "    return SpokenAlternative('the ' + x, source)\n",
            ["m.py:4 ['the ']"],
        ),
        (
            "def speak(x):\n    return SpokenAlternative('the ' + x, 'lexical:en_US')\n",
            ["m.py:3 ['the ']"],
        ),
        (
            "def speak(x, art='the'):\n"
            "    return SpokenAlternative(f'{art} {x}', LEXICAL_SOURCE)\n",
            ["m.py:3 ['the']"],
        ),
        (
            "def _day(d):\n"
            "    return (SpokenAlternative(d, LEXICAL_SOURCE),)\n"
            "def speak(d, m):\n"
            "    p = [_day(d), m]\n"
            "    return _compose(p, 'the {} of {}')\n",
            ["m.py:6 ['the {} of {}']"],
        ),
        (
            "def speak(d, m):\n"
            "    p = [d]\n"
            "    p.append((SpokenAlternative('x', LEXICAL_SOURCE),))\n"
            "    template = 'the {} of {}'\n"
            "    return _compose(p, template)\n",
            ["m.py:4 ['x']", "m.py:6 ['the {} of {}']"],
        ),
    ],
    ids=["keyword", "alias", "local-alias", "literal-label", "default", "compose", "compose-local"],
)
def test_the_constant_check_follows_every_way_a_label_is_given(body, expected):
    """The routes fugu's review found the first version of the check blind to (a
    ``text=`` keyword, an alias of ``LEXICAL_SOURCE``, the literal label, a default
    argument, a ``_compose`` template), each in a made-up module."""
    source = "LEXICAL_SOURCE = 'lexical:en_US'\n" + body
    assert _lexical_constant_sites(source, "m.py") == expected


def test_the_constant_check_passes_a_template_without_a_label():
    """No false positive: a lettered template filled from unlabeled parts, a template
    of bare slots filled from labeled ones, and a label passed on from a value already
    made (``item.provenance``) are not lexical forms written here."""
    source = (
        "LEXICAL_SOURCE = 'lexical:en_US'\n"
        "def speak(n, d, items):\n"
        "    over = _compose([n, d], '{} over {}')\n"
        "    zero = (SpokenAlternative(_lexical('zero.digit', 'en_US'), LEXICAL_SOURCE),)\n"
        "    plain = _compose([n, zero], ' '.join('{}' for _ in n))\n"
        "    passed = [SpokenAlternative(f'{i.text}s', i.provenance) for i in items]\n"
        "    return over, plain, passed\n"
    )
    assert _lexical_constant_sites(source, "m.py") == []


def test_lexical_provenance_string_is_unchanged():
    from frend.verbalize import LEXICAL_SOURCE

    assert LEXICAL_SOURCE == _LEXICAL
    spoken = (_DATA / "en" / "spoken_priors.json").read_text(encoding="utf-8")
    keys = set(re.findall(r'"([^"]*lexical:[^"]*)"', spoken))
    assert keys, "spoken_priors.json names no lexical source"
    assert all(set(re.findall(r"lexical:[^+\"]*", key)) == {_LEXICAL} for key in keys), keys


def _units(text, detectors):
    lattice = resolve_choices(list(detect(text, detectors)), source_text=text)
    run_ids = {
        edge.id
        for edge in lattice.edges
        if edge.kind == "reading" and edge.detection.get("type") == "symbol:run"
    }
    return [
        unit
        for unit in compose_choices(lattice).units
        if unit.verbalized
        and unit.alternatives
        and unit.alternatives[0].provenance != "surface:passthrough"
        and (not run_ids or unit.edge_id in run_ids)
    ]


# One reading per lexical form: the written text, its readers, and a spoken form that
# only the form gives.
_CONSUMERS = [
    (
        ("date.day_first", "date.month_first"),
        "9/30/1908",
        [FlexibleDateDetector("en_US")],
        (
            "the thirtieth of September nineteen o eight",
            "September thirtieth, nineteen o eight",
        ),
    ),
    ("zero.digit", "3.05", [FlexibleNumberDetector("en_US")], "three point o five"),
    ("zero.minute", "10:05", [FlexibleTimeDetector("en_US")], "ten oh five"),
    ("zero.year", "1908", all_detectors("en_US", ("y",)).detectors, "nineteen o eight"),
    ("clock.oclock", "5pm", [FlexibleTimeDetector("en_US")], "five o'clock p m"),
    ("clock.hundred", "20:00", [FlexibleTimeDetector("en_US")], "twenty hundred"),
    ("possessive.suffix", "II's", [FlexibleNumberDetector("en_US")], "two's"),
    ("range.connector", "5-10", [FlexibleNumberDetector("en_US")], "to ten"),
    ("range.separator", "5 - 10", [FlexibleNumberDetector("en_US"), SymbolDetector()], "to"),
    (
        "separator.words",
        "jane@example.org",
        [ElectronicDetector("en_US")],
        "jane at example dot org",
    ),
    (
        "symbol_run.repeated",
        "****",
        [SymbolDetector("en_US")],
        "line of asterisk",
    ),
    (
        "symbol_run.line",
        "\U0001f600\U0001f603\U0001f604\U0001f601",
        [SymbolDetector("en_US")],
        "line of emoji",
    ),
    (
        "symbol_run.mixed_emoji",
        "😀😃😄😁",
        [SymbolDetector("en_US")],
        "line of emoji",
    ),
    (
        "symbol_run.mixed_symbols",
        "+<=>",
        [SymbolDetector("en_US")],
        "line of symbols",
    ),
    (
        ("currency.units", "money.minor_joiner"),
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
    (
        ("fraction.denominators", "fraction.over"),
        "1/2",
        [FlexibleFractionDetector("en_US")],
        ("one half", "one over two"),
    ),
    (
        ("fraction.one", "fraction.mixed"),
        "3 1/2",
        [FlexibleFractionDetector("en_US")],
        "three and a half",
    ),
    ("fraction.plural_suffix", "2/3", [FlexibleFractionDetector("en_US")], "two thirds"),
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


# What each consumer above emits, in order: every alternative's text and provenance,
# unit by unit. Taken from this tree and identical on origin/main (b1de4e2), where
# every one of these forms was written in the code, so the move to the table changed
# no reading, no provenance and no order. The order also pins ``zero.words``, which
# only reweights: without it "three point zero five", "nineteen oh-eight" and "ten oh
# five" lead.
_SYMBOL_EMOJI_GOLDEN = [
    [
        ("line of emoji", "lexical:en_US+icu-name:symbol"),
        (
            "grinning face smiling face with open mouth "
            "smiling face with open mouth and smiling eyes grinning face with smiling eyes",
            "icu-name:symbol",
        ),
        ("", "surface:silence"),
    ],
]


_GOLDEN = {
    (
        "symbol_run.line",
        "\U0001f600\U0001f603\U0001f604\U0001f601",
    ): _SYMBOL_EMOJI_GOLDEN,
    ("symbol_run.repeated", "****"): [
        [
            ("line of asterisk", "lexical:en_US+cldr-symbol:asterisk"),
            ("asterisk asterisk asterisk asterisk", "cldr-symbol:asterisk"),
            ("", "surface:silence"),
        ],
    ],
    ("symbol_run.mixed_emoji", "😀😃😄😁"): [
        [
            ("line of emoji", "lexical:en_US+icu-name:symbol"),
            (
                "grinning face smiling face with open mouth "
                "smiling face with open mouth and smiling eyes grinning face with smiling eyes",
                "icu-name:symbol",
            ),
            ("", "surface:silence"),
        ],
    ],
    ("symbol_run.mixed_symbols", "+<=>"): [
        [
            (
                "line of symbols",
                "lexical:en_US+cldr-symbol:plus sign+cldr-symbol:less-than+"
                "cldr-symbol:equal+cldr-symbol:greater-than",
            ),
            (
                "plus sign less-than equal greater-than",
                "cldr-symbol:plus sign+cldr-symbol:less-than+cldr-symbol:equal+"
                "cldr-symbol:greater-than",
            ),
            ("", "surface:silence"),
        ],
    ],
    ("date.day_first", "9/30/1908"): [
        [
            (
                "September thirtieth, nineteen o eight",
                "icu-datetime:LLLL+icu-rbnf:%spellout-ordinal+lexical:en_US+icu-rbnf:%spellout-numbering-year+lexical:en_US",
            ),
            (
                "the thirtieth of September nineteen o eight",
                "icu-rbnf:%spellout-ordinal+icu-datetime:LLLL+icu-rbnf:%spellout-numbering-year+lexical:en_US",
            ),
            (
                "September thirtieth, nineteen oh-eight",
                "icu-datetime:LLLL+icu-rbnf:%spellout-ordinal+lexical:en_US+icu-rbnf:%spellout-numbering-year",
            ),
            (
                "the thirtieth of September nineteen oh-eight",
                "icu-rbnf:%spellout-ordinal+icu-datetime:LLLL+icu-rbnf:%spellout-numbering-year",
            ),
            (
                "September thirtieth, one thousand nine hundred eight",
                "icu-datetime:LLLL+icu-rbnf:%spellout-ordinal+lexical:en_US+icu-rbnf:%spellout-numbering",
            ),
            (
                "September thirtieth, one thousand nine hundred and eight",
                "icu-datetime:LLLL+icu-rbnf:%spellout-ordinal+lexical:en_US+icu-rbnf:%spellout-numbering-verbose",
            ),
            (
                "the thirtieth of September one thousand nine hundred eight",
                "icu-rbnf:%spellout-ordinal+icu-datetime:LLLL+icu-rbnf:%spellout-numbering",
            ),
            (
                "the thirtieth of September one thousand nine hundred and eight",
                "icu-rbnf:%spellout-ordinal+icu-datetime:LLLL+icu-rbnf:%spellout-numbering-verbose",
            ),
        ],
    ],
    ("zero.digit", "3.05"): [
        [
            (
                "three point o five",
                "icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-cardinal+lexical:en_US+icu-rbnf:%spellout-numbering",
            ),
            (
                "three point zero five",
                "icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-cardinal+icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-numbering",
            ),
        ],
    ],
    ("zero.minute", "10:05"): [
        [
            (
                "ten o five",
                "icu-rbnf:%spellout-numbering+lexical:en_US+icu-rbnf:%spellout-numbering",
            ),
            (
                "ten oh five",
                "icu-rbnf:%spellout-numbering+lexical:en_US+icu-rbnf:%spellout-numbering",
            ),
        ],
    ],
    ("zero.year", "1908"): [
        [
            ("nineteen o eight", "icu-rbnf:%spellout-numbering-year+lexical:en_US"),
            ("nineteen oh-eight", "icu-rbnf:%spellout-numbering-year"),
            ("one thousand nine hundred eight", "icu-rbnf:%spellout-numbering"),
            ("one thousand nine hundred and eight", "icu-rbnf:%spellout-numbering-verbose"),
            ("one nine o eight", "icu-rbnf:%spellout-cardinal+lexical:en_US"),
            ("one nine zero eight", "icu-rbnf:%spellout-cardinal"),
        ],
    ],
    ("clock.oclock", "5pm"): [
        [
            ("five p m", "icu-rbnf:%spellout-numbering+surface:letters"),
            ("five o'clock p m", "icu-rbnf:%spellout-numbering+lexical:en_US+surface:letters"),
        ],
    ],
    ("clock.hundred", "20:00"): [
        [
            ("twenty o'clock", "icu-rbnf:%spellout-numbering+lexical:en_US"),
            ("twenty hundred", "icu-rbnf:%spellout-numbering+lexical:en_US"),
            ("twenty", "icu-rbnf:%spellout-numbering"),
        ],
    ],
    ("possessive.suffix", "II's"): [
        [
            ("two's", "icu-rbnf:%spellout-numbering+lexical:en_US"),
            ("second's", "icu-rbnf:%spellout-ordinal+lexical:en_US"),
            ("the second's", "lexical:en_US+icu-rbnf:%spellout-ordinal+lexical:en_US"),
        ],
    ],
    ("separator.words", "jane@example.org"): [
        [
            ("jane at example dot org", "measured:electronic+lexical:en_US"),
            ("j a n e at example dot org", "measured:electronic+lexical:en_US"),
            ("jane at e x a m p l e dot org", "measured:electronic+lexical:en_US"),
            ("j a n e at e x a m p l e dot org", "measured:electronic+lexical:en_US"),
            ("jane at example dot o r g", "measured:electronic+lexical:en_US"),
            ("j a n e at example dot o r g", "measured:electronic+lexical:en_US"),
            ("jane at e x a m p l e dot o r g", "measured:electronic+lexical:en_US"),
            ("j a n e at e x a m p l e dot o r g", "measured:electronic+lexical:en_US"),
        ],
    ],
    ("currency.units", "$1.50"): [
        [
            (
                "one dollar and fifty cents",
                "icu-rbnf:%spellout-numbering+lexical:en_US+lexical:en_US+icu-rbnf:%spellout-numbering+lexical:en_US",
            ),
            (
                "one united states dollars and fifty cents",
                "icu-rbnf:%spellout-numbering+lexical:en_US+lexical:en_US+icu-rbnf:%spellout-numbering+lexical:en_US",
            ),
            (
                "one US dollars and fifty cents",
                "icu-rbnf:%spellout-numbering+icu-measure:wide+lexical:en_US+icu-rbnf:%spellout-numbering+lexical:en_US",
            ),
        ],
    ],
    ("currency.region_names", "$2"): [
        [
            ("two dollars", "icu-rbnf:%spellout-numbering+lexical:en_US"),
            ("two united states dollars", "icu-rbnf:%spellout-numbering+lexical:en_US"),
            ("two US dollars", "icu-rbnf:%spellout-numbering+icu-measure:wide"),
        ],
    ],
    ("fraction.denominators", "1/2"): [
        [
            ("one half", "icu-rbnf:%spellout-numbering+lexical:en_US"),
            ("one second", "icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-ordinal"),
            (
                "one over two",
                "icu-rbnf:%spellout-numbering+lexical:en_US+icu-rbnf:%spellout-numbering",
            ),
        ],
    ],
    ("fraction.one", "3 1/2"): [
        [
            (
                "three and a half",
                "icu-rbnf:%spellout-numbering+lexical:en_US+lexical:en_US",
            ),
            (
                "three and one second",
                "icu-rbnf:%spellout-numbering+lexical:en_US+icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-ordinal",
            ),
            (
                "three and one half",
                "icu-rbnf:%spellout-numbering+lexical:en_US+icu-rbnf:%spellout-numbering+lexical:en_US",
            ),
            (
                "three and one over two",
                "icu-rbnf:%spellout-numbering+lexical:en_US+icu-rbnf:%spellout-numbering+lexical:en_US+icu-rbnf:%spellout-numbering",
            ),
        ],
    ],
    ("fraction.plural_suffix", "2/3"): [
        [
            (
                "two thirds",
                "icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-ordinal+lexical:en_US",
            ),
            (
                "two over three",
                "icu-rbnf:%spellout-numbering+lexical:en_US+icu-rbnf:%spellout-numbering",
            ),
        ],
    ],
    ("ordinal.article", "V."): [
        [
            ("fifth", "icu-rbnf:%spellout-ordinal"),
            ("the fifth", "lexical:en_US+icu-rbnf:%spellout-ordinal"),
        ],
    ],
    ("sign.plus", "+5"): [
        [
            ("plus five", "lexical:en_US+icu-rbnf:%spellout-numbering"),
            ("five", "icu-rbnf:%spellout-numbering"),
        ],
    ],
    ("numeral.plural", "1990s"): [
        [
            ("nineteen nineties", "icu-rbnf:%spellout-numbering-year+lexical:en_US"),
            ("one thousand nine hundred nineties", "icu-rbnf:%spellout-numbering+lexical:en_US"),
            (
                "one thousand nine hundred and nineties",
                "icu-rbnf:%spellout-numbering-verbose+lexical:en_US",
            ),
        ],
    ],
    ("duration.milliseconds", "1:47.22"): [
        [
            (
                "one hour forty-seven point two two minutes",
                "icu-rbnf:%spellout-numbering+icu-measure:wide+icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-cardinal+icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-numbering+icu-measure:wide+icu-list:units",
            ),
            (
                "one hour and forty-seven point two two minutes",
                "icu-rbnf:%spellout-numbering+icu-measure:wide+icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-cardinal+icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-numbering+icu-measure:wide+icu-list:units",
            ),
        ],
        [
            (
                "one minute forty-seven seconds and twenty-two milliseconds",
                "icu-rbnf:%spellout-numbering+icu-measure:wide+icu-rbnf:%spellout-numbering+icu-measure:wide+icu-rbnf:%spellout-numbering+icu-measure:wide+icu-list:units+lexical:en_US",
            ),
            (
                "one minute forty-seven point two two seconds",
                "icu-rbnf:%spellout-numbering+icu-measure:wide+icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-cardinal+icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-numbering+icu-measure:wide+icu-list:units",
            ),
            (
                "one minute and forty-seven point two two seconds",
                "icu-rbnf:%spellout-numbering+icu-measure:wide+icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-cardinal+icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-numbering+icu-measure:wide+icu-list:units",
            ),
        ],
    ],
    ("measure.per_plural", "578.3/km2"): [
        [
            (
                "five hundred seventy-eight point three per square kilometers",
                "icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-cardinal+icu-rbnf:%spellout-numbering+icu-measure:wide+lexical:en_US",
            ),
            (
                "five hundred seventy-eight point three per square kilometer",
                "icu-rbnf:%spellout-numbering+icu-rbnf:%spellout-cardinal+icu-rbnf:%spellout-numbering+icu-measure:wide",
            ),
            (
                "five hundred and seventy-eight point three per square kilometers",
                "icu-rbnf:%spellout-numbering-verbose+icu-rbnf:%spellout-cardinal+icu-rbnf:%spellout-numbering+icu-measure:wide+lexical:en_US",
            ),
            (
                "five hundred and seventy-eight point three per square kilometer",
                "icu-rbnf:%spellout-numbering-verbose+icu-rbnf:%spellout-cardinal+icu-rbnf:%spellout-numbering+icu-measure:wide",
            ),
        ],
    ],
    ("range.connector", "5-10"): [
        [("five", "icu-rbnf:%spellout-numbering")],
        [
            ("minus ten", "icu-rbnf:%spellout-numbering"),
            ("to ten", "lexical:en_US+icu-rbnf:%spellout-numbering"),
            ("to one o", "lexical:en_US+icu-rbnf:%spellout-cardinal+lexical:en_US"),
            ("to one zero", "lexical:en_US+icu-rbnf:%spellout-cardinal"),
        ],
    ],
    ("range.separator", "5 - 10"): [
        [("five", "icu-rbnf:%spellout-numbering")],
        [
            ("", "surface:silence"),
            ("to", "lexical:en_US"),
            ("hyphen-minus", "cldr-symbol:hyphen-minus"),
            ("dash", "cldr-symbol:dash"),
            ("hyphen", "cldr-symbol:hyphen"),
            ("minus", "cldr-symbol:minus"),
        ],
        [
            ("ten", "icu-rbnf:%spellout-numbering"),
            ("one o", "icu-rbnf:%spellout-cardinal+lexical:en_US"),
            ("one zero", "icu-rbnf:%spellout-cardinal"),
        ],
    ],
    ("zero.digit", "6 0"): [
        [
            ("six o", "icu-rbnf:%spellout-cardinal+lexical:en_US"),
            ("six zero", "icu-rbnf:%spellout-cardinal"),
        ],
    ],
}
_GOLDEN_DIGIT_FORMS = {
    "cardinal": (
        ("one hundred five", "icu-rbnf:%spellout-numbering"),
        ("one hundred and five", "icu-rbnf:%spellout-numbering-verbose"),
    ),
    "year": (
        ("one hundred five", "icu-rbnf:%spellout-numbering-year"),
        ("one hundred and five", "icu-rbnf:%spellout-numbering-verbose"),
    ),
    "digits": (("one zero five", "icu-rbnf:%spellout-cardinal"),),
    "digits_o": (("one o five", "lexical:en_US"),),
}


@pytest.mark.parametrize(("key", "text", "detectors", "form"), _CONSUMERS)
def test_each_lexical_reading_is_emitted_as_before(key, text, detectors, form, no_context_trees):
    """Each form is emitted as before, in frend's own order (no context tree reorders
    it here; ``test_context.py`` covers the trees)."""
    emitted = [
        [(item.text, item.provenance) for item in unit.alternatives]
        for unit in _units(text, detectors)
    ]
    primary = key[0] if isinstance(key, tuple) else key
    assert emitted == _GOLDEN[(primary, text)]


def test_the_electronic_digit_forms_are_emitted_as_before():
    """``electronic.digit_forms`` reads ``zero.digit`` itself."""
    assert digit_forms("105", "en_US") == _GOLDEN_DIGIT_FORMS


def test_the_golden_readings_cover_every_consumer():
    assert set(_GOLDEN) == {
        (key[0] if isinstance(key, tuple) else key, text) for key, text, *_ in _CONSUMERS
    }


def test_every_read_form_has_a_consumer_here():
    en = set(_tables()["en"]["forms"])
    # The zero words and the vowels are read as sets, not emitted: tests below and
    # tests/test_letters.py::test_the_vowels_are_the_locales cover them.
    consumed = {
        item for key, *_ in _CONSUMERS for item in (key if isinstance(key, tuple) else (key,))
    }
    assert consumed | {"zero.words", "letter.vowels"} == en - _UNREAD


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
    expected = () if form is None else form if isinstance(form, tuple) else (form,)
    lexical = {spoken for spoken, provenance in with_table if _LEXICAL in provenance}
    assert lexical, (key, sorted(with_table))
    assert set(expected) <= lexical, (key, sorted(lexical))

    ru = dict(lexical_forms("ru_RU"))
    monkeypatch.setattr(verbalize_module, "_lexical_for", lambda locale: ru)
    without = _spoken(text, detectors)
    spoken_without = {spoken for spoken, _ in without}
    assert not {spoken for spoken, provenance in without if _LEXICAL in provenance}
    assert not lexical & spoken_without
    assert not set(expected) & spoken_without


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
    assert "its intervalFormats patterns hold no literal word" in why
    # The walk pins no pattern count, so the why states none.
    assert not re.search(r"\d+ intervalFormats", why)
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


def test_the_colon_separator_why_says_where_en_writes_a_colon():
    """en's calendar/gregorian writes ':' between numbers only between a time's fields
    (hour, minute, second); the why keeps kal's ruling and says that much, not that
    ICU writes ':' nowhere between numbers in en. (appendItems' ':' follows a
    field's name, as in "{0} ({2}: {1})".)"""
    gregorian = icu.ResourceBundle("", icu.Locale("en")).get("calendar").get("gregorian")
    patterns = dict(_walk(gregorian, "gregorian"))
    joined = {
        pair
        for path, pattern in patterns.items()
        if "/appendItems/" not in path
        for pair in re.findall(r"(.):(.)", _QUOTED.sub("", pattern))
    }
    assert joined and all(a in "HhKkms" and b in "HhKkms" for a, b in joined), joined
    assert any("h:mm:ss" in p for k, p in patterns.items() if "/DateTimePatterns/" in k)
    for key in ("Hm", "hm"):
        assert ":" in patterns[f"gregorian/availableFormats/{key}"]
        assert any(":" in p for k, p in patterns.items() if f"/intervalFormats/{key}/" in k)

    why = _tables()["en"]["forms"]["range.separator"]["why"]
    assert "ICU writes nowhere in en (kal, 2026-09-28)" in why
    assert "only in time patterns" in why
    assert "DateTimePatterns 'h:mm:ss'" in why
    assert "the Hm and hm availableFormats and intervalFormats" in why
    assert "1,419 'to', 41 silent" in why


def test_the_currency_why_says_what_cldr_names():
    """CLDR's en CurrencyPlurals name EUR bare ('euro', 'euros', as the table's EUR major
    row) and every other currency here with its region; no row's minor name is
    CLDR's. The why says so instead of claiming CLDR lacks every bare major name."""
    plurals = icu.ResourceBundle("ICUDATA-curr", icu.Locale("en")).get("CurrencyPlurals")
    units = _tables()["en"]["forms"]["currency.units"]["value"]

    def cldr(code):
        entry = plurals.get(code)
        names = {entry.get(i).getKey(): entry.get(i).getString() for i in range(entry.getSize())}
        return [names["one"], names["other"]]

    bare = {code for code, (major, _) in units.items() if cldr(code) == major}
    assert bare == {"EUR"}
    assert cldr("USD") == ["US dollar", "US dollars"]
    for code, (major, minor) in units.items():
        assert minor != cldr(code), code
        if code != "EUR":
            assert cldr(code)[0].endswith(" " + major[0]), (code, cldr(code))

    why = _tables()["en"]["forms"]["currency.units"]["why"]
    assert "('US dollar'/'US dollars', 'British pound', 'Japanese yen')" in why
    assert "except EUR's 'euro'/'euros'" in why
    assert "for the other nine currencies" in why and len(units) == 10
    assert "EUR's major row equals CLDR's" in why
    assert "not the corpus's bare major names" not in why
