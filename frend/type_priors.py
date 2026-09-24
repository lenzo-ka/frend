"""Corpus base-rate prior over reading types, keyed on surface shape.

This is Layer 2 of the resolver: an empirical tiebreak that runs *after* geometry
has already chosen which spans form a cover, among covers that geometry left
exactly tied. It answers one question -- when the same span survives as readings
of different semantic type (``date:Md`` vs ``number:fraction`` for ``"3/24"``),
which type does a corpus of written surfaces attest more often for that shape?

The table is built offline from a corpus by ``tools/build_type_priors.py`` and
vendored as ``frend/data/type_priors.json`` (raw ``shape -> class -> count``). The
corpus itself is eval-only and never shipped; only the counts are. Runtime loads
the counts once and derives ``P(class | shape)`` and the sample size ``n(shape)``
on demand.

Honesty is load-bearing here (see the resolver's post-rank composition):

* No smoothing. A count is a count; a zero is a zero. Nothing is invented.
* The ``n`` sample size travels with every prior. A base rate is *not* a
  confidence -- ``p`` is "how often this shape read this way in the corpus",
  carried with the ``n`` it was measured over, never a probability of being
  correct.
* ``MIN_N`` floor: a shape seen fewer than ``MIN_N`` times is too sparse to have
  an opinion, so its lookups withhold entirely (return ``None``) rather than
  speak from one or two examples.

A reading is either *supported* or *unsupported* -- a binary support state, never
a fabricated value:

* *Supported* -- the reading's type maps to a corpus class, that class is present
  under the shape, the shape is well sampled (``n >= MIN_N``), and the class count
  is positive. The reading carries a finite ``log P(class | shape)`` in
  ``[log(1/n), 0]`` and its ``p`` is a real base rate over ``n``.
* *Unsupported* -- anything else: an unmapped type, a shape absent from the table
  or below ``MIN_N``, or a well-sampled shape whose class count is exactly zero
  (*attested-zero*). No value is asserted. An attested-zero is treated as "no
  positive base rate to assert" -- **not** impossibility and **not** a fabricated
  small ``p``. It withholds exactly as absence does, so absence can never beat
  measured evidence and a small-sample zero can never fabricate ``-Infinity`` and
  poison a whole cover.

The resolver consumes this per span: within one identical-span equivalence class
the type at each span is chosen independently, ranking that span's candidate
readings with supported readings (by base rate ``p`` descending) above all
unsupported ones. A supported reading therefore beats an unsupported one; two
unsupported readings, or two supported readings sharing the top ``p``, tie
honestly and leave the span ambiguous. Attested-zero is still *observable*: its
:class:`ReadingPrior` reports ``supported=False`` with a known sample size ``n``
(the shape was sampled) and ``p=None``, distinct from a truly absent shape
(``n=None``).

KNOWN-HONEST LIMITATION (a property of how text-normalization corpora write
dates, not of this design): the vendored Google-TN table has no ``N/N`` dates --
every slashed date it contains is ``N/N/N`` (and ``N.N.N`` dates are all-date),
while bare ``N/N`` is attested only as ``fraction`` (n ~1e5, 100% fraction) -- so
``date`` is attested-zero (hence unsupported) on ``"N/N"`` while ``fraction`` is
supported, and Layer 2 alone always ranks ``fraction`` over ``date`` on a bare
``"3/24"``. The earlier NeMo placeholder had this same gap; the corpus swap did
*not* close it, because dates are simply not written ``N/N`` in running text. This
is exactly the gap Layer-3 context cues (e.g. "on 3/24") and reflective,
ICU-generated shape backfill are meant to close. Nothing downstream assumes a
specific corpus.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from frend.shape import shape

__all__ = [
    "MIN_N",
    "CorpusPrior",
    "BlendedPrior",
    "FeatureSource",
    "PriorTable",
    "IcuBackfillTable",
    "ReadingFeature",
    "ReadingPrior",
    "ResolveContext",
    "corpus_classes",
    "corpus_group",
    "load_prior_table",
    "load_icu_backfill_table",
]

# A shape seen fewer than this many times in the corpus is too sparse to speak.
MIN_N = 3

_DATA = Path(__file__).parent / "data" / "type_priors.json"
_ICU_DATA = Path(__file__).parent / "data" / "icu_shape_backfill.json"

# A detection is a mapping (icukit's ValueDetection is a TypedDict); only ``type``
# and ``text`` are read here, and the whole detection is otherwise untouched.
Detection = Mapping[str, Any]


def corpus_group(type_: str) -> str | None:
    """Map a reading ``type`` to the coarse corpus group label it priors under.

    Keyed on the ``group:subtype`` prefix, so ``date:MMMd`` and ``date:yMMM``
    share the label ``"date"`` and therefore an identical prior -- the corpus
    cannot break a same-group sub-type tie, which is correct. Returns ``None``
    for a type that maps to no corpus class.
    """
    return _GROUP_LABELS.get(_classify(type_))


def corpus_classes(type_: str) -> tuple[str, ...]:
    """Map a reading ``type`` to the corpus classes whose counts it sums over.

    ``number:decimal*`` sums ``cardinal`` and ``decimal`` (the shape ``N`` vs
    ``N.N`` already separates them); every other family maps to a single class.
    Returns ``()`` for a type with no corpus support.
    """
    return _classify(type_)


def _classify(type_: str) -> tuple[str, ...]:
    head, _, tail = type_.partition(":")
    if type_ == "letter:name":
        return ("letters",)
    if type_ == "word:single-letter":
        return ("plain",)
    if head == "date":
        return ("date",)
    if head == "time":
        return ("time",)
    if head == "ordinal":
        return ("ordinal",)
    if head == "fraction":
        return ("fraction",)
    if head == "measure":
        return ("measure",)
    if head == "number":
        if tail.startswith("plural"):
            # "1990s", "'90s": the corpus files decades and centuries as DATE.
            return ("date",)
        if tail.startswith("currency"):
            return ("money",)
        if tail.startswith("percent"):
            return ("percent",)
        if tail.startswith("fraction"):
            return ("fraction",)
        if tail.startswith("decimal"):
            return ("cardinal", "decimal")
        if tail.startswith("cardinal") or tail.startswith("int"):
            return ("cardinal",)
    return ()


# The stable label a class tuple is reported under on a ReadingPrior.
_GROUP_LABELS: dict[tuple[str, ...], str] = {
    ("letters",): "letter",
    ("plain",): "word",
    ("date",): "date",
    ("time",): "time",
    ("ordinal",): "ordinal",
    ("fraction",): "fraction",
    ("money",): "money",
    ("cardinal",): "cardinal",
    ("cardinal", "decimal"): "number",
    ("measure",): "measure",
    ("percent",): "percent",
}


@dataclass(frozen=True)
class ReadingPrior:
    """The corpus base rate observed for one reading, fully decomposed, honest.

    ``supported`` is the binary support state (see the module docstring). When
    ``supported`` is ``True``, ``p`` is the empirical base rate
    ``count(group, shape) / n(shape)`` -- a base rate, not a confidence -- and
    ``n`` is the sample size it was measured over. When ``supported`` is
    ``False``, ``p`` is ``None`` (no positive base rate is asserted); ``n`` is
    still the shape's sample size for an *attested-zero* (a well-sampled shape
    whose class count is zero), and ``None`` for a truly absent or too-sparse
    shape. No value is ever fabricated.
    """

    group: str
    shape: str
    p: Decimal | None
    n: int | None
    supported: bool
    tier: Literal["measured", "icu-backfill", "unsupported"] = "unsupported"
    provenance: str = "measured"
    generated_p: Decimal | None = None


@dataclass(frozen=True)
class ReadingFeature:
    """One named, signed log-weight contributed to a reading's prior score.

    The generic currency of the tiebreak axis (Layer 2 today, Layer-3 cues
    later). ``contribution`` is added into the cover's prior score; ``n`` is the
    sample size when the feature is corpus-derived, or ``None`` for a rule cue
    that is not a base rate.
    """

    name: str
    contribution: Decimal
    detail: str
    n: int | None


@dataclass(frozen=True)
class ResolveContext:
    """Read-only neighborhood a feature source may consult.

    A ``Detection.text`` is only the span surface; a context-sensitive cue source
    (Layer 3) needs the surrounding text and the sibling detections to read, for
    example, a month name preceding a slashed date. The corpus prior ignores it.
    """

    source_text: str | None
    all_detections: tuple[Detection, ...]


@runtime_checkable
class FeatureSource(Protocol):
    """Produces the named log-weight features a single reading earns."""

    def features(
        self, detection: Detection, context: ResolveContext
    ) -> tuple[ReadingFeature, ...]: ...


class PriorTable:
    """An immutable view over vendored ``shape -> class -> count`` corpus counts."""

    def __init__(self, counts: Mapping[str, Mapping[str, int]], provenance: Mapping[str, Any]):
        self._counts: dict[str, dict[str, int]] = {
            sh: dict(by_class) for sh, by_class in counts.items()
        }
        self._validate()
        self._n: dict[str, int] = {sh: sum(by.values()) for sh, by in self._counts.items()}
        self.provenance = dict(provenance)

    def _validate(self) -> None:
        """Reject a malformed table: keys must be strings, counts must be
        non-negative integers, and every shape's total must be positive, so a
        stored shape can never carry an empty or negative sample that would corrupt
        ``n`` or a base rate, nor a non-string key that would never match a lookup."""
        for shape_key, by_class in self._counts.items():
            if not isinstance(shape_key, str):
                raise ValueError(f"prior table shape key is not a string: {shape_key!r}")
            total = 0
            for class_name, count in by_class.items():
                if not isinstance(class_name, str):
                    raise ValueError(
                        f"prior table class key under shape {shape_key!r} is not a "
                        f"string: {class_name!r}"
                    )
                if not isinstance(count, int) or isinstance(count, bool):
                    raise ValueError(
                        f"prior table count for shape {shape_key!r} class "
                        f"{class_name!r} is not an integer: {count!r}"
                    )
                if count < 0:
                    raise ValueError(
                        f"prior table count for shape {shape_key!r} class "
                        f"{class_name!r} is negative: {count!r}"
                    )
                total += count
            if total <= 0:
                raise ValueError(f"prior table shape {shape_key!r} has non-positive total {total}")

    def n(self, shape_key: str) -> int | None:
        """Sample size for ``shape_key``, or ``None`` if absent or below the floor."""
        total = self._n.get(shape_key)
        if total is None or total < MIN_N:
            return None
        return total

    def has_raw_shape(self, shape_key: str) -> bool:
        """Whether the measured artifact contains any raw row for the shape."""
        return shape_key in self._counts

    def prior(self, classes: Sequence[str], shape_key: str) -> tuple[Decimal, int] | None:
        """Return ``(p, n)`` for the summed ``classes`` under ``shape_key``.

        ``None`` when the shape has no usable support (absent or below
        :data:`MIN_N`). ``p`` may be exactly ``0`` -- an attested-zero reading --
        which is a real observation, not absence; the support decision is made in
        :meth:`reading_prior`.
        """
        total = self.n(shape_key)
        if total is None:
            return None
        by_class = self._counts[shape_key]
        matched = sum(by_class.get(c, 0) for c in classes)
        return (Decimal(matched) / Decimal(total), total)

    def reading_prior(self, detection: Detection) -> ReadingPrior | None:
        """The :class:`ReadingPrior` for ``detection``, or ``None`` when its type
        maps to no corpus class at all.

        A mapped type always yields a ReadingPrior, carrying the binary support
        state. Supported: ``p > 0`` over a well-sampled shape. Unsupported: an
        attested-zero (``p=None`` but ``n`` known, the shape was sampled) or an
        absent/too-sparse shape (``p=None`` and ``n=None``). Nothing is fabricated.
        """
        type_ = str(detection.get("type", ""))
        classes = corpus_classes(type_)
        if not classes:
            return None
        shape_key = shape(str(detection.get("text", "")))
        group = corpus_group(type_) or shape_key
        total = self.n(shape_key)
        if total is None:
            # Shape absent from the table or below MIN_N: no sample to speak from.
            return ReadingPrior(
                group=group,
                shape=shape_key,
                p=None,
                n=None,
                supported=False,
                tier="measured" if self.has_raw_shape(shape_key) else "unsupported",
                provenance="measured",
            )
        by_class = self._counts[shape_key]
        matched = sum(by_class.get(c, 0) for c in classes)
        if matched == 0:
            # Attested-zero: well-sampled shape, class never occurs. Not impossible,
            # just no positive base rate to assert; the sample size stays observable.
            return ReadingPrior(
                group=group,
                shape=shape_key,
                p=None,
                n=total,
                supported=False,
                tier="measured",
                provenance="measured",
            )
        return ReadingPrior(
            group=group,
            shape=shape_key,
            p=Decimal(matched) / Decimal(total),
            n=total,
            supported=True,
            tier="measured",
            provenance="measured",
        )


@lru_cache(maxsize=1)
def load_prior_table(path: str | None = None) -> PriorTable:
    """Load the vendored prior table once (cached)."""
    source = Path(path) if path is not None else _DATA
    raw = json.loads(source.read_text(encoding="utf-8"))
    return PriorTable(raw.get("counts", {}), raw.get("provenance", {}))


class IcuBackfillTable:
    """Immutable ICU-generated estimates of ``P(shape | class)``."""

    def __init__(
        self,
        counts_by_class: Mapping[str, Mapping[str, int]],
        totals_by_class: Mapping[str, int],
        provenance: Mapping[str, Any],
    ):
        self._counts = {class_: dict(counts) for class_, counts in counts_by_class.items()}
        self._totals = dict(totals_by_class)
        self.provenance = dict(provenance)
        for class_, counts in self._counts.items():
            total = self._totals.get(class_)
            if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
                raise ValueError(f"ICU backfill total for {class_!r} must be a positive integer")
            for shape_key, count in counts.items():
                if not isinstance(shape_key, str) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"invalid ICU backfill count for {class_!r}: {shape_key!r}")

    def has(self, class_: str, shape_key: str) -> bool:
        """Whether ICU generated a positive count for this class and shape."""
        return self._counts.get(class_, {}).get(shape_key, 0) > 0

    def p_shape_given_class(self, class_: str, shape_key: str) -> Decimal | None:
        """Return exact per-class-normalized ``P(shape | class)`` when present."""
        if not self.has(class_, shape_key):
            return None
        return Decimal(self._counts[class_][shape_key]) / Decimal(self._totals[class_])


@lru_cache(maxsize=1)
def load_icu_backfill_table(path: str | None = None) -> IcuBackfillTable:
    """Load the vendored generated estimate once (cached)."""
    source = Path(path) if path is not None else _ICU_DATA
    raw = json.loads(source.read_text(encoding="utf-8"))
    return IcuBackfillTable(
        raw.get("counts_by_class", {}),
        raw.get("totals_by_class", {}),
        raw.get("provenance", {}),
    )


class BlendedPrior:
    """Measured-first prior with ICU backfill only for corpus-silent shapes."""

    def __init__(
        self,
        measured: PriorTable | None = None,
        backfill: IcuBackfillTable | None = None,
        *,
        class_prior: Mapping[str, Decimal | int] | None = None,
        class_prior_source: str | None = None,
    ):
        self.measured = measured if measured is not None else load_prior_table()
        self.backfill = backfill if backfill is not None else load_icu_backfill_table()
        self.class_prior = (
            None
            if class_prior is None
            else {key: Decimal(value) for key, value in class_prior.items()}
        )
        if class_prior is None:
            if class_prior_source not in (None, "flat-policy"):
                raise ValueError("class_prior_source requires a supplied class_prior")
            self.class_prior_source = "flat-policy"
        else:
            self.class_prior_source = class_prior_source or "supplied"
            if self.class_prior_source == "flat-policy":
                raise ValueError("a supplied class_prior cannot use source 'flat-policy'")
            if not self.class_prior_source:
                raise ValueError("class_prior_source must be non-empty")
        if self.class_prior is not None and any(value < 0 for value in self.class_prior.values()):
            raise ValueError("class prior weights must be non-negative")

    def class_weight(self, group: str) -> Decimal:
        return Decimal(1) if self.class_prior is None else self.class_prior.get(group, Decimal(0))

    def reading_prior(self, detection: Detection) -> ReadingPrior | None:
        type_ = str(detection.get("type", ""))
        classes = corpus_classes(type_)
        shape_key = shape(str(detection.get("text", "")))
        group = corpus_group(type_) or type_.partition(":")[0] or shape_key
        if not classes:
            return ReadingPrior(group, shape_key, None, None, False, "unsupported", "unsupported")
        measured_prior = self.measured.reading_prior(detection)
        if self.measured.has_raw_shape(shape_key):
            assert measured_prior is not None
            return measured_prior
        likelihoods = [
            p
            for class_ in classes
            if (p := self.backfill.p_shape_given_class(class_, shape_key)) is not None
        ]
        if likelihoods:
            likelihood = sum(likelihoods, Decimal(0))
            return ReadingPrior(
                group,
                shape_key,
                likelihood,
                None,
                True,
                "icu-backfill",
                f"icu-backfill:{self.class_prior_source}",
                likelihood,
            )
        return ReadingPrior(group, shape_key, None, None, False, "unsupported", "unsupported")

    def features(self, detection: Detection, context: ResolveContext) -> tuple[ReadingFeature, ...]:
        del detection, context
        return ()


class CorpusPrior:
    """The Layer-2 feature source: a base-rate log-weight per supported reading."""

    def __init__(self, table: PriorTable | None = None):
        self._table = table if table is not None else load_prior_table()

    def reading_prior(self, detection: Detection) -> ReadingPrior | None:
        """Expose the decomposed :class:`ReadingPrior`, or ``None`` for no support."""
        return self._table.reading_prior(detection)

    def features(self, detection: Detection, context: ResolveContext) -> tuple[ReadingFeature, ...]:
        """Return the single base-rate feature for a *supported* ``detection``, or
        ``()``.

        Only a supported reading (positive base rate over a well-sampled shape)
        contributes a feature; an unsupported reading -- unmapped, absent,
        too-sparse, or attested-zero -- contributes nothing, so the resolver's
        per-span ranking treats it as unsupported and never fabricates a value. The
        single feature's ``log P`` is monotonic in the base rate, so ranking a
        span's supported readings by score is ranking them by ``p``. Context is
        ignored: a base rate depends only on the reading's own type and shape.
        """
        del context
        prior = self._table.reading_prior(detection)
        if prior is None or not prior.supported:
            return ()
        assert prior.p is not None and prior.n is not None  # invariant of supported
        # A supported reading has p > 0 (matched >= 1). float(p) can nonetheless
        # underflow to 0.0 for an astronomically large sample (total > ~1e308),
        # which would make math.log raise a domain error. Fall back to Decimal's
        # exact ln in that boundary case so the log stays finite and correct.
        p_float = float(prior.p)
        if p_float > 0.0:
            contribution = Decimal(str(math.log(p_float)))
        else:
            contribution = prior.p.ln()
        detail = f"base rate {prior.p} of n={prior.n} for {prior.group} on shape {prior.shape!r}"
        return (
            ReadingFeature(
                name="base_rate",
                contribution=contribution,
                detail=detail,
                n=prior.n,
            ),
        )
