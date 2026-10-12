"""Bounded incremental plain-text normalization."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from icukit import ReflowMode

from frend.input_folds import InputFold, _stable_fold_prefix_length, apply_input_fold
from frend.input_limits import (
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_UNIT_CHARS,
    validate_input,
    validate_unit_length,
)
from frend.locale_data import canonical_locale
from frend.normalize import (
    NormalizedUnit,
    SentenceBase,
    _date_detector_gang,
    _protected_sentence_spans,
    _reading_detectors,
    _sentence,
    _sentence_override,
)
from frend.profiles import GroupOrders, validate_groups, validate_profile
from frend.symbols import DEFAULT_SYMBOL_RUN_THRESHOLD, SymbolDetector, _code_ranges
from frend.work_budgets import (
    DEFAULT_TIERGRAPH_WORK_BUDGET,
    TiergraphWorkBudget,
    work_meter,
)

__all__ = ["NormalizationBatch", "NormalizationStream"]


@dataclass(frozen=True)
class NormalizationBatch:
    """Newly settled spoken text, absolute alignments, and source watermark."""

    text: str
    units: tuple[NormalizedUnit, ...]
    pending_from: int


class NormalizationStream:
    """Normalize arbitrary chunks after icukit's sentence decisions settle.

    Sentence state is held by icukit's linear incremental breaker. Each settled
    sentence is recognized through the retained gang's ``DetectionStream`` before
    frend resolves and verbalizes it. Returned spans are absolute across the stream.
    """

    def __init__(
        self,
        *,
        locale: str = "en_US",
        profile: str | None = None,
        case_variant_lookup: bool = False,
        fold: InputFold | None = "typographic",
        offsets: bool = False,
        max_input_chars: int | None = DEFAULT_MAX_INPUT_CHARS,
        max_unit_chars: int | None = DEFAULT_MAX_UNIT_CHARS,
        sentence_base: SentenceBase | None = None,
        max_sentence_chars: int | None = 1024,
        reflow: ReflowMode = "none",
        symbol_run_threshold: int = DEFAULT_SYMBOL_RUN_THRESHOLD,
        groups: GroupOrders | None = None,
        work_budget: TiergraphWorkBudget = DEFAULT_TIERGRAPH_WORK_BUDGET,
    ) -> None:
        if not isinstance(offsets, bool):
            raise TypeError(f"offsets must be a bool, got {type(offsets).__name__}")
        if not isinstance(case_variant_lookup, bool):
            raise TypeError(
                f"case_variant_lookup must be a bool, got {type(case_variant_lookup).__name__}"
            )
        validate_input("", max_input_chars=max_input_chars)
        validate_unit_length(0, max_unit_chars=max_unit_chars)
        SymbolDetector("root", run_threshold=symbol_run_threshold)
        locale = canonical_locale(locale)
        self.locale = locale
        self.profile = validate_profile(profile)
        self.case_variant_lookup = case_variant_lookup
        self.fold = fold
        self.offsets = offsets
        self.max_input_chars = max_input_chars
        self.max_unit_chars = max_unit_chars
        self.symbol_run_threshold = symbol_run_threshold
        self.groups = None if groups is None else dict(validate_groups(groups) or ())
        self.work_budget = work_meter(work_budget)
        self._sentences = _sentence_override(
            locale, sentence_base, max_sentence_chars, reflow
        ).stream(protection="watermark")
        protection_cap = max_sentence_chars or max_unit_chars or 4096
        self._protection_stream = _date_detector_gang(locale).stream(
            boundary="paragraph",
            max_pending_chars=protection_cap,
        )
        self._source = ""
        self._folded = ""
        self._source_from = 0
        self._fold_fed = 0
        self._breaker_fed = 0
        self._total = 0
        self._previous_content_end = 0
        self._has_content = False
        self._output_at = 0
        self._closed = False

    def _normalize_range(
        self,
        absolute_start: int,
        absolute_end: int,
        symbol_code_ranges: tuple[tuple[int, int], ...],
    ) -> tuple[str, list[NormalizedUnit]]:
        local_end = absolute_end - self._source_from
        folded_span = self._folded[:local_end]
        raw_span = self._source[:local_end]
        left = len(folded_span) - len(folded_span.lstrip())
        right = len(folded_span.rstrip())
        if left >= right:
            return "", []
        text = folded_span[left:right]
        raw_text = raw_span[left:right]
        content_start = absolute_start + left
        content_end = absolute_start + right
        registry = _reading_detectors(self.locale, self.symbol_run_threshold)
        detection_stream = registry.stream(
            boundary="explicit",
            max_pending_chars=max(1, len(text) + 1),
        )
        first = detection_stream.feed(text)
        last = detection_stream.close()
        detections = [*first.detections, *last.detections]
        rendered, spoken = _sentence(
            text,
            source_offset=content_start,
            symbol_code_ranges=symbol_code_ranges,
            raw_text=raw_text,
            fold=self.fold,
            locale=self.locale,
            profile=self.profile,
            case_variant_lookup=self.case_variant_lookup,
            offsets=self.offsets,
            max_input_chars=self.max_input_chars,
            max_unit_chars=self.max_unit_chars,
            symbol_run_threshold=self.symbol_run_threshold,
            groups=self.groups,
            work_budget=self.work_budget,
            detections=detections,
        )
        separator = (
            " " if self._previous_content_end and content_start > self._previous_content_end else ""
        )
        batch_text = separator + spoken
        units: list[NormalizedUnit] = []
        if self.offsets:
            if not self._has_content and content_start:
                units.append(
                    NormalizedUnit(
                        (self._output_at, self._output_at),
                        (0, content_start),
                        None,
                        "surface:boundary-whitespace",
                    )
                )
            if separator:
                units.append(
                    NormalizedUnit(
                        (self._output_at, self._output_at + 1),
                        (self._previous_content_end, content_start),
                        None,
                        "surface:inter-sentence-whitespace",
                    )
                )
            assert rendered is not None
            at = self._output_at + len(separator)
            for unit_text, edge, unit in rendered:
                end = at + len(unit_text)
                reader = None if edge.detection is None else str(edge.detection.get("type"))
                units.append(
                    NormalizedUnit(
                        (at, end),
                        (content_start + edge.start, content_start + edge.end),
                        reader,
                        unit.best.provenance,
                    )
                )
                at = end
        self._previous_content_end = content_end
        self._has_content = True
        self._output_at += len(batch_text)
        return batch_text, units

    def _settle(self, decisions) -> NormalizationBatch:
        parts: list[str] = []
        units: list[NormalizedUnit] = []
        settled = self._source_from
        symbol_code_ranges = tuple(
            (self._source_from + start, self._source_from + end)
            for start, end in _code_ranges(self._folded)
        )
        for decision in decisions:
            if decision["decision"] != "break":
                continue
            end = int(decision["offset"])
            if end <= settled:
                continue
            text, aligned = self._normalize_range(settled, end, symbol_code_ranges)
            parts.append(text)
            units.extend(aligned)
            consumed = end - self._source_from
            self._source = self._source[consumed:]
            self._folded = self._folded[consumed:]
            self._source_from = end
            settled = end
        return NormalizationBatch("".join(parts), tuple(units), self._source_from)

    @staticmethod
    def _combine(*batches: NormalizationBatch) -> NormalizationBatch:
        return NormalizationBatch(
            "".join(batch.text for batch in batches),
            tuple(unit for batch in batches for unit in batch.units),
            batches[-1].pending_from,
        )

    def _feed_protected(self, batch) -> NormalizationBatch:
        """Advance sentence decisions only through date-detection-settled input."""
        end = int(batch.pending_from)
        if end <= self._breaker_fed:
            return NormalizationBatch("", (), self._source_from)
        local_start = self._breaker_fed - self._source_from
        local_end = end - self._source_from
        text = self._folded[local_start:local_end]
        protected = _protected_sentence_spans(
            text,
            self.locale,
            offset=self._breaker_fed,
            date_detections=batch.detections,
        )
        decisions = self._sentences.feed(
            text,
            protected=protected,
            protected_through=end,
        )
        self._breaker_fed = end
        return self._settle(decisions)

    def _feed_stable_fold(self, *, final: bool = False) -> NormalizationBatch:
        """Feed only folded source whose contextual mapping can no longer change."""
        self._folded = apply_input_fold(self._source, self.fold)
        local_end = (
            len(self._source) if final else _stable_fold_prefix_length(self._source, self.fold)
        )
        end = self._source_from + local_end
        if end <= self._fold_fed:
            return NormalizationBatch("", (), self._source_from)
        local_start = self._fold_fed - self._source_from
        chunk = self._folded[local_start:local_end]
        self._fold_fed = end
        return self._feed_protected(self._protection_stream.feed(chunk))

    def _snapshot(self) -> dict[str, object]:
        """Return enough detached state to make one public operation atomic."""
        snapshot = self.__dict__.copy()
        snapshot["_sentences"] = copy.deepcopy(self._sentences)
        snapshot["_protection_stream"] = copy.copy(self._protection_stream)
        snapshot["work_budget"] = copy.deepcopy(self.work_budget)
        return snapshot

    def _restore(self, snapshot: dict[str, object]) -> None:
        self.__dict__.clear()
        self.__dict__.update(snapshot)

    def feed(self, chunk: str, /) -> NormalizationBatch:
        """Append a source chunk and return newly immutable normalized output."""
        if self._closed:
            raise RuntimeError("normalization stream is closed")
        validated = validate_input(chunk, max_input_chars=None)
        assert validated is not None
        prospective = self._total + len(chunk)
        validate_unit_length(
            prospective,
            max_unit_chars=None,
            max_input_chars=self.max_input_chars,
        )
        snapshot = self._snapshot()
        try:
            self._source += chunk
            self._total = prospective
            return self._feed_stable_fold()
        except Exception:
            self._restore(snapshot)
            raise

    def flush(self) -> NormalizationBatch:
        """Settle the current logical segment without closing the stream."""
        if self._closed:
            raise RuntimeError("normalization stream is closed")
        snapshot = self._snapshot()
        try:
            folded = self._feed_stable_fold(final=True)
            settled = self._feed_protected(self._protection_stream.flush())
            flushed = self._settle(self._sentences.flush())
            return self._combine(folded, settled, flushed)
        except Exception:
            self._restore(snapshot)
            raise

    def close(self) -> NormalizationBatch:
        """Settle all remaining input and reject later input."""
        if self._closed:
            return NormalizationBatch("", (), self._source_from)
        snapshot = self._snapshot()
        try:
            folded = self._feed_stable_fold(final=True)
            settled = self._feed_protected(self._protection_stream.close())
            closed = self._settle(self._sentences.close())
            result = self._combine(folded, settled, closed)
            if self.offsets and self._previous_content_end < self._total:
                result = NormalizationBatch(
                    result.text,
                    (
                        *result.units,
                        NormalizedUnit(
                            (self._output_at, self._output_at),
                            (self._previous_content_end, self._total),
                            None,
                            "surface:boundary-whitespace",
                        ),
                    ),
                    result.pending_from,
                )
            self._closed = True
            return result
        except Exception:
            self._restore(snapshot)
            raise
