"""Incremental normalization uses icukit's bounded sentence and detection streams."""

from __future__ import annotations

import importlib

import pytest
from icukit import RegexLimitError

import frend


@pytest.mark.parametrize("reflow", ["none", "lines", "paragraphs", "auto"])
@pytest.mark.parametrize("width", [1, 7, 64])
def test_stream_chunkings_equal_whole_text(reflow, width):
    source = "Dr. Jones paid $12.50 on Jan. 3.\nThe next line contains 45% and 5 ft. of material."
    stream = frend.NormalizationStream(reflow=reflow, offsets=True)
    batches = [stream.feed(source[start : start + width]) for start in range(0, len(source), width)]
    batches.append(stream.close())

    expected = frend.normalize(source, reflow=reflow, offsets=True)
    assert "".join(batch.text for batch in batches) == expected.text
    units = [unit for batch in batches for unit in batch.units]
    assert units == expected.units
    assert all(
        left.output_span[1] == right.output_span[0]
        for left, right in zip(units, units[1:], strict=False)
    )
    assert all(0 <= unit.source_span[0] <= unit.source_span[1] <= len(source) for unit in units)
    assert batches[-1].pending_from == len(source)


def test_stream_waits_for_date_protection_before_sentence_decisions():
    source = "날짜는 2024. 6. 30. 입니다. 다음."
    expected = frend.normalize(source, locale="ko_KR", offsets=True)
    stream = frend.NormalizationStream(locale="ko_KR", offsets=True)
    batches = [stream.feed(character) for character in source]
    batches.append(stream.close())

    assert "".join(batch.text for batch in batches) == expected.text
    assert [unit for batch in batches for unit in batch.units] == expected.units


def test_stream_canonicalizes_locale_alias_once():
    source = "El precio es 3,50 €."
    stream = frend.NormalizationStream(locale="ES-es")

    assert stream.locale == "es_ES"
    assert stream.feed(source).text + stream.close().text == frend.normalize(source, locale="es_ES")


@pytest.mark.parametrize("source", ["  One.  ", " \t\n"])
def test_stream_offsets_cover_discarded_boundary_whitespace(source):
    stream = frend.NormalizationStream(offsets=True)
    batches = [stream.feed(source[:2]), stream.feed(source[2:]), stream.close()]

    expected = frend.normalize(source, offsets=True)
    assert "".join(batch.text for batch in batches) == expected.text
    assert [unit for batch in batches for unit in batch.units] == expected.units


@pytest.mark.parametrize(
    "chunks",
    [
        ("1，", "2."),
        ("1", "，", "2."),
        ("1ｋ", "ｇ."),
        ("1", "ｋ", "ｇ."),
    ],
)
def test_stream_fold_is_identical_across_context_sensitive_chunk_boundaries(chunks):
    source = "".join(chunks)
    stream = frend.NormalizationStream(offsets=True)
    batches = [stream.feed(chunk) for chunk in chunks]
    batches.append(stream.close())

    expected = frend.normalize(source, offsets=True)
    assert "".join(batch.text for batch in batches) == expected.text
    assert [unit for batch in batches for unit in batch.units] == expected.units


def test_stream_retention_stays_bounded_on_unterminated_input():
    stream = frend.NormalizationStream(max_sentence_chars=1024)
    peak = 0
    for _ in range(256):
        stream.feed("x" * 1024)
        peak = max(peak, len(stream._source), len(stream._folded))
    stream.close()

    assert peak <= 2048


def test_stream_uses_retained_detection_stream(monkeypatch):
    normalization = importlib.import_module("frend.normalize")
    calls = 0
    original = normalization._DetectorRegistry.stream

    def recording(self, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, **kwargs)

    monkeypatch.setattr(normalization._DetectorRegistry, "stream", recording)
    stream = frend.NormalizationStream()
    stream.feed("One. Two.")
    stream.close()

    assert calls >= 1


def test_stream_propagates_regex_limit_error_without_retry(monkeypatch):
    stream_module = importlib.import_module("frend.stream")
    error = RegexLimitError("time", 50, "detection stream")
    calls = 0

    class FailingStream:
        def feed(self, _text):
            nonlocal calls
            calls += 1
            raise error

    class FailingRegistry:
        def stream(self, **_kwargs):
            return FailingStream()

    monkeypatch.setattr(stream_module, "_date_detector_gang", lambda _locale: FailingRegistry())
    stream = frend.NormalizationStream()

    with pytest.raises(RegexLimitError) as caught:
        stream.feed("One.")
    assert caught.value is error
    assert calls == 1


def test_stream_is_closed_transactionally_after_close():
    stream = frend.NormalizationStream(max_sentence_chars=8)
    stream.feed("word " * 20)
    stream.close()

    with pytest.raises(RuntimeError, match="closed"):
        stream.feed("later")
