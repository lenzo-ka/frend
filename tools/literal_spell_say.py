"""Shared builder-only classification of literal spell-or-say corpus rows."""

from __future__ import annotations

from collections.abc import Callable

import icu

from frend.letters import spelled
from frend.spoken_priors import normalize_spoken
from tools.google_tn_rows import expected

_NFC = icu.Normalizer2.getNFCInstance()


def normalize_literal(text: str) -> str:
    return _NFC.normalize(text)


def literal_spell_say(
    corpus_class: str,
    written: str,
    spoken: str,
    locale: str,
    *,
    eligible: Callable[[str, str], bool],
) -> str | None:
    """Classify one NFC-normalized row inside the caller's declared population."""
    if corpus_class not in {"LETTERS", "PLAIN"}:
        return None
    written = normalize_literal(written)
    if not eligible(written, locale):
        return None
    target = expected(corpus_class, written, spoken)
    if corpus_class == "PLAIN" and target == normalize_spoken(written):
        return "say"
    letter_form = spelled(written, locale)
    if (
        corpus_class == "LETTERS"
        and letter_form is not None
        and target == normalize_spoken(letter_form.text)
    ):
        return "spell"
    return None
