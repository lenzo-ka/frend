"""Conservatively recognize dotted versions carrying an explicit ``v`` prefix."""

from __future__ import annotations

import re
from dataclasses import dataclass

from icukit.detectors import Capture

from frend.locale_data import canonical_locale

__all__ = ["VersionDetector", "VersionValue"]


_VERSION = re.compile(r"(?<![\w.])(?P<prefix>[vV])(?P<body>[0-9]+(?:\.[0-9]+){2,})(?!\w|\.\w)")
_PART = re.compile(r"[0-9]+|\.")


@dataclass(frozen=True)
class VersionValue:
    """A version prefix and its numeric components in written order."""

    prefix: str
    components: tuple[str, ...]


class VersionDetector:
    """Detect three-or-more-part ASCII dotted versions prefixed by ``v`` or ``V``."""

    def __init__(self, locale: str = "en_US") -> None:
        self.locale = canonical_locale(locale)

    def detect(self, text: str) -> list[dict]:
        found = []
        for match in _VERSION.finditer(text):
            surface = match.group()
            components = tuple(match.group("body").split("."))
            captures = [
                Capture(
                    "version",
                    match.start(),
                    match.end(),
                    surface,
                    components,
                    "version",
                )
            ]
            for part in _PART.finditer(surface):
                start = match.start() + part.start()
                end = match.start() + part.end()
                name = "separator" if part.group() == "." else "component"
                captures.append(Capture(name, start, end, part.group(), part.group(), "numeric"))
            prefix_end = match.start() + 1
            captures.insert(
                1,
                Capture(
                    "prefix",
                    match.start(),
                    prefix_end,
                    match.group("prefix"),
                    match.group("prefix"),
                    "letters",
                ),
            )
            found.append(
                {
                    "text": surface,
                    "start": match.start(),
                    "end": match.end(),
                    "type": "version:dotted",
                    "value": VersionValue(match.group("prefix"), components),
                    "captures": tuple(captures),
                }
            )
        return found
