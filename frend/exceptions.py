"""Opt-in access to frend's curated ICU break-exception layer."""

from __future__ import annotations

from importlib.resources import files
from json import loads
from typing import cast

from icukit.exceptions import (
    ExceptionInventory,
    LoadedExceptionInventory,
    compose_inventories,
)


def english_break_exceptions() -> LoadedExceptionInventory:
    """Load, compose, and witness-test frend's curated English layer."""
    resource = files("frend").joinpath("data/exceptions/en.json")
    layer = cast(ExceptionInventory, loads(resource.read_text(encoding="utf-8")))
    return compose_inventories([layer])


__all__ = ["english_break_exceptions"]
