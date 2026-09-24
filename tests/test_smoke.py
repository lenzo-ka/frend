"""Smoke test for the frend package skeleton."""

from frend import __version__


def test_version_is_a_nonempty_string():
    assert isinstance(__version__, str)
    assert __version__
