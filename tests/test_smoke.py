"""Smoke test for the irn package skeleton."""

from irn import __version__


def test_version_is_a_nonempty_string():
    assert isinstance(__version__, str)
    assert __version__
