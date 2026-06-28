"""Availability probe tests."""

from praisonai_frameworks._availability import is_available


def test_unknown_framework_returns_false():
    assert is_available("unknown_xyz") is False
