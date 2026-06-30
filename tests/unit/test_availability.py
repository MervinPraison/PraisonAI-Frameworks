"""Availability probe tests."""

from praisonai_frameworks._availability import is_available


def test_langgraph_probe_does_not_raise():
    assert isinstance(is_available("langgraph"), bool)


def test_openai_agents_probe_does_not_raise():
    assert isinstance(is_available("openai_agents"), bool)


def test_unknown_framework_returns_false():
    assert is_available("unknown_xyz") is False
