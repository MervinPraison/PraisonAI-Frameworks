"""Entry-point registration tests."""

import importlib.metadata
import pytest


def test_entry_points_registered():
    eps = {
        ep.name
        for ep in importlib.metadata.entry_points(group="praisonai.framework_adapters")
    }
    expected = {"crewai", "autogen", "autogen_v2", "autogen_v4", "ag2", "langgraph", "openai_agents", "agno", "google_adk", "pydantic_ai"}
    assert expected.issubset(eps)


def test_autogen_family_is_router():
    pytest.importorskip("praisonaiagents.frameworks")
    from praisonai_frameworks.autogen.family import AutoGenFamilyAdapter

    adapter = AutoGenFamilyAdapter()
    assert adapter.is_router is True
