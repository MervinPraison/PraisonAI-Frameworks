"""Google ADK adapter integration tests (requires google-adk optional extra)."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")
pytest.importorskip("google.adk")

from praisonai_frameworks.google_adk.adapter import GoogleAdkAdapter


@pytest.mark.integration
def test_google_adk_adapter_is_available_when_installed():
    adapter = GoogleAdkAdapter()
    assert adapter.is_available() is True


@pytest.mark.integration
def test_google_adk_adapter_build_agent_fields(minimal_google_adk_config, mock_llm_config):
    adapter = GoogleAdkAdapter()
    details = minimal_google_adk_config["roles"]["researcher"]
    agent = adapter._build_adk_agent(
        "researcher",
        "research",
        details,
        mock_llm_config,
        {},
        minimal_google_adk_config["topic"],
    )
    assert agent.name == "researcher_research"
    assert "Research Analyst" in agent.instruction
    assert agent.mode == "chat"
