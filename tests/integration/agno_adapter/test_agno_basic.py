"""Agno adapter integration tests (requires agno optional extra)."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")
pytest.importorskip("agno")

from praisonai_frameworks.agno.adapter import AgnoAdapter


@pytest.mark.integration
def test_agno_adapter_is_available_when_installed():
    adapter = AgnoAdapter()
    assert adapter.is_available() is True


@pytest.mark.integration
def test_agno_adapter_build_agent_fields(minimal_agno_config, mock_llm_config):
    adapter = AgnoAdapter()
    details = minimal_agno_config["roles"]["researcher"]
    agent = adapter._build_agent(
        "researcher",
        details,
        mock_llm_config,
        {},
        minimal_agno_config["topic"],
        expected_output="A concise summary",
    )
    assert agent.name == "Research Analyst"
    assert agent.role == "Research Analyst"
    assert "Research Analyst" in agent.instructions
