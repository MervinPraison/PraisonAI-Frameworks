"""Pydantic AI adapter integration tests (requires pydantic-ai optional extra)."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")
pytest.importorskip("pydantic_ai")

from praisonai_frameworks.pydantic_ai.adapter import PydanticAiAdapter


@pytest.mark.integration
def test_pydantic_ai_adapter_is_available_when_installed():
    adapter = PydanticAiAdapter()
    assert adapter.is_available() is True


@pytest.mark.integration
def test_pydantic_ai_adapter_build_agent_fields(minimal_pydantic_ai_config, mock_llm_config):
    adapter = PydanticAiAdapter()
    details = minimal_pydantic_ai_config["roles"]["researcher"]
    agent = adapter._build_agent(
        "researcher",
        details,
        mock_llm_config,
        {},
        minimal_pydantic_ai_config["topic"],
    )
    assert agent.name == "Research Analyst"
    assert "Research Analyst" in agent._instructions
