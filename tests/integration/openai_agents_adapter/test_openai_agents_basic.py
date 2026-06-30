"""OpenAI Agents adapter integration checks (no praisonai wrapper dependency)."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")
pytest.importorskip("agents")

from praisonai_frameworks.openai_agents.adapter import OpenAIAgentsAdapter


@pytest.mark.integration
def test_openai_agents_adapter_available():
    adapter = OpenAIAgentsAdapter()
    assert adapter.is_available() is True
    assert adapter.name == "openai_agents"


@pytest.mark.integration
def test_openai_agents_adapter_run_signature(minimal_openai_agents_config, mock_llm_config):
    adapter = OpenAIAgentsAdapter()
    assert hasattr(adapter, "run")
    assert callable(adapter.run)
