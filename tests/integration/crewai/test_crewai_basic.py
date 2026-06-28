"""CrewAI adapter integration checks (no praisonai wrapper dependency)."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")
crewai = pytest.importorskip("crewai")

from praisonai_frameworks.crewai.adapter import CrewAIAdapter


@pytest.mark.integration
def test_crewai_adapter_available():
    adapter = CrewAIAdapter()
    assert adapter.is_available() is True
    assert adapter.name == "crewai"


@pytest.mark.integration
def test_crewai_adapter_run_signature(minimal_agents_config, mock_llm_config):
    adapter = CrewAIAdapter()
    assert hasattr(adapter, "run")
    assert callable(adapter.run)
