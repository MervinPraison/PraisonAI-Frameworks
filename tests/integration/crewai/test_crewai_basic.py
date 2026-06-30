"""CrewAI adapter integration checks (no praisonai wrapper dependency)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

crewai = pytest.importorskip("crewai")

from praisonai_frameworks.crewai.adapter import CrewAIAdapter


@pytest.mark.integration
def test_crewai_adapter_available():
    adapter = CrewAIAdapter()
    assert adapter.is_available() is True
    assert adapter.name == "crewai"


@pytest.mark.integration
@patch("litellm.completion")
def test_crewai_adapter_run_mocked(mock_completion, minimal_agents_config, mock_llm_config, mock_crewai_completion):
    mock_completion.return_value = mock_crewai_completion
    adapter = CrewAIAdapter()

    mock_crew = MagicMock()
    mock_crew.kickoff.return_value = "Crew output"

    with patch("crewai.Crew", return_value=mock_crew), patch(
        "crewai.Agent", side_effect=lambda **kwargs: MagicMock(**kwargs)
    ), patch("crewai.Task", side_effect=lambda **kwargs: MagicMock(**kwargs)), patch(
        "crewai.telemetry.Telemetry"
    ):
        result = adapter.run(
            minimal_agents_config,
            mock_llm_config,
            minimal_agents_config["topic"],
            tools_dict={},
        )

    assert "### Task Output ###" in result
    assert "Crew output" in result
