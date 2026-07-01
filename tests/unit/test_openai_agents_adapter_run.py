"""OpenAI Agents adapter mocked run tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("praisonaiagents.frameworks")
pytest.importorskip("agents")

from praisonai_frameworks.openai_agents.adapter import OpenAIAgentsAdapter


def test_openai_agents_adapter_run_single_task_mocked():
    config = {
        "framework": "openai_agents",
        "topic": "AI trends",
        "roles": {
            "researcher": {
                "role": "Research Analyst",
                "goal": "Research",
                "backstory": "Expert",
                "tasks": {
                    "research": {
                        "description": "Research {topic}",
                        "expected_output": "Summary",
                    }
                },
            }
        },
    }

    adapter = OpenAIAgentsAdapter()
    with patch.object(adapter, "_ensure_agents_sdk"), patch.object(
        adapter, "_build_agent_registry", return_value=({}, {})
    ), patch.object(adapter, "_run_single_task", return_value="OpenAI summary") as mock_run:
        result = adapter.run(
            config,
            [{"model": "openai/gpt-4o-mini", "api_key": "test-key"}],
            "AI trends",
            tools_dict={},
        )

    assert "### OpenAI Agents Output ###" in result
    assert "OpenAI summary" in result
    mock_run.assert_called_once()


def test_openai_agents_collect_ordered_tasks_respects_context():
    adapter = OpenAIAgentsAdapter()
    config = {
        "roles": {
            "writer": {
                "role": "Writer",
                "goal": "Write",
                "backstory": "Writer",
                "tasks": {
                    "draft": {
                        "description": "Draft",
                        "expected_output": "Draft",
                    },
                    "polish": {
                        "description": "Polish",
                        "expected_output": "Final",
                        "context": ["draft"],
                    },
                },
            }
        }
    }

    ordered = adapter._collect_ordered_tasks(config, "topic")
    assert [task["name"] for task in ordered] == ["draft", "polish"]


def test_openai_agents_run_no_tasks():
    adapter = OpenAIAgentsAdapter()
    with patch.object(adapter, "_ensure_agents_sdk"):
        result = adapter.run({"roles": {}}, [{"api_key": "k"}], "topic")
    assert "No tasks defined." in result


def test_openai_agents_normalise_model_name():
    adapter = OpenAIAgentsAdapter()
    assert adapter._normalise_model_name("openai/gpt-4o-mini") == "gpt-4o-mini"
    assert adapter._normalise_model_name("gpt-4o-mini") == "gpt-4o-mini"


def test_openai_agents_missing_api_key_raises():
    pytest.importorskip("agents")

    adapter = OpenAIAgentsAdapter()
    config = {
        "roles": {
            "helper": {
                "role": "Helper",
                "goal": "Help",
                "backstory": "Helper",
                "tasks": {
                    "answer": {
                        "description": "Hi",
                        "expected_output": "Hi",
                    }
                },
            }
        }
    }
    with patch.object(adapter, "_ensure_agents_sdk"), patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            adapter.run(config, [{}], "topic")


def test_run_config_wires_api_key_and_base_url():
    pytest.importorskip("agents")

    adapter = OpenAIAgentsAdapter()
    llm_config = [{"api_key": "test-key", "base_url": "https://example.test/v1"}]

    with patch("agents.models.openai_provider.OpenAIProvider") as mock_provider, patch(
        "agents.run.RunConfig"
    ) as mock_run_config:
        mock_run_config.return_value = MagicMock()
        adapter._run_config(llm_config)

    mock_provider.assert_called_once_with(api_key="test-key", base_url="https://example.test/v1")
    mock_run_config.assert_called_once()
    assert mock_run_config.call_args.kwargs["tracing_disabled"] is True
