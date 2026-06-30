"""LangGraph adapter mocked run tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.langgraph.adapter import LangGraphAdapter


def test_langgraph_adapter_run_single_task_mocked():
    config = {
        "framework": "langgraph",
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

    mock_message = MagicMock()
    mock_message.content = "LangGraph summary"
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"messages": [mock_message]}

    adapter = LangGraphAdapter()
    with patch.object(adapter, "_resolve_chat_model", return_value=MagicMock()), patch.object(
        adapter, "_create_react_agent", return_value=mock_agent
    ):
        result = adapter.run(
            config,
            [{"model": "openai/gpt-4o-mini", "api_key": "test-key"}],
            "AI trends",
            tools_dict={},
        )

    assert "### LangGraph Output ###" in result
    assert "LangGraph summary" in result
    mock_agent.invoke.assert_called_once()


def test_langgraph_collect_ordered_tasks_respects_context():
    adapter = LangGraphAdapter()
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
