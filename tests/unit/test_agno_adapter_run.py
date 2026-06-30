"""Agno adapter mocked run tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.agno.adapter import AgnoAdapter


def test_agno_adapter_run_single_task_mocked():
    config = {
        "framework": "agno",
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

    adapter = AgnoAdapter()
    with patch.object(adapter, "_ensure_agno"), patch.object(
        adapter, "_run_single_task", return_value="Agno summary"
    ) as mock_run:
        result = adapter.run(
            config,
            [{"model": "openai/gpt-4o-mini", "api_key": "test-key"}],
            "AI trends",
            tools_dict={},
        )

    assert "### Agno Output ###" in result
    assert "Agno summary" in result
    mock_run.assert_called_once()


def test_agno_collect_ordered_tasks_respects_context():
    adapter = AgnoAdapter()
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


def test_agno_run_no_tasks():
    adapter = AgnoAdapter()
    with patch.object(adapter, "_ensure_agno"):
        result = adapter.run({"roles": {}}, [{"api_key": "k"}], "topic")
    assert "No tasks defined." in result


def test_agno_normalise_model_name():
    adapter = AgnoAdapter()
    assert adapter._normalise_model_name("openai/gpt-4o-mini") == "gpt-4o-mini"
    assert adapter._normalise_model_name("gpt-4o-mini") == "gpt-4o-mini"


def test_agno_missing_api_key_raises():
    adapter = AgnoAdapter()
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
    with patch.object(adapter, "_ensure_agno"), patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            adapter.run(config, [{}], "topic")


def test_agno_run_sequential_mocked():
    adapter = AgnoAdapter()
    config = {
        "roles": {
            "writer": {
                "role": "Writer",
                "goal": "Write",
                "backstory": "Writer",
                "tasks": {
                    "draft": {"description": "Draft", "expected_output": "6"},
                    "polish": {
                        "description": "Polish",
                        "expected_output": "12",
                        "context": ["draft"],
                    },
                },
            }
        }
    }
    with patch.object(adapter, "_ensure_agno"), patch.object(
        adapter, "_run_sequential", return_value="12"
    ) as mock_seq:
        result = adapter.run(
            config,
            [{"model": "gpt-4o-mini", "api_key": "test-key"}],
            "topic",
        )
    assert "12" in result
    mock_seq.assert_called_once()


def test_resolve_agno_model_wires_api_key_and_base_url():
    adapter = AgnoAdapter()
    llm_config = [
        {"api_key": "test-key", "base_url": "https://example.test/v1", "model": "gpt-4o-mini"}
    ]
    mock_model = MagicMock()
    mock_module = MagicMock()
    mock_module.OpenAILike = MagicMock(return_value=mock_model)
    mock_module.OpenAIChat = MagicMock()
    with patch.object(adapter, "_resolve_model_name", return_value="gpt-4o-mini"), patch.object(
        adapter, "_require_api_key", return_value="test-key"
    ), patch.dict("sys.modules", {"agno.models.openai": mock_module}):
        result = adapter._resolve_agno_model(None, llm_config)
    mock_module.OpenAILike.assert_called_once_with(
        id="gpt-4o-mini", api_key="test-key", base_url="https://example.test/v1"
    )
    assert result is mock_model


def test_resolve_agno_model_uses_openai_chat_without_base_url():
    adapter = AgnoAdapter()
    llm_config = [{"api_key": "test-key", "model": "gpt-4o-mini"}]
    mock_model = MagicMock()
    mock_module = MagicMock()
    mock_module.OpenAIChat = MagicMock(return_value=mock_model)
    mock_module.OpenAILike = MagicMock()
    with patch.object(adapter, "_resolve_model_name", return_value="gpt-4o-mini"), patch.object(
        adapter, "_require_api_key", return_value="test-key"
    ), patch.dict("sys.modules", {"agno.models.openai": mock_module}):
        result = adapter._resolve_agno_model(None, llm_config)
    mock_module.OpenAIChat.assert_called_once_with(id="gpt-4o-mini", api_key="test-key")
    assert result is mock_model


def test_agno_rejects_not_provided_api_key():
    adapter = AgnoAdapter()
    with patch.object(adapter, "_ensure_agno"), patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            adapter.run(
                {"roles": {"a": {"role": "A", "goal": "G", "backstory": "B", "tasks": {"t": {"description": "Hi"}}}}},
                [{"api_key": "not-provided"}],
                "topic",
            )


def test_agno_handoff_warning(caplog):
    adapter = AgnoAdapter()
    config = {
        "roles": {
            "triage": {
                "role": "Triage",
                "goal": "Route",
                "backstory": "Router",
                "handoff": {"to": ["english"]},
                "tasks": {"route": {"description": "Route", "expected_output": "Done"}},
            },
            "english": {
                "role": "English",
                "goal": "English",
                "backstory": "English",
                "tasks": {"greet": {"description": "Hi", "expected_output": "Hi"}},
            },
        }
    }
    with patch.object(adapter, "_ensure_agno"), patch.object(
        adapter, "_run_sequential", return_value="done"
    ):
        adapter.run(config, [{"api_key": "k"}], "topic")
    assert any("handoff.to" in r.message for r in caplog.records)
