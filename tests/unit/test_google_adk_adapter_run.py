"""Google ADK adapter mocked run tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.google_adk.adapter import GoogleAdkAdapter


def test_google_adk_adapter_run_single_task_mocked():
    config = {
        "framework": "google_adk",
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

    adapter = GoogleAdkAdapter()
    with patch.object(adapter, "_ensure_adk"), patch.object(
        adapter, "_run_single_task", return_value="ADK summary"
    ) as mock_run:
        result = adapter.run(
            config,
            [{"model": "gemini-2.5-flash", "api_key": "test-key"}],
            "AI trends",
            tools_dict={},
        )

    assert "### Google ADK Output ###" in result
    assert "ADK summary" in result
    mock_run.assert_called_once()


def test_google_adk_collect_ordered_tasks_respects_context():
    adapter = GoogleAdkAdapter()
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


def test_google_adk_run_no_tasks():
    adapter = GoogleAdkAdapter()
    with patch.object(adapter, "_ensure_adk"):
        result = adapter.run({"roles": {}}, [{"api_key": "k"}], "topic")
    assert "No tasks defined." in result


def test_google_adk_sanitize_agent_name():
    adapter = GoogleAdkAdapter()
    assert adapter._sanitize_agent_name("researcher_task", "fallback").isidentifier()
    assert adapter._sanitize_agent_name("user", "helper") != "user"
    assert adapter._sanitize_agent_name("123bad", "helper").startswith("agent_")


def test_google_adk_missing_api_key_raises():
    adapter = GoogleAdkAdapter()
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
    with patch.object(adapter, "_ensure_adk"), patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            adapter.run(config, [{"model": "openai/gpt-4o-mini"}], "topic")


def test_google_adk_run_sequential_mocked():
    adapter = GoogleAdkAdapter()
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
    with patch.object(adapter, "_ensure_adk"), patch.object(
        adapter, "_run_sequential", return_value="12"
    ) as mock_seq:
        result = adapter.run(
            config,
            [{"model": "gemini-2.5-flash", "api_key": "test-key"}],
            "topic",
        )
    assert "12" in result
    mock_seq.assert_called_once()


def test_google_adk_resolve_adk_model_gemini_string():
    adapter = GoogleAdkAdapter()
    with patch.object(adapter, "_require_api_key", return_value="test-key"), patch.object(
        adapter, "_resolve_model_name", return_value="gemini-2.5-flash"
    ), patch.object(adapter, "_uses_gemini", return_value=True):
        model = adapter._resolve_adk_model(None, [{"model": "gemini-2.5-flash"}])
    assert model == "gemini-2.5-flash"


def test_google_adk_resolve_adk_model_litellm_openai():
    adapter = GoogleAdkAdapter()
    mock_lite = MagicMock()
    with patch.object(adapter, "_require_api_key", return_value="test-key"), patch.object(
        adapter, "_uses_gemini", return_value=False
    ), patch.dict(
        "sys.modules",
        {"google.adk.models.lite_llm": MagicMock(LiteLlm=mock_lite)},
    ):
        adapter._resolve_adk_model(None, [{"model": "gpt-4o-mini", "api_key": "test-key"}])
    mock_lite.assert_called_once()
    assert mock_lite.call_args.kwargs["model"] == "openai/gpt-4o-mini"


def test_google_adk_rejects_not_provided_api_key():
    adapter = GoogleAdkAdapter()
    with patch.object(adapter, "_ensure_adk"), patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            adapter.run(
                {
                    "roles": {
                        "a": {
                            "role": "A",
                            "goal": "G",
                            "backstory": "B",
                            "tasks": {"t": {"description": "Hi"}},
                        }
                    }
                },
                [{"api_key": "not-provided", "model": "openai/gpt-4o-mini"}],
                "topic",
            )


def test_google_adk_handoff_warning(caplog):
    adapter = GoogleAdkAdapter()
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
    with patch.object(adapter, "_ensure_adk"), patch.object(
        adapter, "_run_sequential", return_value="done"
    ):
        adapter.run(
            config,
            [{"api_key": "k", "model": "gemini-2.5-flash"}],
            "topic",
        )
    assert any("handoff.to" in r.message for r in caplog.records)


def test_google_adk_run_sequential_passes_llm_config():
    adapter = GoogleAdkAdapter()
    specs = [
        {
            "name": "draft",
            "role_key": "writer",
            "role_details": {},
            "description": "Draft",
            "expected_output": "",
            "context": [],
            "tools": [],
        },
        {
            "name": "polish",
            "role_key": "writer",
            "role_details": {},
            "description": "Polish",
            "expected_output": "",
            "context": ["draft"],
            "tools": [],
        },
    ]
    llm_config = [{"model": "gemini-2.5-flash", "api_key": "config-only-key"}]
    mock_agent = MagicMock()
    with patch.object(adapter, "_agent_for_task", return_value=mock_agent), patch.object(
        adapter, "_invoke_agent", side_effect=["draft out", "polish out"]
    ) as mock_invoke:
        result = adapter._run_sequential(specs, llm_config, {}, "topic")

    assert result == "polish out"
    assert mock_invoke.call_count == 2
    for call in mock_invoke.call_args_list:
        assert call.args[2] == llm_config


def test_google_adk_resolve_model_name_role_overrides_global():
    adapter = GoogleAdkAdapter()
    llm_config = [{"model": "gemini-2.5-flash", "api_key": "k"}]
    assert adapter._resolve_model_name({"model": "gpt-4o-mini"}, llm_config) == "gpt-4o-mini"


def test_google_adk_resolve_model_name_falls_back_to_llm_config():
    adapter = GoogleAdkAdapter()
    llm_config = [{"model": "gemini-2.5-flash", "api_key": "k"}]
    assert adapter._resolve_model_name(None, llm_config) == "gemini-2.5-flash"


def test_google_adk_resolve_adk_model_maps_base_url_to_api_base():
    adapter = GoogleAdkAdapter()
    mock_lite = MagicMock()
    with patch.object(adapter, "_require_api_key", return_value="test-key"), patch.object(
        adapter, "_uses_gemini", return_value=False
    ), patch.dict(
        "sys.modules",
        {"google.adk.models.lite_llm": MagicMock(LiteLlm=mock_lite)},
    ):
        adapter._resolve_adk_model(
            None,
            [{"model": "gpt-4o-mini", "api_key": "test-key", "base_url": "https://proxy.example/v1"}],
        )
    assert mock_lite.call_args.kwargs["api_base"] == "https://proxy.example/v1"


def test_google_adk_duplicate_task_name_warning(caplog):
    adapter = GoogleAdkAdapter()
    config = {
        "roles": {
            "writer": {
                "role": "Writer",
                "goal": "Write",
                "backstory": "Writer",
                "tasks": {
                    "draft": {"description": "First draft"},
                },
            },
            "editor": {
                "role": "Editor",
                "goal": "Edit",
                "backstory": "Editor",
                "tasks": {
                    "draft": {"description": "Second draft"},
                },
            },
        }
    }
    ordered = adapter._collect_ordered_tasks(config, "topic")
    assert len(ordered) == 1
    assert ordered[0]["description"] == "Second draft"
    assert any("Duplicate task name" in r.message for r in caplog.records)


def test_google_adk_to_adk_tool_does_not_mutate_shared_callable():
    adapter = GoogleAdkAdapter()
    original_name = "shared_tool"
    original_doc = "Shared doc"

    def shared_tool(query: str) -> str:
        """Shared doc"""
        return query

    shared_tool.__name__ = original_name
    shared_tool.__doc__ = original_doc

    with patch.dict(
        "sys.modules",
        {"google.adk.tools.function_tool": MagicMock(FunctionTool=lambda fn: fn)},
    ):
        wrapped = adapter._to_adk_tool(shared_tool, "fallback")

    assert shared_tool.__name__ == original_name
    assert shared_tool.__doc__ == original_doc
    assert wrapped is not None
    assert wrapped.__name__ == original_name
