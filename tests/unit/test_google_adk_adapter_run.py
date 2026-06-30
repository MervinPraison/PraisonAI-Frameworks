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
