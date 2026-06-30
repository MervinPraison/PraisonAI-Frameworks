"""Pydantic AI adapter run unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.pydantic_ai.adapter import PydanticAiAdapter


def test_pydantic_ai_single_task_handoff_uses_handoff_path():
    adapter = PydanticAiAdapter()
    config = {
        "roles": {
            "triage": {
                "role": "Triage",
                "goal": "Route",
                "backstory": "Router",
                "handoff": {"to": ["English"]},
                "tasks": {"route": {"description": "Route", "expected_output": "Done"}},
            },
            "english": {
                "role": "English",
                "goal": "English",
                "backstory": "English",
            },
        }
    }
    with patch.object(adapter, "_ensure_pydantic_ai"), patch.object(
        adapter, "_require_api_key"
    ), patch.object(
        adapter, "_build_agent_registry", return_value=({}, {})
    ), patch.object(adapter, "_run_with_handoffs", return_value="handoff") as mock_handoff, patch.object(
        adapter, "_run_sequential"
    ) as mock_seq:
        result = adapter.run(config, [{"api_key": "k"}], "topic")
    assert result.endswith("handoff")
    mock_handoff.assert_called_once()
    mock_seq.assert_not_called()


def test_to_pydantic_model_openai():
    adapter = PydanticAiAdapter()
    with patch.object(adapter, "_require_api_key", return_value="k"):
        assert (
            adapter._to_pydantic_model("gpt-4o-mini", [{"api_key": "k"}])
            == "openai:gpt-4o-mini"
        )


def test_to_pydantic_model_gemini():
    adapter = PydanticAiAdapter()
    with patch.object(adapter, "_require_api_key", return_value="k"):
        assert (
            adapter._to_pydantic_model("gemini-2.5-flash", [{"api_key": "k"}])
            == "google:gemini-2.5-flash"
        )


def test_invoke_agent_nested_loop_uses_thread_pool():
    adapter = PydanticAiAdapter()
    mock_agent = MagicMock()
    mock_agent.run_sync.return_value = MagicMock(output="ok")
    mock_agent.model = "openai:gpt-4o-mini"

    with patch.object(adapter, "_push_api_keys", return_value={}), patch.object(
        adapter, "_pop_api_keys"
    ), patch("asyncio.get_running_loop", side_effect=RuntimeError):
        text = adapter._invoke_agent(mock_agent, "hi", [{"api_key": "k"}])
    assert text == "ok"
