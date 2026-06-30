"""Agno adapter handoff unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.agno.adapter import AgnoAdapter


def test_handoff_to_resolves_by_role_string():
    adapter = AgnoAdapter()
    config = {
        "roles": {
            "triage": {
                "role": "Triage Agent",
                "goal": "Route",
                "backstory": "Router",
                "handoff": {"to": ["English Agent"]},
                "tasks": {"route": {"description": "Go", "expected_output": "Done"}},
            },
            "english": {
                "role": "English Agent",
                "goal": "English",
                "backstory": "English",
            },
        }
    }

    mock_triage = MagicMock()
    mock_english = MagicMock()

    def fake_build(role_key, details, llm_config, tools_dict, topic, tool_names=None, expected_output=""):
        if role_key == "triage":
            return mock_triage
        return mock_english

    with patch.object(adapter, "_build_agent", side_effect=fake_build):
        by_key, by_role = adapter._build_agent_registry(
            config, [{"api_key": "k"}], {}, "topic"
        )

    assert by_role["Triage Agent"] is mock_triage
    assert by_role["English Agent"] is mock_english


def test_handoff_undefined_target_raises():
    adapter = AgnoAdapter()
    config = {
        "roles": {
            "triage": {
                "role": "Triage Agent",
                "goal": "Route",
                "backstory": "Router",
                "handoff": {"to": ["Missing Agent"]},
                "tasks": {"route": {"description": "Go", "expected_output": "Done"}},
            },
        }
    }

    with patch.object(adapter, "_build_agent", return_value=MagicMock()):
        with pytest.raises(ValueError, match="Handoff target"):
            adapter._build_agent_registry(config, [{"api_key": "k"}], {}, "topic")


def test_handoff_to_resolves_by_role_key():
    adapter = AgnoAdapter()
    config = {
        "roles": {
            "triage": {
                "role": "Triage Agent",
                "goal": "Route",
                "backstory": "Router",
                "handoff": {"to": ["english"]},
            },
            "english": {
                "role": "English Agent",
                "goal": "English",
                "backstory": "English",
            },
        }
    }

    mock_triage = MagicMock()
    mock_english = MagicMock()

    def fake_build(role_key, details, llm_config, tools_dict, topic, tool_names=None, expected_output=""):
        if role_key == "triage":
            return mock_triage
        return mock_english

    with patch.object(adapter, "_build_agent", side_effect=fake_build):
        by_key, by_role = adapter._build_agent_registry(
            config, [{"api_key": "k"}], {}, "topic"
        )

    assert by_key["english"] is mock_english


def test_allow_delegation_without_handoff_to_warns(caplog):
    adapter = AgnoAdapter()
    config = {
        "roles": {
            "helper": {
                "role": "Helper",
                "goal": "Help",
                "backstory": "Helper",
                "allow_delegation": True,
            }
        }
    }

    with patch.object(adapter, "_build_agent", return_value=MagicMock()):
        adapter._build_agent_registry(config, [{"api_key": "k"}], {}, "topic")

    assert "allow_delegation" in caplog.text


def test_handoff_with_multi_task_uses_sequential():
    adapter = AgnoAdapter()
    config = {
        "roles": {
            "triage": {
                "role": "Triage Agent",
                "goal": "Route",
                "backstory": "Router",
                "handoff": {"to": ["English Agent"]},
                "tasks": {
                    "route": {"description": "First", "expected_output": "A"},
                    "follow": {
                        "description": "Second",
                        "expected_output": "B",
                        "context": ["route"],
                    },
                },
            },
            "english": {
                "role": "English Agent",
                "goal": "English",
                "backstory": "English",
            },
        }
    }

    with patch.object(adapter, "_ensure_agno"), patch.object(
        adapter, "_require_api_key"
    ), patch.object(
        adapter, "_build_agent_registry", return_value=({}, {})
    ), patch.object(adapter, "_run_sequential", return_value="sequential") as mock_seq, patch.object(
        adapter, "_run_with_handoffs"
    ) as mock_handoff:
        result = adapter.run(config, [{"api_key": "k"}], "topic")

    assert result.endswith("sequential")
    mock_seq.assert_called_once()
    mock_handoff.assert_not_called()
