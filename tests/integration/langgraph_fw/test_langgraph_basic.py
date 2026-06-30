"""LangGraph adapter integration checks (no praisonai wrapper dependency).

This test package is intentionally named ``langgraph_fw`` rather than
``langgraph`` so it does not shadow the real ``langgraph`` namespace package
on ``sys.path`` under pytest's prepend import mode.
"""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")
pytest.importorskip("langgraph")

from praisonai_frameworks.langgraph.adapter import LangGraphAdapter


@pytest.mark.integration
def test_langgraph_adapter_available():
    adapter = LangGraphAdapter()
    assert adapter.is_available() is True
    assert adapter.name == "langgraph"


@pytest.mark.integration
def test_langgraph_run_sequential_mock_llm(
    monkeypatch, minimal_agents_config, mock_llm_config
):
    adapter = LangGraphAdapter()

    calls = []

    def _fake_call(self, model, prompt, llm_config):
        calls.append((model, prompt))
        return f"result for {model}"

    monkeypatch.setattr(LangGraphAdapter, "_call_llm", _fake_call)

    config = dict(minimal_agents_config)
    config["roles"] = {
        "researcher": {
            "role": "Research Analyst",
            "goal": "Find facts about {topic}",
            "backstory": "Expert",
            "tasks": {
                "research": {
                    "description": "Research {topic}",
                    "expected_output": "A summary",
                }
            },
        },
        "writer": {
            "role": "Writer",
            "goal": "Write report",
            "backstory": "Skilled",
            "tasks": {
                "write": {
                    "description": "Write about {topic}",
                    "expected_output": "A paragraph",
                }
            },
        },
    }

    result = adapter.run(config, mock_llm_config, "test topic")

    assert isinstance(result, str)
    assert "### Task Output ###" in result
    assert len(calls) == 2
