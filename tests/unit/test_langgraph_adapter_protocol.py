"""LangGraph adapter protocol surface tests."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.langgraph.adapter import LangGraphAdapter


def test_langgraph_adapter_protocol_shape():
    adapter = LangGraphAdapter()
    assert adapter.name == "langgraph"
    assert adapter.requires_tools_extra is True
    assert hasattr(adapter, "run")
    assert hasattr(adapter, "is_available")
    assert callable(adapter.run)


def test_langgraph_adapter_install_hint():
    adapter = LangGraphAdapter()
    assert "langgraph" in adapter.install_hint


def test_langgraph_build_prompt_fills_topic():
    adapter = LangGraphAdapter()
    details = {
        "role": "Research Analyst",
        "goal": "Find facts about {topic}",
        "backstory": "Expert",
        "tasks": {
            "research": {
                "description": "Research {topic}",
                "expected_output": "A summary",
            }
        },
    }
    prompt = adapter._build_prompt("researcher", details, topic="quantum computing")
    assert "quantum computing" in prompt
    assert "Research Analyst" in prompt
    assert "{topic}" not in prompt
