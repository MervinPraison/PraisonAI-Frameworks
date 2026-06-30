"""Live LangGraph adapter tests (requires OPENAI_API_KEY)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("praisonaiagents.frameworks")
pytest.importorskip("langgraph")

from praisonai_frameworks.langgraph.adapter import LangGraphAdapter


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_langgraph_live_single_task():
    adapter = LangGraphAdapter()
    config = {
        "roles": {
            "helper": {
                "role": "Assistant",
                "goal": "Answer briefly",
                "backstory": "Helpful assistant",
                "tasks": {
                    "answer": {
                        "description": "What is 2+2? Reply with only the number.",
                        "expected_output": "4",
                    }
                },
            }
        }
    }
    llm_config = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]
    result = adapter.run(config, llm_config, "math", tools_dict={})
    assert "### LangGraph Output ###" in result
    assert "4" in result


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_langgraph_live_sequential_context():
    adapter = LangGraphAdapter()
    config = {
        "roles": {
            "writer": {
                "role": "Writer",
                "goal": "Write concise content",
                "backstory": "Professional writer",
                "tasks": {
                    "draft": {
                        "description": "Write one word: hello",
                        "expected_output": "hello",
                    },
                    "uppercase": {
                        "description": "Uppercase the prior draft word only.",
                        "expected_output": "HELLO",
                        "context": ["draft"],
                    },
                },
            }
        }
    }
    llm_config = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]
    result = adapter.run(config, llm_config, "text", tools_dict={})
    assert "### LangGraph Output ###" in result
    assert "HELLO" in result.upper()
