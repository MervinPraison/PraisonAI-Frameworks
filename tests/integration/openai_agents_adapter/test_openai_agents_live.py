"""Live OpenAI Agents adapter tests (requires OPENAI_API_KEY)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("praisonaiagents.frameworks")
pytest.importorskip("agents")

from praisonai_frameworks.openai_agents.adapter import OpenAIAgentsAdapter


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_agents_live_single_task():
    adapter = OpenAIAgentsAdapter()
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
    assert "### OpenAI Agents Output ###" in result
    assert "4" in result


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_agents_live_sequential_context():
    adapter = OpenAIAgentsAdapter()
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
    assert "### OpenAI Agents Output ###" in result
    assert "HELLO" in result.upper()


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_agents_live_handoff():
    adapter = OpenAIAgentsAdapter()
    config = {
        "roles": {
            "triage": {
                "role": "Triage Agent",
                "goal": "Route to English for English requests",
                "backstory": "You delegate English questions to the English Agent.",
                "handoff": {"to": ["English Agent"]},
                "tasks": {
                    "route": {
                        "description": "The user says: Hello, how are you? Route appropriately.",
                        "expected_output": "A friendly English reply",
                    }
                },
            },
            "english": {
                "role": "English Agent",
                "goal": "Reply in English only",
                "backstory": "English specialist.",
            },
        }
    }
    llm_config = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]
    result = adapter.run(config, llm_config, "greeting", tools_dict={})
    assert "### OpenAI Agents Output ###" in result
    assert len(result.strip()) > len("### OpenAI Agents Output ###")
