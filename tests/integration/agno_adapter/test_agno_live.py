"""Live Agno adapter tests (requires OPENAI_API_KEY)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("praisonaiagents.frameworks")
pytest.importorskip("agno")

from praisonai_frameworks.agno.adapter import AgnoAdapter


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("PRAISONAI_LIVE_TESTS"), reason="Set PRAISONAI_LIVE_TESTS=1")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_agno_live_single_task():
    adapter = AgnoAdapter()
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
    result = adapter.run(config, llm_config, "math")
    assert "### Agno Output ###" in result
    assert "4" in result


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("PRAISONAI_LIVE_TESTS"), reason="Set PRAISONAI_LIVE_TESTS=1")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_agno_live_sequential_context():
    adapter = AgnoAdapter()
    config = {
        "roles": {
            "writer": {
                "role": "Writer",
                "goal": "Write numbers only",
                "backstory": "Concise writer",
                "tasks": {
                    "draft": {
                        "description": "Reply with only the number 3.",
                        "expected_output": "3",
                    },
                    "polish": {
                        "description": "Add 3 to the previous result. Reply with only the number.",
                        "expected_output": "6",
                        "context": ["draft"],
                    },
                },
            }
        }
    }
    llm_config = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]
    result = adapter.run(config, llm_config, "numbers")
    assert "### Agno Output ###" in result
    assert "6" in result
