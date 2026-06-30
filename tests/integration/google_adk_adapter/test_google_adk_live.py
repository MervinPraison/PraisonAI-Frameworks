"""Live Google ADK adapter tests."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("praisonaiagents.frameworks")
pytest.importorskip("google.adk")

from praisonai_frameworks.google_adk.adapter import GoogleAdkAdapter


def _google_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("PRAISONAI_LIVE_TESTS"), reason="Set PRAISONAI_LIVE_TESTS=1")
@pytest.mark.skipif(not _google_key(), reason="GOOGLE_API_KEY or GEMINI_API_KEY not set")
def test_google_adk_live_single_task_gemini():
    adapter = GoogleAdkAdapter()
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
    llm_config = [{"model": "gemini-2.5-flash", "api_key": _google_key()}]
    result = adapter.run(config, llm_config, "math")
    assert "### Google ADK Output ###" in result
    assert "4" in result


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("PRAISONAI_LIVE_TESTS"), reason="Set PRAISONAI_LIVE_TESTS=1")
@pytest.mark.skipif(not _google_key(), reason="GOOGLE_API_KEY or GEMINI_API_KEY not set")
def test_google_adk_live_sequential_context_gemini():
    adapter = GoogleAdkAdapter()
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
    llm_config = [{"model": "gemini-2.5-flash", "api_key": _google_key()}]
    result = adapter.run(config, llm_config, "numbers")
    assert "### Google ADK Output ###" in result
    assert "6" in result


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("PRAISONAI_LIVE_TESTS"), reason="Set PRAISONAI_LIVE_TESTS=1")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_google_adk_live_single_task_openai_litellm():
    adapter = GoogleAdkAdapter()
    config = {
        "roles": {
            "helper": {
                "role": "Assistant",
                "goal": "Answer briefly",
                "backstory": "Helpful assistant",
                "tasks": {
                    "answer": {
                        "description": "Calculate 3+3. Reply with only the number.",
                        "expected_output": "6",
                    }
                },
            }
        }
    }
    llm_config = [{"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]
    result = adapter.run(config, llm_config, "math")
    assert "### Google ADK Output ###" in result
    assert "6" in result


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("PRAISONAI_LIVE_TESTS"), reason="Set PRAISONAI_LIVE_TESTS=1")
def test_google_adk_live_sequential_llm_config_key_only(monkeypatch):
    key = _google_key()
    if not key:
        pytest.skip("GOOGLE_API_KEY or GEMINI_API_KEY not set")

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    adapter = GoogleAdkAdapter()
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
    llm_config = [{"model": "gemini-2.5-flash", "api_key": key}]
    result = adapter.run(config, llm_config, "numbers")
    assert "### Google ADK Output ###" in result
    assert "6" in result
