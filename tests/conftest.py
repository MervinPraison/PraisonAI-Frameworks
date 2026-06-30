"""Shared fixtures for praisonai-frameworks tests."""

from __future__ import annotations

from pathlib import Path

import pytest

_AGENTS_YAML = """\
framework: crewai
topic: Test topic
roles:
  researcher:
    role: Research Analyst
    goal: Find accurate information
    backstory: Expert researcher
    tasks:
      research:
        description: Research {topic}
        expected_output: A concise summary
"""


@pytest.fixture
def agents_yaml_file(tmp_path: Path) -> Path:
    path = tmp_path / "agents.yaml"
    path.write_text(_AGENTS_YAML, encoding="utf-8")
    return path


@pytest.fixture
def minimal_agents_config():
    return {
        "framework": "crewai",
        "topic": "Test topic",
        "roles": {
            "researcher": {
                "role": "Research Analyst",
                "goal": "Find accurate information",
                "backstory": "Expert researcher",
                "tasks": {
                    "research": {
                        "description": "Research {topic}",
                        "expected_output": "A concise summary",
                    }
                },
            }
        },
    }


@pytest.fixture
def minimal_langgraph_config():
    return {
        "framework": "langgraph",
        "topic": "Test topic",
        "roles": {
            "researcher": {
                "role": "Research Analyst",
                "goal": "Find accurate information",
                "backstory": "Expert researcher",
                "tasks": {
                    "research": {
                        "description": "Research {topic}",
                        "expected_output": "A concise summary",
                    }
                },
            }
        },
    }


@pytest.fixture
def minimal_openai_agents_config():
    return {
        "framework": "openai_agents",
        "topic": "Test topic",
        "roles": {
            "researcher": {
                "role": "Research Analyst",
                "goal": "Find accurate information",
                "backstory": "Expert researcher",
                "tasks": {
                    "research": {
                        "description": "Research {topic}",
                        "expected_output": "A concise summary",
                    }
                },
            }
        },
    }


@pytest.fixture
def minimal_agno_config():
    return {
        "framework": "agno",
        "topic": "Test topic",
        "roles": {
            "researcher": {
                "role": "Research Analyst",
                "goal": "Find accurate information",
                "backstory": "Expert researcher",
                "tasks": {
                    "research": {
                        "description": "Research {topic}",
                        "expected_output": "A concise summary",
                    }
                },
            }
        },
    }


@pytest.fixture
def minimal_google_adk_config():
    return {
        "framework": "google_adk",
        "topic": "Test topic",
        "roles": {
            "researcher": {
                "role": "Research Analyst",
                "goal": "Find accurate information",
                "backstory": "Expert researcher",
                "tasks": {
                    "research": {
                        "description": "Research {topic}",
                        "expected_output": "A concise summary",
                    }
                },
            }
        },
    }


@pytest.fixture
def minimal_pydantic_ai_config():
    return {
        "framework": "pydantic_ai",
        "topic": "Test topic",
        "roles": {
            "researcher": {
                "role": "Research Analyst",
                "goal": "Find accurate information",
                "backstory": "Expert researcher",
                "tasks": {
                    "research": {
                        "description": "Research {topic}",
                        "expected_output": "A concise summary",
                    }
                },
            }
        },
    }


@pytest.fixture
def mock_llm_config():
    return [{"model": "openai/gpt-4o-mini", "api_key": "test-key"}]


@pytest.fixture
def minimal_autogen_config():
    return {
        "framework": "autogen",
        "topic": "Test topic",
        "roles": {
            "assistant": {
                "role": "Assistant",
                "goal": "Help with tasks",
                "backstory": "Helpful assistant",
                "tasks": {
                    "task1": {
                        "description": "Summarise {topic}",
                        "expected_output": "A short summary",
                    }
                },
            }
        },
    }


@pytest.fixture
def mock_crewai_completion():
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Task completed successfully."
    return mock_response
