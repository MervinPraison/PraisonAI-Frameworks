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
def mock_llm_config():
    return [{"model": "openai/gpt-4o-mini", "api_key": "test-key"}]
