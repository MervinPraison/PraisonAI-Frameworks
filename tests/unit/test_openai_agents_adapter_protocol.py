"""OpenAI Agents adapter protocol surface tests."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.openai_agents.adapter import OpenAIAgentsAdapter


def test_openai_agents_adapter_protocol_shape():
    adapter = OpenAIAgentsAdapter()
    assert adapter.name == "openai_agents"
    assert adapter.install_hint == 'pip install "praisonai-frameworks[openai-agents]"'
    assert adapter.requires_tools_extra is True
    assert hasattr(adapter, "run")
    assert hasattr(adapter, "is_available")
