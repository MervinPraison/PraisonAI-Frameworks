"""CrewAI adapter protocol surface tests."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.crewai.adapter import CrewAIAdapter


def test_crewai_adapter_protocol_shape():
    adapter = CrewAIAdapter()
    assert adapter.name == "crewai"
    assert hasattr(adapter, "run")
    assert hasattr(adapter, "is_available")
