"""LangGraph adapter integration checks (no praisonai wrapper dependency)."""

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
def test_langgraph_adapter_run_signature(minimal_langgraph_config, mock_llm_config):
    adapter = LangGraphAdapter()
    assert hasattr(adapter, "run")
    assert callable(adapter.run)
