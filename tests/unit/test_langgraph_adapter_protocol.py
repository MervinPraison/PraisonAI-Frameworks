"""LangGraph adapter protocol surface tests."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.langgraph.adapter import LangGraphAdapter


def test_langgraph_adapter_protocol_shape():
    adapter = LangGraphAdapter()
    assert adapter.name == "langgraph"
    assert adapter.install_hint == 'pip install "praisonai-frameworks[langgraph]"'
    assert adapter.requires_tools_extra is True
    assert hasattr(adapter, "run")
    assert hasattr(adapter, "is_available")
