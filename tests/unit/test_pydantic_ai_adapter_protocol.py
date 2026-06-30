"""Pydantic AI adapter protocol surface tests."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.pydantic_ai.adapter import PydanticAiAdapter


def test_pydantic_ai_adapter_protocol_shape():
    adapter = PydanticAiAdapter()
    assert adapter.name == "pydantic_ai"
    assert adapter.install_hint == 'pip install "praisonai-frameworks[pydantic-ai]"'
    assert adapter.requires_tools_extra is True
    assert hasattr(adapter, "run")
    assert hasattr(adapter, "is_available")
