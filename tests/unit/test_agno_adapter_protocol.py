"""Agno adapter protocol surface tests."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.agno.adapter import AgnoAdapter


def test_agno_adapter_protocol_shape():
    adapter = AgnoAdapter()
    assert adapter.name == "agno"
    assert adapter.install_hint == 'pip install "praisonai-frameworks[agno]"'
    assert adapter.requires_tools_extra is True
    assert hasattr(adapter, "run")
    assert hasattr(adapter, "is_available")
