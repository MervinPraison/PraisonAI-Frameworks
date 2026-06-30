"""Google ADK adapter protocol surface tests."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.google_adk.adapter import GoogleAdkAdapter


def test_google_adk_adapter_protocol_shape():
    adapter = GoogleAdkAdapter()
    assert adapter.name == "google_adk"
    assert adapter.install_hint == 'pip install "praisonai-frameworks[google-adk]"'
    assert adapter.requires_tools_extra is True
    assert hasattr(adapter, "run")
    assert hasattr(adapter, "is_available")
