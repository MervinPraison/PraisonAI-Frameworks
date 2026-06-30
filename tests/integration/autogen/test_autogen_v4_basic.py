"""AutoGen v0.4 adapter integration checks (no praisonai wrapper dependency)."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")
pytest.importorskip("autogen_agentchat")
pytest.importorskip("autogen_ext")

from praisonai_frameworks.autogen.adapter_v4 import AutoGenV4Adapter


@pytest.mark.integration
def test_autogen_v4_adapter_available():
    adapter = AutoGenV4Adapter()
    assert adapter.implemented is True
    assert adapter.is_available() is True
    assert adapter.name == "autogen_v4"


@pytest.mark.integration
def test_autogen_v4_build_model_client():
    adapter = AutoGenV4Adapter()
    client = adapter._build_model_client(
        [{"model": "gpt-4o-mini", "api_key": "sk-test"}]
    )
    assert client is not None
