"""AutoGen adapter integration checks (no praisonai wrapper dependency)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

autogen = pytest.importorskip("autogen")

from praisonai_frameworks.autogen.adapter_v2 import AutoGenAdapter
from praisonai_frameworks.autogen.family import AutoGenFamilyAdapter


@pytest.mark.integration
def test_autogen_v2_adapter_available():
    adapter = AutoGenAdapter()
    assert adapter.is_available() is True
    assert adapter.name == "autogen_v2"


@pytest.mark.integration
def test_autogen_family_has_install_hint():
    family = AutoGenFamilyAdapter()
    assert "praisonai-frameworks" in family.install_hint


@pytest.mark.integration
def test_autogen_family_resolves_to_v2_when_installed():
    family = AutoGenFamilyAdapter()
    assert family.is_available() is True
    resolved = family.resolve()
    assert resolved.name == "autogen_v2"


@pytest.mark.integration
@patch("autogen.AssistantAgent")
@patch("autogen.UserProxyAgent")
def test_autogen_v2_run_mocked(
    mock_user_proxy_cls,
    mock_assistant_cls,
    minimal_autogen_config,
    mock_llm_config,
):
    mock_user_proxy = MagicMock()
    mock_user_proxy_cls.return_value = mock_user_proxy
    mock_assistant_cls.return_value = MagicMock()

    chat_result = MagicMock()
    chat_result.summary = "AutoGen summary"
    mock_user_proxy.initiate_chats.return_value = [chat_result]

    adapter = AutoGenAdapter()
    result = adapter.run(
        minimal_autogen_config,
        mock_llm_config,
        minimal_autogen_config["topic"],
        tools_dict={},
    )

    assert "### AutoGen v0.2 Output ###" in result
    assert "AutoGen summary" in result
