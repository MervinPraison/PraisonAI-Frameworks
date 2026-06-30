"""AutoGen adapter integration checks (no praisonai wrapper dependency)."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")
autogen = pytest.importorskip("autogen")

from praisonai_frameworks.autogen.adapter_v2 import AutoGenAdapter
from praisonai_frameworks.autogen.family import AutoGenFamilyAdapter


@pytest.mark.integration
def test_autogen_v2_adapter_available():
    adapter = AutoGenAdapter()
    assert adapter.is_available() is True
    assert adapter.name == "autogen_v2"


@pytest.mark.integration
def test_autogen_family_resolves_to_v2_when_installed():
    family = AutoGenFamilyAdapter()
    assert family.is_available() is True
    resolved = family.resolve()
    assert resolved.name in {"autogen_v2", "autogen_v4", "ag2"}
