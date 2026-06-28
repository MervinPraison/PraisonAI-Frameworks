"""AutoGen family router resolve() behaviour."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.autogen.family import AutoGenFamilyAdapter


def test_autogen_family_resolve_alias_prefers_v2(monkeypatch):
    monkeypatch.delenv("AUTOGEN_VERSION", raising=False)
    adapter = AutoGenFamilyAdapter()

    class _V2:
        def is_available(self):
            return True

    class _V4:
        def is_available(self):
            return False

    class _AG2:
        def is_available(self):
            return False

    monkeypatch.setattr(
        "praisonai_frameworks.autogen.family.AutoGenAdapter",
        lambda: _V2(),
    )
    monkeypatch.setattr(
        "praisonai_frameworks.autogen.family.AutoGenV4Adapter",
        lambda: _V4(),
    )
    monkeypatch.setattr(
        "praisonai_frameworks.autogen.family.AG2Adapter",
        lambda: _AG2(),
    )

    assert adapter.resolve_alias() == "autogen_v2"
    resolved = adapter.resolve()
    assert resolved.name == "autogen_v2"


def test_autogen_family_run_raises():
    adapter = AutoGenFamilyAdapter()
    with pytest.raises(RuntimeError, match="should not be called directly"):
        adapter.run({}, [], "topic")
