"""AutoGenV4Adapter unit behaviour (no autogen-v4 deps required)."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.autogen.adapter_v4 import AutoGenV4Adapter


def test_v4_marked_implemented():
    assert AutoGenV4Adapter.implemented is True


def test_v4_is_available_tracks_probe(monkeypatch):
    adapter = AutoGenV4Adapter()
    monkeypatch.setattr(
        "praisonai_frameworks.autogen.adapter_v4.is_available",
        lambda name: name == "autogen_v4",
    )
    assert adapter.is_available() is True


def test_v4_sanitize_name():
    assert AutoGenV4Adapter._sanitize_name("Senior Writer") == "Senior_Writer"
    assert AutoGenV4Adapter._sanitize_name("") == "agent"
    assert AutoGenV4Adapter._sanitize_name("123role").startswith("a_")


def test_family_resolves_v4_without_warning(monkeypatch, caplog):
    monkeypatch.delenv("AUTOGEN_VERSION", raising=False)
    from praisonai_frameworks.autogen.family import AutoGenFamilyAdapter

    class _Off:
        def is_available(self):
            return False

    class _On:
        def is_available(self):
            return True

    monkeypatch.setattr(
        "praisonai_frameworks.autogen.family.AutoGenAdapter", lambda: _Off()
    )
    monkeypatch.setattr(
        "praisonai_frameworks.autogen.family.AutoGenV4Adapter", lambda: _On()
    )
    monkeypatch.setattr(
        "praisonai_frameworks.autogen.family.AG2Adapter", lambda: _Off()
    )

    with caplog.at_level("WARNING"):
        assert AutoGenFamilyAdapter().resolve_alias() == "autogen_v4"
    assert "not yet implemented" not in caplog.text
