"""Tests for the Windows UTF-8 console preflight in the CrewAI adapter."""

from __future__ import annotations

import pytest

pytest.importorskip("praisonaiagents.frameworks")

from praisonai_frameworks.crewai import adapter as crewai_adapter


class _FakeStream:
    def __init__(self) -> None:
        self.encoding = "cp1252"
        self.reconfigure_calls: list[dict] = []

    def reconfigure(self, **kwargs) -> None:
        self.reconfigure_calls.append(kwargs)
        if "encoding" in kwargs:
            self.encoding = kwargs["encoding"]


class _NoReconfigureStream:
    encoding = "cp1252"


def test_ensure_utf8_console_noop_off_windows(monkeypatch):
    monkeypatch.setattr(crewai_adapter._sys, "platform", "linux")
    out, err = _FakeStream(), _FakeStream()
    monkeypatch.setattr(crewai_adapter._sys, "stdout", out)
    monkeypatch.setattr(crewai_adapter._sys, "stderr", err)

    crewai_adapter._ensure_utf8_console()

    assert out.reconfigure_calls == []
    assert err.reconfigure_calls == []


def test_ensure_utf8_console_reconfigures_on_windows(monkeypatch):
    monkeypatch.setattr(crewai_adapter._sys, "platform", "win32")
    out, err = _FakeStream(), _FakeStream()
    monkeypatch.setattr(crewai_adapter._sys, "stdout", out)
    monkeypatch.setattr(crewai_adapter._sys, "stderr", err)

    crewai_adapter._ensure_utf8_console()

    assert out.reconfigure_calls == [{"encoding": "utf-8"}]
    assert err.reconfigure_calls == [{"encoding": "utf-8"}]
    assert out.encoding == "utf-8"
    assert err.encoding == "utf-8"


def test_ensure_utf8_console_skips_already_utf8(monkeypatch):
    monkeypatch.setattr(crewai_adapter._sys, "platform", "win32")
    out, err = _FakeStream(), _FakeStream()
    out.encoding = "utf-8"
    err.encoding = "UTF-8"
    monkeypatch.setattr(crewai_adapter._sys, "stdout", out)
    monkeypatch.setattr(crewai_adapter._sys, "stderr", err)

    crewai_adapter._ensure_utf8_console()

    assert out.reconfigure_calls == []
    assert err.reconfigure_calls == []


def test_ensure_utf8_console_handles_none_streams(monkeypatch):
    monkeypatch.setattr(crewai_adapter._sys, "platform", "win32")
    monkeypatch.setattr(crewai_adapter._sys, "stdout", None)
    monkeypatch.setattr(crewai_adapter._sys, "stderr", None)

    crewai_adapter._ensure_utf8_console()


def test_ensure_utf8_console_handles_missing_reconfigure(monkeypatch):
    monkeypatch.setattr(crewai_adapter._sys, "platform", "win32")
    monkeypatch.setattr(crewai_adapter._sys, "stdout", _NoReconfigureStream())
    monkeypatch.setattr(crewai_adapter._sys, "stderr", _NoReconfigureStream())

    crewai_adapter._ensure_utf8_console()


def test_ensure_utf8_console_swallows_reconfigure_errors(monkeypatch):
    monkeypatch.setattr(crewai_adapter._sys, "platform", "win32")

    class _RaisingStream:
        encoding = "cp1252"

        def reconfigure(self, **kwargs):
            raise ValueError("boom")

    monkeypatch.setattr(crewai_adapter._sys, "stdout", _RaisingStream())
    monkeypatch.setattr(crewai_adapter._sys, "stderr", _RaisingStream())

    crewai_adapter._ensure_utf8_console()
