"""Framework availability probes (no PraisonAI wrapper dependency)."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import threading
from typing import Callable

_cache: dict[str, bool] = {}
_lock = threading.Lock()


def _ag2_probe() -> bool:
    try:
        importlib.metadata.distribution("ag2")
    except importlib.metadata.PackageNotFoundError:
        return False
    return importlib.util.find_spec("autogen") is not None


def _autogen_v4_probe() -> bool:
    return (
        importlib.util.find_spec("autogen_agentchat") is not None
        and importlib.util.find_spec("autogen_ext") is not None
    )


def _agno_probe() -> bool:
    try:
        importlib.metadata.distribution("agno")
    except importlib.metadata.PackageNotFoundError:
        return False
    if importlib.util.find_spec("agno") is None:
        return False
    try:
        from agno.agent import Agent  # noqa: F401

        return True
    except ImportError:
        return False


def _openai_agents_probe() -> bool:
    try:
        importlib.metadata.distribution("openai-agents")
    except importlib.metadata.PackageNotFoundError:
        return False
    if importlib.util.find_spec("agents") is None:
        return False
    try:
        from agents import Runner  # noqa: F401

        return True
    except ImportError:
        return False


def _google_adk_probe() -> bool:
    try:
        importlib.metadata.distribution("google-adk")
    except importlib.metadata.PackageNotFoundError:
        return False
    if importlib.util.find_spec("google.adk") is None:
        return False
    try:
        from google.adk.agents import Agent  # noqa: F401

        return True
    except ImportError:
        try:
            from google.adk import Agent  # noqa: F401

            return True
        except ImportError:
            return False


_PROBES: dict[str, Callable[[], bool]] = {
    "crewai": lambda: importlib.util.find_spec("crewai") is not None,
    "autogen": lambda: importlib.util.find_spec("autogen") is not None,
    "autogen_v4": _autogen_v4_probe,
    "ag2": _ag2_probe,
    "langgraph": lambda: (
        importlib.util.find_spec("langgraph") is not None
        and importlib.util.find_spec("langgraph.prebuilt") is not None
    ),
    "openai_agents": _openai_agents_probe,
    "agno": _agno_probe,
    "google_adk": _google_adk_probe,
}


def is_available(name: str) -> bool:
    if name not in _PROBES:
        return False
    cached = _cache.get(name)
    if cached is not None:
        return cached
    with _lock:
        cached = _cache.get(name)
        if cached is None:
            cached = bool(_PROBES[name]())
            _cache[name] = cached
        return cached
