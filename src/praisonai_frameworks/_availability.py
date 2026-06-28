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


_PROBES: dict[str, Callable[[], bool]] = {
    "crewai": lambda: importlib.util.find_spec("crewai") is not None,
    "autogen": lambda: importlib.util.find_spec("autogen") is not None,
    "autogen_v4": _autogen_v4_probe,
    "ag2": _ag2_probe,
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
