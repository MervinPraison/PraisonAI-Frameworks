"""Scoped telemetry disable helper."""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def scoped_telemetry_disable(telemetry_class):
    if not telemetry_class:
        yield
        return

    originals = {}
    noop = lambda *args, **kwargs: None

    for attr_name in dir(telemetry_class):
        attr = getattr(telemetry_class, attr_name)
        if callable(attr) and not attr_name.startswith("__"):
            originals[attr_name] = attr
            setattr(telemetry_class, attr_name, noop)

    try:
        yield
    finally:
        for attr_name, original_method in originals.items():
            setattr(telemetry_class, attr_name, original_method)
