"""Optional observability hooks without importing the praisonai wrapper."""

from __future__ import annotations

import logging
from typing import Callable, List

logger = logging.getLogger(__name__)

FinalizeHook = Callable[[str, str], None]
_finalize_hooks: List[FinalizeHook] = []


def register_finalize_hook(hook: FinalizeHook) -> None:
    """Register an optional callback invoked after framework runs."""
    _finalize_hooks.append(hook)


def finalize_observability(framework_name: str, *, status: str = "Success") -> None:
    if not _finalize_hooks:
        logger.debug("No observability hooks registered; skipping finalize")
        return
    for hook in _finalize_hooks:
        try:
            hook(framework_name, status)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error finalizing observability: %s", exc)
