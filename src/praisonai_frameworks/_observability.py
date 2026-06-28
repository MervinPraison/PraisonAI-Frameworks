"""Optional observability finalisation (wrapper hooks when praisonai is installed)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def finalize_observability(framework_name: str, *, status: str = "Success") -> None:
    try:
        from praisonai.observability.hooks import finalize_observability as _finalize

        _finalize(framework_name, status=status)
    except ImportError:
        logger.debug("praisonai observability not installed; skipping finalize")
    except Exception as exc:  # noqa: BLE001
        logger.error("Error finalizing observability: %s", exc)
