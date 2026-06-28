"""AutoGen family router adapter."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from praisonai_frameworks.autogen.adapter_ag2 import AG2Adapter
from praisonai_frameworks.autogen.adapter_v2 import AutoGenAdapter
from praisonai_frameworks.autogen.adapter_v4 import AutoGenV4Adapter
from praisonai_frameworks.base import BaseFrameworkAdapter

logger = logging.getLogger(__name__)

_ADAPTERS = {
    "autogen_v2": AutoGenAdapter,
    "autogen_v4": AutoGenV4Adapter,
    "ag2": AG2Adapter,
}


class AutoGenFamilyAdapter(BaseFrameworkAdapter):
    name = "autogen"
    is_router = True

    def is_available(self) -> bool:
        return (
            AutoGenAdapter().is_available()
            or AutoGenV4Adapter().is_available()
            or AG2Adapter().is_available()
        )

    def resolve_alias(self) -> str:
        requested = os.getenv("AUTOGEN_VERSION", "auto").lower()

        v2_available = AutoGenAdapter().is_available()
        v4_available = AutoGenV4Adapter().is_available()
        ag2_available = AG2Adapter().is_available()

        if requested == "v0.2":
            return "autogen_v2"
        if requested == "v0.4":
            return "autogen_v4"
        if requested == "ag2":
            return "ag2"

        if v2_available:
            return "autogen_v2"
        if v4_available:
            logger.warning("AutoGen v0.4 is installed but not yet implemented.")
            return "autogen_v4"
        if ag2_available:
            return "ag2"

        raise ImportError(
            "No AutoGen variant is available. Install with:\n"
            "  pip install 'praisonai-frameworks[autogen]' for v0.2\n"
            "  pip install 'praisonai-frameworks[autogen-v4]' for v0.4\n"
            "  pip install 'praisonai-frameworks[ag2]' for AG2"
        )

    def resolve(self, *, config: Optional[Dict[str, Any]] = None) -> BaseFrameworkAdapter:
        adapter_name = self.resolve_alias()
        adapter_cls = _ADAPTERS.get(adapter_name)
        if adapter_cls is None:
            raise ImportError(f"Unknown AutoGen adapter: {adapter_name}")
        logger.info("AutoGenFamilyAdapter resolved to: %s", adapter_name)
        return adapter_cls()

    def run(self, config: Dict[str, Any], llm_config: List[Dict], topic: str) -> str:
        raise RuntimeError(
            "AutoGenFamilyAdapter.run() should not be called directly; "
            "use resolve() first."
        )
