"""AG2 adapter stub (not yet implemented)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from praisonai_frameworks._availability import is_available
from praisonai_frameworks.base import BaseFrameworkAdapter


class AG2Adapter(BaseFrameworkAdapter):
    name = "ag2"
    install_hint = 'pip install "praisonai-frameworks[ag2]"'
    requires_tools_extra = False
    implemented: bool = False

    def is_available(self) -> bool:
        if not self.implemented:
            return False
        return is_available("ag2")

    def run(
        self,
        config: Dict[str, Any],
        llm_config: List[Dict],
        topic: str,
        *,
        tools_dict: Optional[Dict[str, Any]] = None,
        agent_callback: Optional[Callable] = None,
        task_callback: Optional[Callable] = None,
        cli_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError(
            "AG2 adapter is not yet implemented. "
            "Use framework='autogen' with AUTOGEN_VERSION=v0.2."
        )
