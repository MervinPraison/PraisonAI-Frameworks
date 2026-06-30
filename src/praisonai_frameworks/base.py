"""Shared base adapter with CrewAI-friendly LLM resolution."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from praisonaiagents.frameworks.base import BaseFrameworkAdapter as _CoreBase
except ImportError:
    class _CoreBase:
        """Local fallback used when ``praisonaiagents.frameworks`` is unavailable.

        Published ``praisonaiagents`` releases do not yet ship the
        ``frameworks`` package. Falling back to this minimal base keeps
        ``praisonai-frameworks`` importable and usable from a PyPI-only
        install instead of crashing at import time. Once core ships
        ``praisonaiagents.frameworks.base`` the real base is used instead.
        """

        DEFAULT_MODEL = "openai/gpt-4o-mini"
        name: str = ""
        install_hint: str = ""
        is_router: bool = False
        requires_tools_extra: bool = False

        def is_available(self) -> bool:
            raise NotImplementedError

        def run(self, *args: Any, **kwargs: Any) -> str:
            raise NotImplementedError

        @staticmethod
        def _format_template(template: Any, **values: Any) -> Any:
            if isinstance(template, str):
                try:
                    return template.format(**values)
                except (KeyError, IndexError, ValueError):
                    return template
            return template


class BaseFrameworkAdapter(_CoreBase):
    """Extends core base with optional CrewAI LLM objects."""

    def _resolve_llm(self, spec: Any, llm_config: Optional[List[Dict]]):
        base = llm_config[0].get("base_url") if (llm_config and len(llm_config) > 0) else None
        key = llm_config[0].get("api_key") if (llm_config and len(llm_config) > 0) else None

        if isinstance(spec, str) and spec.strip():
            model = spec.strip()
        elif isinstance(spec, dict) and spec.get("model"):
            model = spec["model"]
        else:
            model = os.environ.get("MODEL_NAME") or self.DEFAULT_MODEL

        try:
            from crewai import LLM

            return LLM(model=model, base_url=base, api_key=key)
        except Exception:
            return model
