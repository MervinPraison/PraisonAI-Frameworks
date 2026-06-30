"""Shared base adapter with CrewAI-friendly LLM resolution."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from praisonaiagents.frameworks.base import BaseFrameworkAdapter as _CoreBase


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
