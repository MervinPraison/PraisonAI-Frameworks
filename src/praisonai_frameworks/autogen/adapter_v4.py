"""AutoGen v0.4 adapter (autogen-agentchat / autogen-ext)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from praisonai_frameworks._availability import is_available
from praisonai_frameworks.base import BaseFrameworkAdapter

logger = logging.getLogger(__name__)


class AutoGenV4Adapter(BaseFrameworkAdapter):
    name = "autogen_v4"
    install_hint = 'pip install "praisonai-frameworks[autogen-v4]"'
    requires_tools_extra = True
    implemented: bool = True

    def is_available(self) -> bool:
        return is_available("autogen_v4")

    def _build_model_client(self, llm_config: List[Dict]):
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        entry = llm_config[0] if llm_config else {}
        model = entry.get("model") or os.environ.get("MODEL_NAME") or self.DEFAULT_MODEL
        kwargs: Dict[str, Any] = {"model": model}
        if entry.get("api_key"):
            kwargs["api_key"] = entry["api_key"]
        if entry.get("base_url"):
            kwargs["base_url"] = entry["base_url"]
        return OpenAIChatCompletionClient(**kwargs)

    async def _arun(
        self,
        config: Dict[str, Any],
        llm_config: List[Dict],
        topic: str,
    ) -> str:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
        from autogen_agentchat.teams import RoundRobinGroupChat

        model_client = self._build_model_client(llm_config)

        agents = []
        task_messages: List[str] = []

        for role, details in config.get("roles", {}).items():
            agent_name = self._sanitize_name(
                self._format_template(details.get("role", role), topic=topic)
            )
            system_message = self._format_template(
                details.get("backstory", details.get("goal", "")), topic=topic
            )
            agents.append(
                AssistantAgent(
                    name=agent_name,
                    model_client=model_client,
                    system_message=system_message
                    + ' Reply "TERMINATE" when the task is complete.',
                )
            )

            for _task_name, task_details in details.get("tasks", {}).items():
                task_messages.append(
                    self._format_template(task_details["description"], topic=topic)
                )

        if not agents:
            raise ValueError("AutoGen v0.4 adapter requires at least one role in config.")

        termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(
            max_messages=len(agents) * 4 + 4
        )
        team = RoundRobinGroupChat(agents, termination_condition=termination)

        task = "\n".join(task_messages) if task_messages else (topic or "")
        result = await team.run(task=task)

        try:
            await model_client.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.debug("model_client.close() failed", exc_info=True)

        summary = ""
        if getattr(result, "messages", None):
            last = result.messages[-1]
            summary = getattr(last, "content", "") or ""
        return "### AutoGen v0.4 Output ###\n" + str(summary)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in (name or ""))
        cleaned = cleaned.strip("_") or "agent"
        if cleaned[0].isdigit():
            cleaned = "a_" + cleaned
        return cleaned

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
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._arun(config, llm_config, topic))
        raise RuntimeError(
            "AutoGenV4Adapter.run() called from a running event loop; "
            "use 'await adapter.arun(...)' instead."
        )
