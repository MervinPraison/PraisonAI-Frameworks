"""AutoGen v0.2 adapter."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from praisonai_frameworks._availability import is_available
from praisonai_frameworks.base import BaseFrameworkAdapter

logger = logging.getLogger(__name__)


class AutoGenAdapter(BaseFrameworkAdapter):
    name = "autogen_v2"
    install_hint = 'pip install "praisonai-frameworks[autogen]"'
    requires_tools_extra = True

    def is_available(self) -> bool:
        return is_available("autogen")

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
        import autogen

        llm_config_dict = {"config_list": llm_config}

        user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            is_termination_msg=lambda x: (x.get("content") or "")
            .rstrip()
            .rstrip(".")
            .lower()
            .endswith("terminate")
            or "TERMINATE" in (x.get("content") or ""),
            code_execution_config={"work_dir": "coding", "use_docker": False},
        )

        agents = {}
        tasks = []

        for role, details in config.get("roles", {}).items():
            agent_name = self._format_template(details["role"], topic=topic)
            agents[role] = autogen.AssistantAgent(
                name=agent_name,
                llm_config=llm_config_dict,
                system_message=self._format_template(details["backstory"], topic=topic)
                + '. Must Reply "TERMINATE" in the end when everything is done.',
            )

            if tools_dict and "tools" in details:
                for tool_name in details["tools"]:
                    if tool_name in tools_dict:
                        pass  # tool registration varies by autogen version

            for _task_name, task_details in details.get("tasks", {}).items():
                description_filled = self._format_template(
                    task_details["description"], topic=topic
                )
                tasks.append(
                    {
                        "recipient": agents[role],
                        "message": description_filled,
                        "summary_method": "last_msg",
                    }
                )

        response = user_proxy.initiate_chats(tasks)
        summary = response[-1].summary if hasattr(response[-1], "summary") else ""
        return "### AutoGen v0.2 Output ###\n" + summary
