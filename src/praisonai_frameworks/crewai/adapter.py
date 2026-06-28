"""CrewAI framework adapter."""

from __future__ import annotations

import logging
import sys as _sys
from typing import Any, Callable, Dict, List, Optional

from praisonai_frameworks._availability import is_available
from praisonai_frameworks._observability import finalize_observability
from praisonai_frameworks._telemetry import scoped_telemetry_disable
from praisonai_frameworks.base import BaseFrameworkAdapter

logger = logging.getLogger(__name__)


class CrewAIAdapter(BaseFrameworkAdapter):
    name = "crewai"
    install_hint = 'pip install "praisonai-frameworks[crewai]"'
    requires_tools_extra = True

    def is_available(self) -> bool:
        return is_available("crewai")

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
            from crewai import Agent, Crew, Task
            from crewai.telemetry import Telemetry

            logging.getLogger("crewai.cli.config").setLevel(logging.ERROR)

            with scoped_telemetry_disable(Telemetry):
                agents = {}
                tasks = []
                tasks_dict = {}

                for role, details in config["roles"].items():
                    role_filled = self._format_template(details["role"], topic=topic)
                    goal_filled = self._format_template(details["goal"], topic=topic)
                    backstory_filled = self._format_template(details["backstory"], topic=topic)

                    agent_tools = [
                        tools_dict[tool]
                        for tool in details.get("tools", [])
                        if tools_dict and tool in tools_dict
                    ]

                    llm = self._resolve_llm(details.get("llm"), llm_config)
                    function_calling_llm = self._resolve_llm(
                        details.get("function_calling_llm"), llm_config
                    )

                    agent = Agent(
                        role=role_filled,
                        goal=goal_filled,
                        backstory=backstory_filled,
                        tools=agent_tools,
                        allow_delegation=details.get("allow_delegation", False),
                        llm=llm,
                        function_calling_llm=function_calling_llm,
                        max_iter=details.get("max_iter") or 15,
                        max_rpm=details.get("max_rpm") or None,
                        max_execution_time=details.get("max_execution_time") or None,
                        verbose=details.get("verbose", True),
                        cache=details.get("cache", True),
                        system_template=details.get("system_template") or None,
                        prompt_template=details.get("prompt_template") or None,
                        response_template=details.get("response_template") or None,
                    )

                    if agent_callback:
                        agent.step_callback = agent_callback

                    agents[role] = agent

                    for task_name, task_details in details.get("tasks", {}).items():
                        description_filled = self._format_template(
                            task_details["description"], topic=topic
                        )
                        expected_output_filled = self._format_template(
                            task_details["expected_output"], topic=topic
                        )

                        task_tools = []
                        for tool_name in task_details.get("tools", []):
                            if isinstance(tool_name, str) and tools_dict and tool_name in tools_dict:
                                task_tools.append(tools_dict[tool_name])
                            elif callable(tool_name):
                                task_tools.append(tool_name)

                        task = Task(
                            description=description_filled,
                            expected_output=expected_output_filled,
                            agent=agent,
                            tools=task_tools,
                            async_execution=task_details.get("async_execution", False),
                            context=[],
                            config=task_details.get("config", {}),
                            output_json=task_details.get("output_json"),
                            output_pydantic=task_details.get("output_pydantic"),
                            output_file=task_details.get("output_file", ""),
                            callback=task_details.get("callback"),
                            human_input=task_details.get("human_input", False),
                            create_directory=task_details.get("create_directory", False),
                        )

                        if task_callback:
                            task.callback = task_callback

                        tasks.append(task)
                        tasks_dict[task_name] = task

                for details in config["roles"].values():
                    for task_name, task_details in details.get("tasks", {}).items():
                        task = tasks_dict[task_name]
                        task.context = [
                            tasks_dict[ctx]
                            for ctx in task_details.get("context", [])
                            if ctx in tasks_dict
                        ]

                crew = Crew(agents=list(agents.values()), tasks=tasks, verbose=True)
                response = crew.kickoff()
                return f"### Task Output ###\n{response}"
        finally:
            status = "Failure" if _sys.exc_info()[0] is not None else "Success"
            finalize_observability(self.name, status=status)
