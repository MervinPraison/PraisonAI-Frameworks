"""OpenAI Agents SDK framework adapter."""

from __future__ import annotations

import logging
import os
import sys as _sys
from typing import Any, Callable, Dict, List, Optional

from praisonai_frameworks._availability import is_available
from praisonai_frameworks._observability import finalize_observability
from praisonai_frameworks.base import BaseFrameworkAdapter

logger = logging.getLogger(__name__)


class OpenAIAgentsAdapter(BaseFrameworkAdapter):
    name = "openai_agents"
    install_hint = 'pip install "praisonai-frameworks[openai-agents]"'
    requires_tools_extra = True

    def is_available(self) -> bool:
        return is_available("openai_agents")

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
            self._ensure_agents_sdk()
            task_specs = self._collect_ordered_tasks(config, topic)
            if not task_specs:
                return "### OpenAI Agents Output ###\nNo tasks defined."

            agents_by_key, agents_by_role = self._build_agent_registry(
                config, llm_config, tools_dict, topic
            )

            if self._has_handoffs(config) and len(task_specs) == 1:
                answer = self._run_with_handoffs(
                    config,
                    task_specs,
                    agents_by_key,
                    agents_by_role,
                    llm_config,
                    tools_dict,
                    topic,
                    cli_config,
                )
            elif len(task_specs) == 1:
                answer = self._run_single_task(
                    task_specs[0],
                    agents_by_key,
                    llm_config,
                    tools_dict,
                    topic,
                    cli_config,
                )
            else:
                answer = self._run_sequential(
                    task_specs,
                    agents_by_key,
                    llm_config,
                    tools_dict,
                    topic,
                    cli_config,
                )

            if task_callback:
                task_callback({"result": answer})

            return f"### OpenAI Agents Output ###\n{answer}"
        finally:
            status = "Failure" if _sys.exc_info()[0] is not None else "Success"
            finalize_observability(self.name, status=status)

    def _ensure_agents_sdk(self) -> None:
        try:
            from agents import Runner  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenAI Agents SDK not installed. "
                'Install with: pip install "praisonai-frameworks[openai-agents]"'
            ) from exc

    def _resolve_model_name(self, spec: Any, llm_config: Optional[List[Dict]]) -> str:
        from praisonaiagents.frameworks.base import BaseFrameworkAdapter as CoreBase

        return self._normalise_model_name(CoreBase._resolve_llm(self, spec, llm_config))

    def _normalise_model_name(self, model: str) -> str:
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    def _api_key(self, llm_config: Optional[List[Dict]]) -> Optional[str]:
        key = llm_config[0].get("api_key") if llm_config else None
        return key or os.environ.get("OPENAI_API_KEY")

    def _require_api_key(self, llm_config: Optional[List[Dict]]) -> str:
        key = self._api_key(llm_config)
        if not key:
            raise ValueError(
                "OpenAI Agents requires an API key. "
                "Set OPENAI_API_KEY or pass api_key in llm_config."
            )
        return key

    def _run_config(self, llm_config: Optional[List[Dict]]):
        from agents.models.openai_provider import OpenAIProvider
        from agents.run import RunConfig

        base = llm_config[0].get("base_url") if llm_config else None
        provider = OpenAIProvider(
            api_key=self._require_api_key(llm_config),
            base_url=base,
        )
        return RunConfig(tracing_disabled=True, model_provider=provider)

    def _max_turns(
        self,
        cli_config: Optional[Dict[str, Any]],
        role_details: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        if cli_config:
            depth = cli_config.get("handoff_max_depth")
            if depth is not None:
                return int(depth)
        if role_details and role_details.get("max_iter") is not None:
            return int(role_details["max_iter"])
        return None

    def _system_prompt(self, details: Dict[str, Any], topic: str) -> str:
        role = self._format_template(details["role"], topic=topic)
        goal = self._format_template(details["goal"], topic=topic)
        backstory = self._format_template(details["backstory"], topic=topic)
        return f"You are {role}. Goal: {goal}\n{backstory}"

    def _to_oai_tools(
        self,
        tool_names: Optional[List[Any]],
        tools_dict: Optional[Dict[str, Any]],
    ) -> List[Any]:
        from agents import function_tool

        if not tools_dict or not tool_names:
            return []

        tools: List[Any] = []
        for item in tool_names:
            if not isinstance(item, str) or item not in tools_dict:
                continue
            tool = tools_dict[item]
            if hasattr(tool, "name") and hasattr(tool, "on_invoke_tool"):
                tools.append(tool)
            elif callable(tool):
                name = getattr(tool, "name", None) or getattr(tool, "__name__", item)
                desc = getattr(tool, "description", None) or (tool.__doc__ or item)
                tools.append(
                    function_tool(
                        tool,
                        name_override=name,
                        description_override=str(desc).strip(),
                    )
                )
        return tools

    def _build_agent(
        self,
        role_key: str,
        details: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
        tool_names: Optional[List[Any]] = None,
    ):
        from agents import Agent

        self._require_api_key(llm_config)
        name = self._format_template(details.get("role", role_key), topic=topic)
        instructions = self._system_prompt(details, topic)
        model = self._resolve_model_name(details.get("llm"), llm_config)
        tools = self._to_oai_tools(
            tool_names if tool_names is not None else details.get("tools"),
            tools_dict,
        )
        return Agent(name=name, instructions=instructions, model=model, tools=tools)

    def _invoke_agent(
        self,
        agent,
        message: str,
        llm_config: Optional[List[Dict]],
        *,
        cli_config: Optional[Dict[str, Any]] = None,
        role_details: Optional[Dict[str, Any]] = None,
    ) -> str:
        from agents import Runner

        self._require_api_key(llm_config)
        kwargs: Dict[str, Any] = {
            "run_config": self._run_config(llm_config),
        }
        max_turns = self._max_turns(cli_config, role_details)
        if max_turns is not None:
            kwargs["max_turns"] = max_turns
        try:
            result = Runner.run_sync(agent, message, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"OpenAI Agents invocation failed: {exc}") from exc

        output = result.final_output
        if output is None:
            raise RuntimeError("OpenAI Agents returned no final output")
        return str(output).strip()

    def _agent_for_task(
        self,
        spec: Dict[str, Any],
        agents_by_key: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
    ):
        role_key = spec["role_key"]
        agent = self._build_agent(
            role_key,
            spec["role_details"],
            llm_config,
            tools_dict,
            topic,
            tool_names=spec["tools"],
        )
        base = agents_by_key.get(role_key)
        if base and getattr(base, "handoffs", None):
            agent.handoffs = list(base.handoffs)
        return agent

    def _collect_ordered_tasks(
        self, config: Dict[str, Any], topic: str
    ) -> List[Dict[str, Any]]:
        tasks_dict: Dict[str, Dict[str, Any]] = {}
        ordered_names: List[str] = []

        for role_key, details in config.get("roles", {}).items():
            for task_name, task_details in details.get("tasks", {}).items():
                tasks_dict[task_name] = {
                    "name": task_name,
                    "role_key": role_key,
                    "role_details": details,
                    "description": self._format_template(
                        task_details["description"], topic=topic
                    ),
                    "expected_output": self._format_template(
                        task_details.get("expected_output", ""), topic=topic
                    ),
                    "tools": task_details.get("tools") or details.get("tools") or [],
                    "context": [
                        ctx
                        for ctx in task_details.get("context", [])
                        if isinstance(ctx, str)
                    ],
                }
                ordered_names.append(task_name)

        if not tasks_dict:
            return []

        resolved: List[Dict[str, Any]] = []
        remaining = set(tasks_dict.keys())
        while remaining:
            resolved_names = {task["name"] for task in resolved}
            ready = [
                name
                for name in remaining
                if all(dep in resolved_names for dep in tasks_dict[name]["context"])
            ]
            if not ready:
                ready = [name for name in ordered_names if name in remaining]
            for name in list(ready):
                if name in remaining:
                    resolved.append(tasks_dict[name])
                    remaining.discard(name)

        return resolved

    def _task_message(self, spec: Dict[str, Any], context_outputs: Dict[str, str]) -> str:
        parts: List[str] = []
        for ctx in spec["context"]:
            if ctx in context_outputs:
                parts.append(f"[{ctx}]: {context_outputs[ctx]}")
        message = spec["description"]
        if parts:
            message = "Context from prior tasks:\n" + "\n".join(parts) + "\n\n" + message
        if spec["expected_output"]:
            message += f"\n\nExpected output: {spec['expected_output']}"
        return message

    def _role_field(self, details: Dict[str, Any], role_key: str, topic: str) -> str:
        return self._format_template(details.get("role", role_key), topic=topic)

    def _handoff_targets(self, details: Dict[str, Any]) -> List[str]:
        handoff = details.get("handoff") or {}
        if isinstance(handoff, dict):
            targets = handoff.get("to") or []
            return [t for t in targets if isinstance(t, str)]
        return []

    def _has_handoffs(self, config: Dict[str, Any]) -> bool:
        for details in config.get("roles", {}).values():
            if self._handoff_targets(details):
                return True
        return False

    def _warn_allow_delegation_without_handoffs(
        self, config: Dict[str, Any], topic: str
    ) -> None:
        for role_key, details in config.get("roles", {}).items():
            if details.get("allow_delegation") and not self._handoff_targets(details):
                role_name = self._role_field(details, role_key, topic)
                logger.warning(
                    "Role '%s' has allow_delegation but no handoff.to; "
                    "OpenAI Agents SDK requires explicit handoff targets.",
                    role_name,
                )

    def _build_agent_registry(
        self,
        config: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        self._warn_allow_delegation_without_handoffs(config, topic)
        by_key: Dict[str, Any] = {}
        by_role: Dict[str, Any] = {}

        for role_key, details in config.get("roles", {}).items():
            agent = self._build_agent(role_key, details, llm_config, tools_dict, topic)
            by_key[role_key] = agent
            by_role[self._role_field(details, role_key, topic)] = agent

        for role_key, details in config.get("roles", {}).items():
            targets = self._handoff_targets(details)
            if not targets:
                continue
            agent = by_key[role_key]
            handoff_agents = []
            for target in targets:
                target_agent = by_role.get(target) or by_key.get(target)
                if target_agent is None:
                    available = sorted(set(by_role) | set(by_key))
                    raise ValueError(
                        f"Handoff target '{target}' not found. "
                        f"Available roles: {', '.join(available)}"
                    )
                handoff_agents.append(target_agent)
            agent.handoffs = handoff_agents

        return by_key, by_role

    def _run_single_task(
        self,
        spec: Dict[str, Any],
        agents_by_key: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
        cli_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        agent = self._agent_for_task(spec, agents_by_key, llm_config, tools_dict, topic)
        message = self._task_message(spec, {})
        return self._invoke_agent(
            agent,
            message,
            llm_config,
            cli_config=cli_config,
            role_details=spec["role_details"],
        )

    def _run_sequential(
        self,
        task_specs: List[Dict[str, Any]],
        agents_by_key: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
        cli_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        context_outputs: Dict[str, str] = {}
        answer = ""
        for spec in task_specs:
            message = self._task_message(spec, context_outputs)
            agent = self._agent_for_task(spec, agents_by_key, llm_config, tools_dict, topic)
            answer = self._invoke_agent(
                agent,
                message,
                llm_config,
                cli_config=cli_config,
                role_details=spec["role_details"],
            )
            context_outputs[spec["name"]] = answer
        return answer

    def _run_with_handoffs(
        self,
        _config: Dict[str, Any],
        task_specs: List[Dict[str, Any]],
        agents_by_key: Dict[str, Any],
        _agents_by_role: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        _topic: str,
        cli_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        first_spec = task_specs[0]
        agent = self._agent_for_task(
            first_spec, agents_by_key, llm_config, tools_dict, _topic
        )
        message = self._task_message(first_spec, {})
        return self._invoke_agent(
            agent,
            message,
            llm_config,
            cli_config=cli_config,
            role_details=first_spec["role_details"],
        )
