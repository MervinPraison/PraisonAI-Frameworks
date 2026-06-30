"""Agno framework adapter."""

from __future__ import annotations

import logging
import os
import sys as _sys
from inspect import iscoroutinefunction
from typing import Any, Callable, Dict, List, Optional

from praisonai_frameworks._availability import is_available
from praisonai_frameworks._observability import finalize_observability
from praisonai_frameworks.base import BaseFrameworkAdapter

logger = logging.getLogger(__name__)

_INVALID_API_KEYS = frozenset({"", "not-provided", "not_provided", "none", "null"})


class AgnoAdapter(BaseFrameworkAdapter):
    name = "agno"
    install_hint = 'pip install "praisonai-frameworks[agno]"'
    requires_tools_extra = True

    def is_available(self) -> bool:
        return is_available("agno")

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
        _ = (agent_callback, cli_config)
        try:
            self._ensure_agno()
            self._require_api_key(llm_config)
            self._warn_allow_delegation_without_handoffs(config, topic)
            task_specs = self._collect_ordered_tasks(config, topic)
            if not task_specs:
                return "### Agno Output ###\nNo tasks defined."

            if self._has_handoffs(config):
                logger.warning(
                    "Agno adapter does not support handoff.to yet; "
                    "running collected tasks without handoff wiring."
                )

            if len(task_specs) == 1:
                answer = self._run_single_task(
                    task_specs[0], llm_config, tools_dict, topic
                )
            else:
                answer = self._run_sequential(
                    task_specs, llm_config, tools_dict, topic
                )

            if task_callback:
                task_callback({"result": answer})

            return f"### Agno Output ###\n{answer}"
        finally:
            status = "Failure" if _sys.exc_info()[0] is not None else "Success"
            finalize_observability(self.name, status=status)

    def _ensure_agno(self) -> None:
        try:
            from agno.agent import Agent  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Agno is not installed. "
                'Install with: pip install "praisonai-frameworks[agno]"'
            ) from exc

    def _resolve_model_name(self, spec: Any, llm_config: Optional[List[Dict]]) -> str:
        from praisonaiagents.frameworks.base import BaseFrameworkAdapter as CoreBase

        return self._normalise_model_name(CoreBase._resolve_llm(self, spec, llm_config))

    def _normalise_model_name(self, model: str) -> str:
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    def _api_key(self, llm_config: Optional[List[Dict]]) -> Optional[str]:
        raw = llm_config[0].get("api_key") if llm_config else None
        if isinstance(raw, str) and raw.strip().lower() in _INVALID_API_KEYS:
            raw = None
        return raw or os.environ.get("OPENAI_API_KEY")

    def _require_api_key(self, llm_config: Optional[List[Dict]]) -> str:
        key = self._api_key(llm_config)
        if not key:
            raise ValueError(
                "Agno requires an API key. "
                "Set OPENAI_API_KEY or pass api_key in llm_config."
            )
        return key

    def _resolve_agno_model(self, spec: Any, llm_config: Optional[List[Dict]]):
        api_key = self._require_api_key(llm_config)
        base = llm_config[0].get("base_url") if llm_config else None
        model_id = self._resolve_model_name(spec, llm_config)
        if base:
            from agno.models.openai import OpenAILike

            return OpenAILike(id=model_id, api_key=api_key, base_url=base)
        from agno.models.openai import OpenAIChat

        return OpenAIChat(id=model_id, api_key=api_key)

    def _system_prompt(self, details: Dict[str, Any], topic: str) -> str:
        role = self._format_template(details.get("role", ""), topic=topic)
        goal = self._format_template(details.get("goal", ""), topic=topic)
        backstory = self._format_template(details.get("backstory", ""), topic=topic)
        return f"You are {role}. Goal: {goal}\n{backstory}"

    def _to_agno_tool(self, tool: Any, fallback_name: str) -> Optional[Any]:
        inner = getattr(tool, "_func", None)
        if inner is not None and callable(inner):
            if iscoroutinefunction(inner):
                logger.warning(
                    "Skipping async tool %s; Agno adapter uses synchronous run()",
                    getattr(tool, "name", fallback_name),
                )
                return None
            return inner

        if hasattr(tool, "run") and callable(tool.run):
            name = getattr(tool, "name", fallback_name)
            desc = getattr(tool, "description", name)

            def _bound(**kwargs):
                return tool.run(**kwargs)

            _bound.__name__ = name
            _bound.__doc__ = desc or name
            return _bound

        if callable(tool):
            if iscoroutinefunction(tool):
                logger.warning(
                    "Skipping async tool %s; Agno adapter uses synchronous run()",
                    getattr(tool, "__name__", fallback_name),
                )
                return None
            return tool

        return None

    def _resolve_tools(
        self,
        tool_names: Optional[List[Any]],
        tools_dict: Optional[Dict[str, Any]],
    ) -> List[Any]:
        if not tool_names or not tools_dict:
            return []

        tools: List[Any] = []
        for item in tool_names:
            if not isinstance(item, str):
                continue
            if item not in tools_dict:
                logger.warning("Tool '%s' not found in tools_dict", item)
                continue
            converted = self._to_agno_tool(tools_dict[item], item)
            if converted is not None:
                tools.append(converted)
        return tools

    def _build_agent(
        self,
        role_key: str,
        details: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
        tool_names: Optional[List[Any]] = None,
        expected_output: str = "",
    ):
        self._require_api_key(llm_config)
        from agno.agent import Agent

        role = self._format_template(details.get("role", role_key), topic=topic)
        goal = self._format_template(details.get("goal", ""), topic=topic)
        instructions = self._system_prompt(details, topic)
        model = self._resolve_agno_model(details.get("llm"), llm_config)
        tools = self._resolve_tools(
            tool_names if tool_names is not None else details.get("tools"),
            tools_dict,
        )
        kwargs: Dict[str, Any] = {
            "name": role or role_key,
            "role": role,
            "description": goal,
            "instructions": instructions,
            "model": model,
            "tools": tools,
            "search_knowledge": False,
        }
        if expected_output:
            kwargs["expected_output"] = expected_output
        return Agent(**kwargs)

    def _invoke_agent(self, agent, message: str) -> str:
        try:
            result = agent.run(input=message, stream=False)
        except Exception as exc:
            raise RuntimeError(f"Agno agent invocation failed: {exc}") from exc

        if result is None:
            raise RuntimeError("Agno agent returned no result")

        content = result.get_content_as_string() if hasattr(result, "get_content_as_string") else ""
        if not content or content == "null":
            output = getattr(result, "content", None)
            content = str(output).strip() if output is not None else ""
        if not content:
            messages = getattr(result, "messages", None) or []
            for msg in reversed(messages):
                if getattr(msg, "role", None) == "assistant":
                    text = getattr(msg, "content", None)
                    if text:
                        content = str(text).strip()
                        break
        if not content:
            raise RuntimeError("Agno agent returned empty content")
        return str(content).strip()

    def _agent_for_task(
        self,
        spec: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
    ):
        return self._build_agent(
            spec["role_key"],
            spec["role_details"],
            llm_config,
            tools_dict,
            topic,
            tool_names=spec["tools"],
            expected_output=spec.get("expected_output", ""),
        )

    def _collect_ordered_tasks(
        self, config: Dict[str, Any], topic: str
    ) -> List[Dict[str, Any]]:
        tasks_dict: Dict[str, Dict[str, Any]] = {}
        ordered_names: List[str] = []

        for role_key, details in config.get("roles", {}).items():
            for task_name, task_details in details.get("tasks", {}).items():
                description = task_details.get("description")
                if not description:
                    raise ValueError(
                        f"Task '{task_name}' in role '{role_key}' requires 'description'"
                    )
                tasks_dict[task_name] = {
                    "name": task_name,
                    "role_key": role_key,
                    "role_details": details,
                    "description": self._format_template(description, topic=topic),
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
                logger.warning(
                    "Unresolvable task context dependencies among %s; "
                    "falling back to YAML order.",
                    sorted(remaining),
                )
                ready = [name for name in ordered_names if name in remaining]
            for name in list(ready):
                if name in remaining:
                    resolved.append(tasks_dict[name])
                    remaining.discard(name)

        return resolved

    def _task_message(self, spec: Dict[str, Any], context_outputs: Dict[str, str]) -> str:
        missing = [ctx for ctx in spec["context"] if ctx not in context_outputs]
        if missing:
            logger.warning(
                "Task '%s' missing context from: %s",
                spec["name"],
                ", ".join(missing),
            )
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
                    "Agno requires explicit handoff targets or Team wiring.",
                    role_name,
                )

    def _run_single_task(
        self,
        spec: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
    ) -> str:
        agent = self._agent_for_task(spec, llm_config, tools_dict, topic)
        message = self._task_message(spec, {})
        return self._invoke_agent(agent, message)

    def _run_sequential(
        self,
        task_specs: List[Dict[str, Any]],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
    ) -> str:
        context_outputs: Dict[str, str] = {}
        answer = ""
        for spec in task_specs:
            message = self._task_message(spec, context_outputs)
            agent = self._agent_for_task(spec, llm_config, tools_dict, topic)
            answer = self._invoke_agent(agent, message)
            context_outputs[spec["name"]] = answer
        return answer
