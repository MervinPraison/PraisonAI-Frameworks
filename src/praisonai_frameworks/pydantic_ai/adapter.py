"""Pydantic AI framework adapter."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import re
import sys as _sys
from inspect import iscoroutinefunction
from typing import Any, Callable, Dict, List, Optional

from praisonai_frameworks._availability import is_available
from praisonai_frameworks._observability import finalize_observability
from praisonai_frameworks.base import BaseFrameworkAdapter

logger = logging.getLogger(__name__)

_INVALID_API_KEYS = frozenset({"", "not-provided", "not_provided", "none", "null"})


class PydanticAiAdapter(BaseFrameworkAdapter):
    name = "pydantic_ai"
    install_hint = 'pip install "praisonai-frameworks[pydantic-ai]"'
    requires_tools_extra = True

    def is_available(self) -> bool:
        return is_available("pydantic_ai")

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
            self._ensure_pydantic_ai()
            self._require_api_key(llm_config)
            task_specs = self._collect_ordered_tasks(config, topic)
            if not task_specs:
                return "### Pydantic AI Output ###\nNo tasks defined."

            agents_by_key: Dict[str, Any] = {}
            agents_by_role: Dict[str, Any] = {}
            if self._has_handoffs(config):
                agents_by_key, agents_by_role = self._build_agent_registry(
                    config, llm_config, tools_dict, topic
                )
            else:
                self._warn_allow_delegation_without_handoffs(config, topic)

            if self._has_handoffs(config) and len(task_specs) == 1:
                answer = self._run_with_handoffs(
                    task_specs,
                    agents_by_key,
                    agents_by_role,
                    llm_config,
                    tools_dict,
                    topic,
                )
            elif len(task_specs) == 1:
                answer = self._run_single_task(
                    task_specs[0], llm_config, tools_dict, topic
                )
            else:
                answer = self._run_sequential(
                    task_specs, llm_config, tools_dict, topic
                )

            if task_callback:
                task_callback({"result": answer})

            return f"### Pydantic AI Output ###\n{answer}"
        finally:
            status = "Failure" if _sys.exc_info()[0] is not None else "Success"
            finalize_observability(self.name, status=status)

    def _ensure_pydantic_ai(self) -> None:
        try:
            from pydantic_ai import Agent  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Pydantic AI is not installed. "
                'Install with: pip install "praisonai-frameworks[pydantic-ai]"'
            ) from exc

    def _resolve_model_name(self, spec: Any, llm_config: Optional[List[Dict]]) -> str:
        from praisonaiagents.frameworks.base import BaseFrameworkAdapter as CoreBase

        return CoreBase._resolve_llm(self, spec, llm_config)

    def _uses_gemini(self, model: str, llm_config: Optional[List[Dict]]) -> bool:
        model_lower = model.lower()
        if model_lower.startswith("gemini") or model_lower.startswith("google/"):
            return True
        if model_lower.startswith("google:"):
            return True
        if model_lower.startswith("openai/") or model_lower.startswith("openai:"):
            return False
        if model_lower.startswith("gpt-"):
            return False
        if llm_config:
            raw = llm_config[0].get("api_key")
            if isinstance(raw, str) and raw.strip().lower() not in _INVALID_API_KEYS:
                if raw.startswith("sk-"):
                    return False
                if raw.startswith("AIza"):
                    return True
        google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        if google_key and not openai_key:
            return True
        return False

    def _api_key(
        self, llm_config: Optional[List[Dict]], *, gemini: bool
    ) -> Optional[str]:
        raw = llm_config[0].get("api_key") if llm_config else None
        if isinstance(raw, str) and raw.strip().lower() in _INVALID_API_KEYS:
            raw = None
        if gemini:
            return raw or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        return raw or os.environ.get("OPENAI_API_KEY")

    def _require_api_key(
        self, llm_config: Optional[List[Dict]], model: Optional[str] = None
    ) -> str:
        resolved = model or self._resolve_model_name(None, llm_config)
        gemini = self._uses_gemini(resolved, llm_config)
        key = self._api_key(llm_config, gemini=gemini)
        if not key:
            if gemini:
                raise ValueError(
                    "Pydantic AI requires an API key for Gemini. "
                    "Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass api_key in llm_config."
                )
            raise ValueError(
                "Pydantic AI requires an API key. "
                "Set OPENAI_API_KEY or pass api_key in llm_config."
            )
        return key

    def _to_pydantic_model(self, spec: Any, llm_config: Optional[List[Dict]]) -> str:
        model = self._resolve_model_name(spec, llm_config)
        self._require_api_key(llm_config, model)
        lower = model.lower()
        if lower.startswith("google:") or lower.startswith("openai:"):
            return model
        if lower.startswith("google/gemini") or lower.startswith("gemini"):
            bare = model.split("/", 1)[1] if "/" in model else model
            return f"google:{bare}"
        if lower.startswith("openai/"):
            return f"openai:{model.split('/', 1)[1]}"
        return f"openai:{model}"

    def _push_api_keys(
        self, llm_config: Optional[List[Dict]], model: str
    ) -> Dict[str, Optional[str]]:
        prev: Dict[str, Optional[str]] = {}
        if self._uses_gemini(model, llm_config):
            key = self._api_key(llm_config, gemini=True)
            if key:
                prev["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
                os.environ["GOOGLE_API_KEY"] = key
        else:
            key = self._api_key(llm_config, gemini=False)
            if key:
                prev["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
                os.environ["OPENAI_API_KEY"] = key
        return prev

    def _pop_api_keys(self, prev: Dict[str, Optional[str]]) -> None:
        for env_name, old in prev.items():
            if old is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = old

    def _system_prompt(self, details: Dict[str, Any], topic: str) -> str:
        role = self._format_template(details.get("role", ""), topic=topic)
        goal = self._format_template(details.get("goal", ""), topic=topic)
        backstory = self._format_template(details.get("backstory", ""), topic=topic)
        return f"You are {role}. Goal: {goal}\n{backstory}"

    def _handoff_tool_name(self, target: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", str(target).strip()).strip("_").lower()
        if not safe or safe[0].isdigit():
            safe = f"agent_{safe or 'specialist'}"
        name = f"handoff_to_{safe}"[:63]
        return name.rstrip("_") or "handoff_to_specialist"

    def _to_plain_tool(self, tool: Any, fallback_name: str) -> Optional[Callable[..., Any]]:
        inner = getattr(tool, "_func", None)
        if inner is not None and callable(inner):
            if iscoroutinefunction(inner):
                logger.warning(
                    "Skipping async tool %s; use sync tools with pydantic_ai adapter",
                    getattr(tool, "name", fallback_name),
                )
                return None
            fn = inner
            name = getattr(tool, "name", fallback_name)
            desc = getattr(tool, "description", name) or name
            fn.__name__ = name
            fn.__doc__ = desc
            return fn

        if hasattr(tool, "run") and callable(tool.run):
            name = getattr(tool, "name", fallback_name)
            desc = getattr(tool, "description", name) or name

            def _bound(**kwargs):
                return tool.run(**kwargs)

            _bound.__name__ = name
            _bound.__doc__ = desc
            return _bound

        if callable(tool):
            if iscoroutinefunction(tool):
                logger.warning(
                    "Skipping async tool %s; use sync tools with pydantic_ai adapter",
                    getattr(tool, "__name__", fallback_name),
                )
                return None
            return tool

        return None

    def _resolve_tools(
        self,
        tool_names: Optional[List[Any]],
        tools_dict: Optional[Dict[str, Any]],
    ) -> List[Callable[..., Any]]:
        if not tool_names or not tools_dict:
            return []

        tools: List[Callable[..., Any]] = []
        for item in tool_names:
            if not isinstance(item, str):
                continue
            if item not in tools_dict:
                logger.warning("Tool '%s' not found in tools_dict", item)
                continue
            converted = self._to_plain_tool(tools_dict[item], item)
            if converted is not None:
                tools.append(converted)
        return tools

    def _attach_plain_tools(self, agent, tools: List[Callable[..., Any]]) -> None:
        for fn in tools:
            agent.tool_plain(fn)

    def _build_agent(
        self,
        role_key: str,
        details: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
        tool_names: Optional[List[Any]] = None,
        *,
        extra_instructions: str = "",
    ):
        from pydantic_ai import Agent

        role = self._format_template(details.get("role", role_key), topic=topic)
        instructions = self._system_prompt(details, topic)
        if extra_instructions:
            instructions = f"{instructions}\n\n{extra_instructions}"
        model = self._to_pydantic_model(details.get("llm"), llm_config)
        agent = Agent(
            model,
            name=role or role_key,
            instructions=instructions,
        )
        tools = self._resolve_tools(
            tool_names if tool_names is not None else details.get("tools"),
            tools_dict,
        )
        self._attach_plain_tools(agent, tools)
        return agent

    def _register_handoff_tools(
        self,
        router,
        targets: List[str],
        agents_by_role: Dict[str, Any],
        agents_by_key: Dict[str, Any],
    ) -> List[str]:
        tool_names: List[str] = []
        for target in targets:
            specialist = agents_by_role.get(target) or agents_by_key.get(target)
            if specialist is None:
                continue
            tool_name = self._handoff_tool_name(target)
            tool_names.append(tool_name)
            delegate = self._build_handoff_delegate(specialist, target)
            router.tool(name=tool_name)(delegate)

        return tool_names

    def _build_handoff_delegate(self, specialist: Any, target: str):
        from pydantic_ai import RunContext

        async def _delegate(ctx: RunContext, request: str) -> str:
            result = await specialist.run(request, usage=ctx.usage)
            output = result.output
            return str(output).strip() if output is not None else ""

        tool_name = self._handoff_tool_name(target)
        _delegate.__name__ = tool_name
        _delegate.__doc__ = f"Delegate to {target}. Pass the user request verbatim."
        _delegate.__globals__["RunContext"] = RunContext
        return _delegate

    def _handoff_instructions(self, tool_names: List[str]) -> str:
        if not tool_names:
            return ""
        joined = ", ".join(tool_names)
        return (
            "When the user request fits a specialist, delegate using one of these tools: "
            f"{joined}. After calling a handoff tool, respond with exactly the tool output "
            "unchanged and nothing else."
        )

    def _invoke_agent(
        self,
        agent,
        message: str,
        llm_config: Optional[List[Dict]],
        *,
        model: Optional[str] = None,
    ) -> str:
        resolved_model = model or getattr(agent, "model", None)
        model_str = str(resolved_model) if resolved_model is not None else self._to_pydantic_model(
            None, llm_config
        )

        def _run_sync() -> str:
            prev = self._push_api_keys(llm_config, model_str)
            try:
                result = agent.run_sync(message)
            except Exception as exc:
                raise RuntimeError(f"Pydantic AI agent invocation failed: {exc}") from exc
            finally:
                self._pop_api_keys(prev)

            output = result.output
            if output is None:
                raise RuntimeError("Pydantic AI agent returned no output")
            text = str(output).strip()
            if not text:
                raise RuntimeError("Pydantic AI agent returned empty content")
            return text

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _run_sync()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(_run_sync).result()
        except Exception as exc:
            raise RuntimeError(f"Pydantic AI agent invocation failed: {exc}") from exc

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
                    "Pydantic AI requires explicit handoff targets or delegation tools.",
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
            agent = self._build_agent(
                role_key, details, llm_config, tools_dict, topic
            )
            by_key[role_key] = agent
            by_role[self._role_field(details, role_key, topic)] = agent

        for _role_key, details in config.get("roles", {}).items():
            targets = self._handoff_targets(details)
            if not targets:
                continue
            for target in targets:
                if target not in by_role and target not in by_key:
                    available = sorted(set(by_role) | set(by_key))
                    raise ValueError(
                        f"Handoff target '{target}' not found. "
                        f"Available roles: {', '.join(available)}"
                    )

        return by_key, by_role

    def _run_with_handoffs(
        self,
        task_specs: List[Dict[str, Any]],
        agents_by_key: Dict[str, Any],
        agents_by_role: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
    ) -> str:
        first_spec = task_specs[0]
        targets = self._handoff_targets(first_spec["role_details"])
        tool_names = [
            self._handoff_tool_name(t)
            for t in targets
            if (agents_by_role.get(t) or agents_by_key.get(t)) is not None
        ]
        router = self._build_agent(
            first_spec["role_key"],
            first_spec["role_details"],
            llm_config,
            tools_dict,
            topic,
            tool_names=first_spec["tools"],
            extra_instructions=self._handoff_instructions(tool_names),
        )
        self._register_handoff_tools(router, targets, agents_by_role, agents_by_key)
        message = self._task_message(first_spec, {})
        model = self._to_pydantic_model(
            first_spec["role_details"].get("llm"), llm_config
        )
        return self._invoke_agent(router, message, llm_config, model=model)

    def _run_single_task(
        self,
        spec: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
    ) -> str:
        agent = self._agent_for_task(spec, llm_config, tools_dict, topic)
        message = self._task_message(spec, {})
        model = self._to_pydantic_model(spec["role_details"].get("llm"), llm_config)
        return self._invoke_agent(agent, message, llm_config, model=model)

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
            model = self._to_pydantic_model(spec["role_details"].get("llm"), llm_config)
            answer = self._invoke_agent(agent, message, llm_config, model=model)
            context_outputs[spec["name"]] = answer
        return answer
