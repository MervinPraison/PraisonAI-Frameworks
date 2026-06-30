"""Google ADK framework adapter."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import re
import sys as _sys
import uuid
from inspect import iscoroutinefunction
from typing import Any, Callable, Dict, List, Optional

from praisonai_frameworks._availability import is_available
from praisonai_frameworks._observability import finalize_observability
from praisonai_frameworks.base import BaseFrameworkAdapter

logger = logging.getLogger(__name__)

_INVALID_API_KEYS = frozenset({"", "not-provided", "not_provided", "none", "null"})


class GoogleAdkAdapter(BaseFrameworkAdapter):
    name = "google_adk"
    install_hint = 'pip install "praisonai-frameworks[google-adk]"'
    requires_tools_extra = True

    def is_available(self) -> bool:
        return is_available("google_adk")

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
            self._ensure_adk()
            self._require_api_key(llm_config)
            task_specs = self._collect_ordered_tasks(config, topic)
            if not task_specs:
                return "### Google ADK Output ###\nNo tasks defined."

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

            return f"### Google ADK Output ###\n{answer}"
        finally:
            status = "Failure" if _sys.exc_info()[0] is not None else "Success"
            finalize_observability(self.name, status=status)

    def _ensure_adk(self) -> None:
        try:
            from google.adk import Agent  # noqa: F401
            from google.adk.runners import InMemoryRunner  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Google ADK is not installed. "
                'Install with: pip install "praisonai-frameworks[google-adk]"'
            ) from exc

    def _resolve_model_name(self, spec: Any, llm_config: Optional[List[Dict]]) -> str:
        from praisonaiagents.frameworks.base import BaseFrameworkAdapter as CoreBase

        has_role_spec = (isinstance(spec, str) and spec.strip()) or (
            isinstance(spec, dict) and spec.get("model")
        )
        if has_role_spec:
            return CoreBase._resolve_llm(self, spec, llm_config)
        if llm_config:
            explicit = llm_config[0].get("model")
            if isinstance(explicit, str) and explicit.strip():
                return explicit.strip()
        return CoreBase._resolve_llm(self, spec, llm_config)

    def _uses_gemini(self, model: str, llm_config: Optional[List[Dict]]) -> bool:
        model_lower = model.lower()
        if model_lower.startswith("gemini") or model_lower.startswith("google/gemini"):
            return True
        if model_lower.startswith("openai/") or model_lower.startswith("gpt-"):
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
                    "Google ADK requires an API key for Gemini. "
                    "Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass api_key in llm_config."
                )
            raise ValueError(
                "Google ADK requires an API key for this model. "
                "Set OPENAI_API_KEY or pass api_key in llm_config."
            )
        return key

    def _resolve_adk_model(self, spec: Any, llm_config: Optional[List[Dict]]):
        model = self._resolve_model_name(spec, llm_config)
        self._require_api_key(llm_config, model)
        if self._uses_gemini(model, llm_config):
            if model.lower().startswith("google/"):
                return model.split("/", 1)[1]
            return model
        litellm_model = model if "/" in model else f"openai/{model}"
        from google.adk.models.lite_llm import LiteLlm

        kwargs: Dict[str, Any] = {"model": litellm_model}
        api_key = self._api_key(llm_config, gemini=False)
        if api_key:
            kwargs["api_key"] = api_key
        base = llm_config[0].get("base_url") if llm_config else None
        if base:
            kwargs["api_base"] = base
        return LiteLlm(**kwargs)

    def _sanitize_agent_name(self, name: str, fallback: str = "agent") -> str:
        suffix = uuid.uuid5(uuid.NAMESPACE_DNS, str(name)).hex[:8]
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip()).strip("_")
        if safe and safe[0].isdigit():
            safe = f"agent_{safe}"
        if not safe or not safe.isidentifier() or safe == "user":
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", fallback).strip("_") or "agent"
            if safe[0].isdigit():
                safe = f"agent_{safe}"
        max_base = max(1, 63 - len(suffix) - 1)
        safe = safe[:max_base].rstrip("_") or "agent"
        return f"{safe}_{suffix}"

    def _stable_handoff_name(
        self, details: Dict[str, Any], role_key: str, topic: str
    ) -> str:
        raw = self._role_field(details, role_key, topic)
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", str(raw).strip()).strip("_")
        if safe and safe[0].isdigit():
            safe = f"agent_{safe}"
        if not safe or not safe.isidentifier() or safe == "user":
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", role_key).strip("_") or "agent"
            if safe[0].isdigit():
                safe = f"agent_{safe}"
        return (safe[:63] or "agent").rstrip("_")

    def _system_prompt(self, details: Dict[str, Any], topic: str) -> str:
        role = self._format_template(details.get("role", ""), topic=topic)
        goal = self._format_template(details.get("goal", ""), topic=topic)
        backstory = self._format_template(details.get("backstory", ""), topic=topic)
        return f"You are {role}. Goal: {goal}\n{backstory}"

    def _to_adk_tool(self, tool: Any, fallback_name: str) -> Optional[Any]:
        from google.adk.tools.function_tool import FunctionTool

        inner = getattr(tool, "_func", None)
        if inner is not None and callable(inner):
            if iscoroutinefunction(inner):
                logger.warning(
                    "Skipping async tool %s; Google ADK adapter uses synchronous run()",
                    getattr(tool, "name", fallback_name),
                )
                return None
            name = getattr(tool, "name", fallback_name)
            desc = getattr(tool, "description", name) or name

            def _wrapped(**kwargs):
                return inner(**kwargs)

            _wrapped.__name__ = name
            _wrapped.__doc__ = desc
            return FunctionTool(_wrapped)

        if hasattr(tool, "run") and callable(tool.run):
            name = getattr(tool, "name", fallback_name)
            desc = getattr(tool, "description", name) or name

            def _bound(**kwargs):
                return tool.run(**kwargs)

            _bound.__name__ = name
            _bound.__doc__ = desc
            return FunctionTool(_bound)

        if callable(tool):
            if iscoroutinefunction(tool):
                logger.warning(
                    "Skipping async tool %s; Google ADK adapter uses synchronous run()",
                    getattr(tool, "__name__", fallback_name),
                )
                return None
            name = getattr(tool, "__name__", fallback_name)
            desc = getattr(tool, "__doc__", None) or name

            def _wrapped(*args, **kwargs):
                return tool(*args, **kwargs)

            _wrapped.__name__ = name
            _wrapped.__doc__ = desc
            return FunctionTool(_wrapped)

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
            converted = self._to_adk_tool(tools_dict[item], item)
            if converted is not None:
                tools.append(converted)
        return tools

    def _build_adk_agent(
        self,
        role_key: str,
        task_name: str,
        details: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
        tool_names: Optional[List[Any]] = None,
        *,
        stable_name: bool = False,
    ):
        self._require_api_key(llm_config)
        from google.adk import Agent

        role = self._format_template(details.get("role", role_key), topic=topic)
        goal = self._format_template(details.get("goal", ""), topic=topic)
        instruction = self._system_prompt(details, topic)
        model = self._resolve_adk_model(details.get("llm"), llm_config)
        tools = self._resolve_tools(
            tool_names if tool_names is not None else details.get("tools"),
            tools_dict,
        )
        if stable_name:
            agent_name = self._stable_handoff_name(details, role_key, topic)
        else:
            agent_name = self._sanitize_agent_name(f"{role_key}_{task_name}", role_key)
        return Agent(
            name=agent_name,
            model=model,
            instruction=instruction,
            description=goal or role,
            tools=tools,
            mode="chat",
        )

    async def _invoke_agent_async(
        self, agent, message: str, llm_config: Optional[List[Dict]] = None
    ) -> str:
        from google.adk.runners import InMemoryRunner
        from google.adk.utils.content_utils import extract_text_from_content
        from google.genai import types

        model_name = self._resolve_model_name(None, llm_config)
        prev_google_key = os.environ.get("GOOGLE_API_KEY")
        if self._uses_gemini(model_name, llm_config):
            gemini_key = self._api_key(llm_config, gemini=True)
            if gemini_key:
                os.environ["GOOGLE_API_KEY"] = gemini_key

        runner = InMemoryRunner(agent=agent, app_name="praisonai")
        user_id = "praisonai_user"
        session_id = uuid.uuid4().hex
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        content = types.Content(
            role="user", parts=[types.Part.from_text(text=message)]
        )

        final_text = ""
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
            ):
                if event.is_final_response():
                    text = extract_text_from_content(event.content)
                    if text:
                        final_text = text.strip()
        finally:
            await runner.close()
            if prev_google_key is None:
                os.environ.pop("GOOGLE_API_KEY", None)
            else:
                os.environ["GOOGLE_API_KEY"] = prev_google_key

        if not final_text:
            raise RuntimeError("Google ADK agent returned empty content")
        return final_text

    def _invoke_agent(
        self, agent, message: str, llm_config: Optional[List[Dict]] = None
    ) -> str:
        coro = self._invoke_agent_async(agent, message, llm_config)

        def _run_coro() -> str:
            return asyncio.run(coro)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return _run_coro()
            except Exception as exc:
                raise RuntimeError(f"Google ADK agent invocation failed: {exc}") from exc

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(_run_coro).result()
        except Exception as exc:
            raise RuntimeError(f"Google ADK agent invocation failed: {exc}") from exc

    def _agent_for_task(
        self,
        spec: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
        agents_by_key: Optional[Dict[str, Any]] = None,
    ):
        role_key = spec["role_key"]
        stable = agents_by_key is not None
        agent = self._build_adk_agent(
            role_key,
            spec["name"],
            spec["role_details"],
            llm_config,
            tools_dict,
            topic,
            tool_names=spec["tools"],
            stable_name=stable,
        )
        if agents_by_key:
            base = agents_by_key.get(role_key)
            if base and getattr(base, "sub_agents", None):
                agent.sub_agents = list(base.sub_agents)
        return agent

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
                if task_name in tasks_dict:
                    logger.warning(
                        "Duplicate task name '%s' in role '%s'; "
                        "earlier definition will be overwritten.",
                        task_name,
                        role_key,
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
                    "Google ADK requires explicit handoff targets or sub-agent wiring.",
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
            agent = self._build_adk_agent(
                role_key,
                "handoff",
                details,
                llm_config,
                tools_dict,
                topic,
                stable_name=True,
            )
            by_key[role_key] = agent
            by_role[self._role_field(details, role_key, topic)] = agent

        for role_key, details in config.get("roles", {}).items():
            targets = self._handoff_targets(details)
            if not targets:
                continue
            agent = by_key[role_key]
            sub_agents = []
            for target in targets:
                target_agent = by_role.get(target) or by_key.get(target)
                if target_agent is None:
                    available = sorted(set(by_role) | set(by_key))
                    raise ValueError(
                        f"Handoff target '{target}' not found. "
                        f"Available roles: {', '.join(available)}"
                    )
                sub_agents.append(target_agent)
            agent.sub_agents = sub_agents

        return by_key, by_role

    def _run_with_handoffs(
        self,
        task_specs: List[Dict[str, Any]],
        agents_by_key: Dict[str, Any],
        _agents_by_role: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
    ) -> str:
        first_spec = task_specs[0]
        agent = self._agent_for_task(
            first_spec,
            llm_config,
            tools_dict,
            topic,
            agents_by_key=agents_by_key,
        )
        message = self._task_message(first_spec, {})
        return self._invoke_agent(agent, message, llm_config)

    def _run_single_task(
        self,
        spec: Dict[str, Any],
        llm_config: Optional[List[Dict]],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
    ) -> str:
        agent = self._agent_for_task(spec, llm_config, tools_dict, topic)
        message = self._task_message(spec, {})
        return self._invoke_agent(agent, message, llm_config)

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
            answer = self._invoke_agent(agent, message, llm_config)
            context_outputs[spec["name"]] = answer
        return answer
