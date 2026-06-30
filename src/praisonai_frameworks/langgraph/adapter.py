"""LangGraph framework adapter."""

from __future__ import annotations

import logging
import os
import sys as _sys
import warnings
from typing import Any, Callable, Dict, List, Optional

from praisonai_frameworks._availability import is_available
from praisonai_frameworks._observability import finalize_observability
from praisonai_frameworks.base import BaseFrameworkAdapter

logger = logging.getLogger(__name__)


class LangGraphAdapter(BaseFrameworkAdapter):
    name = "langgraph"
    install_hint = 'pip install "praisonai-frameworks[langgraph]"'
    requires_tools_extra = True

    def is_available(self) -> bool:
        return is_available("langgraph")

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
            task_specs = self._collect_ordered_tasks(config, topic)
            if not task_specs:
                return "### LangGraph Output ###\nNo tasks defined."

            if len(task_specs) == 1:
                answer = self._run_single_task(
                    task_specs[0], llm_config, tools_dict, topic
                )
            else:
                answer = self._run_sequential_graph(
                    task_specs, llm_config, tools_dict, topic
                )

            if task_callback:
                task_callback({"result": answer})

            return f"### LangGraph Output ###\n{answer}"
        finally:
            status = "Failure" if _sys.exc_info()[0] is not None else "Success"
            finalize_observability(self.name, status=status)

    def _resolve_model_name(self, spec: Any, llm_config: Optional[List[Dict]]) -> str:
        return self._normalise_model_name(self._resolve_raw_model_name(spec, llm_config))

    def _resolve_raw_model_name(self, spec: Any, llm_config: Optional[List[Dict]]) -> str:
        from praisonaiagents.frameworks.base import BaseFrameworkAdapter as CoreBase

        return CoreBase._resolve_llm(self, spec, llm_config)

    def _resolve_chat_model(self, spec: Any, llm_config: Optional[List[Dict]]):
        first_config = (
            llm_config[0]
            if llm_config and isinstance(llm_config[0], dict)
            else {}
        )
        base = first_config.get("base_url")
        key = first_config.get("api_key") or os.environ.get("OPENAI_API_KEY")

        raw_model = self._resolve_raw_model_name(spec, llm_config)
        provider = raw_model.split("/", 1)[0] if "/" in raw_model else None
        model = self._normalise_model_name(raw_model)

        non_openai = provider is not None and provider not in ("openai", "azure")
        if non_openai and not base:
            init_chat_model = self._load_init_chat_model()
            if init_chat_model is not None:
                return init_chat_model(model, model_provider=provider)

        from langchain_openai import ChatOpenAI

        kwargs: Dict[str, Any] = {"model": model}
        if key:
            kwargs["api_key"] = key
        if base:
            kwargs["base_url"] = base
        if not key and not base:
            raise ValueError(
                "LangGraph requires an API key. Set OPENAI_API_KEY or pass api_key in llm_config."
            )
        return ChatOpenAI(**kwargs)

    def _load_init_chat_model(self):
        """Return LangChain's init_chat_model if available, else None."""
        for module_name in ("langchain.chat_models", "langchain_core.language_models"):
            try:
                import importlib

                module = importlib.import_module(module_name)
            except ImportError:
                continue
            init_chat_model = getattr(module, "init_chat_model", None)
            if init_chat_model is not None:
                return init_chat_model
        return None

    def _extract_message_content(self, message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "\n".join(part for part in parts if part).strip()
        return str(content).strip()

    def _invoke_agent(self, agent, message: str) -> str:
        from langchain_core.messages import HumanMessage

        try:
            result = agent.invoke({"messages": [HumanMessage(content=message)]})
        except Exception as exc:
            raise RuntimeError(f"LangGraph agent invocation failed: {exc}") from exc

        messages = result.get("messages") if isinstance(result, dict) else None
        if not messages:
            raise RuntimeError("LangGraph agent returned no messages")

        content = self._extract_message_content(messages[-1])
        if not content:
            raise RuntimeError("LangGraph agent returned empty content")
        return content

    def _normalise_model_name(self, model: str) -> str:
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    def _system_prompt(self, details: Dict[str, Any], topic: str) -> str:
        role = self._format_template(details["role"], topic=topic)
        goal = self._format_template(details["goal"], topic=topic)
        backstory = self._format_template(details["backstory"], topic=topic)
        return f"You are {role}. Goal: {goal}\n{backstory}"

    def _to_langchain_tools(
        self,
        tool_names: Optional[List[Any]],
        tools_dict: Optional[Dict[str, Any]],
    ) -> List[Any]:
        from langchain_core.tools import BaseTool, StructuredTool

        if not tools_dict or not tool_names:
            return []

        tools: List[Any] = []
        for item in tool_names:
            if not isinstance(item, str) or item not in tools_dict:
                continue
            tool = tools_dict[item]
            if isinstance(tool, BaseTool):
                tools.append(tool)
            elif callable(tool):
                tools.append(
                    StructuredTool.from_function(
                        func=tool,
                        name=getattr(tool, "__name__", item),
                        description=(getattr(tool, "__doc__", None) or item).strip(),
                    )
                )
        return tools

    def _import_langgraph_module(self, submodule: str):
        """Import langgraph submodules."""
        import importlib

        return importlib.import_module(f"langgraph.{submodule}")

    def _create_react_agent(self, model, tools: List[Any], prompt: str):
        create_react_agent = self._import_langgraph_module("prebuilt").create_react_agent

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return create_react_agent(model, tools, prompt=prompt)

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

    def _run_single_task(
        self,
        spec: Dict[str, Any],
        llm_config: List[Dict],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
    ) -> str:
        details = spec["role_details"]
        model = self._resolve_chat_model(details.get("llm"), llm_config)
        tools = self._to_langchain_tools(spec["tools"], tools_dict)
        prompt = self._system_prompt(details, topic)
        agent = self._create_react_agent(model, tools, prompt)
        message = self._task_message(spec, {})
        return self._invoke_agent(agent, message)

    def _run_sequential_graph(
        self,
        task_specs: List[Dict[str, Any]],
        llm_config: List[Dict],
        tools_dict: Optional[Dict[str, Any]],
        topic: str,
    ) -> str:
        graph_mod = self._import_langgraph_module("graph")
        END = graph_mod.END
        START = graph_mod.START
        StateGraph = graph_mod.StateGraph
        from typing import TypedDict

        class GraphState(TypedDict):
            outputs: Dict[str, str]

        graph_builder = StateGraph(GraphState)

        for spec in task_specs:
            details = spec["role_details"]
            model = self._resolve_chat_model(details.get("llm"), llm_config)
            tools = self._to_langchain_tools(spec["tools"], tools_dict)
            prompt = self._system_prompt(details, topic)
            agent = self._create_react_agent(model, tools, prompt)
            node_name = spec["name"]

            def make_node(agent_ref, spec_ref):
                def node(state: GraphState) -> Dict[str, str]:
                    message = self._task_message(spec_ref, state.get("outputs", {}))
                    content = self._invoke_agent(agent_ref, message)
                    outputs = dict(state.get("outputs", {}))
                    outputs[spec_ref["name"]] = content
                    return {"outputs": outputs}

                return node

            graph_builder.add_node(node_name, make_node(agent, spec))

        graph_builder.add_edge(START, task_specs[0]["name"])
        for index in range(len(task_specs) - 1):
            graph_builder.add_edge(task_specs[index]["name"], task_specs[index + 1]["name"])
        graph_builder.add_edge(task_specs[-1]["name"], END)

        graph = graph_builder.compile()
        final = graph.invoke({"outputs": {}})
        last_name = task_specs[-1]["name"]
        outputs = final.get("outputs", {})
        return outputs.get(last_name, "")
