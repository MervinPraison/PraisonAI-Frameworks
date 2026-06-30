"""LangGraph framework adapter.

Maps a PraisonAI ``agents.yaml`` config onto a LangGraph ``StateGraph`` where
each role becomes a node executed sequentially. Optional framework deps are
imported lazily inside :meth:`run` so importing this module is always cheap.
"""

from __future__ import annotations

import logging
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

    def _build_prompt(
        self, role: str, details: Dict[str, Any], topic: str
    ) -> str:
        role_filled = self._format_template(details.get("role") or role, topic=topic)
        goal_filled = self._format_template(details.get("goal") or "", topic=topic)
        backstory_filled = self._format_template(
            details.get("backstory") or "", topic=topic
        )

        task_lines: List[str] = []
        tasks = details.get("tasks") or {}
        if isinstance(tasks, dict):
            for task_details in tasks.values():
                if not isinstance(task_details, dict):
                    continue
                description = self._format_template(
                    task_details.get("description", ""), topic=topic
                )
                expected = self._format_template(
                    task_details.get("expected_output", ""), topic=topic
                )
                if description:
                    task_lines.append(f"Task: {description}")
                if expected:
                    task_lines.append(f"Expected output: {expected}")

        parts = [
            f"You are {role_filled}.",
            f"Goal: {goal_filled}." if goal_filled else "",
            f"Backstory: {backstory_filled}." if backstory_filled else "",
            *task_lines,
        ]
        return "\n".join(part for part in parts if part)

    def _call_llm(self, model: str, prompt: str, llm_config: List[Dict]) -> str:
        try:
            from litellm import completion
        except Exception:  # noqa: BLE001
            logger.debug("litellm unavailable; echoing prompt as result")
            return prompt

        kwargs: Dict[str, Any] = {"model": model}
        if llm_config and isinstance(llm_config[0], dict):
            cfg = llm_config[0]
            if cfg.get("base_url"):
                kwargs["base_url"] = cfg["base_url"]
            if cfg.get("api_key"):
                kwargs["api_key"] = cfg["api_key"]

        response = completion(
            messages=[{"role": "user", "content": prompt}], **kwargs
        )
        return response["choices"][0]["message"]["content"]

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
        status = "Failure"
        try:
            from langgraph.graph import END, START, StateGraph

            roles: Dict[str, Any] = config.get("roles", {})
            if not roles:
                raise ValueError("LangGraph config requires at least one role")

            graph: StateGraph = StateGraph(dict)
            role_names = list(roles.keys())

            def _make_node(role_name: str, details: Dict[str, Any]) -> Callable:
                prompt = self._build_prompt(role_name, details, topic)
                model = self._resolve_llm(details.get("llm"), llm_config)

                def _node(state: Dict[str, Any]) -> Dict[str, Any]:
                    history = state.get("messages", [])
                    context = "\n\n".join(history)
                    full_prompt = f"{context}\n\n{prompt}" if context else prompt
                    result = self._call_llm(model, full_prompt, llm_config)
                    if agent_callback:
                        agent_callback({"role": role_name, "content": result})
                    return {"messages": history + [result], "output": result}

                return _node

            for role_name, details in roles.items():
                graph.add_node(role_name, _make_node(role_name, details))

            graph.add_edge(START, role_names[0])
            for current, nxt in zip(role_names, role_names[1:]):
                graph.add_edge(current, nxt)
            graph.add_edge(role_names[-1], END)

            compiled = graph.compile()
            final_state = compiled.invoke({"messages": [], "topic": topic})

            output = final_state.get("output")
            if output is None:
                messages = final_state.get("messages", [])
                output = messages[-1] if messages else ""

            if task_callback:
                task_callback({"output": output})

            status = "Success"
            return f"### Task Output ###\n{output}"
        finally:
            finalize_observability(self.name, status=status)
