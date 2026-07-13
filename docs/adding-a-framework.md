# Adding a framework adapter

This guide covers adding a new third-party framework adapter to **praisonai-frameworks**. For agent-runtime rules and repo scope, see [AGENTS.md](../AGENTS.md) at the repository root.

---

## Two paths

| Path | When | Template |
|------|------|----------|
| **In-repo** (this repository) | Official PraisonAI-supported frameworks | Full checklist below |
| **External package** | Community or private bridges | [examples/third_party_adapter/](../examples/third_party_adapter/) |

External adapters depend only on `praisonaiagents`, register their own entry point, and do not require changes here.

---

## Adapter contract

Subclass `BaseFrameworkAdapter` from `praisonaiagents.frameworks.base` (via `praisonai_frameworks.base` for in-repo adapters).

**Gold standard:** [src/praisonai_frameworks/crewai/adapter.py](../src/praisonai_frameworks/crewai/adapter.py)  
**Modern leaf pattern (single / sequential / handoff):** [src/praisonai_frameworks/pydantic_ai/adapter.py](../src/praisonai_frameworks/pydantic_ai/adapter.py)

### Class attributes

| Attribute | Example | Notes |
|-----------|---------|-------|
| `name` | `"pydantic_ai"` | Must match YAML `framework:` and the entry-point key (underscore) |
| `install_hint` | `'pip install "praisonai-frameworks[pydantic-ai]"'` | Shown when the probe fails |
| `requires_tools_extra` | `True` if YAML `tools:` need `praisonai-tools` | Add `praisonai-tools>=0.1.0` to the optional extra |
| `is_router` | `False` (default) | `True` only for family routers (e.g. AutoGen) |

### Required methods

```python
def is_available(self) -> bool:
    return is_available("my_framework")  # from praisonai_frameworks._availability

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
    ...
```

### Leaf adapter rules

1. **Lazy-import** the third-party SDK inside `run()` (or small private helpers) — never at module top.
2. Wrap the body in **`try` / `finally`** and call `finalize_observability(self.name, status=...)` from `praisonai_frameworks._observability`.
3. **Return a string** with a clear sentinel header, e.g. `"### My Framework Output ###\n{content}"`.
4. **Do not override** `arun`, `resolve`, `setup`, or `cleanup` unless you are building a family router.
5. Use `scoped_telemetry_disable` only when the third-party SDK has noisy telemetry (CrewAI pattern).

### LLM resolution

- For backends expecting LangChain or OpenAI clients, use `BaseFrameworkAdapter._resolve_llm` for **model strings**, then build your SDK client.
- Read `api_key` / `base_url` from `llm_config[0]`; fall back to `OPENAI_API_KEY` where appropriate.
- The local `praisonai_frameworks.base._resolve_llm` returns a CrewAI `LLM` object — use it only for the CrewAI adapter.

---

## YAML mapping

Standard `agents.yaml` shape:

```yaml
framework: my_framework
topic: Example task
roles:
  researcher:                    # role key (internal ID)
    role: Research Analyst       # display name
    goal: Gather information
    backstory: Expert researcher
    tools: [search]              # optional; names into tools_dict
    handoff:                     # optional; not all adapters support this
      to: [English Agent]
    tasks:
      research:                  # task name (used for context chains)
        description: Research {topic}
        expected_output: A short summary
        context: [prior_task]    # optional sequential dependency
        tools: [search]
```

### Conventions

| YAML | Maps to |
|------|---------|
| `roles` | Agents |
| Nested `tasks` | Tasks or per-task agent runs |
| `context` | Sequential task dependencies (prior task names) |
| `{topic}` | Template variable via `_format_template` |
| `tools` | Names looked up in `tools_dict` (and inline callables where supported) |

### Execution patterns

Choose the pattern that fits your SDK:

| Pattern | Use when | Reference |
|---------|----------|-----------|
| Framework-native crew | SDK has Agent / Task / Crew | CrewAI |
| Native graph | SDK has a workflow / graph engine | LangGraph |
| Python sequential loop | Agent-centric SDK, no graph API | OpenAI Agents, Agno, Google ADK, Pydantic AI |
| Handoff delegation | YAML has `handoff.to` | Map to SDK primitive (`handoffs`, `sub_agents`, `Team(route)`, etc.) |

Most leaf adapters reuse private helpers: `_collect_ordered_tasks`, `_task_message`, `_system_prompt`, and inherited `_format_template`.

---

## File structure

```
src/praisonai_frameworks/my_framework/
├── __init__.py
├── adapter.py
└── README.md

examples/
└── agents_my_framework.yaml
    # optional: agents_my_framework_multi.yaml, agents_my_framework_handoff.yaml

tests/unit/
├── test_my_framework_adapter_protocol.py
└── test_my_framework_adapter_run.py

tests/integration/my_framework_adapter/
├── test_my_framework_basic.py
└── test_my_framework_live.py   # optional
```

**Integration folder naming:** use `<name>_adapter/` (e.g. `langgraph_adapter/`). Never name a folder `langgraph/` or `agents/` — it shadows PyPI packages and breaks imports.

---

## Registration checklist

Complete every step for an in-repo adapter.

### 1. Adapter module

Create `src/praisonai_frameworks/<name>/adapter.py` (+ `__init__.py`, `README.md`).

### 2. Availability probe

Add a probe to `_PROBES` in [src/praisonai_frameworks/_availability.py](../src/praisonai_frameworks/_availability.py):

```python
"my_framework": lambda: importlib.util.find_spec("my_sdk") is not None,
```

Use a dedicated probe function if the SDK needs distribution or symbol checks (see `_openai_agents_probe`, `_agno_probe`).

### 3. Optional extra (`pyproject.toml`)

Pip extra names use **hyphens**; entry points and YAML use **underscores**.

```toml
[project.optional-dependencies]
my-framework = [
    "some-sdk>=x.y",
    "praisonai-tools>=0.1.0",  # if requires_tools_extra = True
]
```

### 4. Entry point (`pyproject.toml`)

```toml
[project.entry-points."praisonai.framework_adapters"]
my_framework = "praisonai_frameworks.my_framework.adapter:MyFrameworkAdapter"
```

### 5. `all` meta-extra (optional)

Include your extra in `all` only if it is compatible with every other extra in that group. `google-adk` is excluded from `all` because it conflicts with `langgraph` and `openai-agents`.

### 6. Dependency conflicts (`tool.uv`)

If your extra cannot co-install with others, declare conflicts in `pyproject.toml`:

```toml
[tool.uv]
conflicts = [
    [{ extra = "my-framework" }, { extra = "langgraph" }],
    [{ extra = "all" }, { extra = "my-framework" }],
]
```

Run `uv lock` locally before publishing to verify resolution.

### 7. Entry-point unit test

Add `"my_framework"` to the `expected` set in [tests/unit/test_entry_points_registered.py](../tests/unit/test_entry_points_registered.py).

### 8. CI matrix

In [.github/workflows/ci.yml](../.github/workflows/ci.yml):

- Add `"my-framework"` to `matrix.extra`.
- Extend the integration directory mapping if the folder name differs from the extra:

```yaml
matrix.extra == 'my-framework' && 'my_framework_adapter' || ...
```

### 9. Example YAML

Create `examples/agents_my_framework.yaml` with `framework: my_framework`.

### 10. README

Add an install line to [README.md](../README.md):

```bash
pip install praisonai-frameworks[my-framework]
```

---

## Testing

| Layer | File | Runs on | Rule |
|-------|------|---------|------|
| Protocol | `test_<name>_adapter_protocol.py` | Every CI row | Assert `name`, `install_hint`, `requires_tools_extra`, `run`, `is_available` |
| Mocked `run` | `test_<name>_adapter_run.py` | Every CI row | **Patch** internal paths; do not import the optional SDK at module level |
| Integration | `tests/integration/<name>_adapter/` | That extra's CI rows only | `pytest.importorskip("my_sdk")` at top |
| Live | `test_<name>_live.py` | Manual | `@pytest.mark.live`; gate on API keys and optionally `PRAISONAI_LIVE_TESTS=1` |

### Minimum mocked run tests

- Single-task `run()` returns the sentinel header and content.
- `_collect_ordered_tasks` respects `context` ordering.
- Empty config returns `"No tasks defined."`.
- Handoff routing (if supported): `_run_with_handoffs` is used when YAML has `handoff.to`.

### Local commands

```bash
# Base CI (no optional frameworks)
pip install -e path/to/praisonaiagents -e ".[dev]"
pytest tests/unit -q

# Per-framework
pip install -e ".[my-framework]"
pytest tests/integration/my_framework_adapter -q
```

Unit tests must pass on the **empty-extra** CI row — do not require optional deps on rows where that extra is not installed.

---

## Family routers (AutoGen only)

Use a family router when one entry point delegates to version-specific adapters:

- Set `is_router = True`.
- Implement `resolve()` to pick the concrete adapter from config or version.
- Register both the router and leaf entry points in `pyproject.toml`.
- See [src/praisonai_frameworks/autogen/family.py](../src/praisonai_frameworks/autogen/family.py).

Leaf adapters must not override `resolve`.

---

## External third-party adapter

Minimal template for publishers outside this repo:

```python
from praisonaiagents.frameworks.base import BaseFrameworkAdapter

class MyFrameworkAdapter(BaseFrameworkAdapter):
    name = "my_framework"
    install_hint = "pip install my-framework-bridge"
    requires_tools_extra = False

    def is_available(self) -> bool:
        return True

    def run(self, config, llm_config, topic, *, tools_dict=None,
            agent_callback=None, task_callback=None, cli_config=None) -> str:
        return f"my_framework ran: {topic}"
```

Register in your own `pyproject.toml`:

```toml
[project.entry-points."praisonai.framework_adapters"]
my_framework = "my_pkg.adapter:MyFrameworkAdapter"
```

Then use `framework: my_framework` in `agents.yaml`. See [examples/third_party_adapter/](../examples/third_party_adapter/) for a full minimal package.

---

## Gotchas

| Issue | Fix |
|-------|-----|
| Module-level imports of `crewai`, `langgraph`, `agents`, etc. | Lazy-import inside `run()` only |
| Unit tests fail on base CI | Mock/patch; never hard-import optional packages in unit tests |
| Integration folder shadows a PyPI package | Use `<name>_adapter/` suffix |
| `uv lock` fails on publish | Add `[tool.uv] conflicts`; omit incompatible extras from `all` |
| Framework-specific logic in shared `base.py` | Keep helpers local to the adapter module |
| Renaming entry points | Deprecate; do not rename — breaks existing YAML |
| Forgetting `test_entry_points_registered.py` | CI fails on every matrix row |

---

## Which adapter to copy from

| Your SDK looks like… | Start from |
|---------------------|------------|
| Agent + Task + Crew orchestration | CrewAI |
| Graph / workflow engine | LangGraph |
| OpenAI-style agents + handoffs | OpenAI Agents or Pydantic AI |
| Multi-provider (Gemini + OpenAI) + async runner | Google ADK |
| Team routing / delegation | Agno |

---

## Scope

Implement in this repository only:

- Adapter classes, probes, entry points, optional extras, tests, examples, docs.

Do **not** change `praisonaiagents` or `praisonai` here, vendor core SDK source, or add workflow YAML dispatch (`process: workflow` stays in the wrapper).
