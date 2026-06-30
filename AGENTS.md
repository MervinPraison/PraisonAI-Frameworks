# AGENTS.md — PraisonAI Frameworks

> **For AI agents and contributors**: Rules for `MervinPraison/PraisonAI-Frameworks` — optional third-party framework adapters for PraisonAI YAML `framework:` execution.

---

## 1. What this repository is

**PraisonAI-Frameworks** (`praisonai-frameworks` on PyPI) registers adapters via setuptools entry points so users can run:

```yaml
framework: crewai   # or autogen, langgraph, …
```

without shipping CrewAI, AutoGen, or LangGraph inside the core SDK or wrapper wheel.

```
┌─────────────────────────────────────────────────────────────┐
│  praisonai (wrapper)     CLI, registry, YAML, doctor        │
├─────────────────────────────────────────────────────────────┤
│  praisonaiagents (core)  FrameworkAdapterProtocol, base     │
├─────────────────────────────────────────────────────────────┤
│  praisonai-frameworks (THIS REPO)  CrewAI/AutoGen/LangGraph │
└─────────────────────────────────────────────────────────────┘
```

**Philosophy:** Minimal adapter surface • Lazy optional deps • Entry-point discovery • No core SDK duplication.

---

## 2. Scope — implement here vs route elsewhere

| Implement **here** | Route **elsewhere** |
|--------------------|---------------------|
| Framework adapter classes (`run`, `is_available`) | Agent runtime, memory, hooks → `praisonaiagents` |
| `[project.entry-points."praisonai.framework_adapters"]` | CLI `--framework`, workflow engine → `praisonai` wrapper |
| Optional extras in `pyproject.toml` | Heavy tools (Slack, DB, …) → `PraisonAI-Tools` |
| `_availability.py` probes | Protocol changes → `praisonaiagents.frameworks` |
| Unit/integration tests, `examples/agents_*.yaml` | Mintlify docs → `PraisonAIDocs` |

**Reject in this repo:**

- Vendoring or copying `praisonaiagents` source
- Module-level imports of `crewai`, `autogen`, `langgraph`, `langchain_*`
- Bloating `base.py` with framework-specific logic (keep per-adapter helpers local)
- Workflow YAML dispatch (`process: workflow` stays PraisonAI-native in the wrapper)

---

## 3. Adapter contract (mandatory)

Implement `FrameworkAdapterProtocol` via `BaseFrameworkAdapter`:

**Reference:** `src/praisonai_frameworks/crewai/adapter.py` (gold standard)

### Class attributes

| Attribute | Example |
|-----------|---------|
| `name` | `"langgraph"` |
| `install_hint` | `'pip install "praisonai-frameworks[langgraph]"'` |
| `requires_tools_extra` | `True` if YAML tools need `praisonai-tools` |
| `is_router` | `False` for leaf adapters; `True` only for family routers (e.g. AutoGen) |

### Required methods

```python
def is_available(self) -> bool:
    return is_available("langgraph")  # from praisonai_frameworks._availability

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

### Leaf adapter checklist

1. **Lazy-import** third-party packages inside `run()` (or small private helpers), never at module top.
2. **`try` / `finally`** with `finalize_observability(self.name, status=...)` (see CrewAI adapter).
3. **Return string** with a clear sentinel, e.g. `"### LangGraph Output ###\n{content}"`.
4. **Do not override** `arun`, `resolve`, `setup`, `cleanup` unless routing (family adapters only).
5. Register probe in `_availability.py` and entry point + optional extra in `pyproject.toml`.

### LLM resolution

- Use `praisonaiagents.frameworks.base.BaseFrameworkAdapter._resolve_llm` for **model strings**, not CrewAI `LLM` objects, when the backend expects LangChain/OpenAI clients.
- Read `api_key` / `base_url` from `llm_config[0]`; fall back to `OPENAI_API_KEY` where appropriate.

---

## 4. Adding a new framework (checklist)

1. Create `src/praisonai_frameworks/<name>/adapter.py` (+ `__init__.py`, `README.md`).
2. Add probe to `_PROBES` in `_availability.py`.
3. Add optional extra + entry point in `pyproject.toml`.
4. Add `examples/agents_<name>.yaml`.
5. Tests:
   - `tests/unit/test_<name>_adapter_protocol.py` — always runs (no optional deps).
   - `tests/unit/test_<name>_adapter_run.py` — mock `run()`; **must not import optional deps** on base CI matrix (patch helpers or use `pytest.importorskip`).
   - `tests/integration/<name>_adapter/` — **never name the folder `langgraph/`** (shadows PyPI `langgraph`); use `langgraph_adapter/`.
   - Update `tests/unit/test_entry_points_registered.py`.
6. Document install in root `README.md`.

See also: [docs/adding-a-framework.md](docs/adding-a-framework.md), [examples/third_party_adapter/](examples/third_party_adapter/).

---

## 5. Registered frameworks

| Entry point | Extra | Adapter module |
|-------------|-------|----------------|
| `crewai` | `[crewai]` | `crewai.adapter:CrewAIAdapter` |
| `autogen` | `[autogen]` | `autogen.family:AutoGenFamilyAdapter` (router) |
| `autogen_v2` | `[autogen]` | `autogen.adapter_v2:AutoGenAdapter` |
| `autogen_v4` | `[autogen-v4]` | stub / placeholder |
| `ag2` | `[ag2]` | stub / placeholder |
| `langgraph` | `[langgraph]` | `langgraph.adapter:LangGraphAdapter` (when merged) |

Family routers implement `resolve()` to pick a concrete adapter from config/version.

---

## 6. Testing standards

```bash
# Base (no optional frameworks)
pip install -e praisonai-package/src/praisonai-agents -e ".[dev]"
pytest tests/unit -q

# Per-framework
pip install -e ".[langgraph]"
pytest tests/integration/langgraph_adapter -q
```

| Layer | Location | Gate |
|-------|----------|------|
| Protocol smoke | `tests/unit/test_*_protocol.py` | Always |
| Mocked `run` | `tests/unit/test_*_run.py` | No optional imports on all CI rows |
| Integration | `tests/integration/<name>_adapter/` | `pytest.importorskip("<pkg>")` |
| Live API | `test_*_live.py` | `OPENAI_API_KEY`; optional `PRAISONAI_LIVE_TESTS` |

**CI matrix** (`.github/workflows/ci.yml`): `extra: ["", "crewai", "autogen", "langgraph"]` — unit tests run on **every** row; do not require `langchain_core` on the crewai/autogen rows.

---

## 7. Engineering rules

1. **Minimal diffs** — one framework or one fix per PR where possible.
2. **British English** in comments and user-facing strings.
3. **No performance impact** on import: `import praisonai_frameworks` must stay lightweight.
4. **Backward compatible** entry point names; deprecate, do not rename without cycle.
5. **Observability:** call `finalize_observability` in `finally` (CrewAI pattern).
6. **Telemetry:** use `scoped_telemetry_disable` only when the third-party SDK has noisy telemetry (CrewAI).
7. **Tools:** map `tools_dict` by name from YAML; support callables and framework-native tool types.
8. **YAML mapping:** `roles` → agents, nested `tasks`, `context` for sequential chains, `{topic}` template vars via `_format_template`.

---

## 8. Primary paths

```
src/praisonai_frameworks/
├── base.py              # Extends core base (CrewAI LLM helper — avoid for non-CrewAI adapters)
├── _availability.py     # Probes for is_available()
├── _observability.py    # Wrapper hook bridge
├── _telemetry.py        # Optional telemetry suppress
├── crewai/              # CrewAI adapter
├── autogen/             # AutoGen family + v2/v4/ag2
└── langgraph/           # LangGraph adapter

tests/unit/              # Always-on tests
tests/integration/       # Per-extra integration (use *_adapter dir names)
examples/                # agents_*.yaml, run_*.py
docs/                    # Contributor docs
```

---

## 9. Claude / CI workflow

When triggered via `@claude` or merge gate:

1. **Read this file** (`AGENTS.md` at repository root) and `docs/adding-a-framework.md` before coding.
2. **Scope:** only `src/praisonai_frameworks/`, `tests/`, `examples/`, `docs/`, `pyproject.toml`, `README.md`.
3. **Do not** change `praisonaiagents` or `praisonai` in this repo (CI checks out PraisonAI only for `praisonaiagents` install).
4. Run `pytest tests/unit -q` before pushing; run integration tests if you touch an optional extra.
5. Prefer fixing CI failures on the PR branch over broad refactors.

### GitHub Actions secrets

| Secret | Purpose |
|--------|---------|
| `GH_TOKEN` | PAT — review chains, merge gate |
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude Code OAuth |
| `CLAUDE_APP_ID` + `CLAUDE_APP_PRIVATE_KEY` | GitHub App token |

---

## 10. Related repositories

| Repo | Role |
|------|------|
| [MervinPraison/PraisonAI](https://github.com/MervinPraison/PraisonAI) | Core SDK + wrapper; full `AGENTS.md` for ecosystem |
| [MervinPraison/PraisonAI-Tools](https://github.com/MervinPraison/PraisonAI-Tools) | Optional agent tools |
| [MervinPraison/PraisonAIDocs](https://github.com/MervinPraison/PraisonAIDocs) | User documentation |

---

*Adapter-only repository. Keep it thin, lazy, and entry-point driven.*
