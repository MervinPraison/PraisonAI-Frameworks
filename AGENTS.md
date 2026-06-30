# AGENTS.md — PraisonAI Frameworks

Optional agent framework adapters for PraisonAI (`praisonai-frameworks` on PyPI).

## Scope

**Implement here:** CrewAI/AutoGen/LangGraph adapters, `praisonai.framework_adapters` entry points, lazy imports in `run()`.

**Route elsewhere:**

| Layer | Repository | Examples |
|-------|------------|----------|
| Core SDK | MervinPraison/PraisonAI (`praisonaiagents`) | Adapter protocols, Agent runtime APIs |
| Wrapper | MervinPraison/PraisonAI (`praisonai`) | CLI/YAML `framework:` wiring |
| Tools | MervinPraison/PraisonAI-Tools | Agent-callable API tools |
| Plugins | MervinPraison/PraisonAI-Plugins | Hooks, guardrails, policies |

**Reject:** Duplicating core agent logic; heavy deps without optional extras in `pyproject.toml`.

## Rules

- Depends on `praisonaiagents` only — do not vendor the core SDK.
- Lazy-import optional framework deps inside adapter methods.
- Declare optional extras in `pyproject.toml` (e.g. `[crewai]`, `[autogen]`).
- Minimal, focused diffs; add tests under `tests/` for behaviour changes.
- See [docs/adding-a-framework.md](docs/adding-a-framework.md) for adapter patterns.

## Primary paths

- `src/praisonai_frameworks/` — adapters and shared helpers
- `tests/` — unit and integration tests
- `examples/` — usage samples
- `docs/` — contributor docs

## GitHub Actions secrets

Configure in repo Settings → Secrets:

| Secret | Purpose |
|--------|---------|
| `GH_TOKEN` | PAT as `MervinPraison` — chains `@copilot` / `@claude`, merge gate |
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude Code OAuth |
| `CLAUDE_APP_ID` + `CLAUDE_APP_PRIVATE_KEY` | GitHub App token for checkout and Claude action |

Install CodeRabbit, Qodo, and GitHub Copilot for PRs for the review chain.
