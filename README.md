# PraisonAI Frameworks

Optional agent framework adapters for [PraisonAI](https://github.com/MervinPraison/PraisonAI).

Implements the `praisonai.framework_adapters` entry-point group so YAML `framework:` values
(`crewai`, `autogen`, …) resolve without bloating the core SDK or wrapper wheel.

## Install

```bash
# Native PraisonAI only (no third-party frameworks)
pip install praisonaiagents praisonai

# CrewAI backend
pip install praisonai-frameworks[crewai]

# AutoGen v0.2 backend
pip install praisonai-frameworks[autogen]

# LangGraph backend
pip install praisonai-frameworks[langgraph]

# OpenAI Agents SDK backend
pip install praisonai-frameworks[openai-agents]

# Agno backend
pip install praisonai-frameworks[agno]

# Google ADK backend
pip install praisonai-frameworks[google-adk]

# Pydantic AI backend
pip install praisonai-frameworks[pydantic-ai]

# AutoGen v0.4 backend (autogen-agentchat / autogen-ext)
pip install praisonai-frameworks[autogen-v4]
```

### Selecting an AutoGen version

Use `framework: autogen` and set `AUTOGEN_VERSION` to choose a backend:

| `AUTOGEN_VERSION` | Backend |
|-------------------|---------|
| `v0.2`            | AutoGen v0.2 (`autogen`) |
| `v0.4`            | AutoGen v0.4 (`autogen-agentchat`) |
| `ag2`             | AG2 fork |
| `auto` (default)  | First available: v0.2 → v0.4 → ag2 |

## Usage

```yaml
# agents.yaml
framework: crewai
topic: Research AI trends
roles:
  researcher:
    role: Research Analyst
    goal: Find accurate information
    backstory: Expert researcher
    tasks:
      research:
        agent: researcher
        description: Research {topic}
        expected_output: A concise summary
```

```bash
praisonai run agents.yaml
```

## Architecture

- Depends on **`praisonaiagents` only** (protocol + base helpers in `praisonaiagents.frameworks`)
- Registers adapters via setuptools entry points — no wrapper import required
- Lazy-imports CrewAI / AutoGen inside `run()` only

## Adding a framework

See `examples/third_party_adapter/` and `docs/adding-a-framework.md`.
