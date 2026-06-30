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
```

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
