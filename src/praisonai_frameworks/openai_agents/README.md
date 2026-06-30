# OpenAI Agents SDK adapter

Runs PraisonAI YAML configs via [OpenAI Agents Python](https://github.com/openai/openai-agents-python).

## Install

```bash
pip install "praisonai-frameworks[openai-agents]"
```

## YAML

```yaml
framework: openai_agents
topic: math
roles:
  calculator:
    role: Calculator
    goal: Compute exactly
    backstory: Return only the number
    tasks:
      add:
        description: What is 3 + 3?
        expected_output: "6"
```

## Handoffs

Use `handoff.to` with **role field strings** (not YAML dict keys):

```yaml
handoff:
  to:
    - English Agent
    - French Agent
```

Explicit `handoff.to` is required; `allow_delegation: true` alone does not register handoffs.

## CLI

```bash
praisonai agents run --file examples/agents_openai_agents.yaml
```
