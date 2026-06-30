# Agno adapter

Runs PraisonAI `agents.yaml` via [Agno](https://github.com/agno-agi/agno).

## Install

```bash
pip install "praisonai-frameworks[agno]"
```

## YAML

```yaml
framework: agno
topic: math
roles:
  calculator:
    role: Calculator
    goal: Compute exactly
    backstory: Return only the numeric answer.
    tasks:
      add:
        description: What is 3 + 3?
        expected_output: "6"
```

## Phases

- **Phase 1:** single role, single task — `Agent.run`
- **Phase 2:** sequential tasks with `context:` — sequential `Agent.run` loop
- **Phase 3:** `handoff.to` — not implemented yet
