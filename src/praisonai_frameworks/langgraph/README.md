# LangGraph adapter

Optional PraisonAI framework adapter for [LangGraph](https://github.com/langchain-ai/langgraph).

## Install

```bash
pip install "praisonai-frameworks[langgraph]"
```

## Usage

```yaml
# agents_langgraph.yaml
framework: langgraph
topic: Example research task
roles:
  researcher:
    role: Research Analyst
    goal: Gather accurate information on {topic}
    backstory: Expert researcher
    tasks:
      research:
        description: Research {topic}
        expected_output: A short summary
```

```bash
praisonai agents run --file agents_langgraph.yaml --framework langgraph
```

## Notes

- Single-task YAML uses `create_react_agent`.
- Multi-task YAML with `context` chains uses a sequential `StateGraph`.
- Workflow YAML (`process: workflow`) remains PraisonAI-native only.
