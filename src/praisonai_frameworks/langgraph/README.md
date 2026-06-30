# LangGraph adapter

First-party adapter that maps a PraisonAI `agents.yaml` config onto a LangGraph
[`StateGraph`](https://langchain-ai.github.io/langgraph/). It is registered via
the `praisonai.framework_adapters` entry-point group, so `framework: langgraph`
resolves automatically once the optional extra is installed.

## Install

```bash
pip install "praisonai-frameworks[langgraph]"
```

## Use

```yaml
# agents.yaml
framework: langgraph
topic: Example research task
process: sequential
roles:
  researcher:
    role: Research Analyst
    goal: Gather accurate information about {topic}
    backstory: Expert researcher
    tasks:
      research:
        description: Research {topic}
        expected_output: A short summary
```

See [`examples/agents_langgraph.yaml`](../../../examples/agents_langgraph.yaml).

## YAML → graph mapping

| YAML field | LangGraph mapping |
|------------|-------------------|
| `topic` | Initial state input channel |
| `roles.<key>` | Graph node |
| `roles.*.tasks.*` | Node prompt (description + expected output) |
| `process: sequential` | Linear edges between role nodes |

Each role becomes a node that calls the resolved LLM with the formatted prompt,
threading prior outputs through the graph state. Nodes are wired sequentially
(`START → role1 → role2 → … → END`) and the final assistant message is returned.

Optional framework deps (`langgraph`, `langchain-core`) are imported lazily
inside `run()`, so importing this package never requires LangGraph to be present.

## Writing your own external adapter

To ship an adapter in a separate package instead, register an entry point:

```toml
[project.entry-points."praisonai.framework_adapters"]
langgraph = "my_pkg.langgraph:LangGraphAdapter"
```

and implement `FrameworkAdapterProtocol` from `praisonaiagents.frameworks`.
See [`examples/third_party_adapter/`](../../../examples/third_party_adapter/)
and [`docs/adding-a-framework.md`](../../../docs/adding-a-framework.md).
