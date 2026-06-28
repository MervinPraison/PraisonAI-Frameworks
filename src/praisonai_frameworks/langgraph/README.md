# LangGraph adapter placeholder

Publish a third-party package with:

```toml
[project.entry-points."praisonai.framework_adapters"]
langgraph = "my_pkg.langgraph:LangGraphAdapter"
```

Implement `FrameworkAdapterProtocol` from `praisonaiagents.frameworks`.
