# Adding a custom framework adapter

1. Subclass `BaseFrameworkAdapter` from `praisonaiagents.frameworks.base`.
2. Implement `run(..., *, tools_dict=..., agent_callback=..., task_callback=..., cli_config=...)`.
3. Register via setuptools:

```toml
[project.entry-points."praisonai.framework_adapters"]
my_framework = "my_pkg.adapter:MyFrameworkAdapter"
```

4. Use `framework: my_framework` in `agents.yaml`.

See `examples/third_party_adapter/` for a minimal template.
