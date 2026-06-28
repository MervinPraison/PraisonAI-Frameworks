# Third-party framework adapter template

Publish your own adapter without depending on `praisonai-frameworks`.

1. Subclass `BaseFrameworkAdapter` from `praisonaiagents.frameworks.base`.
2. Register an entry point in the `praisonai.framework_adapters` group.
3. Install your package alongside `praisonai` or `praisonaiagents`.

See `my_framework_adapter.py` and `pyproject.toml` in this folder.
