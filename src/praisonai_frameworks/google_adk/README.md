# Google ADK adapter

Runs PraisonAI agents YAML via [Google Agent Development Kit](https://github.com/google/adk-python).

## Install

```bash
pip install "praisonai-frameworks[google-adk]"
```

## YAML

```yaml
framework: google_adk
topic: Quick task
roles:
  helper:
    role: Helper
    goal: Answer briefly
    backstory: Helpful assistant
    tasks:
      answer:
        description: Reply with exactly OK.
        expected_output: OK
```

## Models

- **Gemini (native):** `gemini-2.5-flash` with `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- **OpenAI (via LiteLLM):** `openai/gpt-4o-mini` or `gpt-4o-mini` with `OPENAI_API_KEY`

Requires `google-adk[extensions]` for non-Gemini providers.

## Note

`framework: google_adk` runs agents locally through ADK's `InMemoryRunner`. This is separate from PraisonAI's remote `Session(agent_url=...)` client pattern described in remote-agent examples.

For Gemini models, the adapter may temporarily set `GOOGLE_API_KEY` from `llm_config` during each run. The previous value is restored in a `finally` block.

## Limitations (v1)

- Phase 1–2 only: single task and sequential `context:` chaining
- `handoff.to` is not wired; a warning is logged if present
- Sync `run()` only (uses ADK async runner internally)
