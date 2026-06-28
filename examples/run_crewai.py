"""Run a CrewAI-backed agents YAML via the PraisonAI Python entry point."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    yaml_path = Path(__file__).with_name("agents_crewai.yaml")
    if not yaml_path.exists():
        print(f"Missing example YAML: {yaml_path}", file=sys.stderr)
        return 1

    try:
        from praisonai import run
    except ImportError as exc:
        print(
            "Install praisonai and praisonai-frameworks[crewai] first:\n"
            "  pip install praisonai praisonai-frameworks[crewai]",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    result = run(str(yaml_path), framework="crewai")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
