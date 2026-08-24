#!/usr/bin/env python3
"""Real LLM API smoke test for binder-harness.

This script intentionally performs a network call through the configured
OpenAI-compatible endpoint. It is not part of the offline regression suite.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.llm import LLMConfigError, OpenAICompatibleClient  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call a real configured LLM endpoint once and validate the JSON response.")
    parser.add_argument("--llm-config", default="configs/llm_endpoints.local.json", help="Path to ignored local LLM config JSON.")
    parser.add_argument("--llm-model", default=None, help="Endpoint key in the config. Defaults to default_model.")
    parser.add_argument("--llm-thinking", default=None, help="Optional reasoning/thinking override for the selected endpoint.")
    parser.add_argument("--out", default=None, help="Optional path to write a redacted JSON result artifact.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config_path = Path(args.llm_config)
    client = OpenAICompatibleClient.from_json(config_path)
    if client is None:
        raise RuntimeError(f"LLM config was not loaded: {config_path}")
    client.configure_default(model_key=args.llm_model, thinking=args.llm_thinking)
    result: Dict[str, Any] = client.preflight()
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except LLMConfigError as exc:
        raise SystemExit(str(exc)) from exc
