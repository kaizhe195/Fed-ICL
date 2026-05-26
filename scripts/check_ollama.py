"""Check a local Ollama model through the HTTP generate API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import OllamaCompatibleLMClient, parse_mmlu_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local Ollama generation for MMLU-style answers.")
    parser.add_argument("--model", default="gemma3:1b", help="Ollama model name, for example gemma3:1b.")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL.")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="HTTP timeout for the test request.")
    args = parser.parse_args()

    client = OllamaCompatibleLMClient(
        model_name=args.model,
        base_url=args.base_url,
        temperature=0.0,
        max_tokens=8,
        timeout_seconds=args.timeout_seconds,
    )
    prompt = (
        "Answer with only one letter: A, B, C, or D.\n"
        "Question: Which option is the letter A?\n"
        "A. A\n"
        "B. B\n"
        "C. C\n"
        "D. D\n"
        "Answer:"
    )
    try:
        response = client.generate(prompt)
    except RuntimeError as exc:
        raise SystemExit(f"Ollama check failed: {exc}") from exc

    parsed = parse_mmlu_answer(response)
    print("Raw response:")
    print(response)
    print(f"Parsed MMLU answer: {parsed if parsed is not None else 'None'}")


if __name__ == "__main__":
    main()
