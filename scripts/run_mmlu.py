"""Command line entry point for MMLU Fed-ICL experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment import run_mmlu_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MMLU Fed-ICL reproduction workflow.")
    parser.add_argument("--config", type=str, default="configs/default_mmlu.yaml")
    args = parser.parse_args()

    summary = run_mmlu_experiment(PROJECT_ROOT / args.config)
    print("Final summary")
    print(f"  number of clients: {summary['num_clients']}")
    print(f"  number of server queries: {summary['num_server_queries']}")
    print(f"  number of rounds: {summary['num_rounds']}")
    print(f"  alpha: {summary['alpha']}")
    print(f"  selection method: {summary['selection_method']}")
    print(f"  fixed ordering method: {summary['ordering_method']}")
    print(f"  backend: {summary['backend']}")
    print(f"  final accuracy: {summary['final_accuracy']:.4f}")
    print(f"  output path: {summary['output_path']}")
    if summary["backend"] == "mock":
        print("  MockLM note: pipeline verification only; not real model performance.")


if __name__ == "__main__":
    main()
