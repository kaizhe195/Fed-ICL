"""Summarize the formal main-experiment runs into one CSV."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, load_yaml

CONFIGS = [
    "configs/main_mmlu_random_seed42.yaml",
    "configs/main_mmlu_random_seed43.yaml",
    "configs/main_mmlu_random_seed44.yaml",
    "configs/main_mmlu_knn_seed42.yaml",
    "configs/main_mmlu_knn_seed43.yaml",
    "configs/main_mmlu_knn_seed44.yaml",
]

SUMMARY_COLUMNS = [
    "run_name",
    "selection_method",
    "seed",
    "final_accuracy",
    "num_rounds",
    "num_server_queries",
    "output_dir",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _row_from_config(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    experiment_cfg = config.get("experiment", {})
    run_name = str(experiment_cfg.get("name", config_path.stem))
    output_dir = PROJECT_ROOT / str(experiment_cfg["output_dir"])
    metrics = _read_json(output_dir / "metrics" / "final_metrics.json")
    return {
        "run_name": run_name,
        "selection_method": experiment_cfg.get("selection_method", ""),
        "seed": int(experiment_cfg.get("seed", 42)),
        "final_accuracy": metrics.get("final_accuracy", ""),
        "num_rounds": metrics.get("num_rounds", ""),
        "num_server_queries": metrics.get("num_server_queries", ""),
        "output_dir": str(experiment_cfg.get("output_dir", "")),
    }


def main() -> None:
    rows = [_row_from_config(PROJECT_ROOT / config_path) for config_path in CONFIGS]
    output_path = PROJECT_ROOT / "outputs" / "main_experiment_summary.csv"
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)


if __name__ == "__main__":
    main()
