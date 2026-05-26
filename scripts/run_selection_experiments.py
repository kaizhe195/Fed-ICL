"""Run selection-focused MMLU Fed-ICL experiments sequentially."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment import run_mmlu_experiment
from src.result_collector import collect_results
from src.utils import load_yaml

DEFAULT_CONFIGS = [
    "configs/selection_random_mmlu.yaml",
    "configs/selection_knn_mmlu.yaml",
]

TABLE_COLUMNS = [
    "experiment_name",
    "selection_method",
    "num_clients",
    "num_rounds",
    "alpha",
    "num_context_examples",
    "backend",
    "final_accuracy",
    "output_dir",
]


def _seeds_from_config(config: dict[str, Any]) -> list[int]:
    experiment_cfg = config.get("experiment", {})
    seeds = experiment_cfg.get("seeds")
    if seeds is None:
        return [int(experiment_cfg.get("seed", 42))]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("experiment.seeds must be a non-empty list when provided.")
    return [int(seed) for seed in seeds]


def _validate_selection_config(config: dict[str, Any], config_path: Path) -> None:
    experiment_cfg = config.get("experiment", {})
    ordering_method = experiment_cfg.get("ordering_method", "similarity_desc")
    if ordering_method != "similarity_desc":
        raise ValueError(
            f"{config_path} sets ordering_method={ordering_method!r}. "
            "Selection experiments keep ordering fixed as 'similarity_desc'."
        )


def _seed_output_dir(base_output_dir: str, seed: int, multiple_seeds: bool) -> str:
    if not multiple_seeds:
        return base_output_dir
    return str(Path(base_output_dir) / f"seed_{seed}")


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No experiments were run.")
        return
    formatted_rows: list[dict[str, str]] = []
    for row in rows:
        formatted = {column: str(row.get(column, "")) for column in TABLE_COLUMNS}
        if "final_accuracy" in row and row.get("final_accuracy") != "":
            formatted["final_accuracy"] = f"{float(row['final_accuracy']):.4f}"
        formatted_rows.append(formatted)
    widths = {
        column: max(len(column), *(len(row[column]) for row in formatted_rows))
        for column in TABLE_COLUMNS
    }
    header = "  ".join(column.ljust(widths[column]) for column in TABLE_COLUMNS)
    print(header)
    print("  ".join("-" * widths[column] for column in TABLE_COLUMNS))
    for row in formatted_rows:
        print("  ".join(row[column].ljust(widths[column]) for column in TABLE_COLUMNS))


def run_selection_configs(config_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Run each config, expanding experiment.seeds when more than one seed is provided."""
    summaries: list[dict[str, Any]] = []
    output_dirs: list[str] = []
    for config_path in config_paths:
        resolved_path = PROJECT_ROOT / config_path
        config = load_yaml(resolved_path)
        _validate_selection_config(config, resolved_path)
        experiment_cfg = config.get("experiment", {})
        seeds = _seeds_from_config(config)
        multiple_seeds = len(seeds) > 1
        base_output_dir = str(experiment_cfg.get("output_dir", "outputs/mmlu"))
        for seed in seeds:
            output_dir = _seed_output_dir(base_output_dir, seed, multiple_seeds)
            summary = run_mmlu_experiment(
                resolved_path,
                seed_override=seed,
                output_dir_override=output_dir,
            )
            if multiple_seeds:
                summary["experiment_name"] = f"{summary['experiment_name']}_seed_{seed}"
            summaries.append(summary)
            output_dirs.append(summary["output_dir"])
    collect_results(output_dirs, PROJECT_ROOT / "outputs" / "selection_summary.csv")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selection-focused MMLU Fed-ICL experiments.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    args = parser.parse_args()

    rows = run_selection_configs(args.configs)
    print("Selection experiment comparison")
    _print_table(rows)
    print("Summary CSV: outputs/selection_summary.csv")
    if any(str(row.get("backend", "")) == "mock" for row in rows):
        print("MockLM note: these results verify the pipeline only and are not real model performance.")


if __name__ == "__main__":
    main()
