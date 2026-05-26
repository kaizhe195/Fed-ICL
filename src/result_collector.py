"""Collect selection experiment results into a compact CSV summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .utils import ensure_dir, load_yaml, project_root

SUMMARY_COLUMNS = [
    "experiment_name",
    "dataset",
    "backend",
    "selection_method",
    "ordering_method",
    "num_clients",
    "num_rounds",
    "alpha",
    "num_context_examples",
    "use_local_filtering",
    "final_accuracy",
    "output_dir",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _load_saved_config(output_dir: Path) -> dict[str, Any]:
    resolved = output_dir / "config.resolved.json"
    if resolved.exists():
        return _read_json(resolved)
    yaml_config = output_dir / "config.yaml"
    if yaml_config.exists():
        return load_yaml(yaml_config)
    return {}


def _metrics_path(output_dir: Path) -> Path:
    nested = output_dir / "metrics" / "final_metrics.json"
    if nested.exists():
        return nested
    direct = output_dir / "final_metrics.json"
    if direct.exists():
        return direct
    raise FileNotFoundError(f"Could not find final_metrics.json under {output_dir}")


def summarize_output_dir(output_dir: str | Path) -> dict[str, Any]:
    """Read one experiment output directory and return a summary row."""
    original_path = Path(output_dir)
    path = original_path if original_path.is_absolute() else project_root() / original_path
    config = _load_saved_config(path)
    metrics = _read_json(_metrics_path(path))
    dataset_cfg = config.get("dataset", {})
    experiment_cfg = config.get("experiment", {})
    llm_cfg = config.get("llm", {})
    experiment_name = experiment_cfg.get("name", path.name)
    if path.name.startswith("seed_"):
        experiment_name = f"{experiment_name}_{path.name}"
    return {
        "experiment_name": experiment_name,
        "dataset": dataset_cfg.get("name", ""),
        "backend": llm_cfg.get("backend", ""),
        "selection_method": experiment_cfg.get("selection_method", ""),
        "ordering_method": experiment_cfg.get("ordering_method", ""),
        "num_clients": experiment_cfg.get("num_clients", ""),
        "num_rounds": metrics.get("num_rounds", experiment_cfg.get("num_rounds", "")),
        "alpha": experiment_cfg.get("alpha", ""),
        "num_context_examples": experiment_cfg.get("num_context_examples", ""),
        "use_local_filtering": experiment_cfg.get("use_local_filtering", ""),
        "final_accuracy": metrics.get("final_accuracy", ""),
        "output_dir": original_path.as_posix(),
    }


def collect_results(
    output_dirs: list[str | Path],
    summary_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Collect result rows and write outputs/selection_summary.csv by default."""
    root = project_root()
    destination = Path(summary_path) if summary_path is not None else root / "outputs" / "selection_summary.csv"
    rows = [summarize_output_dir(output_dir) for output_dir in output_dirs]
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
