"""General utility helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("The 'PyYAML' package is required to load YAML config files.") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON with stable formatting."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def save_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Save rows as JSON Lines."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load rows from JSON Lines."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]
