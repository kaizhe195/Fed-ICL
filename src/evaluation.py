"""Evaluation and output persistence."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .utils import ensure_dir, save_json, save_jsonl


def accuracy(server_queries: list[dict[str, Any]], predictions: dict[str, str]) -> float:
    """Compute exact answer accuracy."""
    if not server_queries:
        return 0.0
    correct = 0
    for query in server_queries:
        if predictions.get(str(query["id"])) == query.get("answer"):
            correct += 1
    return correct / len(server_queries)


def per_subject_accuracy(server_queries: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, float]:
    """Compute accuracy by subject."""
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for query in server_queries:
        subject = str(query.get("subject", "unknown"))
        totals[subject] += 1
        if predictions.get(str(query["id"])) == query.get("answer"):
            correct[subject] += 1
    return {subject: correct[subject] / totals[subject] for subject in sorted(totals)}


def confusion_counts(server_queries: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, int]:
    """Count gold-to-predicted answer pairs."""
    counts: Counter[str] = Counter()
    for query in server_queries:
        gold = str(query.get("answer", "unknown"))
        pred = str(predictions.get(str(query["id"]), "missing"))
        counts[f"{gold}->{pred}"] += 1
    return dict(sorted(counts.items()))


def save_round_predictions(
    predictions: dict[str, str],
    vote_counts: dict[str, dict[str, int]],
    round_index: int,
    output_dir: str | Path,
) -> None:
    """Save aggregated round predictions."""
    rows = []
    for query_id in sorted(predictions):
        rows.append(
            {
                "round": round_index,
                "query_id": query_id,
                "predicted_answer": predictions[query_id],
                "vote_counts": vote_counts.get(query_id, {}),
            }
        )
    save_jsonl(rows, Path(output_dir) / "predictions" / f"round_{round_index}.jsonl")


def save_metrics(
    round_metrics: list[dict[str, Any]],
    final_metrics: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Save per-round metrics CSV and final metrics JSON."""
    output_dir = Path(output_dir)
    metrics_dir = ensure_dir(output_dir / "metrics")
    csv_path = metrics_dir / "per_round_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["round", "accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in round_metrics:
            writer.writerow({key: row.get(key) for key in fieldnames})
    save_json(final_metrics, metrics_dir / "final_metrics.json")
    with (output_dir / "per_round_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["round", "accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in round_metrics:
            writer.writerow({key: row.get(key) for key in fieldnames})
    save_json(final_metrics, output_dir / "final_metrics.json")
