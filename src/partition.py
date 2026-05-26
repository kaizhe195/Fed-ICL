"""Federated client data partitioning.客户数据"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .utils import ensure_dir, save_json, save_jsonl


def dirichlet_partition(
    examples: list[dict[str, Any]],
    num_clients: int,
    alpha: float,
    seed: int,
    partition_key: str = "subject",
) -> dict[str, list[dict[str, Any]]]:
    """Partition examples across clients with a Dirichlet distribution per group."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    rng = np.random.default_rng(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[str(example.get(partition_key, "unknown"))].append(example)

    clients = {f"client_{i}": [] for i in range(num_clients)}
    for _, group_examples in sorted(grouped.items()):
        shuffled = list(group_examples)
        rng.shuffle(shuffled)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        counts = rng.multinomial(len(shuffled), proportions)
        offset = 0
        for client_index, count in enumerate(counts):
            client_id = f"client_{client_index}"
            clients[client_id].extend(shuffled[offset : offset + count])
            offset += count
    return clients


def client_statistics(clients: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Compute per-client size, subject distribution, and label distribution."""
    stats: dict[str, dict[str, Any]] = {}
    for client_id, examples in sorted(clients.items()):
        subject_counts = Counter(str(example.get("subject", "unknown")) for example in examples)
        label_counts = Counter(str(example.get("answer", "unknown")) for example in examples)
        stats[client_id] = {
            "num_examples": len(examples),
            "subject_distribution": dict(sorted(subject_counts.items())),
            "label_distribution": dict(sorted(label_counts.items())),
        }
    return stats


def save_client_partitions(
    clients: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Save client datasets and distribution statistics."""
    base = ensure_dir(output_dir)
    for client_id, examples in sorted(clients.items()):
        save_jsonl(examples, base / f"{client_id}.jsonl")

    stats = client_statistics(clients)
    save_json(stats, base / "client_stats.json")

    csv_path = base / "client_stats.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "num_examples", "subject_distribution", "label_distribution"])
        writer.writeheader()
        for client_id, row in stats.items():
            writer.writerow(
                {
                    "client_id": client_id,
                    "num_examples": row["num_examples"],
                    "subject_distribution": row["subject_distribution"],
                    "label_distribution": row["label_distribution"],
                }
            )
    return stats
