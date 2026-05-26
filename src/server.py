"""Fed-ICL server orchestration.服务器逻辑"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .aggregation import aggregate_predictions
from .evaluation import accuracy, confusion_counts, per_subject_accuracy, save_metrics, save_round_predictions
from .utils import ensure_dir, save_json, save_jsonl

ANSWER_LETTERS = ["A", "B", "C", "D"]


@dataclass
class Server:
    """Central server that receives only client predictions."""

    server_queries: list[dict[str, Any]]
    clients: list[Any]
    num_rounds: int
    output_dir: str | Path
    aggregation_method: str = "majority_vote"
    seed: int = 42
    mode: str = "fed_icl"
    save_per_subject_accuracy: bool = True
    round_metrics: list[dict[str, Any]] = field(default_factory=list)
    global_context: list[dict[str, Any]] = field(default_factory=list)
    selection_records: list[dict[str, Any]] = field(default_factory=list)
    round_prediction_records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        ensure_dir(self.output_dir / "predictions")
        ensure_dir(self.output_dir / "metrics")
        ensure_dir(self.output_dir / "logs")
        self.global_context = self.initialize_global_context()

    def initialize_global_context(self) -> list[dict[str, Any]]:
        """Create C_1 with seeded random labels for server queries."""
        rng = random.Random(self.seed)
        context: list[dict[str, Any]] = []
        for query in self.server_queries:
            label = rng.choice(ANSWER_LETTERS)
            item = dict(query)
            item["answer"] = label
            item["answer_index"] = ANSWER_LETTERS.index(label)
            context.append(item)
        return context

    def _effective_rounds(self) -> int:
        if self.mode == "zero_context_baseline":
            return 1
        if self.mode == "one_round_fed_icl":
            return 1
        if self.mode in {"fed_icl", "fed_icl_free", "lite_multiround_fed_icl"}:
            return self.num_rounds
        raise ValueError(f"Unsupported experiment mode: {self.mode}")

    def run(self) -> dict[str, Any]:
        """Run the server-client Fed-ICL loop."""
        final_predictions: dict[str, str] = {}
        final_vote_counts: dict[str, dict[str, int]] = {}
        previous_predictions: dict[str, str] = {}
        previous_vote_counts: dict[str, dict[str, int]] = {}
        for round_index in range(1, self._effective_rounds() + 1):
            client_predictions: list[dict[str, str]] = []
            for client in self.clients:
                client_predictions.extend(
                    client.run_round(
                        self.global_context,
                        self.server_queries,
                        mode=self.mode,
                        previous_aggregated_answers=previous_predictions if round_index > 1 else None,
                        previous_vote_counts=previous_vote_counts if round_index > 1 else None,
                    )
                )
                self.selection_records.extend(
                    {
                        "round": round_index,
                        **record,
                    }
                    for record in getattr(client, "last_selection_records", [])
                )

            final_predictions, final_vote_counts = aggregate_predictions(client_predictions, self.aggregation_method)
            self._update_global_context(final_predictions)
            round_accuracy = accuracy(self.server_queries, final_predictions)
            self.round_metrics.append({"round": round_index, "accuracy": round_accuracy})
            self.round_prediction_records.extend(
                self._round_prediction_records(
                    round_index,
                    client_predictions,
                    final_predictions,
                    final_vote_counts,
                    previous_predictions,
                    previous_vote_counts,
                )
            )
            save_round_predictions(final_predictions, final_vote_counts, round_index, self.output_dir)
            previous_predictions = dict(final_predictions)
            previous_vote_counts = dict(final_vote_counts)

        final_metrics: dict[str, Any] = {
            "final_accuracy": accuracy(self.server_queries, final_predictions),
            "num_rounds": self._effective_rounds(),
            "num_server_queries": len(self.server_queries),
            "confusion_counts": confusion_counts(self.server_queries, final_predictions),
        }
        if self.save_per_subject_accuracy:
            final_metrics["per_subject_accuracy"] = per_subject_accuracy(self.server_queries, final_predictions)
        save_metrics(self.round_metrics, final_metrics, self.output_dir)
        save_json(self._sanitized_global_context(), self.output_dir / "predictions" / "final_global_context.json")
        save_jsonl(self.selection_records, self.output_dir / "selected_examples.jsonl")
        save_jsonl(self.round_prediction_records, self.output_dir / "round_predictions.jsonl")
        return final_metrics

    def _update_global_context(self, predictions: dict[str, str]) -> None:
        updated: list[dict[str, Any]] = []
        for query in self.server_queries:
            label = predictions.get(str(query["id"]), "A")
            item = dict(query)
            item["answer"] = label
            item["answer_index"] = ANSWER_LETTERS.index(label)
            updated.append(item)
        self.global_context = updated

    def _sanitized_global_context(self) -> list[dict[str, str]]:
        """Return saved server context without raw question text or choices."""
        return [
            {
                "query_id": str(item["id"]),
                "predicted_answer": str(item["answer"]),
            }
            for item in self.global_context
        ]

    def _round_prediction_records(
        self,
        round_index: int,
        client_predictions: list[dict[str, str]],
        aggregated_predictions: dict[str, str],
        vote_counts: dict[str, dict[str, int]],
        previous_predictions: dict[str, str],
        previous_vote_counts: dict[str, dict[str, int]],
    ) -> list[dict[str, Any]]:
        """Build detailed per-query prediction records for analysis."""
        client_answers_by_query: dict[str, dict[str, str]] = {}
        for row in client_predictions:
            query_id = str(row.get("query_id", ""))
            client_id = str(row.get("client_id", ""))
            answer = str(row.get("predicted_answer", ""))
            if query_id and client_id:
                client_answers_by_query.setdefault(query_id, {})[client_id] = answer

        rows: list[dict[str, Any]] = []
        for query in self.server_queries:
            query_id = str(query["id"])
            aggregated_answer = aggregated_predictions.get(query_id, "")
            gold_answer = str(query.get("answer", ""))
            previous_answer = previous_predictions.get(query_id)
            previous_counts = previous_vote_counts.get(query_id)
            rows.append(
                {
                    "round": round_index,
                    "query_id": query_id,
                    "query_text": str(query.get("question", "")),
                    "client_answers": client_answers_by_query.get(query_id, {}),
                    "aggregated_answer": aggregated_answer,
                    "vote_counts": vote_counts.get(query_id, {}),
                    "previous_vote_counts_used": previous_counts,
                    "gold_answer": gold_answer,
                    "is_correct": aggregated_answer == gold_answer,
                    "answer_changed_from_previous_round": None
                    if previous_answer is None
                    else aggregated_answer != previous_answer,
                }
            )
        return rows
