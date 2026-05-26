"""Aggregation methods for federated predictions.实现majority_vote"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

ANSWER_LETTERS = ["A", "B", "C", "D"]


def majority_vote(predictions: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Aggregate MMLU client predictions with deterministic A-B-C-D tie-breaking."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in predictions:
        answer = row.get("predicted_answer")
        query_id = row.get("query_id")
        if query_id is None or answer not in ANSWER_LETTERS:
            continue
        grouped[str(query_id)].append(str(answer))

    final_answers: dict[str, str] = {}
    vote_counts: dict[str, dict[str, int]] = {}
    for query_id, answers in grouped.items():
        counts = Counter(answers)
        vote_counts[query_id] = {letter: counts.get(letter, 0) for letter in ANSWER_LETTERS}
        max_count = max(vote_counts[query_id].values())
        for letter in ANSWER_LETTERS:
            if vote_counts[query_id][letter] == max_count:
                final_answers[query_id] = letter
                break
    return final_answers, vote_counts


def aggregate_predictions(
    predictions: list[dict[str, Any]],
    method: str = "majority_vote",
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Dispatch aggregation by method."""
    if method == "majority_vote":
        return majority_vote(predictions)
    raise ValueError(f"Unsupported aggregation method: {method}")


def simple_select(candidates: list[str]) -> str | None:
    """Placeholder for future open-ended answer fusion."""
    return candidates[0] if candidates else None


def judge_select(candidates: list[str]) -> str | None:
    """Placeholder for future judge-based TruthfulQA answer selection."""
    raise NotImplementedError("judge_select is reserved for the future TruthfulQA branch.")


def genfuser_placeholder(candidates: list[str]) -> str | None:
    """Placeholder for future generated answer fusion."""
    raise NotImplementedError("Generated fusion is reserved for the future TruthfulQA branch.")
