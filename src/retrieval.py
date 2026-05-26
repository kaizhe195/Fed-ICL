"""Example selection and ordering utilities.选择逻辑"""

from __future__ import annotations

import random
from typing import Any


def lexical_similarity(a: str, b: str) -> float:
    """A lightweight fallback similarity for mock and test workflows."""
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def select_examples(
    query_example: dict[str, Any],
    local_examples: list[dict[str, Any]],
    num_examples: int,
    method: str,
    rng: random.Random,
    embedding_index: Any | None = None,
    text_key: str = "question",
) -> list[tuple[dict[str, Any], float]]:
    """Select local examples for one query."""
    if num_examples <= 0 or not local_examples:
        return []
    if method == "random":
        selected = rng.sample(local_examples, k=min(num_examples, len(local_examples)))
        return [(example, 0.0) for example in selected]
    if method == "knn":
        if embedding_index is not None:
            return embedding_index.search(str(query_example.get(text_key, "")), num_examples)
        scored = [
            (example, lexical_similarity(str(query_example.get(text_key, "")), str(example.get(text_key, ""))))
            for example in local_examples
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: min(num_examples, len(scored))]
    raise ValueError(f"Unsupported selection method: {method}")


def order_examples(
    selected: list[tuple[dict[str, Any], float]],
    method: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Order selected examples before prompt construction."""
    return [example for example, _ in order_selected_pairs(selected, method, rng)]


def order_selected_pairs(
    selected: list[tuple[dict[str, Any], float]],
    method: str,
    rng: random.Random,
) -> list[tuple[dict[str, Any], float]]:
    """Order selected example-score pairs before prompt construction."""
    if method == "similarity_desc":
        ordered = sorted(selected, key=lambda item: item[1], reverse=True)
    elif method == "similarity_asc":
        ordered = sorted(selected, key=lambda item: item[1])
    elif method == "random":
        ordered = list(selected)
        rng.shuffle(ordered)
    elif method == "original":
        ordered = list(selected)
    else:
        raise ValueError(f"Unsupported ordering method: {method}")
    return ordered
