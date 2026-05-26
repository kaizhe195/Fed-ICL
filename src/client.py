"""Fed-ICL client implementation.客户逻辑"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from .embeddings import EmbeddingIndex
from .llm_client import LMClient, parse_mmlu_answer
from .prompts import build_mmlu_prompt, build_mmlu_refinement_prompt
from .retrieval import lexical_similarity, order_selected_pairs, select_examples

ANSWER_LETTERS = ["A", "B", "C", "D"]


@dataclass
class Client:
    """Federated in-context learning client with private local data."""

    client_id: str
    local_dataset: list[dict[str, Any]]
    lm_client: LMClient
    selection_method: str = "knn"
    ordering_method: str = "similarity_desc"
    num_context_examples: int = 5
    use_local_filtering: bool = True
    local_filter_top_k: int = 50
    text_key: str = "question"
    embedding_model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    seed: int = 42
    embedding_index: EmbeddingIndex | None = None
    last_selection_records: list[dict[str, Any]] = field(default_factory=list, init=False)
    rng: random.Random = field(init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(f"{self.seed}:{self.client_id}")
        self.logger = logging.getLogger("fed_icl")

    def _fallback_answer(self, item_id: str, phase: str) -> str:
        fallback = self.rng.choice(ANSWER_LETTERS)
        self.logger.warning(
            "Unable to parse MMLU answer for client_id=%s item_id=%s phase=%s; using deterministic fallback=%s",
            self.client_id,
            item_id,
            phase,
            fallback,
        )
        return fallback

    def _build_index_if_needed(self, examples: list[dict[str, Any]]) -> EmbeddingIndex | None:
        if self.selection_method != "knn":
            return None
        if self.embedding_index is None:
            try:
                self.embedding_index = EmbeddingIndex(
                    model_name=self.embedding_model_name,
                    batch_size=self.embedding_batch_size,
                )
                self.embedding_index.build_index(examples, self.text_key)
            except Exception:
                self.embedding_index = None
        elif self.embedding_index.examples != examples:
            self.embedding_index.build_index(examples, self.text_key)
        return self.embedding_index

    def filter_local_dataset(self, server_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Retain local examples relevant to the server query set without exposing them."""
        if not self.use_local_filtering or not self.local_dataset:
            return list(self.local_dataset)

        retained: dict[str, dict[str, Any]] = {}
        index = self._build_index_if_needed(self.local_dataset)
        if index is not None:
            for query in server_queries:
                for example, _ in index.search(str(query.get(self.text_key, "")), self.local_filter_top_k):
                    retained[str(example["id"])] = example
        else:
            for query in server_queries:
                scored = [
                    (example, lexical_similarity(str(query.get(self.text_key, "")), str(example.get(self.text_key, ""))))
                    for example in self.local_dataset
                ]
                scored.sort(key=lambda item: item[1], reverse=True)
                for example, _ in scored[: self.local_filter_top_k]:
                    retained[str(example["id"])] = example
        return list(retained.values())

    def relabel_local_dataset(
        self,
        global_context: list[dict[str, Any]],
        filtered_dataset: list[dict[str, Any]] | None = None,
        allow_original_label_fallback: bool = False,
    ) -> list[dict[str, Any]]:
        """Relabel local examples using server global context as in-context examples."""
        source_dataset = filtered_dataset if filtered_dataset is not None else self.local_dataset
        relabeled: list[dict[str, Any]] = []
        for example in source_dataset:
            prompt = build_mmlu_prompt(global_context, example)
            parsed = parse_mmlu_answer(self.lm_client.generate(prompt))
            new_example = dict(example)
            original_answer = new_example.get("answer")
            if allow_original_label_fallback and original_answer in ANSWER_LETTERS:
                new_example["original_answer"] = original_answer
            if parsed in ANSWER_LETTERS:
                new_example["answer"] = parsed
            elif allow_original_label_fallback and original_answer in ANSWER_LETTERS:
                new_example["answer"] = original_answer
            else:
                new_example["answer"] = self._fallback_answer(str(example.get("id", "")), "relabel")
            new_example["answer_index"] = ANSWER_LETTERS.index(new_example["answer"])
            relabeled.append(new_example)
        return relabeled

    def answer_server_queries(
        self,
        server_queries: list[dict[str, Any]],
        relabeled_dataset: list[dict[str, Any]],
        zero_context: bool = False,
        previous_aggregated_answers: dict[str, str] | None = None,
        previous_vote_counts: dict[str, dict[str, int]] | None = None,
    ) -> list[dict[str, str]]:
        """Return only predictions for server query ids."""
        predictions: list[dict[str, str]] = []
        self.last_selection_records = []
        query_index = self._build_query_index(relabeled_dataset) if not zero_context else None
        for query in server_queries:
            if zero_context:
                context_examples: list[dict[str, Any]] = []
                ordered_selected: list[tuple[dict[str, Any], float]] = []
            else:
                selected = select_examples(
                    query,
                    relabeled_dataset,
                    self.num_context_examples,
                    self.selection_method,
                    self.rng,
                    query_index,
                    self.text_key,
                )
                ordered_selected = order_selected_pairs(selected, self.ordering_method, self.rng)
                context_examples = [example for example, _ in ordered_selected]
            self.last_selection_records.append(
                self._selection_record(
                    query=query,
                    ordered_selected=ordered_selected,
                    similarity_scores_available=(self.selection_method == "knn"),
                )
            )
            query_id = str(query["id"])
            previous_answer = (previous_aggregated_answers or {}).get(query_id)
            if previous_answer in ANSWER_LETTERS:
                prompt = build_mmlu_refinement_prompt(
                    context_examples,
                    query,
                    previous_answer,
                    (previous_vote_counts or {}).get(query_id),
                )
            else:
                prompt = build_mmlu_prompt(context_examples, query)
            parsed = parse_mmlu_answer(self.lm_client.generate(prompt))
            predictions.append(
                {
                    "query_id": query_id,
                    "client_id": self.client_id,
                    "predicted_answer": parsed
                    if parsed in ANSWER_LETTERS
                    else self._fallback_answer(query_id, "prediction"),
                }
            )
        return predictions

    def answer_with_original_dataset(
        self,
        server_queries: list[dict[str, Any]],
        previous_aggregated_answers: dict[str, str] | None = None,
        previous_vote_counts: dict[str, dict[str, int]] | None = None,
    ) -> list[dict[str, str]]:
        """Answer server queries using the private original local dataset."""
        return self.answer_server_queries(
            server_queries,
            self.local_dataset,
            zero_context=False,
            previous_aggregated_answers=previous_aggregated_answers,
            previous_vote_counts=previous_vote_counts,
        )

    def answer_with_relabeled_dataset(
        self,
        global_context: list[dict[str, Any]],
        server_queries: list[dict[str, Any]],
        allow_original_label_fallback: bool = False,
    ) -> list[dict[str, str]]:
        """Answer server queries using locally filtered and relabeled examples."""
        filtered = self.filter_local_dataset(server_queries)
        relabeled = self.relabel_local_dataset(
            global_context,
            filtered,
            allow_original_label_fallback=allow_original_label_fallback,
        )
        return self.answer_server_queries(server_queries, relabeled, zero_context=False)

    def answer_without_context(self, server_queries: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Answer server queries without local in-context examples."""
        return self.answer_server_queries(server_queries, [], zero_context=True)

    def _selection_record(
        self,
        query: dict[str, Any],
        ordered_selected: list[tuple[dict[str, Any], float]],
        similarity_scores_available: bool,
    ) -> dict[str, Any]:
        return {
            "query_id": str(query.get("id", "")),
            "query_text": str(query.get(self.text_key, "")),
            "client_id": self.client_id,
            "selection_method": self.selection_method,
            "selected_example_ids": [str(example.get("id", "")) for example, _ in ordered_selected],
            "selected_example_texts": [str(example.get(self.text_key, "")) for example, _ in ordered_selected],
            "selected_example_labels": [str(example.get("answer", "")) for example, _ in ordered_selected],
            "similarity_scores": [
                float(score) if similarity_scores_available else None
                for _, score in ordered_selected
            ],
        }

    def _build_query_index(self, examples: list[dict[str, Any]]) -> EmbeddingIndex | None:
        if self.selection_method != "knn" or not examples:
            return None
        try:
            index = EmbeddingIndex(
                model_name=self.embedding_model_name,
                batch_size=self.embedding_batch_size,
            )
            index.build_index(examples, self.text_key)
            return index
        except Exception:
            return None

    def run_round(
        self,
        global_context: list[dict[str, Any]],
        server_queries: list[dict[str, Any]],
        mode: str = "fed_icl",
        previous_aggregated_answers: dict[str, str] | None = None,
        previous_vote_counts: dict[str, dict[str, int]] | None = None,
    ) -> list[dict[str, str]]:
        """Run one Fed-ICL client round."""
        if mode == "zero_context_baseline":
            return self.answer_without_context(server_queries)
        if mode == "one_round_fed_icl":
            return self.answer_with_original_dataset(server_queries)
        if mode == "lite_multiround_fed_icl":
            return self.answer_with_original_dataset(
                server_queries,
                previous_aggregated_answers=previous_aggregated_answers,
                previous_vote_counts=previous_vote_counts,
            )
        if mode == "fed_icl":
            return self.answer_with_relabeled_dataset(
                global_context,
                server_queries,
                allow_original_label_fallback=True,
            )
        if mode == "fed_icl_free":
            return self.answer_with_relabeled_dataset(
                global_context,
                server_queries,
                allow_original_label_fallback=False,
            )
        raise ValueError(f"Unsupported experiment mode: {mode}")
