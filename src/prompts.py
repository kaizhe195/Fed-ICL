"""Prompt construction for MMLU multiple-choice QA.把 examples 变成模型输入"""

from __future__ import annotations

from typing import Any

ANSWER_LETTERS = ["A", "B", "C", "D"]


def format_mmlu_example(example: dict[str, Any], include_answer: bool = True) -> str:
    """Format one MMLU example in a stable multiple-choice layout."""
    choices = example.get("choices")
    if not isinstance(choices, list) or len(choices) < 4:
        raise ValueError("MMLU example must contain at least four choices.")
    lines = [f"Question: {example['question']}"]
    for letter, choice in zip(ANSWER_LETTERS, choices[:4]):
        lines.append(f"{letter}. {choice}")
    if include_answer:
        answer = str(example.get("answer", "")).strip()
        if answer not in ANSWER_LETTERS:
            raise ValueError(f"MMLU answer must be one of A, B, C, D. Got: {answer}")
        lines.append(f"Answer: {answer}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def build_mmlu_prompt(context_examples: list[dict[str, Any]], query_example: dict[str, Any]) -> str:
    """Build an in-context prompt for one MMLU query."""
    instruction = "Choose the correct answer. Reply with only one letter: A, B, C, or D."
    parts = [instruction]
    for example in context_examples:
        parts.append(format_mmlu_example(example, include_answer=True))
    parts.append(format_mmlu_example(query_example, include_answer=False))
    return "\n\n".join(parts)


def build_mmlu_refinement_prompt(
    context_examples: list[dict[str, Any]],
    query_example: dict[str, Any],
    previous_aggregated_answer: str,
    previous_vote_counts: dict[str, int] | None = None,
) -> str:
    """Build a refinement-round prompt for one MMLU query."""
    if previous_aggregated_answer not in ANSWER_LETTERS:
        raise ValueError(
            "Previous aggregated answer must be one of A, B, C, D. "
            f"Got: {previous_aggregated_answer}"
        )
    instruction = (
        "This is a refinement round for an MMLU multiple-choice question. "
        "Review the local examples and the previous server aggregated answer. "
        "Keep the previous answer if it seems correct, or change it if the local examples "
        "suggest a better answer. Reply with only one letter: A, B, C, or D."
    )
    parts = [instruction]
    for example in context_examples:
        parts.append(format_mmlu_example(example, include_answer=True))
    previous_lines = [f"Previous server aggregated answer: {previous_aggregated_answer}"]
    if previous_vote_counts:
        counts = ", ".join(f"{letter}: {int(previous_vote_counts.get(letter, 0))}" for letter in ANSWER_LETTERS)
        previous_lines.append(f"Previous vote counts: {counts}")
    parts.append("\n".join(previous_lines))
    parts.append(format_mmlu_example(query_example, include_answer=False))
    return "\n\n".join(parts)
