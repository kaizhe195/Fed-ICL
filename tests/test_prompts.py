from src.llm_client import parse_mmlu_answer
from src.prompts import build_mmlu_prompt, build_mmlu_refinement_prompt, format_mmlu_example, format_vote_counts


def _example(answer="A"):
    return {
        "id": "q1",
        "subject": "demo",
        "question": "Which option is correct?",
        "choices": ["Alpha", "Beta", "Gamma", "Delta"],
        "answer": answer,
        "answer_index": 0,
    }


def test_prompt_contains_choices():
    prompt = build_mmlu_prompt([_example("B")], _example())
    assert "A. Alpha" in prompt
    assert "B. Beta" in prompt
    assert "C. Gamma" in prompt
    assert "D. Delta" in prompt
    assert prompt.strip().endswith("Answer:")


def test_refinement_prompt_contains_previous_server_answer_and_vote_counts():
    prompt = build_mmlu_refinement_prompt(
        [_example("B")],
        _example(),
        "C",
        {"A": 0, "B": 1, "C": 2, "D": 0},
    )

    assert "The previous server aggregated answer was: C." in prompt
    assert "The previous client vote counts were: A:0, B:1, C:2, D:0." in prompt
    assert "This previous answer may be correct or incorrect." in prompt
    assert "If the previous answer seems correct, keep it." in prompt
    assert "Answer with only one letter: A, B, C, or D." in prompt
    assert prompt.strip().endswith("Answer:")


def test_vote_counts_are_formatted_in_abcd_order():
    assert format_vote_counts({"B": 2, "A": 1}) == "A:1, B:2, C:0, D:0"


def test_format_includes_answer_when_requested():
    text = format_mmlu_example(_example("C"), include_answer=True)
    assert "Answer: C" in text


def test_answer_parsing():
    assert parse_mmlu_answer("B") == "B"
    assert parse_mmlu_answer("The answer is D.") == "D"
    assert parse_mmlu_answer("unknown") is None


def test_answer_parsing_handles_common_llm_outputs():
    assert parse_mmlu_answer("A.") == "A"
    assert parse_mmlu_answer("Answer: A") == "A"
    assert parse_mmlu_answer("The answer is A.") == "A"
    assert parse_mmlu_answer("I think the correct answer is B.") == "B"
    assert parse_mmlu_answer("A is plausible, but the answer is C.") == "C"
    assert parse_mmlu_answer("No valid option is provided.") is None
