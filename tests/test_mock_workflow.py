import json
from pathlib import Path

from src.client import Client
from src.llm_client import LMClient
from src.server import Server


class RecordingLMClient(LMClient):
    def __init__(self, answer: str = "A") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def _fake_example(index: int, subject: str, answer: str) -> dict:
    return {
        "id": f"{subject}_{index}",
        "subject": subject,
        "question": f"Question {index} about {subject}",
        "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
        "answer": answer,
        "answer_index": ["A", "B", "C", "D"].index(answer),
    }


def _make_clients() -> tuple[list[Client], list[RecordingLMClient]]:
    lm_0 = RecordingLMClient("A")
    lm_1 = RecordingLMClient("B")
    clients = [
        Client(
            client_id="client_0",
            local_dataset=[_fake_example(2, "math", "A"), _fake_example(3, "math", "A")],
            lm_client=lm_0,
            selection_method="random",
            ordering_method="original",
            num_context_examples=1,
            use_local_filtering=False,
            seed=1,
        ),
        Client(
            client_id="client_1",
            local_dataset=[_fake_example(4, "history", "B"), _fake_example(5, "history", "B")],
            lm_client=lm_1,
            selection_method="random",
            ordering_method="original",
            num_context_examples=1,
            use_local_filtering=False,
            seed=2,
        ),
    ]
    return clients, [lm_0, lm_1]


def _server_queries() -> list[dict]:
    return [_fake_example(0, "math", "A"), _fake_example(1, "history", "B")]


def _run_server(tmp_path: Path, mode: str, num_rounds: int = 2) -> tuple[dict, list[RecordingLMClient]]:
    clients, lms = _make_clients()
    server = Server(
        server_queries=_server_queries(),
        clients=clients,
        num_rounds=num_rounds,
        output_dir=tmp_path,
        seed=3,
        mode=mode,
    )
    return server.run(), lms


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_one_round_fed_icl_runs_once_and_uses_original_dataset(tmp_path: Path):
    metrics, lms = _run_server(tmp_path, mode="one_round_fed_icl", num_rounds=5)

    assert metrics["num_rounds"] == 1
    assert (tmp_path / "predictions" / "round_1.jsonl").exists()
    assert not (tmp_path / "predictions" / "round_2.jsonl").exists()

    prompts = [prompt for lm in lms for prompt in lm.prompts]
    assert len(prompts) == 4
    assert all(prompt.count("Question:") == 2 for prompt in prompts)
    assert not any("previous server aggregated answer" in prompt.lower() for prompt in prompts)
    assert not any("previous client vote counts" in prompt.lower() for prompt in prompts)


def test_lite_multiround_single_round_has_no_refinement_prompt(tmp_path: Path):
    metrics, lms = _run_server(tmp_path, mode="lite_multiround_fed_icl", num_rounds=1)

    assert metrics["num_rounds"] == 1
    prompts = [prompt for lm in lms for prompt in lm.prompts]
    assert len(prompts) == 4
    assert not any("previous server aggregated answer" in prompt.lower() for prompt in prompts)
    assert not any("previous client vote counts" in prompt.lower() for prompt in prompts)


def test_zero_context_baseline_uses_no_context_examples(tmp_path: Path):
    metrics, lms = _run_server(tmp_path, mode="zero_context_baseline", num_rounds=3)

    assert metrics["num_rounds"] == 1
    prompts = [prompt for lm in lms for prompt in lm.prompts]
    assert len(prompts) == 4
    assert all(prompt.count("Question:") == 1 for prompt in prompts)


def test_fed_icl_runs_multiple_rounds(tmp_path: Path):
    metrics, lms = _run_server(tmp_path, mode="fed_icl", num_rounds=2)

    assert metrics["num_rounds"] == 2
    assert (tmp_path / "predictions" / "round_1.jsonl").exists()
    assert (tmp_path / "predictions" / "round_2.jsonl").exists()
    prompts = [prompt for lm in lms for prompt in lm.prompts]
    assert len(prompts) > 4


def test_lite_multiround_writes_one_metrics_row_per_round(tmp_path: Path):
    metrics, _ = _run_server(tmp_path, mode="lite_multiround_fed_icl", num_rounds=3)

    assert metrics["num_rounds"] == 3
    csv_lines = (tmp_path / "per_round_metrics.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 4
    assert csv_lines[0] == "round,accuracy"
    assert [line.split(",")[0] for line in csv_lines[1:]] == ["1", "2", "3"]


def test_lite_multiround_saves_detailed_round_predictions(tmp_path: Path):
    _run_server(tmp_path, mode="lite_multiround_fed_icl", num_rounds=3)

    rows = _load_jsonl(tmp_path / "round_predictions.jsonl")
    assert rows
    assert len(rows) == 6
    first = rows[0]
    assert {
        "round",
        "query_id",
        "query_text",
        "client_answers",
        "aggregated_answer",
        "vote_counts",
        "previous_vote_counts_used",
        "gold_answer",
        "is_correct",
        "answer_changed_from_previous_round",
    } <= set(first.keys())
    assert first["answer_changed_from_previous_round"] is None
    assert first["previous_vote_counts_used"] is None
    assert any(row["round"] == 2 and isinstance(row["answer_changed_from_previous_round"], bool) for row in rows)
    assert any(row["round"] == 2 and row["previous_vote_counts_used"] == {"A": 1, "B": 1, "C": 0, "D": 0} for row in rows)


def test_refinement_round_prompts_include_previous_aggregated_answer(tmp_path: Path):
    _, lms = _run_server(tmp_path, mode="lite_multiround_fed_icl", num_rounds=2)

    prompts = [prompt for lm in lms for prompt in lm.prompts]
    assert len(prompts) == 8
    for lm in lms:
        assert not any("previous server aggregated answer" in prompt.lower() for prompt in lm.prompts[:2])
        assert not any("previous client vote counts" in prompt.lower() for prompt in lm.prompts[:2])
        assert all("The previous server aggregated answer was:" in prompt for prompt in lm.prompts[2:])
        assert all("The previous client vote counts were: A:1, B:1, C:0, D:0." in prompt for prompt in lm.prompts[2:])
        assert all("This previous answer may be correct or incorrect." in prompt for prompt in lm.prompts[2:])


def test_prediction_outputs_do_not_contain_raw_questions_or_choices(tmp_path: Path):
    _run_server(tmp_path, mode="fed_icl", num_rounds=2)

    for path in (tmp_path / "predictions").glob("*.json*"):
        text = path.read_text(encoding="utf-8").lower()
        assert "question" not in text
        assert "choices" not in text
        assert "choice a" not in text


def test_client_predictions_have_only_allowed_fields():
    clients, _ = _make_clients()
    predictions = clients[0].run_round([], _server_queries(), mode="one_round_fed_icl")

    assert predictions
    assert all(set(row.keys()) == {"query_id", "client_id", "predicted_answer"} for row in predictions)


def test_saved_round_predictions_have_only_server_safe_fields(tmp_path: Path):
    _run_server(tmp_path, mode="one_round_fed_icl", num_rounds=1)

    rows = _load_jsonl(tmp_path / "predictions" / "round_1.jsonl")
    assert rows
    assert all(set(row.keys()) == {"round", "query_id", "predicted_answer", "vote_counts"} for row in rows)


def test_selected_examples_are_saved_with_query_and_selection_details(tmp_path: Path):
    _run_server(tmp_path, mode="one_round_fed_icl", num_rounds=1)

    rows = _load_jsonl(tmp_path / "selected_examples.jsonl")
    assert rows
    first = rows[0]
    assert {
        "round",
        "query_id",
        "query_text",
        "client_id",
        "selection_method",
        "selected_example_ids",
        "selected_example_texts",
        "selected_example_labels",
        "similarity_scores",
    } <= set(first.keys())
    assert isinstance(first["selected_example_ids"], list)
    assert isinstance(first["selected_example_texts"], list)
    assert isinstance(first["selected_example_labels"], list)


def test_root_level_metrics_files_are_saved(tmp_path: Path):
    _run_server(tmp_path, mode="one_round_fed_icl", num_rounds=1)

    assert (tmp_path / "final_metrics.json").exists()
    assert (tmp_path / "per_round_metrics.csv").exists()
