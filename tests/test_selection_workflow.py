import json
import random
from pathlib import Path

from scripts import run_selection_experiments
from src.retrieval import select_examples
from src.result_collector import collect_results
from src.utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_selection_config_files_exist_and_load():
    expected = [
        "configs/selection_random_mmlu.yaml",
        "configs/selection_knn_mmlu.yaml",
        "configs/selection_smoke_mmlu.yaml",
        "configs/main_mmlu_random_seed42.yaml",
        "configs/main_mmlu_random_seed43.yaml",
        "configs/main_mmlu_random_seed44.yaml",
        "configs/main_mmlu_knn_seed42.yaml",
        "configs/main_mmlu_knn_seed43.yaml",
        "configs/main_mmlu_knn_seed44.yaml",
        "configs/multiround_mmlu_random_seed42.yaml",
        "configs/multiround_mmlu_knn_seed42.yaml",
    ]
    for relative_path in expected:
        config_path = PROJECT_ROOT / relative_path
        assert config_path.exists()
        config = load_yaml(config_path)
        assert config["dataset"]["name"] == "cais/mmlu"
        assert config["experiment"]["ordering_method"] == "similarity_desc"


def test_multiround_configs_use_lite_mode_and_requested_limits():
    expected = {
        "configs/multiround_mmlu_random_seed42.yaml": (
            "random",
            "outputs/multiround_mmlu_random_seed42",
        ),
        "configs/multiround_mmlu_knn_seed42.yaml": (
            "knn",
            "outputs/multiround_mmlu_knn_seed42",
        ),
    }
    for relative_path, (selection_method, output_dir) in expected.items():
        config = load_yaml(PROJECT_ROOT / relative_path)
        assert config["dataset"]["name"] == "cais/mmlu"
        assert config["dataset"]["split"] == "test"
        assert config["dataset"]["fallback_split"] == "validation"
        assert config["dataset"]["max_subjects"] == 5
        assert config["dataset"]["num_server_samples_per_subject"] == 5
        assert config["experiment"]["mode"] == "lite_multiround_fed_icl"
        assert config["experiment"]["num_clients"] == 3
        assert config["experiment"]["num_rounds"] == 3
        assert config["experiment"]["num_context_examples"] == 3
        assert config["experiment"]["selection_method"] == selection_method
        assert config["experiment"]["ordering_method"] == "similarity_desc"
        assert config["experiment"]["seed"] == 42
        assert config["experiment"]["output_dir"] == output_dir
        assert config["llm"]["backend"] == "ollama"


def test_knn_configs_define_local_filtering_defaults():
    for relative_path in [
        "configs/default_mmlu.yaml",
        "configs/selection_knn_mmlu.yaml",
        "configs/main_mmlu_knn_seed42.yaml",
        "configs/main_mmlu_knn_seed43.yaml",
        "configs/main_mmlu_knn_seed44.yaml",
        "configs/multiround_mmlu_knn_seed42.yaml",
    ]:
        config = load_yaml(PROJECT_ROOT / relative_path)
        assert config["local_filtering"] == {
            "enabled": True,
            "filter_top_k_per_query": 5,
            "max_filtered_examples": 100,
            "scoring": "max_similarity",
        }


def test_random_and_basic_knn_selection_methods_still_select_examples():
    query = {"id": "q", "question": "linear algebra vector space"}
    examples = [
        {"id": "a", "question": "ancient history empire"},
        {"id": "b", "question": "linear algebra matrix vector"},
        {"id": "c", "question": "biology cell structure"},
    ]

    random_selected = select_examples(query, examples, 2, "random", random.Random(1))
    basic_knn_selected = select_examples(query, examples, 2, "basic_knn", random.Random(1), None)

    assert len(random_selected) == 2
    assert [example["id"] for example, _ in basic_knn_selected][0] == "b"


def test_main_experiment_configs_use_real_backend_and_single_round():
    for relative_path in [
        "configs/main_mmlu_random_seed42.yaml",
        "configs/main_mmlu_random_seed43.yaml",
        "configs/main_mmlu_random_seed44.yaml",
        "configs/main_mmlu_knn_seed42.yaml",
        "configs/main_mmlu_knn_seed43.yaml",
        "configs/main_mmlu_knn_seed44.yaml",
    ]:
        config = load_yaml(PROJECT_ROOT / relative_path)
        assert config["llm"]["backend"] == "ollama"
        assert config["experiment"]["num_rounds"] == 1
        assert config["dataset"]["num_server_samples_per_subject"] == 5
        assert config["dataset"]["max_subjects"] == 10


def test_selection_runner_accepts_smoke_config_without_network(monkeypatch, tmp_path):
    calls = []

    def fake_run_mmlu_experiment(config_path, seed_override=None, output_dir_override=None):
        calls.append((config_path, seed_override, output_dir_override))
        output_dir = tmp_path / str(output_dir_override)
        (output_dir / "metrics").mkdir(parents=True)
        (output_dir / "config.resolved.json").write_text(
            json.dumps(
                {
                    "dataset": {"name": "cais/mmlu"},
                    "experiment": {
                        "name": "selection_smoke_mmlu",
                        "selection_method": "random",
                        "ordering_method": "similarity_desc",
                        "num_clients": 3,
                        "num_rounds": 2,
                        "alpha": 100.0,
                        "num_context_examples": 3,
                        "use_local_filtering": False,
                    },
                    "llm": {"backend": "mock"},
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "metrics" / "final_metrics.json").write_text(
            json.dumps({"final_accuracy": 0.5, "num_rounds": 2}),
            encoding="utf-8",
        )
        return {
            "experiment_name": "selection_smoke_mmlu",
            "selection_method": "random",
            "num_clients": 3,
            "num_rounds": 2,
            "alpha": 100.0,
            "num_context_examples": 3,
            "backend": "mock",
            "final_accuracy": 0.5,
            "output_dir": str(output_dir),
        }

    collected = {}

    def fake_collect_results(output_dirs, summary_path):
        collected["output_dirs"] = output_dirs
        collected["summary_path"] = summary_path
        return []

    monkeypatch.setattr(run_selection_experiments, "run_mmlu_experiment", fake_run_mmlu_experiment)
    monkeypatch.setattr(run_selection_experiments, "collect_results", fake_collect_results)

    rows = run_selection_experiments.run_selection_configs(["configs/selection_smoke_mmlu.yaml"])

    assert len(rows) == 1
    assert calls[0][1] == 42
    assert calls[0][2] == "outputs/selection_smoke"
    assert collected["output_dirs"] == [rows[0]["output_dir"]]


def test_result_collector_writes_selection_summary(tmp_path):
    output_dir = tmp_path / "selection_random_mmlu"
    (output_dir / "metrics").mkdir(parents=True)
    (output_dir / "config.resolved.json").write_text(
        json.dumps(
            {
                "dataset": {"name": "cais/mmlu"},
                "experiment": {
                    "name": "selection_random_mmlu",
                    "selection_method": "random",
                    "ordering_method": "similarity_desc",
                    "num_clients": 3,
                    "num_rounds": 6,
                    "alpha": 100.0,
                    "num_context_examples": 5,
                    "use_local_filtering": False,
                },
                "llm": {"backend": "mock"},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "metrics" / "final_metrics.json").write_text(
        json.dumps({"final_accuracy": 0.25, "num_rounds": 6}),
        encoding="utf-8",
    )

    summary_path = tmp_path / "outputs" / "selection_summary.csv"
    rows = collect_results([output_dir], summary_path)

    assert summary_path.exists()
    text = summary_path.read_text(encoding="utf-8")
    assert "selection_random_mmlu" in text
    assert rows[0]["ordering_method"] == "similarity_desc"


def test_ordering_is_not_reported_as_runner_variable():
    assert "ordering_method" not in run_selection_experiments.TABLE_COLUMNS


def test_readme_contains_mocklm_warning():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "MockLM is only for pipeline verification" in readme
    assert "Selection comparison under MockLM is not meaningful" in readme
