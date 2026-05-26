from pathlib import Path

import pytest

from src.llm_client import MockLMClient, create_lm_client
from src.utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_create_lm_client_returns_mock_by_default():
    client = create_lm_client({"backend": "mock"}, seed=123)

    assert isinstance(client, MockLMClient)


def test_create_lm_client_rejects_unsupported_backend():
    with pytest.raises(ValueError, match="Unsupported LLM backend"):
        create_lm_client({"backend": "missing_backend"})


def test_ollama_smoke_config_exists_and_loads():
    config = load_yaml(PROJECT_ROOT / "configs" / "ollama_smoke_mmlu.yaml")

    assert config["dataset"]["max_subjects"] == 3
    assert config["experiment"]["selection_method"] == "random"
    assert config["experiment"]["ordering_method"] == "similarity_desc"
    assert config["llm"]["backend"] == "ollama"
    assert config["llm"]["model_name"] == "gemma3:1b"
    assert config["llm"]["base_url"] == "http://localhost:11434"
