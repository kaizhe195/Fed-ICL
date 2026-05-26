"""Dataset loading and preprocessing.数据集和预处理"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .utils import ensure_dir, save_json, save_jsonl

ANSWER_LETTERS = ["A", "B", "C", "D"]


def _load_hf_dataset(dataset_name: str, split: str, fallback_split: str | None) -> Any:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load MMLU. "
            "Install project requirements before running the full experiment."
        ) from exc

    errors: list[str] = []
    candidate_configs: list[str | None] = ["all", None]
    try:
        configs = get_dataset_config_names(dataset_name)
        if "all" in configs:
            candidate_configs = ["all"]
        elif configs:
            candidate_configs = configs
    except Exception as exc:
        errors.append(str(exc))

    for requested_split in [split, fallback_split]:
        if requested_split is None:
            continue
        if len(candidate_configs) > 1 and "all" not in candidate_configs:
            combined_rows: list[dict[str, Any]] = []
            successful_configs = 0
            for config_name in candidate_configs:
                try:
                    dataset = load_dataset(dataset_name, config_name, split=requested_split)
                    for row in dataset:
                        row_dict = dict(row)
                        row_dict.setdefault("subject", str(config_name))
                        combined_rows.append(row_dict)
                    successful_configs += 1
                except Exception as exc:
                    errors.append(f"{dataset_name}/{config_name}/{requested_split}: {exc}")
            if successful_configs > 0:
                return combined_rows
        for config_name in candidate_configs:
            try:
                if config_name is None:
                    return load_dataset(dataset_name, split=requested_split)
                return load_dataset(dataset_name, config_name, split=requested_split)
            except Exception as exc:
                errors.append(f"{dataset_name}/{config_name}/{requested_split}: {exc}")

    joined = "\n".join(errors[-8:])
    raise RuntimeError(f"Unable to load dataset '{dataset_name}'. Recent errors:\n{joined}")


def _as_choice_list(raw: dict[str, Any]) -> list[str]:
    if isinstance(raw.get("choices"), list):
        choices = [str(x) for x in raw["choices"]]
    elif all(key in raw for key in ["A", "B", "C", "D"]):
        choices = [str(raw[key]) for key in ["A", "B", "C", "D"]]
    elif all(key in raw for key in ["option_a", "option_b", "option_c", "option_d"]):
        choices = [str(raw[key]) for key in ["option_a", "option_b", "option_c", "option_d"]]
    else:
        raise ValueError(f"Cannot find four answer choices in example keys: {sorted(raw.keys())}")
    if len(choices) < 4:
        raise ValueError("MMLU examples must contain at least four choices.")
    return choices[:4]


def _answer_to_index(answer: Any) -> int:
    if isinstance(answer, int):
        index = answer
    elif isinstance(answer, str):
        stripped = answer.strip()
        if stripped in ANSWER_LETTERS:
            index = ANSWER_LETTERS.index(stripped)
        elif stripped.isdigit():
            index = int(stripped)
        else:
            upper = stripped.upper()
            if upper in ANSWER_LETTERS:
                index = ANSWER_LETTERS.index(upper)
            else:
                raise ValueError(f"Unsupported MMLU answer label: {answer}")
    else:
        raise ValueError(f"Unsupported MMLU answer type: {type(answer)}")
    if index not in range(4):
        raise ValueError(f"MMLU answer index must be 0, 1, 2, or 3. Got {index}.")
    return index


def normalize_mmlu_example(raw: dict[str, Any], row_index: int, default_subject: str = "unknown") -> dict[str, Any]:
    """Normalize a Hugging Face MMLU row to the internal schema."""
    question = raw.get("question") or raw.get("input") or raw.get("prompt")
    if not question:
        raise ValueError(f"Cannot find question text in example keys: {sorted(raw.keys())}")
    subject = raw.get("subject") or raw.get("category") or raw.get("task") or default_subject
    choices = _as_choice_list(raw)
    answer_index = _answer_to_index(raw.get("answer", raw.get("label")))
    example_id = raw.get("id") or raw.get("example_id") or f"{subject}_{row_index}"
    return {
        "id": str(example_id),
        "subject": str(subject),
        "question": str(question),
        "choices": choices,
        "answer": ANSWER_LETTERS[answer_index],
        "answer_index": answer_index,
    }


def load_mmlu(
    dataset_name: str = "cais/mmlu",
    split: str = "test",
    fallback_split: str | None = "validation",
    max_subjects: int | None = None,
) -> list[dict[str, Any]]:
    """Load and normalize MMLU from Hugging Face datasets."""
    dataset = _load_hf_dataset(dataset_name, split, fallback_split)
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        examples.append(normalize_mmlu_example(dict(row), index))

    if max_subjects is not None:
        subjects = sorted({example["subject"] for example in examples})[:max_subjects]
        keep = set(subjects)
        examples = [example for example in examples if example["subject"] in keep]
    return examples


def build_server_query_set(
    examples: list[dict[str, Any]],
    samples_per_subject: int = 2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select server queries per subject and remove them from the remaining pool."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[example["subject"]].append(example)

    rng = random.Random(seed)
    selected_ids: set[str] = set()
    server_queries: list[dict[str, Any]] = []

    for subject in sorted(grouped):
        subject_examples = list(grouped[subject])
        rng.shuffle(subject_examples)
        selected = subject_examples[: min(samples_per_subject, len(subject_examples))]
        server_queries.extend(selected)
        selected_ids.update(example["id"] for example in selected)

    remaining_pool = [example for example in examples if example["id"] not in selected_ids]
    metadata = {
        "num_total_examples": len(examples),
        "num_subjects": len(grouped),
        "num_server_queries": len(server_queries),
        "num_remaining_examples": len(remaining_pool),
        "samples_per_subject": samples_per_subject,
        "expected_mmlu_server_queries_if_57_subjects": 114,
    }
    return server_queries, remaining_pool, metadata


def prepare_mmlu_data(config: dict[str, Any], output_base: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load MMLU, build server queries, and save processed files."""
    dataset_cfg = config["dataset"]
    examples = load_mmlu(
        dataset_name=dataset_cfg.get("name", "cais/mmlu"),
        split=dataset_cfg.get("split", "test"),
        fallback_split=dataset_cfg.get("fallback_split", "validation"),
        max_subjects=dataset_cfg.get("max_subjects"),
    )
    server_queries, remaining_pool, metadata = build_server_query_set(
        examples,
        samples_per_subject=int(dataset_cfg.get("num_server_samples_per_subject", 2)),
        seed=int(config.get("experiment", {}).get("seed", 42)),
    )

    processed_dir = ensure_dir(Path(output_base) / "data" / "processed")
    save_jsonl(server_queries, processed_dir / "mmlu_server_queries.jsonl")
    save_jsonl(remaining_pool, processed_dir / "mmlu_remaining_pool.jsonl")
    save_json(metadata, processed_dir / "mmlu_metadata.json")
    return server_queries, remaining_pool
