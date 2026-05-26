"""Experiment runner for the Fed-ICL reproduction prototype.启动器"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

from .client import Client
from .data_loader import prepare_mmlu_data
from .llm_client import create_lm_client
from .logging_utils import setup_logging
from .partition import dirichlet_partition, save_client_partitions
from .server import Server
from .utils import ensure_dir, load_yaml, save_json, set_seed


def run_mmlu_experiment(
    config_path: str | Path,
    seed_override: int | None = None,
    output_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full MMLU Fed-ICL workflow."""
    config_path = Path(config_path)
    config = copy.deepcopy(load_yaml(config_path))
    experiment_cfg = config["experiment"]
    if seed_override is not None:
        experiment_cfg["seed"] = int(seed_override)
    if output_dir_override is not None:
        experiment_cfg["output_dir"] = str(output_dir_override)
    seed = int(experiment_cfg.get("seed", 42))
    set_seed(seed)

    project_root = Path(__file__).resolve().parents[1]
    configured_output_dir = str(experiment_cfg.get("output_dir", "outputs/mmlu"))
    output_dir = project_root / configured_output_dir
    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")
    logger = setup_logging(output_dir / "logs")
    logger.info("Starting MMLU Fed-ICL experiment")
    backend = str(config.get("llm", {}).get("backend", "mock"))
    if backend == "mock":
        logger.warning(
            "MockLM is active. This run verifies the pipeline only; selection accuracy "
            "must not be interpreted as real model performance."
        )
    shutil.copyfile(config_path, output_dir / "config.yaml")
    shutil.copyfile(config_path, output_dir / "run_config.yaml")
    save_json(config, output_dir / "config.resolved.json")

    logger.info("Loading and preprocessing MMLU")
    server_queries, remaining_pool = prepare_mmlu_data(config, project_root)

    logger.info("Partitioning remaining pool into clients")
    clients_data = dirichlet_partition(
        remaining_pool,
        num_clients=int(experiment_cfg["num_clients"]),
        alpha=float(experiment_cfg["alpha"]),
        seed=seed,
        partition_key=experiment_cfg.get("partition_key", "subject"),
    )
    save_client_partitions(clients_data, project_root / "data" / "clients" / "mmlu")

    clients: list[Client] = []
    for index, (client_id, local_dataset) in enumerate(sorted(clients_data.items())):
        lm_client = create_lm_client(config.get("llm", {}), seed=seed + index)
        embedding_cfg = config.get("embedding", {})
        local_filtering_cfg = config.get("local_filtering", {})
        clients.append(
            Client(
                client_id=client_id,
                local_dataset=local_dataset,
                lm_client=lm_client,
                selection_method=experiment_cfg.get("selection_method", "knn"),
                ordering_method=experiment_cfg.get("ordering_method", "similarity_desc"),
                num_context_examples=int(experiment_cfg.get("num_context_examples", 5)),
                use_local_filtering=bool(experiment_cfg.get("use_local_filtering", True)),
                local_filter_top_k=int(experiment_cfg.get("local_filter_top_k", 50)),
                local_filtering_enabled=bool(local_filtering_cfg.get("enabled", True)),
                filter_top_k_per_query=int(local_filtering_cfg.get("filter_top_k_per_query", 5)),
                max_filtered_examples=local_filtering_cfg.get("max_filtered_examples", 100),
                local_filtering_scoring=str(local_filtering_cfg.get("scoring", "max_similarity")),
                text_key=embedding_cfg.get("text_key", "question"),
                embedding_model_name=embedding_cfg.get(
                    "model_name", "sentence-transformers/paraphrase-MiniLM-L6-v2"
                ),
                embedding_batch_size=int(embedding_cfg.get("batch_size", 32)),
                seed=seed + index,
            )
        )

    server = Server(
        server_queries=server_queries,
        clients=clients,
        num_rounds=int(experiment_cfg.get("num_rounds", 6)),
        output_dir=output_dir,
        aggregation_method=config.get("aggregation", {}).get("method", "majority_vote"),
        seed=seed,
        mode=experiment_cfg.get("mode", "fed_icl"),
        save_per_subject_accuracy=bool(config.get("evaluation", {}).get("save_per_subject_accuracy", True)),
    )
    final_metrics = server.run()
    summary = {
        "num_clients": len(clients),
        "num_server_queries": len(server_queries),
        "num_rounds": final_metrics["num_rounds"],
        "alpha": float(experiment_cfg["alpha"]),
        "num_context_examples": int(experiment_cfg.get("num_context_examples", 5)),
        "selection_method": experiment_cfg.get("selection_method", "knn"),
        "ordering_method": experiment_cfg.get("ordering_method", "similarity_desc"),
        "backend": backend,
        "final_accuracy": final_metrics["final_accuracy"],
        "experiment_name": experiment_cfg.get("name", config_path.stem),
        "output_path": configured_output_dir,
        "output_dir": configured_output_dir,
    }
    logger.info("Final summary: %s", summary)
    return summary
