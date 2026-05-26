# fed_icl_reproduction

This project is a Python research prototype for **Fed-ICL example selection under federated constraints**. The current main implemented branch is MMLU multiple-choice QA.

The goal is to provide a clean baseline for comparing example selection methods in the same Fed-ICL workflow. The implementation simulates a central server, multiple clients with private QA datasets, local in-context prompting, client prediction sharing, server-side majority voting, and iterative refinement.

## What is implemented

- MMLU loading from Hugging Face with normalized multiple-choice examples.
- Server query construction with two examples per subject when available.
- Dirichlet non-IID partitioning of the remaining examples across clients.
- Local kNN or random example selection.
- Fixed example ordering for selection-focused experiments. `ordering_method` remains an implementation parameter, but it is not a main research variable.
- Client-side prompt construction and local answer generation.
- Round-based Fed-ICL server-client workflow.
- Experiment modes for zero-context, one-round, iterative Fed-ICL, and a simplified Fed-ICL-free branch.
- Majority voting aggregation with deterministic A before B before C before D tie-breaking.
- Per-round and final accuracy outputs.
- MockLM backend for tests and dry runs without API keys. MockLM is only for pipeline verification.
- Extension points for OpenAI-compatible APIs and a local Ollama HTTP backend.
- A TruthfulQA placeholder branch for later open-ended generation and fusion work.

## What is not implemented

- Traditional FedAvg.
- LLM fine-tuning or parameter training.
- Sending raw client examples to the server.
- Blockchain, adversarial attacks, privacy attacks, agent tool-use, or unrelated federated fine-tuning.
- Claims that this prototype reproduces all reported paper numbers.
- Full TruthfulQA generation, judging, or open-ended answer fusion.

## Privacy simulation rule

The server receives only client predictions for server queries, not raw client examples.

Server-side prediction files contain query ids, predicted labels, and vote counts. Client local examples are used only inside the client.

## Installation

```bash
cd fed_icl_reproduction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with:

```bash
source .venv/bin/activate
```

## Run MMLU

```bash
python scripts/run_mmlu.py --config configs/default_mmlu.yaml
```

The default config uses `llm.backend: "mock"`, so no external API key is required. MockLM results are not real model results and must not be interpreted as selection quality. The first full MMLU run may still download the Hugging Face dataset and the sentence-transformers embedding model.

## Selection-focused experiment workflow

The current MSc project focus is example selection in Fed-ICL. The main comparison is random selection versus kNN selection under the same federated workflow.

Ordering is fixed with `ordering_method: "similarity_desc"` in the selection configs and is not treated as a main experimental factor. Random selection may still preserve a fixed internal order for prompt construction, but the experiment runner reports selection method rather than ordering as the variable of interest.

MockLM is only for pipeline verification. Selection comparison under MockLM is not meaningful for model quality, because MockLM is a deterministic heuristic backend rather than a real language model. A real LLM backend is required before interpreting selection performance.

Smoke run:

```bash
python scripts/run_mmlu.py --config configs/selection_smoke_mmlu.yaml
```

Run the main selection comparison:

```bash
python scripts/run_selection_experiments.py --configs configs/selection_random_mmlu.yaml configs/selection_knn_mmlu.yaml
```

The selection experiment runner writes a compact CSV summary to:

```text
outputs/selection_summary.csv
```

## Running with Ollama

Use Ollama only after the MockLM pipeline checks pass. Start or install Ollama first, then pull a small local model. The default smoke config uses `gemma3:1b` because it is small and suitable for a first local check.

```bash
ollama pull gemma3:1b
```

Check the Ollama HTTP backend:

```bash
python scripts/check_ollama.py --model gemma3:1b --base-url http://localhost:11434
```

Run the small real-backend MMLU smoke test:

```bash
python scripts/run_mmlu.py --config configs/ollama_smoke_mmlu.yaml
```

This is a real-backend smoke test only. It uses a tiny MMLU subset, random example selection, fixed ordering, and one Fed-ICL round. It is too small for research conclusions. Run the full random versus kNN selection comparison only after this smoke test works reliably.

Do not mix MockLM and Ollama results in the same analysis table. MockLM verifies code paths, while Ollama produces real local model outputs under a different backend.

## Main config fields

Important fields in `configs/default_mmlu.yaml`:

- `experiment.num_clients`
- `experiment.num_rounds`
- `experiment.alpha`
- `experiment.num_context_examples`
- `experiment.selection_method`
- `experiment.ordering_method` for fixed prompt construction order
- `embedding.model_name`
- `llm.backend`
- `experiment.seed`

Supported modes:

- `zero_context_baseline`
- `one_round_fed_icl`
- `fed_icl`
- `fed_icl_free`

Mode behavior:

- `zero_context_baseline`: clients answer server queries without local context examples.
- `one_round_fed_icl`: clients answer once using original local ground-truth examples as context, and no local relabeling is performed.
- `fed_icl`: clients filter local data, relabel local examples using the current server context, answer with the relabeled local examples, and the server iterates.
- `fed_icl_free`: a simplified experimental branch where clients rely on relabeled local data for query answering and do not use original labels as a fallback.

## Outputs

Processed and partitioned data:

- `data/processed/mmlu_server_queries.jsonl`
- `data/processed/mmlu_remaining_pool.jsonl`
- `data/processed/mmlu_metadata.json`
- `data/clients/mmlu/client_0.jsonl`
- `data/clients/mmlu/client_1.jsonl`
- `data/clients/mmlu/client_stats.json`
- `data/clients/mmlu/client_stats.csv`

Experiment outputs:

- `outputs/mmlu/config.yaml`
- `outputs/mmlu/config.resolved.json`
- `outputs/mmlu/logs/run.log`
- `outputs/mmlu/predictions/round_*.jsonl`
- `outputs/mmlu/predictions/final_global_context.json`
- `outputs/mmlu/metrics/per_round_metrics.csv`
- `outputs/mmlu/metrics/final_metrics.json`

## Tests

```bash
python -m pytest
```

The tests use synthetic MMLU-like examples and the mock backend, so they do not require downloading MMLU or using an external LLM.

## Limitations

- MockLM does not reflect real LLM performance.
- MockLM is only for checking the pipeline.
- Selection comparison under MockLM is not meaningful for model quality.
- A real LLM backend must be used before meaningful selection conclusions.
- This prototype reproduces the main Fed-ICL workflow, not all reported paper results.
- TruthfulQA is only a placeholder for future extension.
- Ollama smoke results are not final experiment results because the smoke config is intentionally tiny.

## Future extensions

- Add a real LLM backend.
- Add TruthfulQA generation and answer fusion.
