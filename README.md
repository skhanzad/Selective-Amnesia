# Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM Agents

A knowledge graph system with a forgetting module for augmenting LLM responses with persistent user memory, evaluated against the LongMemEval benchmark.

## Prerequisites

* Python 3.10+

* [Ollama](https://ollama.com) running locally with the required models pulled:

  ```bash
  ollama pull nomic-embed-text
  ```

* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

* Create a `.env` file in the project root and add your OpenAI API key:

  ```env
  OPENAI_API_KEY=your_openai_api_key_here
  ```

* Download the `longmemeval_oracle.json` dataset from [Hugging Face](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned):

  1. Open the [LongMemEval cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/main).
  2. Download `longmemeval_oracle.json`.
  3. Create a `data` folder in the root of the project.
  4. Place `longmemeval_oracle.json` inside the `data` folder.

  Your project structure should look like:

  ```text
  project-root/
  ├── data/
  │   └── longmemeval_oracle.json
  ├── .env
  ├── requirements.txt
  └── ...
  ```

## Running

### Combined Benchmark

`combined_benchmark.py` runs all four RAG experiment variants in a single benchmark: per-question Graph RAG, per-question baseline RAG, persistent Graph RAG without forgetting, and persistent Graph RAG with forgetting. It shares extraction results across the graph variants, uses checkpointing for safe resume, and produces a combined performance and storage comparison.

**Run:**

```bash
python combined_benchmark.py [--limit N] [--data PATH] [--out-dir DIR] \
    [--per-type N] [--seed N] [--forget-every-n N] \
    [--forget-threshold F] [--forget-node-count N] \
    [--forget-recency-half-life-days F] [--forget-turns-half-life F] \
    [--skip-phase2]
```

**Arguments:**

* `--data PATH` — Path to the dataset JSON (default: `data/longmemeval_oracle.json`).
* `--out-dir DIR` — Directory for all output files (default: `combined/`).
* `--limit N` — Maximum total number of questions.
* `--per-type N` — Sample `N` questions per `question_type`.
* `--seed N` — Random seed for subsampling (default: `42`).
* `--forget-every-n N` — Trigger forgetting every `N` turns.
* `--forget-threshold F` — Importance-score threshold used for pruning.
* `--forget-node-count N` — Trigger forgetting when the graph reaches `N` nodes.
* `--forget-recency-half-life-days F` — Recency half-life, in days, used by the forgetting module.
* `--forget-turns-half-life F` — Turn-based half-life used by the forgetting module.
* `--skip-phase2` — Run only Phase 1 (ingestion and per-question Graph/Baseline RAG evaluation). Phase 2 can be run by rerunning the command without this flag.

The benchmark saves checkpoints, per-variant results, persistent graph states, and a final `summary.json` under the specified output directory. Rerunning the same command resumes from the checkpoint and skips completed work.


---

### Knowledge Graph Benchmark

`benchmark.py` runs the LongMemEval benchmark using a knowledge graph: it ingests all user and assistant turns into a fresh in-memory graph for each question, retrieves a relevant subgraph as context, and evaluates the generated answer against the ground truth. It optionally applies the forgetting module during ingestion and records both retrieval/answer metrics and graph-forgetting statistics.

**Run:**

```bash
python -m benchmark [--limit N] [--per-type N] [--seed SEED] [--data PATH] \
    [--verbose] [--forget] [--forget-every-n N] [--forget-threshold T] \
    [--forget-node-count N] [--type QUESTION_TYPE] [--resume] [--output PATH]
```

**Arguments:**

* `--limit N` — Maximum number of questions to evaluate, applied after other filtering/subsampling.
* `--per-type N` — Sample `N` questions per `question_type`.
* `--seed SEED` — Random seed for subsampling (default: `42`).
* `--data PATH` — Path to the dataset JSON (default: `data/longmemeval_oracle.json`).
* `--verbose` — Print detailed results for each question.
* `--forget` — Enable the forgetting module during ingestion.
* `--forget-every-n N` — Trigger forgetting every `N` ingested turns (default: `50`).
* `--forget-threshold T` — Importance-score threshold below which graph nodes are pruned (default: `0.15`).
* `--forget-node-count N` — Trigger forgetting when the graph reaches `N` nodes (default: `300`).
* `--type QUESTION_TYPE` — Run only questions of a specific type, such as `knowledge-update`, `multi-session`, or `temporal-reasoning`.
* `--resume` — Resume from an existing results file, retaining successful results and continuing from the first errored entry.
* `--output PATH` — Output file path (default: `results.json`).

Results are saved to the specified output file, and the full knowledge graph for each question is saved under `graphs/<question_id>.json`. The benchmark reports token-level F1, precision, recall, and LLM-judged correctness, with results also broken down by question type.


---

### Baseline RAG Benchmark

`baseline_rag.py` runs a baseline Retrieval-Augmented Generation (RAG) benchmark using a flat vector store, where each conversation turn is embedded and the top-k most similar chunks are retrieved as context for answering each question. This provides a graph-free baseline for comparison with the knowledge-graph-based benchmark.

**Run:**

```bash
python -m baseline_rag [--limit N] [--per-type N] [--seed SEED] [--data PATH] [--top-k K] [--verbose]
```

**Arguments:**

* `--limit N` — Maximum number of questions to evaluate.
* `--per-type N` — Sample `N` questions per `question_type`.
* `--seed SEED` — Random seed for subsampling (default: `42`).
* `--data PATH` — Path to the dataset JSON file (defaults to the project's configured `DATA_PATH`).
* `--top-k K` — Number of chunks to retrieve for each question (default: `5`).
* `--verbose` — Print detailed results for each question.

Results are saved to `baseline_results.json`.


---

### Lifetime Graph Experiment

`experiment_lifetime.py` evaluates the forgetting module in a long-lived memory setting by ingesting all sessions from a subset of questions into one shared knowledge graph in chronological order. It runs the same questions against both an unbounded graph and a periodically pruned graph, then compares F1 and LLM-judged correctness to measure the effect of forgetting on retrieval quality.

**Run:**

```bash
python experiment_lifetime.py [--data PATH] [--per-type N] [--limit N] \
    [--seed SEED] [--regen-cache] [--forget-every-n N] \
    [--forget-threshold T] [--forget-node-count N]
```

**Arguments:**

* `--data PATH` — Path to the dataset JSON (default: `data/longmemeval_oracle.json`).
* `--per-type N` — Sample `N` questions per `question_type`.
* `--limit N` — Maximum total number of questions.
* `--seed SEED` — Random seed for subsampling (default: `42`).
* `--regen-cache` — Re-extract all user turns instead of using the existing `lifetime_cache.json`.
* `--forget-every-n N` — Trigger forgetting every `N` turns.
* `--forget-threshold T` — Importance-score threshold used for pruning.
* `--forget-node-count N` — Trigger forgetting when the graph reaches this many nodes.

The experiment writes the extraction cache to `lifetime_cache.json`, saves the two graphs as `graphs/lifetime_no_forget.json` and `graphs/lifetime_forget.json`, and writes the final comparison to `lifetime_results.json`.


---

### Interactive extraction

Accepts a single user message, runs extraction, and prints the resulting graph triples as JSON.

```
python main.py
```

You will be prompted to enter a message. Useful for testing the extraction pipeline in isolation.