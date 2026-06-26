# GraphRAG

A knowledge graph system for augmenting LLM responses with persistent user memory, evaluated against the LongMemEval benchmark.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the required models pulled:
  ```
  ollama pull gemma2
  ollama pull nomic-embed-text
  ```
- Install dependencies:

```
pip install -r requirements.txt
```

## Running

### `main.py` — Interactive extraction

Accepts a single user message, runs extraction, and prints the resulting graph triples as JSON.

```
python main.py
```

You will be prompted to enter a message. Useful for testing the extraction pipeline in isolation.

---

### `benchmark.py` — Graph RAG benchmark

Ingests user turns from the LongMemEval dataset into a fresh knowledge graph per question, then answers each question using subgraph retrieval and scores the result.

```
python -m benchmark [--limit N] [--per-type N] [--seed N] [--data PATH] [--verbose]
```

| Flag | Default | Description |
|---|---|---|
| `--limit N` | all | Max total questions (applied after subsampling) |
| `--per-type N` | all | Sample this many questions per question type |
| `--seed N` | 42 | Random seed for subsampling |
| `--data PATH` | `data/longmemeval_oracle.json` | Path to the dataset JSON |
| `--verbose` | off | Print each question result as it runs |

---

### `baseline_rag.py` — Flat vector baseline

Same benchmark, but uses a flat FAISS vector store over raw text chunks instead of a knowledge graph. Run this to isolate the contribution of the graph structure — any performance difference vs. `benchmark.py` is attributable to the graph.

```
python -m baseline_rag [--limit N] [--per-type N] [--seed N] [--data PATH] [--top-k K] [--verbose]
```

| Flag | Default | Description |
|---|---|---|
| `--limit N` | all | Max total questions |
| `--per-type N` | all | Sample this many questions per question type |
| `--seed N` | 42 | Random seed for subsampling |
| `--data PATH` | `data/longmemeval_oracle.json` | Path to the dataset JSON |
| `--top-k K` | 5 | Chunks to retrieve per question |
| `--verbose` | off | Print each question result as it runs |
