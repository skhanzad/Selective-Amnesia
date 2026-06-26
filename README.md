# Selective Amnesia: Knowledge Graph Augmented Retrieval

This repository contains two related graph-RAG systems for conversational memory, each evaluated on a different benchmark.

## LoCoMo pipeline (LangGraph)

The main chat pipeline extends RAG with a **knowledge-graph-based memory system** that maintains a persistent, structured representation of everything discussed across a conversation. When a user sends a message, the system retrieves a semantically relevant subgraph and injects it as structured context before generating a response. After the response is generated, the user's message is processed by a second LLM call that extracts new nodes and edges and merges them into the graph.

A **forgetting module** runs periodically to prune low-importance nodes, preventing the graph from growing unboundedly and keeping retrieval focused on what is most relevant.

Built with **LangGraph** for the main flow, **Gemma2 via Ollama** as the LLM, and **nomic-embed-text via Ollama** for embeddings. Evaluated on the **LoCoMo** benchmark.

### Prerequisites (LoCoMo)

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- The LoCoMo dataset file `data/locomo10.json` (not included — place it manually) [https://github.com/snap-research/locomo/blob/main/data/locomo10.json](https://github.com/snap-research/locomo/blob/main/data/locomo10.json)

### Setup

```bash
ollama pull gemma2
ollama pull nomic-embed-text
pip install -r requirements.txt
mkdir -p data/benchmarks
```

### Running the live chat (`main.py`)

**Interactive mode:**

```bash
python main.py
```

Type messages at the `You:` prompt. Type `quit` or press `Ctrl-C` to exit. The knowledge graph is saved to `data/graph.json` and a turn-by-turn log is written to `data/chat_log.json`.

**Script mode (batch prompts from a file):**

```bash
python main.py --script test_prompts.txt
```

**Reset the graph before starting:**

```bash
python main.py --reset
```

### LoCoMo benchmark

Ingest builds a per-sample knowledge graph from the LoCoMo conversation histories:

```bash
python benchmark/ingest.py --data data/locomo10.json --samples 1
python benchmark/ingest.py --data data/locomo10.json
```

Evaluate queries each sample's graph with the LoCoMo QA pairs:

```bash
python benchmark/evaluate.py --data data/locomo10.json --samples 1
python benchmark/evaluate.py --data data/locomo10.json --samples 10
```

Results are saved to `data/benchmarks/results.json` by default.

### Visualizing a knowledge graph

```bash
python data/visualize_graph.py data/graph.json
python data/visualize_graph.py data/benchmarks/kg_<sample_id>.json
```

---

## LongMemEval pipeline

A separate graph-RAG pipeline with session-based graph storage, evaluated against the **LongMemEval** benchmark. Uses `graph/session_graph.py` for its graph schema (distinct from the LoCoMo pipeline's `graph/graph_manager.py`).

### Running LongMemEval benchmarks

**Graph RAG benchmark** — ingests user turns into a fresh knowledge graph per question, then answers using subgraph retrieval:

```bash
python -m benchmark [--limit N] [--per-type N] [--seed N] [--data PATH] [--verbose]
```

| Flag | Default | Description |
|---|---|---|
| `--limit N` | all | Max total questions (applied after subsampling) |
| `--per-type N` | all | Sample this many questions per question type |
| `--seed N` | 42 | Random seed for subsampling |
| `--data PATH` | `data/longmemeval_oracle.json` | Path to the dataset JSON |
| `--verbose` | off | Print each question result as it runs |

**Flat vector baseline** — same benchmark using a flat FAISS-style vector store over raw text chunks (no graph):

```bash
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

### Testing extraction in isolation

```bash
python graph/ingest.py
```

Accepts a single user message, runs extraction and ingestion, and prints the resulting subgraph context.
