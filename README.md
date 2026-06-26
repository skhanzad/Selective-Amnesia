# KG-RAG: Knowledge Graph Augmented Retrieval for Conversational Memory

## Introduction

This project extends RAG with a **knowledge-graph-based memory system** that maintains a persistent, structured representation of everything discussed across a conversation. When a user sends a message, the system retrieves a semantically relevant subgraph and injects it as structured context before generating a response. After the response is generated, the user's message is processed by a second LLM call that extracts new nodes and edges and merges them into the graph.

A **forgetting module** runs periodically to prune low-importance nodes, preventing the graph from growing unboundedly and keeping retrieval focused on what is most relevant.

The system is built in Python using **LangGraph** for the main flow, **Gemma2 via Ollama** as the LLM, and **nomic-embed-text via Ollama** for embeddings. Evaluation is performed on the **LoCoMo** benchmark.


## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- The LoCoMo dataset file `data/locomo10.json` (not included in the repository — place it manually) [https://github.com/snap-research/locomo/blob/main/data/locomo10.json](https://github.com/snap-research/locomo/blob/main/data/locomo10.json)

---

## 1. Install Ollama

Download and install Ollama from [https://ollama.com/download](https://ollama.com/download) for your OS.

After installation, verify it is running:

```bash
ollama --version
```

Ollama must be running in the background before using this project. On most systems it starts automatically after installation. If not, run:

```bash
ollama serve
```

---

## 2. Pull the Required Models

```bash
ollama pull gemma2
ollama pull nomic-embed-text
```

Both models are required. `gemma2` is the LLM used for generation and extraction; `nomic-embed-text` produces node embeddings.

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Create Required Directories

The `data/` and `data/benchmarks/` directories must exist before running anything:

```bash
mkdir -p data/benchmarks
```

Place the LoCoMo dataset file at `data/locomo10.json` before running ingestion or evaluation.

---

## 5. Running the Live Chat (`main.py`)

### Interactive mode

```bash
python main.py
```

Type messages at the `You:` prompt. Type `quit` or press `Ctrl-C` to exit. The knowledge graph is saved to `data/graph.json` and a turn-by-turn log is written to `data/chat_log.json`.

### Script mode (batch prompts from a file)

Create a plain text file with one prompt per line (lines starting with `#` are skipped):

```bash
python main.py --script test_prompts.txt
```

### Reset the graph before starting

```bash
python main.py --reset
```

This deletes `data/graph.json` and `data/chat_log.json` before the session begins. Can be combined with `--script`.

---

## 6. Ingesting LoCoMo into Knowledge Graphs (`benchmark/ingest.py`)

Ingest builds a per-sample knowledge graph from the LoCoMo conversation histories. Each sample produces a graph file at `data/benchmarks/kg_<sample_id>.json`. You must ingest before evaluating.

### Ingest 1 sample

```bash
python benchmark/ingest.py --data data/locomo10.json --samples 1
```

### Ingest all 10 samples

```bash
python benchmark/ingest.py --data data/locomo10.json
```

## 7. Evaluating on LoCoMo (`benchmark/evaluate.py`)

Evaluation queries each sample's graph with the LoCoMo QA pairs and reports token F1 scores by category. **Ingest must be run first** for the graphs to exist.

### Evaluate 1 sample

```bash
python benchmark/evaluate.py --data data/locomo10.json --samples 1
```

### Evaluate all 10 samples

```bash
python benchmark/evaluate.py --data data/locomo10.json --samples 10
```

Results are saved to `data/benchmarks/results.json` by default.

## 8. Visualizing a Knowledge Graph (`data/visualize_graph.py`)
Pass the path to any graph JSON file as a required positional argument. Must be run from the **repository root**:

```bash
python data/visualize_graph.py data/graph.json
python data/visualize_graph.py data/benchmarks/kg_<sample_id>.json
```

A window will open displaying the graph. Close the window to exit.
