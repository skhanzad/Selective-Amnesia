from pathlib import Path

import pytest
import requests

from graph.extractor import extract, merge_extraction_into_graph
from graph.graph_manager import load_graph, save_graph, increment_interactions


def _ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return any("gemma2" in m for m in models) and any("nomic-embed-text" in m for m in models)
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama with gemma2 and nomic-embed-text is not available",
)


PROMPTS = [
    # Turn 1 — simple entity introduction
    "My friend Daniel just started working as a software engineer at Shopify in Toronto.",

    # Turn 2 — background / education, same entity
    "Daniel graduated from the University of Waterloo with a degree in computer engineering.",

    # Turn 3 — episodic event linking user and Daniel
    "Last weekend I went to a Raptors game at Scotiabank Arena with Daniel.",

    # Turn 4 — user's own learning context, no Daniel
    "I've been learning about reinforcement learning and neural networks for my AI course.",

    # Turn 5 — Daniel + technology overlap with turn 4 topic
    "Daniel mentioned that Shopify is using machine learning models to improve product recommendations.",

    # Turn 6 — user's personal project, implicit connection to the system being built
    "Yesterday I spent three hours debugging a Python project related to graph-based memory systems.",

    # Turn 7 — new person introduced, shared domain with turn 6
    "I met a professor from the University of Toronto who researches knowledge graphs and LLMs.",

    # Turn 8 — Daniel re-appears with new technical detail
    "That reminds me, Daniel once told me he worked on a graph database project during his internship.",

    # Turn 9 — social event, Daniel + location + topic
    "Two days ago I had coffee with Daniel at a cafe near Queen Street and we talked about AI startups.",

    # Turn 10 — technical concept, connects to prior ML/embedding threads
    "I'm trying to better understand how embeddings work in vector databases like Pinecone.",
]

OUTPUT_PATH = Path("data/simulation_graph.json")

@requires_ollama
class TestSimulation:

    @pytest.fixture(scope="class")
    def final_graph(self):
        graph = load_graph(str(OUTPUT_PATH))  # start fresh each run
        # Reset to empty so repeated runs are deterministic
        graph["nodes"] = {}
        graph["edges"] = []
        graph["metadata"]["total_interactions"] = 0

        for i, prompt in enumerate(PROMPTS, start=1):
            print(f"\n--- Turn {i} ---")
            print(f"Prompt: {prompt}")

            extraction = extract(prompt)
            print(f"Extracted {len(extraction['nodes'])} node(s), "
                  f"{len(extraction['edges'])} edge(s)")

            merge_extraction_into_graph(graph, extraction, prompt)
            increment_interactions(graph)

            print(f"Graph now has {len(graph['nodes'])} node(s), "
                  f"{len(graph['edges'])} edge(s)")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_graph(graph, str(OUTPUT_PATH))
        print(f"\nGraph saved to {OUTPUT_PATH}")
        return graph