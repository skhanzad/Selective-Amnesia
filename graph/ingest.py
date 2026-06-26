import uuid

from graph.embedder import get_embeddings
from graph.extract import extract
from graph.session_graph import (
    DEDUP_THRESHOLD,
    _load_graph,
    _save_graph,
    build_descriptor,
    create_session,
    increment_prompt_count,
    search_similar,
    write_edges,
    write_nodes,
)
from graph.retrieval import (
    build_adjacency,
    collect_subgraph,
    rank_candidates,
    serialize_subgraph,
    update_access,
)


def _pipeline(user_message: str, graph: dict) -> tuple[str, dict]:
    """
    Core pipeline logic operating on an already-loaded graph dict.
    Does not persist to disk — callers decide whether to save.

      1. Extract nodes and edges (one LLM call)
      2. Embed each extracted node
      3. Search vector_index: dedup (>=0.92) and retrieval candidates (>=0.75)
      4. Build and serialise subgraph from retrieval candidates
      5. Write new/merged nodes and edges into the graph dict

    Returns (context_str, graph).
    """
    extraction = extract(user_message)

    session_id = create_session(graph)
    session_created_at = graph["sessions"][session_id]["created_at"]

    temp_to_real: dict[str, str] = {}
    vectors: dict[str, list[float]] = {}
    candidate_lists: list[list[tuple[str, float]]] = []

    nodes_list = list(extraction.get("nodes", {}).items())
    descriptors = [
        build_descriptor(
            label=node_data.get("label", ""),
            title=node_data.get("title", ""),
            content=node_data.get("content"),
        )
        for _, node_data in nodes_list
    ]

    all_vectors = get_embeddings(descriptors) if descriptors else []

    for (temp_id, node_data), vector in zip(nodes_list, all_vectors):
        vectors[temp_id] = vector

        ranked = search_similar(graph, vector)
        candidate_lists.append(ranked)

        if ranked and ranked[0][1] >= DEDUP_THRESHOLD:
            temp_to_real[temp_id] = ranked[0][0]
        else:
            temp_to_real[temp_id] = str(uuid.uuid4())

    root_ids = rank_candidates(candidate_lists, graph=graph)
    adj = build_adjacency(graph)
    subgraph_nodes, subgraph_edges = collect_subgraph(graph, adj, root_ids)
    update_access(graph, set(subgraph_nodes.keys()))
    context_str = serialize_subgraph(subgraph_nodes, subgraph_edges, full_graph=graph) if subgraph_nodes else ""

    write_nodes(graph, extraction, temp_to_real, vectors, session_id, session_created_at)
    write_edges(graph, extraction, temp_to_real, session_id, session_created_at)
    increment_prompt_count(graph, session_id)

    return context_str, graph


def run(user_message: str) -> tuple[str, dict]:
    """Load graph from disk, run pipeline, persist, return (context_str, graph)."""
    graph = _load_graph()
    context_str, graph = _pipeline(user_message, graph)
    _save_graph(graph)
    return context_str, graph


def run_in_memory(user_message: str, graph: dict) -> tuple[str, dict]:
    """Run pipeline on a provided in-memory graph dict without touching disk."""
    return _pipeline(user_message, graph)


if __name__ == "__main__":
    message = input("Enter message: ")
    context, updated_graph = run(message)
    print(f"\nGraph: {len(updated_graph['nodes'])} nodes, {len(updated_graph['edges'])} edges")
    if context:
        print(f"\n{context}")
