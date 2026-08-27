import uuid
from datetime import datetime

from graph.embedder import get_embeddings
from graph.extract import extract, extract_for_retrieval
from graph.forget import DEFAULT_CONFIG as DEFAULT_FORGET_CONFIG
from graph.forget import run_forgetting_pass, should_trigger
from graph.graph_manager import (
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


def _pipeline(
    user_message: str,
    graph: dict,
    *,
    ingest_only: bool = False,
    retrieve_only: bool = False,
    extraction: dict | None = None,
    forget_config: dict | None = None,
    session_date: "datetime | None" = None,
) -> tuple[str, dict]:
    """
    Core pipeline logic operating on an already-loaded graph dict.
    Does not persist to disk — callers decide whether to save.

      1. Extract nodes and edges (one LLM call, skipped if extraction provided
         or retrieve_only=True)
      2. Embed each extracted node
      3. Search vector_index: dedup (>=0.92) and retrieval candidates (>=0.75)
      4. Build and serialise subgraph from retrieval candidates (skipped if ingest_only)
      5. Write new/merged nodes and edges into the graph dict (skipped if retrieve_only)

    retrieve_only=True: run only the retrieval path — no extraction LLM call, no
    writes to the graph. Use this when querying a graph that must not be modified
    (e.g. querying persistent graphs in Experiment 2).

    Returns (context_str, graph).
    """
    if retrieve_only:
        retrieval_entities = extract_for_retrieval(user_message)
        retrieval_descriptors = [
            build_descriptor(e.get("label", ""), e.get("title", ""), e.get("content"))
            for e in retrieval_entities
        ]
        retrieval_vectors = get_embeddings(retrieval_descriptors) if retrieval_descriptors else []
        retrieval_candidates = [search_similar(graph, v) for v in retrieval_vectors]

        title_index = graph.get("title_index", {})
        title_hits = [
            title_index[e.get("title", "").lower().strip()]
            for e in retrieval_entities
            if e.get("title", "").lower().strip() in title_index
        ]

        root_ids = rank_candidates(retrieval_candidates, graph=graph, extra_ids=title_hits)
        adj = build_adjacency(graph)
        subgraph_nodes, subgraph_edges = collect_subgraph(graph, adj, root_ids)
        update_access(graph, set(subgraph_nodes.keys()), now=session_date)
        context_str = serialize_subgraph(subgraph_nodes, subgraph_edges, full_graph=graph) if subgraph_nodes else ""
        return context_str, graph

    if extraction is None:
        extraction = extract(user_message)

    for node_data in extraction.get("nodes", {}).values():
        if node_data.get("label") == "Person" and node_data.get("title", "").lower() == "user":
            node_data["content"] = "The user"

    session_id = create_session(graph, created_at=session_date)
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

    title_index = graph.get("title_index", {})
    dedup_hit_ids: set[str] = set()

    for (temp_id, node_data), vector in zip(nodes_list, all_vectors):
        vectors[temp_id] = vector

        # Title-based dedup: same label + title = same node, regardless of embedding.
        title_key = node_data.get("title", "").lower().strip()
        label = node_data.get("label", "")
        title_hit = title_index.get(title_key)
        if title_hit and graph["nodes"].get(title_hit, {}).get("label") == label:
            temp_to_real[temp_id] = title_hit
            dedup_hit_ids.add(title_hit)
            candidate_lists.append([])
            continue

        ranked = search_similar(graph, vector)
        candidate_lists.append(ranked)

        if ranked and ranked[0][1] >= DEDUP_THRESHOLD:
            temp_to_real[temp_id] = ranked[0][0]
            dedup_hit_ids.add(ranked[0][0])
        else:
            temp_to_real[temp_id] = str(uuid.uuid4())

    if dedup_hit_ids:
        update_access(graph, dedup_hit_ids, now=session_date)

    if ingest_only:
        context_str = ""
    else:
        retrieval_entities = extract_for_retrieval(user_message)
        retrieval_descriptors = [
            build_descriptor(e.get("label", ""), e.get("title", ""), e.get("content"))
            for e in retrieval_entities
        ]
        retrieval_vectors = get_embeddings(retrieval_descriptors) if retrieval_descriptors else []
        retrieval_candidates = [search_similar(graph, v) for v in retrieval_vectors]

        title_index = graph.get("title_index", {})
        title_hits = [
            title_index[e.get("title", "").lower().strip()]
            for e in retrieval_entities
            if e.get("title", "").lower().strip() in title_index
        ]

        root_ids = rank_candidates(retrieval_candidates, graph=graph, extra_ids=title_hits)
        adj = build_adjacency(graph)
        subgraph_nodes, subgraph_edges = collect_subgraph(graph, adj, root_ids)
        update_access(graph, set(subgraph_nodes.keys()), now=session_date)
        context_str = serialize_subgraph(subgraph_nodes, subgraph_edges, full_graph=graph) if subgraph_nodes else ""

    write_nodes(graph, extraction, temp_to_real, vectors, session_id, session_created_at)
    write_edges(graph, extraction, temp_to_real, session_id, session_created_at)
    increment_prompt_count(graph, session_id)

    if forget_config is not None:
        cfg = {**DEFAULT_FORGET_CONFIG, **forget_config}
        if should_trigger(graph, cfg):
            stats = run_forgetting_pass(graph, cfg, now=session_date)
            print(
                f"  [forget] scored={stats['scored']} pruned={stats['pruned']} "
                f"edges_removed={stats['edges_removed']}",
                flush=True,
            )

    return context_str, graph


def run(user_message: str) -> tuple[str, dict]:
    """Load graph from disk, run pipeline, persist, return (context_str, graph)."""
    graph = _load_graph()
    context_str, graph = _pipeline(user_message, graph)
    _save_graph(graph)
    return context_str, graph


def run_in_memory(
    user_message: str,
    graph: dict,
    *,
    ingest_only: bool = False,
    retrieve_only: bool = False,
    extraction: dict | None = None,
    forget_config: dict | None = None,
    session_date: datetime | None = None,
) -> tuple[str, dict]:
    """Run pipeline on a provided in-memory graph dict without touching disk."""
    return _pipeline(
        user_message, graph,
        ingest_only=ingest_only,
        retrieve_only=retrieve_only,
        extraction=extraction,
        forget_config=forget_config,
        session_date=session_date,
    )


if __name__ == "__main__":
    message = input("Enter message: ")
    context, updated_graph = run(message)
    print(f"\nGraph: {len(updated_graph['nodes'])} nodes, {len(updated_graph['edges'])} edges")
    if context:
        print(f"\n{context}")
