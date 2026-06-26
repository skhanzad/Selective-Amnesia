from collections import deque
from datetime import datetime

from graph.embedder import cosine_similarity

SUBGRAPH_MAX_HOPS = 2
SUBGRAPH_MAX_NODES = 15


def find_entry_nodes(
    graph: dict, query_embedding: list[float], k: int = 3
) -> list[str]:
    scores: list[tuple[float, str]] = []
    for node_id, node in graph["nodes"].items():
        emb = node.get("embedding")
        if not emb:
            continue
        scores.append((cosine_similarity(query_embedding, emb), node_id))
    scores.sort(reverse=True)
    return [node_id for _, node_id in scores[:k]]


def find_entry_node(graph: dict, query_embedding: list[float]) -> str | None:
    results = find_entry_nodes(graph, query_embedding, k=1)
    return results[0] if results else None


def extract_subgraph(
    graph: dict,
    entry_node_ids: str | list[str],
    hops: int = SUBGRAPH_MAX_HOPS,
    max_nodes: int = SUBGRAPH_MAX_NODES,
) -> dict:
    nodes = graph["nodes"]
    edges = graph["edges"]

    if isinstance(entry_node_ids, str):
        seeds = [entry_node_ids]
    else:
        seeds = [nid for nid in entry_node_ids if nid in nodes]

    if not seeds:
        return {"nodes": {}, "edges": []}

    adjacency: dict[str, set[str]] = {nid: set() for nid in nodes}
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in adjacency:
            adjacency[src].add(tgt)
        if tgt in adjacency:
            adjacency[tgt].add(src)

    visited: dict[str, int] = {nid: 0 for nid in seeds}
    queue: deque[str] = deque(seeds)
    while queue:
        current = queue.popleft()
        current_hop = visited[current]
        if current_hop >= hops:
            continue
        for neighbour in adjacency.get(current, set()):
            if neighbour not in visited:
                visited[neighbour] = current_hop + 1
                queue.append(neighbour)

    collected_ids = list(visited.keys())


    if len(collected_ids) > max_nodes:
        collected_ids.sort(
            key=lambda nid: (
                0 if nodes[nid].get("type") == "episodic" else 1,
                -nodes[nid].get("importance_score", 0.0),
            ),
        )
        collected_ids = collected_ids[:max_nodes]

    kept_set = set(collected_ids)

    kept_edges = [
        e for e in edges if e["source"] in kept_set and e["target"] in kept_set
    ]

    return {
        "nodes": {nid: nodes[nid] for nid in collected_ids},
        "edges": kept_edges,
    }


def _format_attr_value(value: object) -> str:
    s = str(value)
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s[:len(fmt)], fmt)
            return dt.strftime("%d %B %Y").lstrip("0")
        except ValueError:
            continue
    return s


def format_subgraph_for_prompt(subgraph: dict) -> str:
    lines: list[str] = ["[Knowledge Graph Context]"]

    for node in subgraph["nodes"].values():
        label = node["label"]
        ntype = node["type"]
        attrs = node.get("attributes", {})
        attr_str = (
            ", ".join(f"{k}={_format_attr_value(v)}" for k, v in attrs.items())
            if attrs else ""
        )
        line = f"  Node: {label} ({ntype})"
        if attr_str:
            line += f" [{attr_str}]"
        lines.append(line)

    for edge in subgraph["edges"]:
        src_id = edge["source"]
        tgt_id = edge["target"]
        src_label = subgraph["nodes"].get(src_id, {}).get("label", src_id)
        tgt_label = subgraph["nodes"].get(tgt_id, {}).get("label", tgt_id)
        lines.append(f"  Edge: {src_label} --[{edge['relation']}]--> {tgt_label}")

    if len(lines) == 1:
        lines.append("  (empty)")

    return "\n".join(lines)