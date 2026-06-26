from collections import deque
from datetime import datetime, timezone

RETRIEVAL_THRESHOLD = 0.75
TOP_K = 10
# Labels that are too generic to be useful retrieval roots on their own.
# The user node connects to everything and floods BFS, crowding out specific nodes.
_GENERIC_LABELS = {"Person"}
MAX_HOPS = 2
MAX_NODES = 15


# ── Adjacency index ────────────────────────────────────────────────────────────


def build_adjacency(graph: dict) -> dict[str, list[str]]:
    """Build an in-memory {node_id: [edge_id, ...]} index from all edges."""
    adj: dict[str, list[str]] = {}
    for edge_id, edge in graph["edges"].items():
        adj.setdefault(edge["source"], []).append(edge_id)
        adj.setdefault(edge["target"], []).append(edge_id)
    return adj


# ── Ranking ────────────────────────────────────────────────────────────────────


def rank_candidates(
    candidate_lists: list[list[tuple[str, float]]],
    graph: dict | None = None,
) -> list[str]:
    """
    Merge per-entity candidate lists: keep the highest score per node,
    filter below RETRIEVAL_THRESHOLD, sort descending, return top-k node IDs.
    Generic hub nodes (Person) are deprioritised — appended after specific nodes
    so they only fill remaining slots, preventing BFS from flooding the cap.
    """
    best: dict[str, float] = {}
    for candidates in candidate_lists:
        for node_id, score in candidates:
            if score >= RETRIEVAL_THRESHOLD and score > best.get(node_id, -1.0):
                best[node_id] = score

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)

    if graph:
        specific = [nid for nid, _ in ranked if graph["nodes"].get(nid, {}).get("label") not in _GENERIC_LABELS]
        generic = [nid for nid, _ in ranked if graph["nodes"].get(nid, {}).get("label") in _GENERIC_LABELS]
        return (specific + generic)[:TOP_K]

    return [node_id for node_id, _ in ranked[:TOP_K]]


# ── BFS subgraph traversal ─────────────────────────────────────────────────────


def collect_subgraph(
    graph: dict,
    adj: dict[str, list[str]],
    root_ids: list[str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    BFS from each root node up to MAX_HOPS hops, capped at MAX_NODES total.
    Returns (nodes_dict, edges_dict) with full records for the subgraph.
    """
    visited_nodes: set[str] = set()
    visited_edges: set[str] = set()
    result_nodes: dict[str, dict] = {}
    result_edges: dict[str, dict] = {}

    queue: deque[tuple[str, int]] = deque()

    for root_id in root_ids:
        if root_id in graph["nodes"] and root_id not in visited_nodes:
            visited_nodes.add(root_id)
            result_nodes[root_id] = graph["nodes"][root_id]
            queue.append((root_id, 0))

    while queue and len(visited_nodes) < MAX_NODES:
        node_id, depth = queue.popleft()

        if depth >= MAX_HOPS:
            continue

        for edge_id in adj.get(node_id, []):
            if edge_id in visited_edges:
                continue
            visited_edges.add(edge_id)

            edge = graph["edges"][edge_id]
            result_edges[edge_id] = edge

            for neighbour_id in (edge["source"], edge["target"]):
                if neighbour_id not in visited_nodes and len(visited_nodes) < MAX_NODES:
                    visited_nodes.add(neighbour_id)
                    if neighbour_id in graph["nodes"]:
                        result_nodes[neighbour_id] = graph["nodes"][neighbour_id]
                    queue.append((neighbour_id, depth + 1))

    return result_nodes, result_edges


# ── Access metadata ────────────────────────────────────────────────────────────


def update_access(graph: dict, node_ids: set[str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for node_id in node_ids:
        if node_id in graph["nodes"]:
            graph["nodes"][node_id]["access_count"] += 1
            graph["nodes"][node_id]["last_accessed_at"] = now


# ── Serialisation ──────────────────────────────────────────────────────────────


def serialize_subgraph(
    nodes: dict[str, dict],
    edges: dict[str, dict],
    full_graph: dict | None = None,
) -> str:
    """
    Compact text for LLM injection:
      Nodes:  [label] title (content) {k: v, ...}
      Edges:  source_title -[relationship {k: v}]-> target_title

    full_graph is used to resolve titles for nodes referenced by edges but
    not included in the subgraph (e.g. nodes beyond the hop/cap limit).
    """
    all_nodes: dict[str, dict] = {}
    if full_graph:
        all_nodes.update(full_graph.get("nodes", {}))
    all_nodes.update(nodes)

    lines: list[str] = []

    lines.append("=== Context Nodes ===")
    for node in nodes.values():
        attrs_str = ""
        if node.get("attributes"):
            attrs_str = " {" + ", ".join(f"{k}: {v}" for k, v in node["attributes"].items()) + "}"
        content_str = f" ({node['content']})" if node.get("content") else ""
        lines.append(f"[{node['label']}] {node['title']}{content_str}{attrs_str}")

    lines.append("=== Context Edges ===")
    for edge in edges.values():
        src_node = all_nodes.get(edge["source"])
        tgt_node = all_nodes.get(edge["target"])
        src_title = src_node["title"] if src_node else edge["source"]
        tgt_title = tgt_node["title"] if tgt_node else edge["target"]
        attrs_str = ""
        if edge.get("attributes"):
            attrs_str = " {" + ", ".join(f"{k}: {v}" for k, v in edge["attributes"].items()) + "}"
        lines.append(f"{src_title} -[{edge['relationship']}{attrs_str}]-> {tgt_title}")

    return "\n".join(lines)
