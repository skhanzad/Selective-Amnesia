import math
from collections import Counter
from datetime import datetime, timezone

PRUNE_THRESHOLD = 0.15
FORGET_EVERY_N = 10

DECAY_RATES = {"episodic": 0.08, "semantic": 0.01}
IMPORTANCE_WEIGHTS = {"recency": 0.4, "frequency": 0.3, "centrality": 0.2, "semantic_boost": 0.1}


def _days_since(iso_timestamp: str) -> float:
    last = datetime.fromisoformat(iso_timestamp)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - last
    return max(delta.total_seconds() / 86400.0, 0.0)


def compute_importance(node: dict, all_nodes: dict, degree_map: dict) -> float:
    # Recency
    decay_rate = DECAY_RATES.get(node["type"], DECAY_RATES["semantic"])
    days = _days_since(node["stats"]["last_accessed_at"])
    recency = math.exp(-decay_rate * days)

    # Frequency
    access_count = node["stats"]["access_count"]
    max_access = max(n["stats"]["access_count"] for n in all_nodes.values()) if all_nodes else 1
    frequency = math.log(1 + access_count) / math.log(1 + max_access) if max_access > 0 else 0.0

    # Centrality
    degree = degree_map.get(node["id"], 0)
    max_degree = max(degree_map.values()) if degree_map else 0
    centrality = degree / max_degree if max_degree > 0 else 0.0

    # Semantic boost
    semantic_boost = 1.0 if node["type"] == "semantic" else 0.7

    return round(
        IMPORTANCE_WEIGHTS["recency"] * recency
        + IMPORTANCE_WEIGHTS["frequency"] * frequency
        + IMPORTANCE_WEIGHTS["centrality"] * centrality
        + IMPORTANCE_WEIGHTS["semantic_boost"] * semantic_boost,
        6,
    )


def run_forgetting(graph: dict) -> dict:
    all_nodes = graph["nodes"]
    all_edges = graph["edges"]

    degree_map = Counter()
    for e in all_edges:
        degree_map[e["source"]] += 1
        degree_map[e["target"]] += 1

    for node in all_nodes.values():
        node["importance_score"] = compute_importance(node, all_nodes, degree_map)

    pruned_ids = {
        node_id
        for node_id, node in all_nodes.items()
        if node["importance_score"] < PRUNE_THRESHOLD
    }

    for node_id in pruned_ids:
        del graph["nodes"][node_id]

    graph["edges"] = [
        e for e in all_edges
        if e["source"] not in pruned_ids and e["target"] not in pruned_ids
    ]

    return graph


def should_forget(graph: dict) -> bool:
    total = graph["metadata"].get("total_interactions", 0)
    return total > 0 and total % FORGET_EVERY_N == 0