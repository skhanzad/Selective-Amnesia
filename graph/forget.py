"""
Forgetting and compression module (deliverable 16).

Recomputes importance scores across all graph nodes using recency, frequency,
centrality, and turns-since-last-access, then prunes or compresses low-importance
nodes to prevent unbounded graph growth.

Triggered by run_forgetting_pass(graph, config). Call should_trigger() first to
check whether conditions are met.
"""

import logging
import math
from datetime import datetime, timezone

from graph.graph_manager import _total_turns

logger = logging.getLogger(__name__)

# ── Default configuration ──────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    # Formula weights (must sum to 1.0)
    "w_recency": 0.35,
    "w_frequency": 0.25,
    "w_centrality": 0.20,
    "w_turns": 0.20,
    # Decay half-lives — calibrated for the LongMemEval dataset which spans ~33 months.
    # A 7-day recency half-life would make anything older than 3 weeks effectively dead.
    # A 20-turn half-life would make anything older than ~100 turns effectively dead across
    # a 9,000-turn run. Both values are far too aggressive for long-term memory benchmarks.
    "recency_half_life_days": 90.0,  # recency halves every 3 months
    "turns_half_life": 1000.0,  # turns_decay halves every ~28 questions worth of turns
    #                                  (calibrated for ~18k total turns including assistant)
    # Type weights
    "episodic_type_weight": 0.5,  # episodic nodes decay twice as fast
    "semantic_type_weight": 1.0,
    # Pruning
    "pruning_threshold": 0.10,  # nodes below this score are pruned/flagged
    # Trigger conditions (both checked; either can fire)
    "trigger_every_n_turns": 400,  # fire roughly every 11 questions (at ~36 turns/question
    #                                  including both user and assistant turns)
    "trigger_node_count": 50000,  # safety cap for very large graphs; rely on trigger_every_n_turns instead
}


# ── Helpers ────────────────────────────────────────────────────────────────────


def _build_degree_map(graph: dict) -> dict[str, int]:
    degree: dict[str, int] = {}
    for edge in graph["edges"].values():
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1
    return degree


def _recency_decay(
    last_accessed_at: str | None,
    created_at: str,
    now: datetime,
    half_life_days: float,
) -> float:
    ref_str = last_accessed_at or created_at
    if not ref_str:
        return 0.0
    try:
        ref = datetime.fromisoformat(ref_str)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    days = max(0.0, (now - ref).total_seconds() / 86400.0)
    return math.exp(-math.log(2) * days / half_life_days)


def _turns_decay(
    turns_at_last_access: int | None,
    turns_at_creation: int | None,
    total_turns: int,
    half_life_turns: float,
) -> float:
    if turns_at_last_access is not None:
        turns_since = max(0, total_turns - turns_at_last_access)
    elif turns_at_creation is not None:
        turns_since = max(0, total_turns - turns_at_creation)
    else:
        turns_since = total_turns
    return math.exp(-math.log(2) * turns_since / half_life_turns)


def _centrality_score(degree: int) -> float:
    # Logarithmic scaling so a node with 1 edge scores ~0.3, 10 edges ~1.0
    return min(math.log(1 + degree) / math.log(11), 1.0)


# ── Core scoring ───────────────────────────────────────────────────────────────


def compute_importance(
    node: dict,
    degree: int,
    now: datetime,
    total_turns: int,
    max_access_count: int,
    config: dict,
) -> float:
    """
    Compute the importance score for a single node.

    Score components:
      recency    — exponential decay from last_accessed_at (or created_at)
      frequency  — log-normalised access_count
      centrality — log-scaled edge degree
      turns      — exponential decay from turns_at_last_access

    Episodic nodes have their score halved by the type weight.
    """
    recency = _recency_decay(
        node.get("last_accessed_at"),
        node.get("created_at", ""),
        now,
        config["recency_half_life_days"],
    )

    denom = math.log(1 + max_access_count + 1)
    frequency = math.log(1 + node.get("access_count", 0)) / denom if denom else 0.0

    centrality = _centrality_score(degree)

    turns = _turns_decay(
        node.get("turns_at_last_access"),
        node.get("turns_at_creation"),
        total_turns,
        config["turns_half_life"],
    )

    is_episodic = node.get("type", "").lower() == "episodic"
    type_weight = (
        config["episodic_type_weight"]
        if is_episodic
        else config["semantic_type_weight"]
    )

    raw = (
        config["w_recency"] * recency
        + config["w_frequency"] * frequency
        + config["w_centrality"] * centrality
        + config["w_turns"] * turns
    )
    return round(type_weight * raw, 6)


# ── Trigger check ──────────────────────────────────────────────────────────────


def should_trigger(graph: dict, config: dict) -> bool:
    """Return True if at least one trigger condition is met."""
    total = _total_turns(graph)
    every_n = config.get("trigger_every_n_turns", 0)
    node_threshold = config.get("trigger_node_count", 0)

    if every_n and total > 0 and total % every_n == 0:
        return True
    return bool(node_threshold and len(graph["nodes"]) >= node_threshold)


# ── Compression helper ─────────────────────────────────────────────────────────


def _find_semantic_neighbour(graph: dict, node_id: str) -> str | None:
    """Return the ID of the first semantic neighbour of node_id via any edge."""
    for edge in graph["edges"].values():
        if edge["source"] == node_id:
            candidate = edge["target"]
        elif edge["target"] == node_id:
            candidate = edge["source"]
        else:
            continue
        if (
            candidate in graph["nodes"]
            and graph["nodes"][candidate].get("type", "").lower() != "episodic"
        ):
            return candidate
    return None


def _compress_episodic(graph: dict, node_id: str, node: dict) -> bool:
    """
    Try to compress an episodic node by writing a summary attribute onto its
    nearest semantic neighbour. Returns True if compression succeeded.
    """
    semantic_id = _find_semantic_neighbour(graph, node_id)
    if not semantic_id:
        return False

    title = node.get("title", node_id)
    attrs = node.get("attributes", {})
    summary_parts = [title]
    for k, v in list(attrs.items())[:5]:
        summary_parts.append(f"{k}={v}")
    summary = "; ".join(summary_parts)

    key = f"compressed_event_{node_id[:8]}"
    graph["nodes"][semantic_id]["attributes"][key] = summary
    return True


# ── Main forgetting pass ───────────────────────────────────────────────────────


def run_forgetting_pass(
    graph: dict, config: dict | None = None, now: datetime | None = None
) -> dict:
    """
    Recompute importance scores for every node, then prune all nodes below the
    pruning threshold along with their edges.

    now: the reference datetime for recency decay. Pass the current session's
    haystack date so recency is computed relative to conversation time, not
    wall-clock time. Defaults to UTC now if omitted.

    Returns a stats dict: {scored, pruned, edges_removed}.
    """
    if config is None:
        config = DEFAULT_CONFIG

    if now is None:
        now = datetime.now(timezone.utc)
    total_turns = _total_turns(graph)
    degree_map = _build_degree_map(graph)
    threshold = config.get("pruning_threshold", 0.10)

    max_access = max(
        (n.get("access_count", 0) for n in graph["nodes"].values()), default=0
    )

    # Pass 1: recompute all importance scores
    for node_id, node in graph["nodes"].items():
        node["importance_score"] = compute_importance(
            node,
            degree_map.get(node_id, 0),
            now,
            total_turns,
            max_access,
            config,
        )

    stats = {"scored": len(graph["nodes"]), "pruned": 0, "edges_removed": 0}

    to_prune = [
        node_id
        for node_id, node in graph["nodes"].items()
        if node["importance_score"] < threshold
    ]

    for node_id in to_prune:
        score = graph["nodes"][node_id]["importance_score"]

        edges_to_remove = [
            eid
            for eid, e in graph["edges"].items()
            if e["source"] == node_id or e["target"] == node_id
        ]
        for eid in edges_to_remove:
            del graph["edges"][eid]
        stats["edges_removed"] += len(edges_to_remove)

        graph["vector_index"].pop(node_id, None)
        del graph["nodes"][node_id]
        stats["pruned"] += 1

        logger.info("Pruned node %s (importance=%.4f)", node_id, score)

    return stats
