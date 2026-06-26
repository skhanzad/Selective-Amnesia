import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_graph() -> dict:
    now = _now_iso()
    return {
        "nodes": {},
        "edges": [],
        "metadata": {
            "created_at": now,
            "last_updated_at": now,
            "total_interactions": 0,
        },
    }


def load_graph(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return _empty_graph()
    with p.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return _empty_graph()


def save_graph(graph: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)


def add_node(
    graph: dict,
    label: str,
    type: str,
    attributes: dict | None = None,
    embedding: list | None = None,
) -> str:
    node_id = str(uuid.uuid4())
    now = _now_iso()
    graph["nodes"][node_id] = {
        "id": node_id,
        "label": label,
        "type": type,
        "attributes": attributes or {},
        "embedding": embedding or [],
        "importance_score": 1.0,
        "stats": {
            "access_count": 1,
            "created_at": now,
            "last_accessed_at": now,
        },
    }
    graph["metadata"]["last_updated_at"] = now
    return node_id


def add_edge(
    graph: dict,
    source_id: str,
    target_id: str,
    relation: str,
) -> None:
    if source_id == target_id:
        return
    for edge in graph["edges"]:
        if (
            edge["source"] == source_id
            and edge["target"] == target_id
            and edge["relation"] == relation
        ):
            return
    now = _now_iso()
    graph["edges"].append(
        {
            "id": str(uuid.uuid4()),
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "created_at": now,
        }
    )
    graph["metadata"]["last_updated_at"] = now


def find_node_by_label(graph: dict, label: str) -> str | None:
    target = label.lower()
    for node_id, node in graph["nodes"].items():
        if node["label"].lower() == target:
            return node_id
    return None


def merge_node(graph: dict, existing_id: str, new_attributes: dict) -> None:
    node = graph["nodes"][existing_id]
    node["attributes"].update(new_attributes)
    node["stats"]["access_count"] += 1
    node["stats"]["last_accessed_at"] = _now_iso()
    graph["metadata"]["last_updated_at"] = _now_iso()


def increment_interactions(graph: dict) -> None:
    graph["metadata"]["total_interactions"] += 1
    graph["metadata"]["last_updated_at"] = _now_iso()