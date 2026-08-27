import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from graph.embedder import cosine_similarity

DEDUP_THRESHOLD = 0.92

GRAPH_PATH = Path(__file__).parent.parent / "graph.json"


# ── Graph I/O ──────────────────────────────────────────────────────────────────


def empty_graph() -> dict:
    return {
        "nodes": {},
        "edges": {},
        "vector_index": {},
        "attribute_index": {},
        "title_index": {},
        "sessions": {},
    }


def _load_graph() -> dict:
    if GRAPH_PATH.exists():
        return json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    return empty_graph()


def _save_graph(graph: dict) -> None:
    GRAPH_PATH.write_text(json.dumps(graph, indent=2), encoding="utf-8")


# ── Session ────────────────────────────────────────────────────────────────────


def create_session(graph: dict, created_at: datetime | None = None) -> str:
    session_id = str(uuid.uuid4())
    ts = (created_at or datetime.now(timezone.utc)).isoformat()
    graph["sessions"][session_id] = {
        "id": session_id,
        "created_at": ts,
        "prompt_count": 0,
    }
    return session_id


def increment_prompt_count(graph: dict, session_id: str) -> None:
    graph["sessions"][session_id]["prompt_count"] += 1


# ── Date normalisation ─────────────────────────────────────────────────────────

_MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_WEEKDAY_NAMES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def normalize_date(raw_date: str, session_created_at: str) -> dict:
    """
    Convert a raw date string to an absolute ISO 8601 date.

    Returns:
        {date: str | None, date_approximate: bool, date_raw: str | None}
    """
    anchor = datetime.fromisoformat(session_created_at)
    s = raw_date.strip().lower()

    # yesterday / today / tomorrow
    if s == "yesterday":
        return {
            "date": (anchor - timedelta(days=1)).date().isoformat(),
            "date_approximate": False,
            "date_raw": None,
        }
    if s == "today":
        return {
            "date": anchor.date().isoformat(),
            "date_approximate": False,
            "date_raw": None,
        }
    if s == "tomorrow":
        return {
            "date": (anchor + timedelta(days=1)).date().isoformat(),
            "date_approximate": False,
            "date_raw": None,
        }

    # last <weekday>  e.g. "last saturday"
    m = re.match(r"last\s+(\w+)", s)
    if m and m.group(1) in _WEEKDAY_NAMES:
        target_wd = _WEEKDAY_NAMES[m.group(1)]
        days_back = (anchor.weekday() - target_wd) % 7 or 7
        return {
            "date": (anchor - timedelta(days=days_back)).date().isoformat(),
            "date_approximate": False,
            "date_raw": None,
        }

    # <N> days/weeks/months/years ago
    m = re.match(r"(\d+)\s+(day|week|month|year)s?\s+ago", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "day":
            result = anchor - timedelta(days=n)
        elif unit == "week":
            result = anchor - timedelta(weeks=n)
        elif unit == "month":
            month = anchor.month - n
            year = anchor.year + (month - 1) // 12
            month = ((month - 1) % 12) + 1
            result = anchor.replace(year=year, month=month)
        else:  # year
            result = anchor.replace(year=anchor.year - n)
        return {
            "date": result.date().isoformat(),
            "date_approximate": False,
            "date_raw": None,
        }

    # approximate: "a few days ago", "recently", "a while ago"
    if re.search(r"\b(few|couple|recent|while|some time)\b", s):
        approx = (anchor - timedelta(days=7)).date().isoformat()
        return {"date": approx, "date_approximate": True, "date_raw": raw_date}

    # "March 15th", "March 15", "15th March" — month name present
    for month_name, month_num in _MONTH_NAMES.items():
        pattern = rf"\b{month_name}\b\s+(\d{{1,2}})(?:st|nd|rd|th)?"
        m = re.search(pattern, s)
        if not m:
            m = re.search(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+\b{month_name}\b", s)
        if m:
            day = int(m.group(1))
            # assume current year; if the date is in the future assume last year
            year = anchor.year
            try:
                candidate = datetime(year, month_num, day, tzinfo=timezone.utc)
                if candidate > anchor:
                    candidate = candidate.replace(year=year - 1)
                return {
                    "date": candidate.date().isoformat(),
                    "date_approximate": False,
                    "date_raw": None,
                }
            except ValueError:
                pass

    # unresolvable
    return {"date": None, "date_approximate": False, "date_raw": raw_date}


# ── Descriptor construction ────────────────────────────────────────────────────


def build_descriptor(label: str, title: str, content: str | None = None) -> str:
    if content:
        return f"{label}: {title}, {content}"
    return f"{label}: {title}"


# ── Attribute index helpers ────────────────────────────────────────────────────


# ── Core ingest ────────────────────────────────────────────────────────────────


def _flatten_attributes(attributes: dict) -> dict:
    return dict(attributes)


def search_similar(graph: dict, vector: list[float]) -> list[tuple[str, float]]:
    """
    Linear cosine scan of the vector_index. Returns all (node_id, score) pairs
    sorted by score descending. Callers apply their own threshold.
    """
    results = []
    for node_id, existing_vec in graph["vector_index"].items():
        score = cosine_similarity(vector, existing_vec)
        results.append((node_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _merge_node(
    graph: dict, node_id: str, new_content: str | None, new_attrs: dict
) -> None:
    """Merge new content and attributes into an existing node without overwriting."""
    node = graph["nodes"][node_id]
    if new_content and not node.get("content"):
        node["content"] = new_content
    node["attributes"].update(new_attrs)


def _process_raw_dates(raw_attrs: dict, session_created_at: str) -> dict:
    """Normalise any raw_date key in-place and return the modified dict."""
    if "raw_date" in raw_attrs:
        raw_str = raw_attrs.pop("raw_date")
        if not isinstance(raw_str, str):
            return raw_attrs
        if raw_str is not None:
            date_result = normalize_date(raw_str, session_created_at)
            raw_attrs.update({k: v for k, v in date_result.items() if v is not None})
    return raw_attrs


def _total_turns(graph: dict) -> int:
    return sum(s["prompt_count"] for s in graph["sessions"].values())


def summarize_graph(graph: dict) -> str:
    """Compact text summary of existing nodes and edges for context-aware extraction."""
    if not graph.get("nodes"):
        return ""
    lines = ["=== Existing Knowledge ===", "Nodes:"]
    for node in graph["nodes"].values():
        attrs = node.get("attributes", {})
        attr_str = (
            (" {" + ", ".join(f"{k}: {v}" for k, v in attrs.items()) + "}")
            if attrs
            else ""
        )
        lines.append(f"  [{node['label']}] {node['title']}{attr_str}")
    lines.append("Edges:")
    for edge in graph["edges"].values():
        src = graph["nodes"].get(edge["source"], {}).get("title", "?")
        tgt = graph["nodes"].get(edge["target"], {}).get("title", "?")
        attrs = edge.get("attributes", {})
        attr_str = (
            (" {" + ", ".join(f"{k}: {v}" for k, v in attrs.items()) + "}")
            if attrs
            else ""
        )
        lines.append(f"  {src} -[{edge['relationship']}{attr_str}]-> {tgt}")
    return "\n".join(lines)


def write_nodes(
    graph: dict,
    extraction: dict,
    temp_to_real: dict[str, str],
    vectors: dict[str, list[float]],
    session_id: str,
    session_created_at: str,
) -> None:
    """
    Write resolved nodes to the graph. temp_to_real and vectors are pre-computed
    by the unified pipeline in ingest.py; this function only handles storage.
    Nodes already mapped to an existing ID are merged instead of created.
    """
    now = datetime.now(timezone.utc).isoformat()

    for temp_id, node_data in extraction.get("nodes", {}).items():
        real_id = temp_to_real[temp_id]
        raw_attrs: dict = dict(node_data.get("attributes", {}))
        raw_attrs = _process_raw_dates(raw_attrs, session_created_at)
        flat_attrs = _flatten_attributes(raw_attrs)
        content = node_data.get("content")

        if real_id in graph["nodes"]:
            _merge_node(graph, real_id, content, flat_attrs)
        else:
            graph["vector_index"][real_id] = vectors[temp_id]
            graph["nodes"][real_id] = {
                "id": real_id,
                "type": node_data.get("node_type", "semantic"),
                "label": node_data.get("label", ""),
                "title": node_data.get("title", ""),
                "content": content,
                "attributes": flat_attrs,
                "created_at": now,
                "session_id": session_id,
                "importance_score": 1.0,
                "access_count": 0,
                "last_accessed_at": None,
            }

        _index_attributes(graph, real_id, raw_attrs)


def add_edge(
    graph: dict,
    source_id: str,
    target_id: str,
    relation: str,
) -> None:
    """Write edges to the graph using the resolved temp_to_real node ID map."""
    now = datetime.now(timezone.utc).isoformat()

    for edge_data in extraction.get("edges", {}).values():
        edge_id = str(uuid.uuid4())
        source_real = temp_to_real.get(edge_data["source"], edge_data["source"])
        target_real = temp_to_real.get(edge_data["target"], edge_data["target"])

        raw_attrs: dict = dict(edge_data.get("attributes", {}))
        raw_attrs = _process_raw_dates(raw_attrs, session_created_at)
        flat_attrs = _flatten_attributes(raw_attrs)

        graph["edges"][edge_id] = {
            "id": edge_id,
            "source": source_real,
            "relationship": edge_data.get("relationship", ""),
            "target": target_real,
            "attributes": flat_attrs,
            "created_at": now,
            "session_id": session_id,
        }

        _index_attributes(graph, edge_id, raw_attrs)
