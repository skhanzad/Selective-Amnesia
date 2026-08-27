import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 output on Windows so emoji in LLM responses don't crash the script
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from graph.graph_manager import load_graph
from pipeline.flow import app

GRAPH_PATH = "data/graph.json"
LOG_PATH = "data/chat_log.json"


def _graph_snapshot(graph: dict) -> dict:
    node_labels = {nid: n["label"] for nid, n in graph["nodes"].items()}
    return {
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "turns": graph["metadata"]["total_interactions"],
        "nodes": [
            {
                "label": n["label"],
                "type": n["type"],
                "importance_score": n["importance_score"],
                "access_count": n["stats"]["access_count"],
                "attributes": n.get("attributes", {}),
            }
            for n in graph["nodes"].values()
        ],
        "edges": [
            {
                "source": node_labels.get(e["source"], e["source"]),
                "relation": e["relation"],
                "target": node_labels.get(e["target"], e["target"]),
            }
            for e in graph["edges"]
        ],
    }


def _print_graph_state(graph: dict) -> None:
    nodes = graph["nodes"]
    edges = graph["edges"]
    meta = graph["metadata"]

    print("\n--- Knowledge Graph ---")
    print(f"  Nodes ({len(nodes)}):")
    for node in nodes.values():
        attrs = node.get("attributes", {})
        attr_str = f" {attrs}" if attrs else ""
        print(
            f"    [{node['type'][:3].upper()}] {node['label']}"
            f"{attr_str}  (score={node['importance_score']:.2f},"
            f" access={node['stats']['access_count']})"
        )
    print(f"  Edges ({len(edges)}):")
    node_labels = {nid: n["label"] for nid, n in nodes.items()}
    for edge in edges:
        src = node_labels.get(edge["source"], edge["source"])
        tgt = node_labels.get(edge["target"], edge["target"])
        print(f"    {src} --[{edge['relation']}]--> {tgt}")
    print(f"  Turns: {meta['total_interactions']}")
    print("-----------------------\n")


def _load_log() -> list:
    p = Path(LOG_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def _append_log(entry: dict) -> None:
    log = _load_log()
    log.append(entry)
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(LOG_PATH).write_text(json.dumps(log, indent=2), encoding="utf-8")


def run_turn(user_input: str) -> None:
    print(f"You: {user_input}")
    try:
        result = app.invoke({"kg_path": GRAPH_PATH, "user_input": user_input})
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        _append_log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user_input,
            "assistant": None,
            "error": str(exc),
            "graph_after": None,
        })
        return

    graph = load_graph(GRAPH_PATH)
    print(f"\nAssistant: {result['llm_response']}\n")
    _print_graph_state(graph)

    _append_log({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user_input,
        "assistant": result["llm_response"],
        "graph_after": _graph_snapshot(graph),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script",
        metavar="FILE",
        help="Path to a text file with one prompt per line. "
             "Lines starting with # are treated as comments and skipped.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete data/graph.json and data/chat_log.json before starting.",
    )
    args = parser.parse_args()

    if args.reset:
        for path in (GRAPH_PATH, LOG_PATH):
            p = Path(path)
            if p.exists():
                p.unlink()
                print(f"[reset] Deleted {path}")
        print()

    if args.script:
        prompts = [
            line.strip()
            for line in Path(args.script).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        print(f"KG-RAG Script  ({len(prompts)} prompts)\n")
        for prompt in prompts:
            run_turn(prompt)
        print(f"Done. Log written to {LOG_PATH}")
        return

    # Interactive mode
    print("KG-RAG Chat  (type 'quit' or Ctrl-C to exit)\n")
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit"}:
                break
            run_turn(user_input)
    except KeyboardInterrupt:
        pass
    print(f"Bye. Log written to {LOG_PATH}")


if __name__ == "__main__":
    main()
