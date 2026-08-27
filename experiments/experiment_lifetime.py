"""
Lifetime graph experiment — tests the forgetting module.

Unlike benchmark.py (which builds a fresh per-question graph), this experiment
ingests ALL sessions from a subset of items into ONE shared graph in
chronological order — simulating a long-lived user memory store.

Two runs are executed back-to-back against the same extraction cache:
  Run A — no forgetting   : graph grows unbounded
  Run B — with forgetting : forgetting module prunes stale nodes periodically

Both runs answer the same questions, scored with F1 and judge_correct.
The comparison shows whether forgetting improves (or hurts) retrieval from
a large, mixed-content graph.

Usage:
    python experiment_lifetime.py [--per-type N] [--limit N] [--data PATH]
    python experiment_lifetime.py --per-type 5          # ~30 questions
    python experiment_lifetime.py --regen-cache         # re-extract even if cache exists

Tune forgetting:
    python experiment_lifetime.py --forget-every-n 30 --forget-threshold 0.20
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

# ── Reuse scoring and LLM helpers from benchmark ──────────────────────────────
from experiments.benchmark import (
    METRICS,
    _subsample,
    generate_answer,
    llm_judge,
    token_scores,
)
from graph.extract import extract_batch
from graph.forget import DEFAULT_CONFIG as DEFAULT_FORGET_CONFIG
from graph.forget import run_forgetting_pass
from graph.graph_manager import empty_graph
from graph.ingest import run_in_memory

DATA_PATH = Path("data/longmemeval_oracle.json")
CACHE_PATH = Path("lifetime_cache.json")
GRAPHS_DIR = Path("graphs")
RESULTS_PATH = Path("lifetime_results.json")


# ── Date parsing ───────────────────────────────────────────────────────────────


def _parse_haystack_date(raw: str) -> str:
    """
    Convert '2023/04/10 (Mon) 17:50' to sortable ISO string '2023-04-10T17:50'.
    Returns raw string on failure so sorting degrades gracefully.
    """
    try:
        parts = raw.split()
        date_part = parts[0].replace("/", "-")
        time_part = parts[2] if len(parts) >= 3 else "00:00"
        return f"{date_part}T{time_part}"
    except Exception:
        return raw


# ── Session collection and deduplication ─────────────────────────────────────


def collect_sessions(items: list[dict]) -> list[dict]:
    """
    Gather all sessions from all items, deduplicated by session_id and sorted
    chronologically by haystack_date.

    Returns a list of:
        {session_id, date_iso, turns: [{role, content}], item_ids: [str]}
    """
    seen: dict[str, dict] = {}

    for item in items:
        session_ids = item.get("haystack_session_ids", [])
        haystack_dates = item.get("haystack_dates", [])
        sessions = item.get("haystack_sessions", [])

        for sid, raw_date, turns in zip(session_ids, haystack_dates, sessions):
            if sid not in seen:
                seen[sid] = {
                    "session_id": sid,
                    "date_iso": _parse_haystack_date(raw_date),
                    "turns": turns,
                    "item_ids": [],
                }
            seen[sid]["item_ids"].append(item["question_id"])

    ordered = sorted(seen.values(), key=lambda s: s["date_iso"])
    return ordered


# ── Extraction cache ───────────────────────────────────────────────────────────


def _turn_key(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def build_extraction_cache(sessions: list[dict], cache_path: Path) -> dict[str, dict]:
    """
    Extract all unique user turns from sessions and cache results to disk.
    Returns {content_hash: extraction_dict}.
    """
    all_user_turns: list[str] = []
    seen_keys: set[str] = set()

    for session in sessions:
        for turn in session["turns"]:
            if turn["role"] != "user":
                continue
            key = _turn_key(turn["content"])
            if key not in seen_keys:
                seen_keys.add(key)
                all_user_turns.append(turn["content"])

    print(f"Extracting {len(all_user_turns)} unique user turns...", flush=True)
    t0 = time.time()
    extractions = extract_batch(all_user_turns, batch_size=6)
    elapsed = round(time.time() - t0, 1)
    print(f"Extraction complete in {elapsed}s", flush=True)

    cache: dict[str, dict] = {}
    for content, extraction in zip(all_user_turns, extractions):
        cache[_turn_key(content)] = extraction

    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(f"Cache saved to {cache_path}", flush=True)
    return cache


def load_or_build_cache(
    sessions: list[dict], cache_path: Path, regen: bool
) -> dict[str, dict]:
    if not regen and cache_path.exists():
        print(f"Loading extraction cache from {cache_path}", flush=True)
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return build_extraction_cache(sessions, cache_path)


# ── Graph build ────────────────────────────────────────────────────────────────


def build_lifetime_graph(
    sessions: list[dict],
    cache: dict[str, dict],
    forget_config: dict | None,
    label: str,
) -> dict:
    """
    Ingest all sessions in chronological order into a fresh graph.
    Returns the completed graph dict.
    """
    graph = empty_graph()
    total_sessions = len(sessions)
    total_turns = sum(1 for s in sessions for t in s["turns"] if t["role"] == "user")
    print(
        f"\n[{label}] building lifetime graph: {total_sessions} sessions, "
        f"{total_turns} user turns",
        flush=True,
    )

    turn_count = 0
    for si, session in enumerate(sessions):
        user_turns = [t for t in session["turns"] if t["role"] == "user"]
        for ti, turn in enumerate(user_turns):
            key = _turn_key(turn["content"])
            extraction = cache.get(key)
            if extraction is None:
                print(
                    f"  WARNING: no cached extraction for turn {key}, skipping",
                    flush=True,
                )
                continue

            _, graph = run_in_memory(
                turn["content"],
                graph,
                ingest_only=True,
                extraction=extraction,
                forget_config=forget_config,
            )
            turn_count += 1

        if (si + 1) % 10 == 0 or si == total_sessions - 1:
            print(
                f"  [{label}] session {si + 1}/{total_sessions} "
                f"({turn_count} turns ingested, "
                f"{len(graph['nodes'])} nodes, {len(graph['edges'])} edges)",
                flush=True,
            )

    # Final explicit forgetting pass if enabled
    if forget_config is not None:
        cfg = {**DEFAULT_FORGET_CONFIG, **forget_config}
        stats = run_forgetting_pass(graph, cfg)
        print(
            f"  [{label}] final forget pass: scored={stats['scored']} "
            f"pruned={stats['pruned']} compressed={stats['compressed']} "
            f"flagged={stats['flagged']} edges_removed={stats['edges_removed']}",
            flush=True,
        )

    return graph


# ── Query phase ────────────────────────────────────────────────────────────────


def query_all(items: list[dict], graph: dict, label: str) -> list[dict]:
    """
    For each item, retrieve context from the shared graph, generate an answer,
    and score against ground truth.
    """
    results = []
    total = len(items)

    for i, item in enumerate(items):
        qid = item["question_id"]
        question = item["question"]
        ground_truth = item["answer"]
        qtype = item.get("question_type", "unknown")

        print(
            f"  [{label}] [{i + 1}/{total}] {qid} type={qtype}",
            flush=True,
        )
        t0 = time.time()

        context, _ = run_in_memory(question, graph)
        predicted = generate_answer(question, context)
        scores = token_scores(predicted, ground_truth)
        scores.update(llm_judge(question, ground_truth, predicted))

        elapsed = round(time.time() - t0, 1)
        print(
            f"    F1={scores['f1']:.3f} judge={'✓' if scores['judge_correct'] else '✗'} "
            f"({elapsed}s)  predicted={predicted!r}",
            flush=True,
        )

        results.append(
            {
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "run": label,
                "context_nodes": context.count("\n[") if context else 0,
                **scores,
            }
        )

    return results


# ── Reporting ──────────────────────────────────────────────────────────────────


def _avg(items: list[dict], key: str) -> float:
    vals = [i[key] for i in items if key in i]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def compare_runs(
    results_a: list[dict],
    results_b: list[dict],
    label_a: str,
    label_b: str,
    graph_a: dict,
    graph_b: dict,
) -> dict:
    """Build and print a side-by-side comparison of two runs."""
    by_type_a: dict[str, list] = defaultdict(list)
    by_type_b: dict[str, list] = defaultdict(list)
    for r in results_a:
        by_type_a[r["question_type"]].append(r)
    for r in results_b:
        by_type_b[r["question_type"]].append(r)

    all_types = sorted(set(by_type_a) | set(by_type_b))

    print(f"\n{'=' * 70}")
    print(f"LIFETIME GRAPH EXPERIMENT  ({len(results_a)} questions per run)")
    print(f"{'=' * 70}")
    print(
        f"Graph sizes:  {label_a}={len(graph_a['nodes'])} nodes / "
        f"{len(graph_a['edges'])} edges   "
        f"{label_b}={len(graph_b['nodes'])} nodes / "
        f"{len(graph_b['edges'])} edges"
    )
    print()

    col = 14
    header = f"{'Metric':<16}  {label_a:>{col}}  {label_b:>{col}}  {'delta':>{col}}"
    print(header)
    print("-" * len(header))

    comparison: dict = {"overall": {}, "by_type": {}}
    for metric in METRICS:
        a_val = _avg(results_a, metric)
        b_val = _avg(results_b, metric)
        delta = b_val - a_val
        sign = "+" if delta >= 0 else ""
        print(
            f"{metric:<16}  {a_val:>{col}.4f}  {b_val:>{col}.4f}  {sign}{delta:>{col - 1}.4f}"
        )
        comparison["overall"][metric] = {
            "no_forget": a_val,
            "forget": b_val,
            "delta": round(delta, 4),
        }

    print()
    print(
        f"{'By type':<16}  {'f1 ' + label_a:>{col}}  {'f1 ' + label_b:>{col}}  {'delta':>{col}}"
    )
    print("-" * len(header))
    for qtype in all_types:
        a_val = _avg(by_type_a.get(qtype, []), "f1")
        b_val = _avg(by_type_b.get(qtype, []), "f1")
        delta = b_val - a_val
        sign = "+" if delta >= 0 else ""
        count = len(by_type_a.get(qtype, []))
        print(
            f"{qtype[:16]:<16}  {a_val:>{col}.4f}  {b_val:>{col}.4f}  {sign}{delta:>{col - 1}.4f}  (n={count})"
        )
        comparison["by_type"][qtype] = {
            "count": count,
            "f1": {"no_forget": a_val, "forget": b_val, "delta": round(delta, 4)},
            "judge_correct": {
                "no_forget": _avg(by_type_a.get(qtype, []), "judge_correct"),
                "forget": _avg(by_type_b.get(qtype, []), "judge_correct"),
            },
        }

    print()
    ctx_a = _avg(results_a, "context_nodes")
    ctx_b = _avg(results_b, "context_nodes")
    print(f"Avg context nodes retrieved:  {label_a}={ctx_a:.1f}  {label_b}={ctx_b:.1f}")

    return comparison


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lifetime graph experiment: compare forgetting vs no-forgetting on a shared graph"
    )
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument(
        "--per-type",
        type=int,
        default=None,
        help="Sample this many questions per question_type",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max total questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regen-cache",
        action="store_true",
        help="Re-extract even if lifetime_cache.json already exists",
    )
    parser.add_argument(
        "--forget-every-n",
        type=int,
        default=DEFAULT_FORGET_CONFIG["trigger_every_n_turns"],
        help=f"Trigger forgetting every N turns (default: {DEFAULT_FORGET_CONFIG['trigger_every_n_turns']})",
    )
    parser.add_argument(
        "--forget-threshold",
        type=float,
        default=DEFAULT_FORGET_CONFIG["pruning_threshold"],
        help=f"Pruning threshold (default: {DEFAULT_FORGET_CONFIG['pruning_threshold']})",
    )
    parser.add_argument(
        "--forget-node-count",
        type=int,
        default=DEFAULT_FORGET_CONFIG["trigger_node_count"],
        help=f"Trigger forgetting at this node count (default: {DEFAULT_FORGET_CONFIG['trigger_node_count']})",
    )
    args = parser.parse_args()

    forget_config = {
        "trigger_every_n_turns": args.forget_every_n,
        "pruning_threshold": args.forget_threshold,
        "trigger_node_count": args.forget_node_count,
    }
    print(
        f"Forgetting config: every_n={args.forget_every_n} "
        f"threshold={args.forget_threshold} node_count={args.forget_node_count}"
    )

    # ── Load and subsample items ───────────────────────────────────────────────
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))

    if args.per_type:
        print(f"Subsampling {args.per_type} per type (seed={args.seed}):")
        data = _subsample(data, args.per_type, seed=args.seed)
    if args.limit:
        data = data[: args.limit]

    print(f"\nUsing {len(data)} questions")

    # ── Collect sessions in chronological order ────────────────────────────────
    sessions = collect_sessions(data)
    print(
        f"Collected {len(sessions)} unique sessions "
        f"({sum(1 for s in sessions for t in s['turns'] if t['role'] == 'user')} user turns)"
    )
    if sessions:
        print(f"  date range: {sessions[0]['date_iso']} → {sessions[-1]['date_iso']}")

    # ── Extract (or load cache) ────────────────────────────────────────────────
    cache = load_or_build_cache(sessions, CACHE_PATH, regen=args.regen_cache)
    print(f"Cache: {len(cache)} unique turn extractions")

    # ── Run A: no forgetting ───────────────────────────────────────────────────
    t0 = time.time()
    graph_a = build_lifetime_graph(
        sessions, cache, forget_config=None, label="no-forget"
    )
    print(
        f"\n[no-forget] graph complete: {len(graph_a['nodes'])} nodes, "
        f"{len(graph_a['edges'])} edges  ({round(time.time() - t0, 1)}s)",
        flush=True,
    )
    GRAPHS_DIR.mkdir(exist_ok=True)
    (GRAPHS_DIR / "lifetime_no_forget.json").write_text(
        json.dumps(graph_a, indent=2), encoding="utf-8"
    )

    # ── Run B: with forgetting ─────────────────────────────────────────────────
    t0 = time.time()
    graph_b = build_lifetime_graph(
        sessions, cache, forget_config=forget_config, label="forget"
    )
    print(
        f"\n[forget] graph complete: {len(graph_b['nodes'])} nodes, "
        f"{len(graph_b['edges'])} edges  ({round(time.time() - t0, 1)}s)",
        flush=True,
    )
    (GRAPHS_DIR / "lifetime_forget.json").write_text(
        json.dumps(graph_b, indent=2), encoding="utf-8"
    )

    # ── Query both graphs ──────────────────────────────────────────────────────
    print(f"\nQuerying {len(data)} questions against no-forget graph...", flush=True)
    results_a = query_all(data, graph_a, label="no-forget")

    print(f"\nQuerying {len(data)} questions against forget graph...", flush=True)
    results_b = query_all(data, graph_b, label="forget")

    # ── Report ─────────────────────────────────────────────────────────────────
    comparison = compare_runs(
        results_a, results_b, "no-forget", "forget", graph_a, graph_b
    )

    output = {
        "config": {
            "questions": len(data),
            "sessions": len(sessions),
            "forget_config": forget_config,
        },
        "graph_sizes": {
            "no_forget": {
                "nodes": len(graph_a["nodes"]),
                "edges": len(graph_a["edges"]),
            },
            "forget": {"nodes": len(graph_b["nodes"]), "edges": len(graph_b["edges"])},
        },
        "comparison": comparison,
        "results_no_forget": results_a,
        "results_forget": results_b,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
