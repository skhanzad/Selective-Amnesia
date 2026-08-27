"""
combined_benchmark.py — Single-pass runner for all 4 experiment variants.

Experiments
-----------
  1a. Graph RAG    — fresh per-question graph             (Experiment 1 treatment)
  1b. Baseline RAG — fresh per-question flat FAISS store  (Experiment 1 control)
  2a. Persistent graph, no forgetting                     (Experiment 2 control)
  2b. Persistent graph, with forgetting                   (Experiment 2 treatment)

Phase 1 — processes questions in order:
  • Calls extract() ONCE per turn, reuses the result for the per-question graph
    AND both persistent graphs (saves ~2/3 of extraction LLM calls).
  • Simultaneously feeds turns into: per-question graph, persistent-no-forget,
    persistent-forget, and a fresh per-question baseline flat store.
  • Answers and scores each question from the per-question graph + baseline.
  • Saves all state after every question so a crash loses at most one question.

Phase 2 — runs after all phase-1 ingestion is complete:
  • Answers all questions from both persistent graphs using retrieve_only mode
    (no writes to the graph — the persistent graphs are read-only in this phase).
  • Saves results after every question.

Resume
------
  Restart with exactly the same command. The checkpoint in <out-dir>/checkpoint.json
  tracks which questions are done in each phase. Completed work is skipped.

Usage
-----
    python combined_benchmark.py [--limit N] [--data PATH] [--out-dir DIR]
                                  [--per-type N] [--seed N]
                                  [--forget-every-n N] [--forget-threshold F]
                                  [--forget-node-count N]
                                  [--forget-recency-half-life-days F]
                                  [--forget-turns-half-life F]
"""

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from experiments.baseline_rag import empty_store, ingest_chunk, retrieve_chunks
from experiments.benchmark import (
    DATA_PATH,
    METRICS,
    _build_summary,
    _parse_haystack_date,
    _subsample,
    generate_answer,
    llm_judge,
    token_scores,
)
from graph.extract import extract
from graph.forget import DEFAULT_CONFIG as DEFAULT_FORGET_CONFIG
from graph.graph_manager import empty_graph
from graph.ingest import run_in_memory

# ── Output file names (all written under --out-dir) ────────────────────────────

CHECKPOINT_FILE = "checkpoint.json"
EXP1_GRAPH_FILE = "exp1_graph_rag.json"
EXP1_BASELINE_FILE = "exp1_baseline.json"
EXP2_NO_FORGET_FILE = "exp2_no_forget.json"
EXP2_FORGET_FILE = "exp2_forget.json"
PERSISTENT_NO_FORGET_GRAPH_FILE = "persistent_no_forget.json"
PERSISTENT_FORGET_GRAPH_FILE = "persistent_forget.json"


# ── Atomic file I/O ────────────────────────────────────────────────────────────


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON to a .tmp file then atomically rename — prevents partial writes."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default() if callable(default) else default


# ── Checkpoint ─────────────────────────────────────────────────────────────────


def _load_checkpoint(path: Path) -> dict:
    raw = _load_json(path, {})
    return {
        "phase1_done": set(raw.get("phase1_done", [])),
        "exp2_no_forget_done": set(raw.get("exp2_no_forget_done", [])),
        "exp2_forget_done": set(raw.get("exp2_forget_done", [])),
    }


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    _atomic_write(
        path,
        {
            "phase1_done": sorted(checkpoint["phase1_done"]),
            "exp2_no_forget_done": sorted(checkpoint["exp2_no_forget_done"]),
            "exp2_forget_done": sorted(checkpoint["exp2_forget_done"]),
        },
    )


# ── Result helpers ─────────────────────────────────────────────────────────────


def _load_results(path: Path) -> list[dict]:
    raw = _load_json(path, {})
    return raw.get("questions", []) if isinstance(raw, dict) else []


def _append_and_save(results: list[dict], new_result: dict, path: Path) -> None:
    results.append(new_result)
    _atomic_write(path, {"summary": _build_summary(results), "questions": results})


# ── Persistent graph I/O ───────────────────────────────────────────────────────


def _load_graph(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return empty_graph()


def _save_graph(graph: dict, path: Path) -> None:
    _atomic_write(path, graph)


# ── Scoring ────────────────────────────────────────────────────────────────────


def _score(question: str, ground_truth: str, predicted: str) -> dict:
    scores = token_scores(predicted, ground_truth)
    scores.update(llm_judge(question, ground_truth, predicted))
    return scores


def _error_result(item: dict, elapsed: float, exc: Exception) -> dict:
    return {
        "question_id": item["question_id"],
        "question_type": item.get("question_type", "unknown"),
        "question": item["question"],
        "ground_truth": item.get("answer", ""),
        "predicted": None,
        "time_seconds": elapsed,
        "error": str(exc),
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "judge_correct": 0,
        "judge_reason": "",
    }


# ── Phase 1: per-question ingestion + answering ────────────────────────────────


def _ingest_sessions(
    sessions: list,
    haystack_dates: list[str],
    per_q_graph: dict,
    persistent_no_forget: dict,
    persistent_forget: dict,
    baseline_store: dict,
    forget_config: dict,
) -> int:
    """
    Ingest all user turns from all sessions into all stores.

    extract() is called once per turn. The same extraction dict is passed to
    the per-question graph and both persistent graphs, avoiding redundant LLM calls.
    The baseline store uses its own embedding call (no LLM involved).

    Returns the total number of user turns ingested.
    """
    total_turns = sum(
        1 for s in sessions for t in s if t["role"] in ("user", "assistant")
    )
    turn_index = 0

    for s_idx, session in enumerate(sessions):
        session_date = (
            _parse_haystack_date(haystack_dates[s_idx])
            if s_idx < len(haystack_dates)
            else None
        )

        for turn in session:
            if turn["role"] not in ("user", "assistant"):
                continue

            preview = turn["content"][:60].replace("\n", " ")
            t0 = time.time()
            print(
                f"  turn {turn_index + 1}/{total_turns} [{turn['role']}]: {preview!r}",
                flush=True,
            )

            # Tag content with speaker role so the extraction prompt handles
            # user vs assistant turns correctly, then extract once and reuse.
            tagged = f"[Role: {turn['role']}]\n{turn['content']}"
            extraction = extract(tagged)

            run_in_memory(
                tagged,
                per_q_graph,
                ingest_only=True,
                extraction=extraction,
                session_date=session_date,
            )
            run_in_memory(
                tagged,
                persistent_no_forget,
                ingest_only=True,
                extraction=extraction,
                session_date=session_date,
            )
            run_in_memory(
                tagged,
                persistent_forget,
                ingest_only=True,
                extraction=extraction,
                session_date=session_date,
                forget_config=forget_config,
            )

            # Baseline: embedding only, no LLM call
            ingest_chunk(turn["content"], baseline_store)

            print(f"    done in {time.time() - t0:.1f}s", flush=True)
            turn_index += 1

    return total_turns


def run_phase1(
    data: list[dict],
    out: Path,
    checkpoint: dict,
    persistent_no_forget: dict,
    persistent_forget: dict,
    forget_config: dict,
) -> tuple[list[dict], list[dict]]:
    """
    For each question: build a fresh per-question graph + fresh baseline store,
    ingest turns (sharing extraction with persistent graphs), answer from both,
    score, and save. Returns (exp1_graph_results, exp1_baseline_results).
    """
    exp1_graph_results = _load_results(out / EXP1_GRAPH_FILE)
    exp1_baseline_results = _load_results(out / EXP1_BASELINE_FILE)
    done = checkpoint["phase1_done"]

    for i, item in enumerate(data):
        q_id = item["question_id"]
        if q_id in done:
            print(
                f"\n[{i + 1}/{len(data)}] {q_id} — already done, skipping", flush=True
            )
            continue

        print(
            f"\n[{i + 1}/{len(data)}] {q_id}  type={item.get('question_type', '?')}",
            flush=True,
        )
        print(f"  Q: {item['question']}", flush=True)
        q_start = time.time()

        sessions = item["haystack_sessions"]
        haystack_dates = item.get("haystack_dates", [])
        question_date = (
            _parse_haystack_date(item["question_date"])
            if item.get("question_date")
            else None
        )

        try:
            # Fresh graphs/store for this question's Experiment 1 answer
            per_q_graph = empty_graph()
            baseline_store = (
                empty_store()
            )  # fresh per question — matches baseline_rag.py

            total_turns = _ingest_sessions(
                sessions,
                haystack_dates,
                per_q_graph,
                persistent_no_forget,
                persistent_forget,
                baseline_store,
                forget_config,
            )

            # Answer from per-question graph (retrieve_only — no writes to graph)
            t0 = time.time()
            print("  [graph rag] retrieving...", flush=True)
            graph_context, _ = run_in_memory(
                item["question"],
                per_q_graph,
                retrieve_only=True,
                session_date=question_date,
            )
            graph_predicted = generate_answer(item["question"], graph_context)
            graph_scores = _score(item["question"], item["answer"], graph_predicted)
            print(
                f"  [graph rag] F1={graph_scores['f1']:.3f}  "
                f"Judge={'✓' if graph_scores['judge_correct'] else '✗'}  "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
            print(f"     expected:  {item['answer']}", flush=True)
            print(f"     predicted: {graph_predicted}", flush=True)

            # Answer from baseline (embedding retrieval, no LLM for retrieval)
            t0 = time.time()
            print("  [baseline] retrieving...", flush=True)
            baseline_context = retrieve_chunks(item["question"], baseline_store)
            baseline_predicted = generate_answer(item["question"], baseline_context)
            baseline_scores = _score(
                item["question"], item["answer"], baseline_predicted
            )
            print(
                f"  [baseline] F1={baseline_scores['f1']:.3f}  "
                f"Judge={'✓' if baseline_scores['judge_correct'] else '✗'}  "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
            print(f"     expected:  {item['answer']}", flush=True)
            print(f"     predicted: {baseline_predicted}", flush=True)

            elapsed = round(time.time() - q_start, 1)

            graph_result = {
                "question_id": q_id,
                "question_type": item.get("question_type", "unknown"),
                "question": item["question"],
                "ground_truth": item["answer"],
                "predicted": graph_predicted,
                "time_seconds": elapsed,
                "turns_ingested": total_turns,
                **graph_scores,
            }
            baseline_result = {
                "question_id": q_id,
                "question_type": item.get("question_type", "unknown"),
                "question": item["question"],
                "ground_truth": item["answer"],
                "predicted": baseline_predicted,
                "time_seconds": elapsed,
                "turns_ingested": total_turns,
                **baseline_scores,
            }

        except Exception as exc:
            elapsed = round(time.time() - q_start, 1)
            print(f"  ERROR after {elapsed}s: {exc}", flush=True)
            traceback.print_exc()
            graph_result = _error_result(item, elapsed, exc)
            baseline_result = _error_result(item, elapsed, exc)

        # Save order: results → persistent graphs → checkpoint.
        # If we crash between graphs and checkpoint, the question is retried on
        # resume (safe — results are overwritten with identical values).
        _append_and_save(exp1_graph_results, graph_result, out / EXP1_GRAPH_FILE)
        _append_and_save(
            exp1_baseline_results, baseline_result, out / EXP1_BASELINE_FILE
        )
        _save_graph(persistent_no_forget, out / PERSISTENT_NO_FORGET_GRAPH_FILE)
        _save_graph(persistent_forget, out / PERSISTENT_FORGET_GRAPH_FILE)

        if "error" not in graph_result:
            done.add(q_id)
            _save_checkpoint(out / CHECKPOINT_FILE, checkpoint)

    return exp1_graph_results, exp1_baseline_results


# ── Phase 2: answer from persistent graphs ─────────────────────────────────────


def _answer_from_graph(
    item: dict,
    graph: dict,
    question_date: datetime | None,
    label: str,
) -> dict:
    """
    Retrieve context from a persistent graph and generate an answer.
    Uses retrieve_only=True so the graph is never modified.
    """
    q_start = time.time()
    try:
        print(f"  [{label}] retrieving...", flush=True)
        context, _ = run_in_memory(
            item["question"],
            graph,
            retrieve_only=True,
            session_date=question_date,
        )
        # Persistent graphs can produce very large subgraphs — truncate to ~80k
        # chars (~20k tokens) to stay well within gpt-4o-mini's 128k context limit.
        if len(context) > 80_000:
            context = context[:80_000] + "\n[context truncated]"
        predicted = generate_answer(item["question"], context)
        scores = _score(item["question"], item["answer"], predicted)
        print(
            f"  [{label}] F1={scores['f1']:.3f}  "
            f"Judge={'✓' if scores['judge_correct'] else '✗'}  "
            f"({time.time() - q_start:.1f}s)",
            flush=True,
        )
        print(f"     expected:  {item['answer']}", flush=True)
        print(f"     predicted: {predicted}", flush=True)
        return {
            "question_id": item["question_id"],
            "question_type": item.get("question_type", "unknown"),
            "question": item["question"],
            "ground_truth": item["answer"],
            "predicted": predicted,
            "time_seconds": round(time.time() - q_start, 1),
            **scores,
        }
    except Exception as exc:
        elapsed = round(time.time() - q_start, 1)
        print(f"  [{label}] ERROR after {elapsed}s: {exc}", flush=True)
        traceback.print_exc()
        return _error_result(item, elapsed, exc)


def run_phase2(
    data: list[dict],
    out: Path,
    checkpoint: dict,
    persistent_no_forget: dict,
    persistent_forget: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Answer every question from both persistent graphs. The graphs are treated as
    read-only (retrieve_only=True) — no modification, safe to resume mid-way.
    Returns (exp2_no_forget_results, exp2_forget_results).
    """
    exp2_no_forget_results = _load_results(out / EXP2_NO_FORGET_FILE)
    exp2_forget_results = _load_results(out / EXP2_FORGET_FILE)

    for i, item in enumerate(data):
        q_id = item["question_id"]
        question_date = (
            _parse_haystack_date(item["question_date"])
            if item.get("question_date")
            else None
        )

        print(
            f"\n[phase2 {i + 1}/{len(data)}] {q_id}  type={item.get('question_type', '?')}",
            flush=True,
        )
        print(f"  Q: {item['question']}", flush=True)

        if q_id not in checkpoint["exp2_no_forget_done"]:
            result = _answer_from_graph(
                item, persistent_no_forget, question_date, "no-forget"
            )
            _append_and_save(exp2_no_forget_results, result, out / EXP2_NO_FORGET_FILE)
            if "error" not in result:
                checkpoint["exp2_no_forget_done"].add(q_id)
                _save_checkpoint(out / CHECKPOINT_FILE, checkpoint)
        else:
            print(f"  [no-forget] already done, skipping", flush=True)

        if q_id not in checkpoint["exp2_forget_done"]:
            result = _answer_from_graph(
                item, persistent_forget, question_date, "forget"
            )
            _append_and_save(exp2_forget_results, result, out / EXP2_FORGET_FILE)
            if "error" not in result:
                checkpoint["exp2_forget_done"].add(q_id)
                _save_checkpoint(out / CHECKPOINT_FILE, checkpoint)
        else:
            print(f"  [forget] already done, skipping", flush=True)

    return exp2_no_forget_results, exp2_forget_results


# ── Summary report ─────────────────────────────────────────────────────────────


def _storage_metrics(graph: dict, path: Path) -> dict:
    """Collect storage metrics for a persistent graph."""
    node_count = len(graph["nodes"])
    edge_count = len(graph["edges"])
    file_size_kb = round(path.stat().st_size / 1024, 1) if path.exists() else 0.0
    flagged = sum(1 for n in graph["nodes"].values() if n.get("flagged_for_review"))
    type_counts: dict[str, int] = {}
    for n in graph["nodes"].values():
        t = n.get("type", n.get("label", "unknown"))
        type_counts[t] = type_counts.get(t, 0) + 1
    return {
        "nodes": node_count,
        "edges": edge_count,
        "file_size_kb": file_size_kb,
        "flagged_for_review": flagged,
        "node_types": type_counts,
    }


def _print_comparison(
    exp1_graph: list[dict],
    exp1_baseline: list[dict],
    exp2_no_forget: list[dict],
    exp2_forget: list[dict],
    out: Path,
    persistent_no_forget: dict,
    persistent_forget: dict,
) -> None:
    variants = [
        ("Graph RAG (per-q)", exp1_graph),
        ("Baseline RAG (per-q)", exp1_baseline),
        ("Persistent (no-forget)", exp2_no_forget),
        ("Persistent (forget)", exp2_forget),
    ]

    print(f"\n{'=' * 84}")
    print("COMBINED RESULTS")
    print(f"{'=' * 84}")

    # ── Performance metrics ────────────────────────────────────────────────────
    col_w = 22
    print(f"\n--- Performance ---")
    print(f"{'Metric':<16}", end="")
    for name, _ in variants:
        print(f"  {name[:col_w]:>{col_w}}", end="")
    print()
    print("-" * (16 + (col_w + 2) * len(variants)))

    for metric in METRICS:
        print(f"{metric:<16}", end="")
        for _, results in variants:
            vals = [r[metric] for r in results if metric in r and "error" not in r]
            avg = sum(vals) / len(vals) if vals else 0.0
            print(f"  {avg:>{col_w}.3f}", end="")
        print()

    print(f"\nQuestion counts (answered / total):")
    for name, results in variants:
        ok = [r for r in results if "error" not in r]
        print(f"  {name:<28} {len(ok)}/{len(results)}")

    # ── Storage metrics (Experiment 2 only) ───────────────────────────────────
    print(f"\n--- Storage (Experiment 2 persistent graphs) ---")
    nf = _storage_metrics(persistent_no_forget, out / PERSISTENT_NO_FORGET_GRAPH_FILE)
    fg = _storage_metrics(persistent_forget, out / PERSISTENT_FORGET_GRAPH_FILE)

    rows = [
        ("Nodes", nf["nodes"], fg["nodes"]),
        ("Edges", nf["edges"], fg["edges"]),
        ("File size (KB)", nf["file_size_kb"], fg["file_size_kb"]),
        ("Flagged for review", nf["flagged_for_review"], fg["flagged_for_review"]),
    ]
    print(f"{'Metric':<24}  {'No-forget':>12}  {'Forget':>12}  {'Reduction':>10}")
    print("-" * 64)
    for label, nf_val, fg_val in rows:
        reduction = ""
        if isinstance(nf_val, (int, float)) and nf_val > 0:
            pct = (nf_val - fg_val) / nf_val * 100
            reduction = f"{pct:+.1f}%"
        print(f"{label:<24}  {nf_val:>12}  {fg_val:>12}  {reduction:>10}")

    print(f"\nNode type breakdown:")
    all_types = sorted(set(nf["node_types"]) | set(fg["node_types"]))
    print(f"  {'Type':<20}  {'No-forget':>10}  {'Forget':>10}")
    for t in all_types:
        print(
            f"  {t:<20}  {nf['node_types'].get(t, 0):>10}  {fg['node_types'].get(t, 0):>10}"
        )

    # Save summary to file
    summary = {
        "performance": {
            name: {
                metric: round(
                    sum(r[metric] for r in results if metric in r and "error" not in r)
                    / max(
                        len([r for r in results if metric in r and "error" not in r]), 1
                    ),
                    4,
                )
                for metric in METRICS
            }
            for name, results in variants
        },
        "question_counts": {
            name: {
                "answered": len([r for r in results if "error" not in r]),
                "total": len(results),
            }
            for name, results in variants
        },
        "storage": {"no_forget": nf, "forget": fg},
    }
    _atomic_write(out / "summary.json", summary)
    print(f"\nSummary saved to {out / 'summary.json'}")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined 4-variant benchmark")
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument(
        "--out-dir",
        type=str,
        default="combined",
        help="Directory for all output files (default: combined/)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max total questions")
    parser.add_argument(
        "--per-type",
        type=int,
        default=None,
        help="Sample this many questions per question_type",
    )
    parser.add_argument("--seed", type=int, default=42)
    # Forgetting hyperparameters (override DEFAULT_CONFIG for the forget variant)
    parser.add_argument(
        "--forget-every-n",
        type=int,
        default=DEFAULT_FORGET_CONFIG["trigger_every_n_turns"],
    )
    parser.add_argument(
        "--forget-threshold",
        type=float,
        default=DEFAULT_FORGET_CONFIG["pruning_threshold"],
    )
    parser.add_argument(
        "--forget-node-count",
        type=int,
        default=DEFAULT_FORGET_CONFIG["trigger_node_count"],
    )
    parser.add_argument(
        "--forget-recency-half-life-days",
        type=float,
        default=DEFAULT_FORGET_CONFIG["recency_half_life_days"],
    )
    parser.add_argument(
        "--forget-turns-half-life",
        type=float,
        default=DEFAULT_FORGET_CONFIG["turns_half_life"],
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Only run Phase 1 (ingestion + Exp 1 answering). "
        "Use this for smoke tests so Phase 2 runs on the "
        "complete persistent graphs after all Phase 1 is done.",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    forget_config = {
        "trigger_every_n_turns": args.forget_every_n,
        "pruning_threshold": args.forget_threshold,
        "trigger_node_count": args.forget_node_count,
        "recency_half_life_days": args.forget_recency_half_life_days,
        "turns_half_life": args.forget_turns_half_life,
    }

    print(f"Output directory : {out.resolve()}")
    print(f"Forget config    : {forget_config}")

    # Load persistent state
    checkpoint = _load_checkpoint(out / CHECKPOINT_FILE)
    persistent_no_forget = _load_graph(out / PERSISTENT_NO_FORGET_GRAPH_FILE)
    persistent_forget = _load_graph(out / PERSISTENT_FORGET_GRAPH_FILE)

    already_done = len(checkpoint["phase1_done"])
    if already_done:
        print(f"\nResuming: {already_done} questions already completed in phase 1.")
        print(
            f"  Persistent graph (no-forget): {len(persistent_no_forget['nodes'])} nodes, "
            f"{len(persistent_no_forget['edges'])} edges"
        )
        print(
            f"  Persistent graph (forget):    {len(persistent_forget['nodes'])} nodes, "
            f"{len(persistent_forget['edges'])} edges"
        )

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))

    if args.per_type:
        print(f"Subsampling {args.per_type} per type (seed={args.seed}):")
        data = _subsample(data, args.per_type, seed=args.seed)

    if args.limit:
        data = data[: args.limit]

    remaining = len(data) - len(checkpoint["phase1_done"])
    print(f"\nPhase 1: {len(data)} questions total, {remaining} remaining")

    exp1_graph, exp1_baseline = run_phase1(
        data,
        out,
        checkpoint,
        persistent_no_forget,
        persistent_forget,
        forget_config,
    )

    print(f"\nPhase 1 complete.")
    print(
        f"  Persistent graph (no-forget): {len(persistent_no_forget['nodes'])} nodes, "
        f"{len(persistent_no_forget['edges'])} edges"
    )
    print(
        f"  Persistent graph (forget):    {len(persistent_forget['nodes'])} nodes, "
        f"{len(persistent_forget['edges'])} edges"
    )

    if args.skip_phase2:
        print(
            "\n--skip-phase2 set: skipping Phase 2. Re-run without this flag "
            "once all Phase 1 questions are complete."
        )
        print(f"\nPhase 1 results saved to {out.resolve()}/")
        return

    exp2_remaining_nf = len(data) - len(checkpoint["exp2_no_forget_done"])
    exp2_remaining_f = len(data) - len(checkpoint["exp2_forget_done"])
    print(
        f"\nPhase 2: {exp2_remaining_nf} no-forget + {exp2_remaining_f} forget questions remaining"
    )

    exp2_no_forget, exp2_forget = run_phase2(
        data,
        out,
        checkpoint,
        persistent_no_forget,
        persistent_forget,
    )

    _print_comparison(
        exp1_graph,
        exp1_baseline,
        exp2_no_forget,
        exp2_forget,
        out,
        persistent_no_forget,
        persistent_forget,
    )
    print(f"\nAll results saved to {out.resolve()}/")


if __name__ == "__main__":
    main()
