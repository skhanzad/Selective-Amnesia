"""
Baseline RAG benchmark runner.

Stores each user turn as a raw text chunk in a flat vector store.
At question time, embeds the question and retrieves the top-k most
similar chunks to use as context — no knowledge graph involved.

Intended to isolate the contribution of the graph structure: any
performance difference vs. benchmark.py is attributable to the graph.

Usage:
    python -m baseline_rag [--limit N] [--data PATH] [--top-k K]
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark import (
    DATA_PATH,
    METRICS,
    _build_summary,
    _save_output,
    _subsample,
    exact_match,
    generate_answer,
    llm_judge,
    report,
    token_scores,
)
from graph.embedder import get_embedding, get_embeddings

TOP_K = 5


# ── Flat vector store ──────────────────────────────────────────────────────────


def empty_store() -> dict:
    return {"chunks": [], "vectors": []}


def ingest_chunk(text: str, store: dict) -> None:
    vector = get_embedding(text)
    store["chunks"].append(text)
    store["vectors"].append(vector)


def retrieve_chunks(question: str, store: dict, top_k: int = TOP_K) -> str:
    if not store["chunks"]:
        return ""

    q_vec = np.asarray(get_embedding(question))
    matrix = np.asarray(store["vectors"])

    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_vec)
    norms = np.where(norms == 0, 1e-10, norms)
    scores = matrix @ q_vec / norms

    top_indices = np.argsort(scores)[::-1][:top_k]
    return "\n\n---\n\n".join(store["chunks"][i] for i in top_indices)


# ── Per-question pipeline ──────────────────────────────────────────────────────


def run_question(item: dict, top_k: int = TOP_K, verbose: bool = False) -> dict:
    store = empty_store()
    q_start = time.time()

    sessions = item["haystack_sessions"]
    total_user_turns = sum(1 for s in sessions for t in s if t["role"] == "user")
    turn_num = 0

    for session in sessions:
        for turn in session:
            if turn["role"] != "user":
                continue
            turn_num += 1
            preview = turn["content"][:60].replace("\n", " ")
            t0 = time.time()
            print(f"  ingesting turn {turn_num}/{total_user_turns}: {preview!r}", flush=True)
            ingest_chunk(turn["content"], store)
            print(f"    done in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    print("  retrieving context for question...", flush=True)
    context = retrieve_chunks(item["question"], store, top_k=top_k)
    print(f"  context: {top_k} chunks retrieved ({time.time() - t0:.1f}s)", flush=True)
    if context:
        print(f"\n  --- Retrieved chunks ---\n{context[:500]}{'...' if len(context) > 500 else ''}\n  ---", flush=True)

    t0 = time.time()
    print("  generating answer...", flush=True)
    predicted = generate_answer(item["question"], context)
    print(f"  answer generated ({time.time() - t0:.1f}s)", flush=True)

    ground_truth = item["answer"]
    scores = token_scores(predicted, ground_truth)
    scores.update(llm_judge(item["question"], ground_truth, predicted))

    result = {
        "question_id": item["question_id"],
        "question_type": item.get("question_type", "unknown"),
        "question": item["question"],
        "ground_truth": ground_truth,
        "predicted": predicted,
        "time_seconds": round(time.time() - q_start, 1),
        "turns_ingested": total_user_turns,
        "chunks_retrieved": top_k,
        **scores,
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"ID:         {result['question_id']}")
        print(f"Type:       {result['question_type']}")
        print(f"Question:   {result['question']}")
        print(f"Expected:   {ground_truth}")
        print(f"Predicted:  {predicted}")
        print(f"F1: {scores['f1']:.3f}  P: {scores['precision']:.3f}  R: {scores['recall']:.3f}  EM: {scores.get('exact_match', 0)}")

    return result


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline RAG benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Max total questions")
    parser.add_argument("--per-type", type=int, default=None, help="Sample this many questions per question_type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling (default: 42)")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to dataset JSON")
    parser.add_argument("--output", type=str, default=None, help="Save per-question results to JSON")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Chunks to retrieve per question (default: {TOP_K})")
    parser.add_argument("--verbose", action="store_true", help="Print each question result")
    args = parser.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))

    if args.per_type:
        print(f"Subsampling {args.per_type} questions per type (seed={args.seed}):")
        data = _subsample(data, args.per_type, seed=args.seed)

    if args.limit:
        data = data[: args.limit]

    print(f"Evaluating {len(data)} questions (baseline RAG, top-k={args.top_k})...")

    results = []
    for i, item in enumerate(data):
        print(f"\n[{i+1}/{len(data)}] {item['question_id']}  type={item.get('question_type', '?')}", flush=True)
        print(f"  Q: {item['question']}", flush=True)
        q_start = time.time()
        try:
            result = run_question(item, top_k=args.top_k, verbose=args.verbose)
            results.append(result)
            print(
                f"  => F1={result['f1']:.3f}  P={result['precision']:.3f}  R={result['recall']:.3f}"
                f"  Judge={'✓' if result['judge_correct'] else '✗'}  ({result['time_seconds']}s)",
                flush=True,
            )
            print(f"     judge: {result.get('judge_reason', '')}", flush=True)
            print(f"     expected:  {result['ground_truth']}", flush=True)
            print(f"     predicted: {result['predicted']}", flush=True)
        except Exception as exc:
            import traceback

            elapsed_s = round(time.time() - q_start, 1)
            print(f"  ERROR after {elapsed_s}s: {exc}", flush=True)
            traceback.print_exc()
            results.append({
                "question_id": item["question_id"],
                "question_type": item.get("question_type", "unknown"),
                "question": item["question"],
                "ground_truth": item.get("answer", ""),
                "predicted": None,
                "time_seconds": elapsed_s,
                "error": str(exc),
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "exact_match": 0,
            })

        if args.output:
            _save_output(results, args.output)

    report(results)
    if args.output:
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
