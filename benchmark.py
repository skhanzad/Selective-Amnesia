"""
LongMemEval benchmark runner.

For each question in longmemeval_oracle.json:
  1. Build a fresh in-memory knowledge graph by ingesting all user turns
     across all haystack sessions for that question.
  2. Ask the question with the retrieved subgraph as context.
  3. Score the LLM answer against the ground-truth using token-level
     F1, precision, recall, and exact match.

Usage:
    python -m benchmark [--limit N] [--data PATH]
"""

import argparse
import json
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests

from graph.ingest import run_in_memory

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_PATH = Path("data/longmemeval_oracle.json")
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "gemma2"

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to personal memory context about the user. "
    "Use the provided context to answer the question as accurately and concisely as possible. "
    "If the context does not contain enough information, answer based on your best judgement "
    "but keep the answer short. Reply with only the key fact or phrase — do not wrap it in "
    "a full sentence. For example, if the answer is a name, return just the name."
)

JUDGE_SYSTEM_PROMPT = (
    "You are an answer quality judge. Given a question, a ground truth answer, and a predicted "
    "answer, decide whether the predicted answer is correct.\n"
    "A predicted answer is correct if it conveys the same core information as the ground truth, "
    "even if phrased differently. Minor differences in wording, capitalisation, or sentence "
    "structure should not count as wrong.\n"
    "Reply with a single JSON object and nothing else:\n"
    "{\"correct\": true, \"reason\": \"<one sentence>\"}\n"
    "or\n"
    "{\"correct\": false, \"reason\": \"<one sentence>\"}"
)

EMPTY_GRAPH = lambda: {
    "nodes": {},
    "edges": {},
    "vector_index": {},
    "attribute_index": {},
    "sessions": {},
}


# ── LLM answer generation ──────────────────────────────────────────────────────


def llm_judge(question: str, ground_truth: str, predicted: str) -> dict:
    """Ask the LLM whether predicted conveys the same information as ground_truth."""
    prompt = f"Question: {question}\nGround truth: {ground_truth}\nPredicted: {predicted}"
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"] or ""
        result = json.loads(raw)
        return {"judge_correct": int(bool(result.get("correct"))), "judge_reason": result.get("reason", "")}
    except Exception as exc:
        return {"judge_correct": 0, "judge_reason": f"judge error: {exc}"}


def generate_answer(question: str, context: str) -> str:
    user_content = question
    if context:
        user_content = f"Context from memory:\n{context}\n\nQuestion: {question}"

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


# ── Scoring ────────────────────────────────────────────────────────────────────

_ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in _ARTICLES]


def token_scores(prediction: str, ground_truth: str) -> dict:
    pred_tokens = normalize(prediction)
    gt_tokens = normalize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)

    overlap = sum((pred_counts & gt_counts).values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"f1": f1, "precision": precision, "recall": recall}


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize(prediction) == normalize(ground_truth))


# ── Per-question pipeline ──────────────────────────────────────────────────────


def run_question(item: dict, verbose: bool = False) -> dict:
    """
    Ingest all user turns for this question, retrieve context, generate an
    answer, and return a result dict with scores.
    """
    import time

    def elapsed(t0):
        return f"{time.time() - t0:.1f}s"

    graph = EMPTY_GRAPH()
    q_start = time.time()

    sessions = item["haystack_sessions"]
    total_user_turns = sum(1 for s in sessions for t in s if t["role"] == "user")
    turn_num = 0

    for s_idx, session in enumerate(sessions):
        for turn in session:
            if turn["role"] != "user":
                continue
            turn_num += 1
            preview = turn["content"][:60].replace("\n", " ")
            t0 = time.time()
            print(f"  ingesting turn {turn_num}/{total_user_turns}: {preview!r}", flush=True)
            _, graph = run_in_memory(turn["content"], graph)
            print(f"    done in {elapsed(t0)}", flush=True)

    # Save the full graph for inspection
    graphs_dir = Path("graphs")
    graphs_dir.mkdir(exist_ok=True)
    graph_path = graphs_dir / f"{item['question_id']}.json"
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"  graph saved to {graph_path} ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)", flush=True)

    t0 = time.time()
    print(f"  retrieving context for question...", flush=True)
    from graph.ingest import _pipeline
    context, _ = _pipeline(item["question"], graph)
    context_node_count = context.count("\n[") if context else 0
    print(f"  context: {context_node_count} nodes retrieved ({elapsed(t0)})", flush=True)
    if context:
        print(f"\n  --- Retrieved subgraph ---\n{context}\n  ---", flush=True)

    t0 = time.time()
    print(f"  generating answer...", flush=True)
    predicted = generate_answer(item["question"], context)
    print(f"  answer generated ({elapsed(t0)})", flush=True)
    ground_truth = item["answer"]

    scores = token_scores(predicted, ground_truth)
    scores.update(llm_judge(item["question"], ground_truth, predicted))

    result = {
        "question_id": item["question_id"],
        "question_type": item.get("question_type", "unknown"),
        "question": item["question"],
        "ground_truth": ground_truth,
        "predicted": predicted,
        "graph_path": str(graph_path),
        "time_seconds": round(time.time() - q_start, 1),
        "turns_ingested": total_user_turns,
        "context_nodes": context.count("\n[") if context else 0,
        **scores,
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"ID:         {result['question_id']}")
        print(f"Type:       {result['question_type']}")
        print(f"Question:   {result['question']}")
        print(f"Expected:   {ground_truth}")
        print(f"Predicted:  {predicted}")
        print(f"F1: {scores['f1']:.3f}  P: {scores['precision']:.3f}  R: {scores['recall']:.3f}  EM: {scores['exact_match']}")

    return result


# ── Aggregate reporting ────────────────────────────────────────────────────────

METRICS = ["f1", "precision", "recall", "judge_correct"]


def _build_summary(results: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    def avg(items, key):
        vals = [i[key] for i in items if key in i]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "total_questions": len(results),
        "overall": {m: avg(results, m) for m in METRICS},
        "by_type": {
            qtype: {"count": len(items), **{m: avg(items, m) for m in METRICS}}
            for qtype, items in sorted(by_type.items())
        },
    }


def _save_output(results: list[dict], output_path: str) -> None:
    payload = {
        "summary": _build_summary(results),
        "questions": results,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def report(results: list[dict]) -> None:
    summary = _build_summary(results)
    by_type = summary["by_type"]

    print(f"\n{'='*60}")
    print(f"RESULTS  ({summary['total_questions']} questions)")
    print(f"{'='*60}")
    print(f"{'Metric':<16} {'Overall':>8}", end="")
    for qtype in by_type:
        print(f"  {qtype[:14]:>14}", end="")
    print()
    print("-" * (16 + 9 + 16 * len(by_type)))

    for metric in METRICS:
        print(f"{metric:<16} {summary['overall'][metric]:>8.3f}", end="")
        for qtype in by_type:
            print(f"  {by_type[qtype][metric]:>14.3f}", end="")
        print()

    print(f"\nQuestion type counts:")
    for qtype, vals in by_type.items():
        print(f"  {qtype:<35} {vals['count']}")


# ── Entry point ────────────────────────────────────────────────────────────────


def _subsample(data: list[dict], per_type: int, seed: int = 42) -> list[dict]:
    """Sample up to per_type questions from each question_type, preserving order within types."""
    import random
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        by_type[item.get("question_type", "unknown")].append(item)
    sampled = []
    for qtype in sorted(by_type):
        pool = by_type[qtype]
        chosen = rng.sample(pool, min(per_type, len(pool)))
        sampled.extend(chosen)
        print(f"  {qtype:<35} {len(chosen)}/{len(pool)} sampled")
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Max total questions (applied after subsampling)")
    parser.add_argument("--per-type", type=int, default=None, help="Sample this many questions per question_type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling (default: 42)")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to dataset JSON")
    parser.add_argument("--output", type=str, default=None, help="Save per-question results to JSON")
    parser.add_argument("--verbose", action="store_true", help="Print each question result")
    args = parser.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))

    if args.per_type:
        print(f"Subsampling {args.per_type} questions per type (seed={args.seed}):")
        data = _subsample(data, args.per_type, seed=args.seed)

    if args.limit:
        data = data[: args.limit]

    print(f"Evaluating {len(data)} questions...")

    import time
    results = []
    for i, item in enumerate(data):
        print(f"\n[{i+1}/{len(data)}] {item['question_id']}  type={item.get('question_type', '?')}", flush=True)
        print(f"  Q: {item['question']}", flush=True)
        q_start = time.time()
        try:
            result = run_question(item, verbose=args.verbose)
            results.append(result)
            print(f"  => F1={result['f1']:.3f}  P={result['precision']:.3f}  R={result['recall']:.3f}  Judge={'✓' if result['judge_correct'] else '✗'}  ({result['time_seconds']}s)", flush=True)
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
                "graph_path": str(Path("graphs") / f"{item['question_id']}.json"),
                "time_seconds": elapsed_s,
                "error": str(exc),
                "f1": 0.0, "precision": 0.0, "recall": 0.0, "exact_match": 0,
            })

        if args.output:
            _save_output(results, args.output)

    report(results)
    if args.output:
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
