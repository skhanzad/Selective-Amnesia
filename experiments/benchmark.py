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
import os
import re
import string
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from graph.forget import DEFAULT_CONFIG as DEFAULT_FORGET_CONFIG
from graph.forget import run_forgetting_pass
from graph.graph_manager import empty_graph
from graph.ingest import run_in_memory

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_PATH = Path("data/longmemeval_oracle.json")
LLM_MODEL = "gpt-4o-mini"

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

_client = OpenAI(api_key=os.environ["OPEN_AI_API_KEY"], max_retries=8)

ANSWER_SYSTEM_PROMPT = (
    "You are a personal memory assistant. The context block provided to you IS your memory "
    "of past conversations with this user — treat every fact in it as something you personally "
    "know and remember. Never say you lack access to personal information, never say you cannot "
    "recall past conversations, and never ask the user to check elsewhere. If the context "
    "contains the answer, state it directly and concisely. Reply with only the key fact or "
    "phrase — no full sentences, no preamble, no caveats. If the answer is a name, return "
    "just the name. If the answer is a number, return just the number. If the context does "
    "not contain enough information, give your best short guess based on what is available."
)

JUDGE_SYSTEM_PROMPT = (
    "You are an answer quality judge. Given a question, a ground truth answer, and a predicted "
    "answer, decide whether the predicted answer is correct.\n"
    "A predicted answer is correct if it conveys the same core information as the ground truth, "
    "even if phrased differently. Minor differences in wording, capitalisation, or sentence "
    "structure should not count as wrong.\n"
    "Reply with a single JSON object and nothing else:\n"
    '{"correct": true, "reason": "<one sentence>"}\n'
    "or\n"
    '{"correct": false, "reason": "<one sentence>"}'
)

# ── LLM answer generation ──────────────────────────────────────────────────────


def llm_judge(question: str, ground_truth: str, predicted: str) -> dict:
    """Ask the LLM whether predicted conveys the same information as ground_truth."""
    prompt = (
        f"Question: {question}\nGround truth: {ground_truth}\nPredicted: {predicted}"
    )
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        result = json.loads(raw)
        return {
            "judge_correct": int(bool(result.get("correct"))),
            "judge_reason": result.get("reason", ""),
        }
    except Exception as exc:
        return {"judge_correct": 0, "judge_reason": f"judge error: {exc}"}


def generate_answer(question: str, context: str) -> str:
    user_content = question
    if context:
        user_content = f"Context from memory:\n{context}\n\nQuestion: {question}"

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ── Scoring ────────────────────────────────────────────────────────────────────

_ARTICLES = {"a", "an", "the"}


def normalize(text) -> list[str]:
    text = str(text).lower()
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
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {"f1": f1, "precision": precision, "recall": recall}


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize(prediction) == normalize(ground_truth))


# ── Date parsing ──────────────────────────────────────────────────────────────


def _parse_haystack_date(date_str: str) -> datetime:
    """Parse '2023/04/10 (Mon) 17:50' → UTC datetime."""
    cleaned = re.sub(r"\s*\([^)]+\)", "", date_str).strip()
    return datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)


# ── Per-question pipeline ──────────────────────────────────────────────────────


def run_question(
    item: dict, verbose: bool = False, forget_config: dict | None = None
) -> dict:
    """
    Ingest all user turns for this question, retrieve context, generate an
    answer, and return a result dict with scores.
    """
    import time

    def elapsed(t0):
        return f"{time.time() - t0:.1f}s"

    graph = empty_graph()
    q_start = time.time()

    sessions = item["haystack_sessions"]
    haystack_dates = item.get("haystack_dates", [])
    question_date = (
        _parse_haystack_date(item["question_date"])
        if item.get("question_date")
        else None
    )
    total_turns = sum(
        1 for s in sessions for t in s if t["role"] in ("user", "assistant")
    )

    # Extract and ingest each turn sequentially, tagging each turn with its
    # session's haystack date so recency decay uses conversation time, not
    # wall-clock time.
    print(f"  ingesting {total_turns} turns (context-aware extraction)...", flush=True)
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
                f"  ingesting turn {turn_index + 1}/{total_turns} [{turn['role']}]: {preview!r}",
                flush=True,
            )
            tagged = f"[Role: {turn['role']}]\n{turn['content']}"
            _, graph = run_in_memory(
                tagged,
                graph,
                ingest_only=True,
                forget_config=forget_config,
                session_date=session_date,
            )
            print(f"    done in {elapsed(t0)}", flush=True)
            turn_index += 1

    # Final forgetting pass (captures end-of-ingestion state regardless of turn triggers)
    forget_stats: dict = {
        "scored": 0,
        "pruned": 0,
        "compressed": 0,
        "flagged": 0,
        "edges_removed": 0,
    }
    if forget_config is not None:
        cfg = {**DEFAULT_FORGET_CONFIG, **forget_config}
        forget_stats = run_forgetting_pass(graph, cfg, now=question_date)
        print(
            f"  [forget final] nodes_before={forget_stats['scored']} pruned={forget_stats['pruned']} "
            f"compressed={forget_stats['compressed']} flagged={forget_stats['flagged']}",
            flush=True,
        )

    # Save the full graph for inspection
    graphs_dir = Path("graphs")
    graphs_dir.mkdir(exist_ok=True)
    graph_path = graphs_dir / f"{item['question_id']}.json"
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(
        f"  graph saved to {graph_path} ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)",
        flush=True,
    )

    t0 = time.time()
    print(f"  retrieving context for question...", flush=True)
    context, _ = run_in_memory(item["question"], graph, session_date=question_date)
    context_node_count = context.count("\n[") if context else 0
    print(
        f"  context: {context_node_count} nodes retrieved ({elapsed(t0)})", flush=True
    )
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
        "turns_ingested": total_turns,
        "context_nodes": context.count("\n[") if context else 0,
        "forget_pruned": forget_stats["pruned"],
        "forget_compressed": forget_stats["compressed"],
        "forget_flagged": forget_stats["flagged"],
        **scores,
    }

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"ID:         {result['question_id']}")
        print(f"Type:       {result['question_type']}")
        print(f"Question:   {result['question']}")
        print(f"Expected:   {ground_truth}")
        print(f"Predicted:  {predicted}")
        print(
            f"F1: {scores['f1']:.3f}  P: {scores['precision']:.3f}  R: {scores['recall']:.3f}  EM: {scores['exact_match']}"
        )

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

    print(f"\n{'=' * 60}")
    print(f"RESULTS  ({summary['total_questions']} questions)")
    print(f"{'=' * 60}")
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max total questions (applied after subsampling)",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=None,
        help="Sample this many questions per question_type",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for subsampling (default: 42)"
    )
    parser.add_argument(
        "--data", type=str, default=str(DATA_PATH), help="Path to dataset JSON"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print each question result"
    )
    parser.add_argument(
        "--forget",
        action="store_true",
        help="Enable forgetting module during ingestion",
    )
    parser.add_argument(
        "--forget-every-n",
        type=int,
        default=50,
        help="Trigger forgetting every N ingested turns (default: 50)",
    )
    parser.add_argument(
        "--forget-threshold",
        type=float,
        default=0.15,
        help="Importance score below which nodes are pruned (default: 0.15)",
    )
    parser.add_argument(
        "--forget-node-count",
        type=int,
        default=300,
        help="Trigger forgetting when graph reaches this many nodes (default: 300)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        dest="question_type",
        help="Only run questions of this type (e.g. knowledge-update, multi-session, temporal-reasoning)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from results.json: keep clean results, drop errored entries, continue from first error",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file path (default: results.json)",
    )
    args = parser.parse_args()

    output_path = args.output
    forget_config: dict | None = None
    if args.forget:
        forget_config = {
            "trigger_every_n_turns": args.forget_every_n,
            "pruning_threshold": args.forget_threshold,
            "trigger_node_count": args.forget_node_count,
        }
        print(
            f"Forgetting enabled: every_n={args.forget_every_n} "
            f"threshold={args.forget_threshold} node_count={args.forget_node_count}"
        )

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))

    # Resume: load existing clean results and skip their question IDs
    existing_results: list[dict] = []
    if args.resume and Path(output_path).exists():
        prior = json.loads(Path(output_path).read_text(encoding="utf-8"))
        existing_results = [r for r in prior.get("questions", []) if "error" not in r]
        done_ids = {r["question_id"] for r in existing_results}
        original_count = len(prior.get("questions", []))
        data = [q for q in data if q["question_id"] not in done_ids]
        print(
            f"Resuming: kept {len(existing_results)} clean results, "
            f"dropped {original_count - len(existing_results)} errors, "
            f"{len(data)} questions remaining"
        )

    if args.question_type:
        data = [q for q in data if q.get("question_type") == args.question_type]
        print(f"Filtered to type '{args.question_type}': {len(data)} questions")

    if args.per_type:
        print(f"Subsampling {args.per_type} questions per type (seed={args.seed}):")
        data = _subsample(data, args.per_type, seed=args.seed)

    if args.limit:
        data = data[: args.limit]

    print(f"Evaluating {len(data)} questions...")

    import time

    results: list[dict] = list(existing_results)
    for i, item in enumerate(data):
        print(
            f"\n[{i + 1}/{len(data)}] {item['question_id']}  type={item.get('question_type', '?')}",
            flush=True,
        )
        print(f"  Q: {item['question']}", flush=True)
        q_start = time.time()
        try:
            result = run_question(
                item, verbose=args.verbose, forget_config=forget_config
            )
            results.append(result)
            print(
                f"  => F1={result['f1']:.3f}  P={result['precision']:.3f}  R={result['recall']:.3f}  Judge={'✓' if result['judge_correct'] else '✗'}  ({result['time_seconds']}s)",
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
            results.append(
                {
                    "question_id": item["question_id"],
                    "question_type": item.get("question_type", "unknown"),
                    "question": item["question"],
                    "ground_truth": item.get("answer", ""),
                    "predicted": None,
                    "graph_path": str(Path("graphs") / f"{item['question_id']}.json"),
                    "time_seconds": elapsed_s,
                    "error": str(exc),
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "exact_match": 0,
                }
            )

        _save_output(results, output_path)

    report(results)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
