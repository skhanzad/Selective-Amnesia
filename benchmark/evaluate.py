import argparse
import json
import string
import sys
from collections import Counter
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.embedder import get_embedding
from graph.graph_manager import load_graph
from graph.retriever import extract_subgraph, find_entry_nodes, format_subgraph_for_prompt

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "gemma2"

CATEGORY_NAMES = {
    1: "Multi-hop",
    2: "Single-hop",
    3: "Temporal",
    4: "Open-domain",
    5: "Adversarial",
}

_NO_INFO_PHRASES = [
    "no information",
    "don't have",
    "do not have",
    "don't know",
    "do not know",
    "cannot find",
    "can't find",
    "no record",
    "not mentioned",
    "not available",
    "not found",
    "i'm not sure",
    "i am not sure",
    "unable to find",
    "no relevant",
]


def _normalize(text: str) -> list[str]:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction)
    gold_tokens = _normalize(str(ground_truth))
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def adversarial_score(prediction: str) -> float:
    pred_lower = prediction.lower()
    return 1.0 if any(phrase in pred_lower for phrase in _NO_INFO_PHRASES) else 0.0


def score_qa(prediction: str, qa: dict) -> float:
    category = qa.get("category", 1)
    if category == 5:
        return adversarial_score(prediction)
    return token_f1(prediction, qa.get("answer", ""))


def generate_answer(question: str, kg_path: str) -> str:
    graph = load_graph(kg_path)
    if not graph["nodes"]:
        return "no information available"

    query_embedding = get_embedding(question)
    entry_node_ids = find_entry_nodes(graph, query_embedding, k=3)
    if not entry_node_ids:
        subgraph = {"nodes": {}, "edges": []}
    else:
        subgraph = extract_subgraph(graph, entry_node_ids)

    subgraph_context = format_subgraph_for_prompt(subgraph)
    system_content = (
        "You are a helpful assistant with access to a knowledge graph built from "
        "a long conversation between two people. "
        "Answer the question concisely — a few words or a short phrase. "
        "Use the exact labels and attribute values from the graph; do not paraphrase "
        "or summarise them. Base your answer only on the knowledge graph context below. "
        "If the answer cannot be found in the graph, respond with exactly: "
        "'no information available'\n\n"
        + subgraph_context
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ],
        "stream": False,
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _qa_covered_by_sessions(qa: dict, max_sessions: int) -> bool:
    """True if all evidence dia_ids for this QA fall within the ingested sessions.

    Evidence items look like "D3:7" — the number after 'D' is the session index.
    """
    for eid in qa.get("evidence", []):
        try:
            session_num = int(eid.lstrip("D").split(":")[0])
        except (ValueError, IndexError):
            return False
        if session_num > max_sessions:
            return False
    return True


def evaluate_sample(sample: dict, kg_path: str, max_sessions: int | None = None) -> list[dict]:
    all_qa = sample.get("qa", [])
    if max_sessions is not None:
        qa_pairs = [qa for qa in all_qa if _qa_covered_by_sessions(qa, max_sessions)]
    else:
        qa_pairs = all_qa
    results = []

    for qa in tqdm(qa_pairs, desc="  QA", leave=False):
        question = qa["question"]
        category = qa.get("category", 1)
        gold = qa.get("answer", qa.get("adversarial_answer", ""))

        try:
            predicted = generate_answer(question, kg_path)
        except Exception as exc:
            print(f"  [eval] WARNING: generation failed — {exc}")
            predicted = ""

        sc = score_qa(predicted, qa)
        results.append(
            {
                "question": question,
                "gold": str(gold),
                "predicted": predicted,
                "category": category,
                "score": sc,
            }
        )

    return results


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KG-RAG on LoCoMo")
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples to evaluate (default: 1)",
    )
    parser.add_argument("--kg_dir", default="data/benchmarks")
    parser.add_argument(
        "--results",
        default="data/benchmarks/results.json",
        help="Path to save the results JSON",
    )
    parser.add_argument(
        "--max_sessions",
        type=int,
        default=None,
        help="Only evaluate QA pairs whose evidence is fully within the first N sessions",
    )
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset[: args.samples]
    print(f"Evaluating {len(samples)} sample(s)  (kg_dir={args.kg_dir})")

    all_results: list[dict] = []
    per_sample_summaries: list[dict] = []

    for i, sample in enumerate(samples):
        sample_id = sample.get("sample_id", i)
        kg_path = f"{args.kg_dir}/kg_{sample_id}.json"

        if not Path(kg_path).exists():
            print(
                f"\n[{i + 1}/{len(samples)}] SKIP {sample_id} — "
                f"no graph at {kg_path}. Run ingest.py first."
            )
            continue

        all_qa = sample.get("qa", [])
        if args.max_sessions is not None:
            filtered_count = sum(1 for qa in all_qa if _qa_covered_by_sessions(qa, args.max_sessions))
        else:
            filtered_count = len(all_qa)
        print(
            f"\n[{i + 1}/{len(samples)}] {sample_id}  "
            f"({filtered_count}/{len(all_qa)} questions after session filter)"
        )

        results = evaluate_sample(sample, kg_path, max_sessions=args.max_sessions)
        for r in results:
            r["sample_id"] = sample_id
        all_results.extend(results)

        # Per-sample summary
        by_cat: dict[int, list[float]] = {}
        for r in results:
            by_cat.setdefault(r["category"], []).append(r["score"])

        summary: dict = {"sample_id": sample_id}
        for cat in sorted(by_cat):
            avg = _avg(by_cat[cat])
            summary[f"cat_{cat}_f1"] = round(avg, 4)
            summary[f"cat_{cat}_n"] = len(by_cat[cat])
            label = CATEGORY_NAMES.get(cat, f"Cat{cat}")
            print(f"  Cat {cat} ({label:12s}): F1={avg:.4f}  n={len(by_cat[cat])}")

        overall = _avg([r["score"] for r in results])
        summary["overall_f1"] = round(overall, 4)
        print(f"  Overall F1: {overall:.4f}")
        per_sample_summaries.append(summary)

    print("\n" + "=" * 50)
    print("AGGREGATE RESULTS")
    print("=" * 50)

    agg_by_cat: dict[int, list[float]] = {}
    for r in all_results:
        agg_by_cat.setdefault(r["category"], []).append(r["score"])

    aggregate: dict = {}
    for cat in sorted(agg_by_cat):
        avg = _avg(agg_by_cat[cat])
        aggregate[f"cat_{cat}_f1"] = round(avg, 4)
        label = CATEGORY_NAMES.get(cat, f"Cat{cat}")
        print(f"  Cat {cat} ({label:12s}): F1={avg:.4f}  n={len(agg_by_cat[cat])}")

    all_scores = [r["score"] for r in all_results]
    aggregate["overall_f1"] = round(_avg(all_scores), 4)
    print(f"  Overall F1 : {aggregate['overall_f1']:.4f}")

    print("\n  Reference baselines (LoCoMo paper, token F1):")
    print("    Mistral-7B  : 0.139")
    print("    GPT-3.5     : 0.245")
    print("    GPT-4       : 0.321")
    print("    Human ceil  : 0.879")

    output = {
        "aggregate": aggregate,
        "per_sample": per_sample_summaries,
        "predictions": all_results,
    }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved -> {args.results}")


if __name__ == "__main__":
    main()
