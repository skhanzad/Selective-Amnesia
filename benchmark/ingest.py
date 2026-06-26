import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.extractor import extract, merge_extraction_into_graph
from graph.graph_manager import increment_interactions, load_graph, save_graph


def _session_keys_ordered(conv: dict) -> list[str]:
    return sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )


def ingest_sample(
    sample: dict, kg_path: str, max_sessions: int | None = None,
    skip_aliases: bool = False, verbose: bool = False
) -> None:
    conv = sample["conversation"]
    graph = load_graph("/nonexistent/__fresh__")

    session_keys = _session_keys_ordered(conv)
    if max_sessions is not None:
        session_keys = session_keys[:max_sessions]

    all_turns: list[tuple[str, dict]] = []
    for sk in session_keys:
        date_str = conv.get(f"{sk}_date_time", "")
        for turn in conv[sk]:
            all_turns.append((date_str, turn))

    for _date, turn in tqdm(all_turns, desc="  turns", leave=False, disable=not verbose):
        text = turn.get("text", "").strip()
        if not text:
            continue
        speaker = turn.get("speaker", "")
        utterance = f"{speaker}: {text}" if speaker else text
        try:
            extraction = extract(utterance, session_date=_date)
            merge_extraction_into_graph(graph, extraction, utterance, skip_aliases=skip_aliases)
            increment_interactions(graph)
        except Exception as exc:
            print(
                f"  [ingest] WARNING: skipping turn {turn.get('dia_id', '?')}: {exc}"
            )

    save_graph(graph, kg_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest LoCoMo samples into KGs")
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="How many samples to ingest (default: all 10)",
    )
    parser.add_argument("--out_dir", default="data/benchmarks")
    parser.add_argument("--verbose", action="store_true", help="Show per-turn progress bars")
    parser.add_argument(
        "--max_sessions",
        type=int,
        default=None,
        help="Only ingest the first N sessions per sample (default: all)",
    )
    parser.add_argument(
        "--skip_aliases",
        action="store_true",
        help="Skip LLM alias resolution (Stage 0) — ~2x faster ingest, minor quality tradeoff",
    )
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset[: args.samples] if args.samples is not None else dataset
    print(f"Ingesting {len(samples)} sample(s) -> {args.out_dir}/")

    for i, sample in enumerate(samples):
        sample_id = sample.get("sample_id", i)
        kg_path = f"{args.out_dir}/kg_{sample_id}.json"
        num_sessions = len(_session_keys_ordered(sample["conversation"]))
        effective_sessions = min(num_sessions, args.max_sessions) if args.max_sessions else num_sessions
        effective_turns = sum(
            len(sample["conversation"][sk])
            for sk in _session_keys_ordered(sample["conversation"])[:effective_sessions]
        )
        print(
            f"[{i + 1}/{len(samples)}] {sample_id}  "
            f"({effective_sessions}/{num_sessions} sessions, {effective_turns} turns) -> {kg_path}"
        )
        ingest_sample(sample, kg_path, max_sessions=args.max_sessions, skip_aliases=args.skip_aliases, verbose=args.verbose)
        graph_path = Path(kg_path)
        if graph_path.exists():
            with open(graph_path) as gf:
                g = json.load(gf)
            print(
                f"  Done — {len(g['nodes'])} nodes, {len(g['edges'])} edges, "
                f"{g['metadata']['total_interactions']} interactions"
            )

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
