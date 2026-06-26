from typing import TypedDict

import requests
from langgraph.graph import END, StateGraph

from graph.embedder import get_embedding
from graph.extractor import extract, merge_extraction_into_graph
from graph.forgetting import run_forgetting, should_forget
from graph.graph_manager import increment_interactions, load_graph, save_graph
from graph.retriever import extract_subgraph, find_entry_nodes, format_subgraph_for_prompt

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "gemma2"


class GraphState(TypedDict):
    user_input: str
    retrieved_subgraph: dict
    llm_response: str
    extracted_nodes: dict
    kg_path: str


def retrieve_node(state: GraphState) -> dict:
    graph = load_graph(state["kg_path"])
    query_embedding = get_embedding(state["user_input"])
    entry_node_ids = find_entry_nodes(graph, query_embedding, k=3)
    if not entry_node_ids:
        subgraph = {"nodes": {}, "edges": []}
    else:
        subgraph = extract_subgraph(graph, entry_node_ids)
    return {"retrieved_subgraph": subgraph}


def generate_node(state: GraphState) -> dict:
    subgraph_context = format_subgraph_for_prompt(state["retrieved_subgraph"])
    system_content = (
        "You are a helpful conversational assistant with access to a knowledge graph "
        "that captures the user's past conversations and relationships. "
        "The graph context below contains nodes (entities) and edges (relationships). "
        "When answering, draw on ALL nodes and edges shown — not just the most "
        "prominent ones. Every fact in the graph is equally available to you. "
        "Use the exact labels and attribute values from the graph rather than "
        "paraphrasing or summarising them.\n\n"
        + subgraph_context
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": state["user_input"]},
        ],
        "stream": False,
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return {"llm_response": response.json()["message"]["content"]}


def extract_node(state: GraphState) -> dict:
    graph = load_graph(state["kg_path"])
    extraction = extract(state["user_input"])
    label_to_id = merge_extraction_into_graph(graph, extraction, state["user_input"])
    increment_interactions(graph)
    save_graph(graph, state["kg_path"])
    return {"extracted_nodes": label_to_id}


def forget_node(state: GraphState) -> dict:
    graph = load_graph(state["kg_path"])
    if should_forget(graph):
        graph = run_forgetting(graph)
        save_graph(graph, state["kg_path"])
    return {}


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("forget", forget_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "extract")
    workflow.add_edge("extract", "forget")
    workflow.add_edge("forget", END)
    return workflow.compile()


app = build_graph()
