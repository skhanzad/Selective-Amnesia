import json
import re

import requests

from graph.embedder import get_embedding, cosine_similarity
from graph.graph_manager import (
    add_edge,
    add_node,
    find_node_by_label,
    merge_node,
)

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "gemma2"
EMBED_SIM_THRESHOLD = 0.85

_SYSTEM_PROMPT = """\
You are a knowledge graph extractor. Given a user message, extract all entities, \
concepts, and events as a structured JSON object.

## Node classification

**semantic** — a persistent, named entity or concept: any person, organisation, \
location, technology, role, academic domain, or product that could be referenced \
again in a future message.

**episodic** — a specific, time-bound event or experience explicitly described \
in this message (something that happened at a particular time or place).

## Rules

1. Every named entity — person, organisation, location, technology, role, concept, \
or domain — is its own node. Never absorb a named entity into another node or \
collapse distinct entities together.
2. Attributes are reserved for atomic scalar properties that cannot stand alone \
as entities: a numeric value, a date, a single-word category tag. \
If a value could itself have relationships with other entities, it must be a node \
with an edge, not an attribute.
3. When a concrete, time-bound activity is described, extract a dedicated episodic \
node for it and connect all participants, locations, and topics to that event node \
with appropriate edges. Named entities involved remain their own semantic nodes.
3a. Every episodic node MUST include a "when" attribute with the date or time of \
the event. If a session date is provided, resolve relative expressions \
(yesterday, last week, next month, etc.) to absolute dates. \
Example: if session date is "2023-05-07" and the message says "yesterday", \
set "when" to "2023-05-06". If no date can be determined, omit the attribute.
4. If the message carries no meaningful entities, concepts, or events \
(e.g. greetings, filler phrases, acknowledgements), output empty lists: \
{"nodes": [], "edges": []}.
5. Always include "User" (type: semantic) as the speaker node, unless rule 4 applies.
5a. Words that describe how two entities relate to each other (e.g. "friend", \
"colleague", "boss", "rival", "mentor") are not entities — they describe the \
nature of an edge. Express them as a relation label on an edge or as an attribute \
of the connecting node, never as a standalone node.
6. Relation labels must be UPPER_SNAKE_CASE verb phrases.
7. If the same entity appears under different surface forms, use one canonical label.
8. Never use abbreviations or acronyms as node labels. Always expand to the full name \
(e.g. "Computer Science" not "CS", "Los Angeles" not "LA", \
"University of California Los Angeles" not "UCLA").
9. Output ONLY valid JSON — no prose, no markdown fences.

## Output schema

{"nodes": [{"label": "<string>", "type": "semantic|episodic", "attributes": {}}],
 "edges": [{"source": "<label>", "target": "<label>", "relation": "<RELATION>"}]}

## Examples

### Example 1 — semantic entities only

Input: "My colleague Bob works at DeepMind in London as a research scientist."

Output:
{"nodes": [
  {"label": "User",              "type": "semantic", "attributes": {}},
  {"label": "Bob",               "type": "semantic", "attributes": {}},
  {"label": "DeepMind",          "type": "semantic", "attributes": {}},
  {"label": "London",            "type": "semantic", "attributes": {}},
  {"label": "Research Scientist", "type": "semantic", "attributes": {}}
 ],
 "edges": [
  {"source": "User", "target": "Bob",               "relation": "KNOWS"},
  {"source": "Bob",  "target": "DeepMind",          "relation": "WORKS_AT"},
  {"source": "Bob",  "target": "Research Scientist", "relation": "HAS_ROLE"},
  {"source": "DeepMind", "target": "London",        "relation": "LOCATED_IN"}
 ]
}

### Example 2 — semantic and episodic nodes together (with session date resolution)

Session date: 10:00 am on 17 January, 2024
Message to extract from: User: Yesterday I attended a guest lecture on computer vision at McGill University.

Output:
{"nodes": [
  {"label": "User",                                      "type": "semantic", "attributes": {}},
  {"label": "Computer Vision",                           "type": "semantic", "attributes": {}},
  {"label": "McGill University",                         "type": "semantic", "attributes": {}},
  {"label": "Attended guest lecture on computer vision", "type": "episodic", "attributes": {"when": "16 January 2024"}}
 ],
 "edges": [
  {"source": "User", "target": "Attended guest lecture on computer vision", "relation": "PARTICIPATED_IN"},
  {"source": "Attended guest lecture on computer vision", "target": "Computer Vision",   "relation": "COVERED_TOPIC"},
  {"source": "Attended guest lecture on computer vision", "target": "McGill University", "relation": "TOOK_PLACE_AT"}
 ]
}

### Example 3 — no meaningful content

Input: "Okay, sounds good!"

Output:
{"nodes": [], "edges": []}"""


def _node_text(label: str, attributes: dict, sentence_context: str = "") -> str:
    """Build a descriptor string for embedding.

    Appends the originating sentence so that short labels like 'Pinecone' or
    'Queen Street' are embedded in their actual semantic context rather than as
    bare tokens. nomic-embed-text returns near-identical vectors for all short
    strings in isolation; the surrounding sentence breaks that degeneracy.
    The type tag is still omitted — it creates a shared cluster center.
    """
    parts = [label]
    if attributes:
        attr_str = ", ".join(f"{k}: {v}" for k, v in attributes.items())
        parts.append(attr_str)
    result = ": ".join(parts) if len(parts) > 1 else parts[0]
    if sentence_context:
        result = f"{result} — {sentence_context}"
    return result


def _strip_fences(text: str) -> str:
    """Extract the JSON object from LLM output.

    Handles three Gemma2 quirks:
    1. Markdown fences (```json ... ```) wrapping the JSON.
    2. Trailing prose or a second fence appended after the closing brace.
    3. Trailing commas on the last item of an array or object, which
       Python's json module rejects.
    """
    text = text.strip()
    # Strip any leading fence so the JSON object starts at '{'
    text = re.sub(r"^```(?:json)?\s*", "", text)
    # Extract the outermost {...} by tracking brace depth, discarding
    # everything before the first '{' and after the matching '}'.
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    text = text[start : i + 1]
                    break
    # Remove trailing commas that Python's json module rejects.
    text = re.sub(r",\s*([\]\}])", r"\1", text)
    return text.strip()


def extract(user_input: str, session_date: str = "") -> dict:
    """Call Gemma2 via Ollama and return parsed {nodes, edges} extraction.

    session_date: ISO date string for the session (e.g. "2023-05-07T10:00:00").
    When provided it is injected into the prompt so relative time expressions
    (yesterday, next week, etc.) can be resolved to absolute dates.
    """
    if session_date:
        user_content = f"Session date: {session_date}\nMessage to extract from: {user_input}"
    else:
        user_content = f"Message to extract from: {user_input}"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"]
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[extractor] WARNING: could not parse LLM response as JSON. Raw:\n{raw!r}")
        return {"nodes": [], "edges": []}


_ALIAS_PROMPT = """\
You are a knowledge graph deduplication assistant.

Given a list of existing node labels and a list of newly extracted labels, \
identify which existing label each new label is an abbreviation, acronym, or \
alternate surface form of — if any.

A match is valid ONLY when the two labels refer to the IDENTICAL real-world entity \
and the difference is purely a naming convention:
- Acronym expansion: "UCLA" → "University of California Los Angeles", \
"LA" → "Los Angeles", "CS" → "Computer Science", "USA" → "United States of America"
- Alternate spelling or capitalisation: "New York City" → "NYC"
- Known alias for the same thing: "Google" → "Alphabet"

A match is INVALID when the labels are merely related, similar, or in the same domain:
- "Software Engineer" and "Senior Engineer" are DIFFERENT roles — do NOT match them.
- "Computer Science" and "Software Engineer" are DIFFERENT concepts — do NOT match them.
- "UCLA" and "Los Angeles" are DIFFERENT entities (a university vs a city) — do NOT match them.
- "Hackathon" and "UCLA" are DIFFERENT entities — do NOT match them.

A match is INVALID when one label is a component or substring of the other, even if \
they share words. Shared words alone do not make two labels the same entity — they may \
simply share a common term. For example: "Los Angeles" ≠ "Los Angeles Dodgers", \
"Apple" ≠ "Apple Records", "Amazon" ≠ "Amazon River", "Stanford" ≠ "Stanford Hospital". \
Apply this principle universally — do not enumerate exceptions, reason from it.

When in doubt, output null. A false negative (missing a merge) is far less harmful \
than a false positive (merging distinct entities).

Rules:
- Only match if you are certain the labels are alternate names for the identical entity.
- Every new label must appear as a key in the output, mapped to a matching existing \
  label string or null.
- Output ONLY valid JSON — no prose, no markdown fences.

Output schema: {"<new_label>": "<matched_existing_label_or_null>", ...}"""


def _resolve_aliases(new_labels: list[str], existing_labels: list[str]) -> dict[str, str | None]:
    if not new_labels or not existing_labels:
        return {}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _ALIAS_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Existing labels: {json.dumps(existing_labels)}\n"
                    f"New labels: {json.dumps(new_labels)}"
                ),
            },
        ],
        "stream": False,
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"]
        result = json.loads(_strip_fences(raw))
        existing_lower = {l.lower(): l for l in existing_labels}
        resolved: dict[str, str | None] = {}
        for new_label, match in result.items():
            if match is None:
                resolved[new_label] = None
            elif match.lower() in existing_lower:
                resolved[new_label] = existing_lower[match.lower()]
            else:
                resolved[new_label] = None
        return resolved
    except Exception:
        return {}


def merge_extraction_into_graph(
    graph: dict, extraction: dict, user_input: str = "", skip_aliases: bool = False
) -> dict:
    label_to_id: dict[str, str] = {}

    existing_embeddings: list[tuple[str, list[float]]] = [
        (node_id, node["embedding"])
        for node_id, node in graph["nodes"].items()
        if node.get("embedding")
    ]

    # Stage 0 — LLM alias resolution (batched for the whole extraction)
    new_labels = [n["label"] for n in extraction.get("nodes", [])]
    existing_labels = [n["label"] for n in graph["nodes"].values()]
    alias_map = {} if skip_aliases else _resolve_aliases(new_labels, existing_labels)

    for node_spec in extraction.get("nodes", []):
        label: str = node_spec["label"]
        ntype: str = node_spec.get("type", "semantic")
        attributes: dict = node_spec.get("attributes", {})

        # Stage 0 — LLM alias resolution
        alias_target = alias_map.get(label)
        if alias_target is not None:
            existing_id = find_node_by_label(graph, alias_target)
            if existing_id is not None:
                merge_node(graph, existing_id, attributes)
                label_to_id[label] = existing_id
                continue

        # Stage 1 — exact label match
        existing_id = find_node_by_label(graph, label)
        if existing_id is not None:
            merge_node(graph, existing_id, attributes)
            label_to_id[label] = existing_id
            continue

        # Stage 2 — embedding similarity with sentence context.
        embed_text = _node_text(label, attributes, user_input)
        label_embedding = get_embedding(embed_text)
        best_id: str | None = None
        best_sim: float = 0.0
        for node_id, emb in existing_embeddings:
            sim = cosine_similarity(label_embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_id = node_id

        if best_id is not None and best_sim > EMBED_SIM_THRESHOLD:
            merge_node(graph, best_id, attributes)
            label_to_id[label] = best_id
        else:
            new_id = add_node(graph, label, ntype, attributes, label_embedding)
            label_to_id[label] = new_id

    for edge_spec in extraction.get("edges", []):
        src_label: str = edge_spec["source"]
        tgt_label: str = edge_spec["target"]
        relation: str = edge_spec["relation"]

        src_id = label_to_id.get(src_label)
        tgt_id = label_to_id.get(tgt_label)
        if src_id and tgt_id:
            add_edge(graph, src_id, tgt_id, relation)

    return label_to_id