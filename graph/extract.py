import json
import os
import re
from pathlib import Path

from openai import OpenAI

from graph.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
)

# ── Configuration ─────────────────────────────────────────────────────────────

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

LLM_MODEL = "gpt-4o-mini"
_client = OpenAI(api_key=os.environ["OPEN_AI_API_KEY"], max_retries=8)

# Structured output schema passed to Ollama's format parameter.
# Constrains grammar-level sampling so structural violations are impossible.
_ATTRIBUTE_SCHEMA = {
    "type": "object",
    "additionalProperties": {"type": "string"},
}

_NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "title": {"type": "string"},
        "content": {"type": "string"},
        "attributes": _ATTRIBUTE_SCHEMA,
    },
    "required": ["label", "title", "content", "attributes"],
}

_EDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "target": {"type": "string"},
        "relationship": {"type": "string"},
        "attributes": _ATTRIBUTE_SCHEMA,
    },
    "required": ["source", "target", "relationship", "attributes"],
}

EXTRACTION_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": _NODE_SCHEMA,
        },
        "edges": {
            "type": "array",
            "items": _EDGE_SCHEMA,
        },
    },
    "required": ["nodes", "edges"],
}


def _batch_format_schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "extractions": {
                "type": "array",
                "items": EXTRACTION_FORMAT_SCHEMA,
                "minItems": n,
                "maxItems": n,
            },
        },
        "required": ["extractions"],
    }


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _repair_json(raw: str) -> str:
    """Best-effort repair of common gemma2 JSON failures."""
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)
    repaired = re.sub(r",\s*$", "", repaired.rstrip())
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    return repaired + ("}" * max(open_braces, 0)) + ("]" * max(open_brackets, 0))


def _normalize_extraction(parsed: dict) -> dict:
    """Convert nodes/edges arrays (schema-enforced) to keyed dicts for downstream use."""
    for key in ("nodes", "edges"):
        val = parsed.get(key, [])
        if isinstance(val, list):
            prefix = key.rstrip("s")
            parsed[key] = {f"{prefix}_{i}": item for i, item in enumerate(val)}
    return parsed


def _parse_extraction(raw: str) -> dict:
    """Parse and normalize a single extraction JSON string."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(_repair_json(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned non-JSON output: {exc}\n\nRaw output:\n{raw}"
            ) from exc

    return _normalize_extraction(parsed)


RETRIEVAL_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["label", "title", "content"],
            },
        }
    },
    "required": ["entities"],
}


# ── Single-turn extraction ─────────────────────────────────────────────────────


def extract(user_message: str) -> dict:
    """
    Run the main extraction prompt against the user message and return the
    parsed JSON dict produced by the LLM.

    Raises:
        ValueError: if the response cannot be parsed as JSON.
    """
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    print(f"  [extract] tokens={response.usage.total_tokens}", flush=True)

    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned an empty response")

    return _parse_extraction(content)


def extract_for_retrieval(question: str) -> list[dict]:
    """
    Lightweight extraction for the retrieval path.
    Returns a list of entity dicts: [{label, title, content}, ...].
    These are embedded and searched against the graph vector index to find
    context nodes — no storage side effects.
    """
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {}

    entities = parsed.get("entities", []) if isinstance(parsed, dict) else []
    print(
        f"  [retrieve] tokens={response.usage.total_tokens} entities={[e.get('title') for e in entities]}",
        flush=True,
    )
    return entities


# ── Batch extraction ───────────────────────────────────────────────────────────


def _extract_batch_chunk(messages: list[str]) -> list[dict]:
    """
    Send N messages in one LLM call and return N extraction dicts.

    Falls back to individual extract() calls per message if the model returns
    malformed or unexpected output.
    """
    n = len(messages)
    parts = "\n\n".join(
        f"=== Message {i + 1} ===\n{msg}" for i, msg in enumerate(messages)
    )
    user_content = (
        f"Extract from each of the {n} user messages below. "
        f"Return a JSON array containing exactly {n} extraction objects "
        f"(one per message, in order). Each element must follow the same "
        f"schema as a single extraction.\n\n{parts}"
    )

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    print(f"  [extract_batch n={n}] tokens={response.usage.total_tokens}", flush=True)

    content = (response.choices[0].message.content or "").strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(_repair_json(content))
        except json.JSONDecodeError:
            parsed = None

    # Unwrap {"extractions": [...]} if the model wrapped the array
    if isinstance(parsed, dict):
        for key in ("extractions", "messages", "results"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list) or len(parsed) != n:
        print(
            f"  [extract_batch] unexpected shape (got {type(parsed).__name__}, "
            f"len={len(parsed) if isinstance(parsed, list) else '?'}), "
            f"falling back to individual calls",
            flush=True,
        )
        return [extract(msg) for msg in messages]

    total_nodes = sum(
        len(item.get("nodes", [])) for item in parsed if isinstance(item, dict)
    )
    if total_nodes == 0:
        print(
            "  [extract_batch] all extractions empty, falling back to individual calls",
            flush=True,
        )
        return [extract(msg) for msg in messages]

    return [_normalize_extraction(item) for item in parsed]


def extract_batch(messages: list[str], batch_size: int = 4) -> list[dict]:
    """
    Extract nodes/edges from multiple messages using fewer LLM calls.
    The system prompt is paid once per batch instead of once per message.

    batch_size=4 is a good default for local Ollama. Each failed batch
    falls back to individual extract() calls, so correctness is guaranteed.
    """
    results = []
    for i in range(0, len(messages), batch_size):
        chunk = messages[i : i + batch_size]
        if len(chunk) == 1:
            results.append(extract(chunk[0]))
        else:
            results.extend(_extract_batch_chunk(chunk))
    return results
