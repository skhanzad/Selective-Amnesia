import json
import re
import requests

from graph.prompts import _MAIN_EXTRACTION_SYSTEM_PROMPT

# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "gemma2"


# ── Main extraction ────────────────────────────────────────────────────────────


def extract(user_message: str) -> dict:
    """
    Run the main extraction prompt against the user message and return the
    parsed JSON dict produced by the LLM.

    Raises:
        requests.HTTPError: if the Ollama API call fails.
        ValueError: if the response cannot be parsed as JSON.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": _MAIN_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()

    payload = response.json()
    print(
        f"  [extract] done={payload.get('done')} done_reason={payload.get('done_reason')} tokens={payload.get('eval_count')}",
        flush=True,
    )

    content = payload["message"]["content"]
    if not content:
        print(
            f"  [extract] WARNING: empty content. Full payload: {payload}", flush=True
        )
        raise ValueError("LLM returned an empty response")
    raw = content.strip()

    # Strip markdown fences if the model added them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Common gemma2 failures: trailing commas and truncated/unclosed output.
    # 1. Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)
    # 2. Remove trailing comma at end-of-string (output was truncated mid-object)
    repaired = re.sub(r",\s*$", "", repaired.rstrip())
    # 3. Close any unclosed braces/brackets
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    repaired = repaired + ("}" * max(open_braces, 0)) + ("]" * max(open_brackets, 0))

    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned non-JSON output: {exc}\n\nRaw output:\n{raw}"
        ) from exc

    # Defensive: model sometimes returns nodes/edges as a list instead of a dict.
    # Convert to the expected {node_0: {...}, ...} dict format.
    for key in ("nodes", "edges"):
        if isinstance(parsed.get(key), list):
            prefix = key.rstrip("s")  # "nodes" -> "node", "edges" -> "edge"
            parsed[key] = {f"{prefix}_{i}": item for i, item in enumerate(parsed[key])}

    return parsed
