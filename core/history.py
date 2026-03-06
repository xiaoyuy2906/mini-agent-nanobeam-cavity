"""History management — SWE-agent style sliding window.

Key ideas from SWE-agent:
- LastNObservations: keep only last N tool results verbatim, compress older ones
- Observation truncation: cap individual tool output length
- The LLM never sees unbounded conversation history
"""

from __future__ import annotations
import json
import copy

# How many recent tool-result turns to keep verbatim
KEEP_LAST_N_OBSERVATIONS = 10

# Max characters per individual tool result
MAX_OBSERVATION_LENGTH = 4000

# Truncation notice (like SWE-agent's truncated_observation_template)
TRUNCATION_NOTICE = "[Output truncated: {omitted} chars omitted. Use view_history for details.]"


def truncate_observation(text: str) -> str:
    """Truncate a single observation if it exceeds MAX_OBSERVATION_LENGTH."""
    if len(text) <= MAX_OBSERVATION_LENGTH:
        return text
    omitted = len(text) - MAX_OBSERVATION_LENGTH
    return text[:MAX_OBSERVATION_LENGTH] + "\n" + TRUNCATION_NOTICE.format(omitted=omitted)


def compress_history(messages: list[dict]) -> list[dict]:
    """Apply SWE-agent-style LastNObservations compression.

    Walks messages, finds tool_result content blocks, and replaces
    old ones with a one-line summary. Keeps the last N observations
    verbatim.

    Returns a new list (does not mutate input).
    """
    messages = copy.deepcopy(messages)

    # Find indices of user messages that contain tool_result blocks
    tool_result_indices = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        ):
            tool_result_indices.append(i)

    # Keep last N verbatim, compress the rest
    if len(tool_result_indices) <= KEEP_LAST_N_OBSERVATIONS:
        return messages

    to_compress = tool_result_indices[:-KEEP_LAST_N_OBSERVATIONS]

    for idx in to_compress:
        content = messages[idx]["content"]
        if not isinstance(content, list):
            continue
        compressed = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                # Extract key metrics from the original content
                summary = _summarize_tool_result(block)
                compressed.append({
                    "type": "tool_result",
                    "tool_use_id": block.get("tool_use_id", ""),
                    "content": summary,
                })
            else:
                compressed.append(block)
        messages[idx]["content"] = compressed

    return messages


def _summarize_tool_result(block: dict) -> str:
    """Create a one-line summary of a tool result (like SWE-agent's elision)."""
    raw = block.get("content", "")
    if not isinstance(raw, str):
        raw = str(raw)

    # Try to parse as JSON and extract key metrics
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            parts = []
            if "iteration" in data:
                parts.append(f"iter={data['iteration']}")
            if "result" in data and isinstance(data["result"], dict):
                r = data["result"]
                if r.get("Q"):
                    parts.append(f"Q={r['Q']:,.0f}")
                if r.get("V"):
                    parts.append(f"V={r['V']:.3f}")
                if r.get("qv_ratio"):
                    parts.append(f"Q/V={r['qv_ratio']:,.0f}")
                if r.get("resonance_nm"):
                    parts.append(f"res={r['resonance_nm']:.1f}nm")
            if data.get("ok") is False:
                parts.append(f"error={data.get('error', 'unknown')[:80]}")
            if parts:
                return f"[Old result: {', '.join(parts)}]"
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: just show length
    return f"[Old observation: {len(raw)} chars omitted]"
