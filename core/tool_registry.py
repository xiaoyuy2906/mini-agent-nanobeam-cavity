"""Tool registry — decorator-based tool registration.

Each tool is defined once: schema + handler in the same place.
No more if-else dispatch chains.
"""

from __future__ import annotations
from typing import Any, Callable

_REGISTRY: dict[str, dict[str, Any]] = {}


def tool(
    name: str,
    description: str,
    input_schema: dict,
    *,
    required_state: str | None = None,
):
    """Register an async tool handler with its Anthropic-compatible schema.

    Args:
        name: Tool name (must match what the LLM calls).
        description: Shown to the LLM.
        input_schema: JSON Schema for tool input.
        required_state: If set, tool will fail unless agent.state has this attr truthy.
    """

    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = {
            "schema": {
                "name": name,
                "description": description,
                "input_schema": input_schema,
            },
            "handler": fn,
            "required_state": required_state,
        }
        return fn

    return decorator


def get_all_schemas() -> list[dict]:
    return [entry["schema"] for entry in _REGISTRY.values()]


def get_handler(name: str) -> Callable | None:
    entry = _REGISTRY.get(name)
    return entry["handler"] if entry else None


def get_required_state(name: str) -> str | None:
    entry = _REGISTRY.get(name)
    return entry["required_state"] if entry else None


async def dispatch(name: str, args: dict, agent: Any) -> dict:
    """Dispatch a tool call. Returns result dict."""
    handler = get_handler(name)
    if handler is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}

    req = get_required_state(name)
    if req and not getattr(agent.state, req, None):
        return {"ok": False, "error": f"Must configure {req} first (call set_unit_cell)."}

    try:
        result = await handler(agent, args)
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    return result
