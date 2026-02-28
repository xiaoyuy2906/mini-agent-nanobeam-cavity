#!/usr/bin/env python
"""
agent_server.py — stdin/stdout JSON bridge using claude_agent_sdk.

Protocol:
  STDIN  (from JS): {"type": "user_message", "content": "..."}
  STDOUT (to JS):   {"type": "ready"|"text"|"tool_start"|"tool_end"|"done"|"error", ...}

Run via: uv run python agent_server.py
"""
import sys
import json
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    UserMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)
from core.agent import CavityAgent
from core.state import CavityDesignState
from tools.toolset import Toolset
from tools.cavity_mcp import create_cavity_mcp_server


def emit(event: dict) -> None:
    print(json.dumps(event, default=str), flush=True)


def build_agent_and_server():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        emit({"type": "error", "message": "ANTHROPIC_API_KEY not set in .env"})
        sys.exit(1)

    toolset = Toolset()
    agent = CavityAgent(toolset=toolset, state=CavityDesignState())
    mcp_server = create_cavity_mcp_server(agent)
    return mcp_server, agent.system_prompt


async def handle_message(
    content: str,
    mcp_server,
    system_prompt: str,
    session_id: str | None,
) -> str | None:
    """Run one user turn via claude_agent_sdk, emit events, return new session_id."""
    options = ClaudeAgentOptions(
        mcp_servers={"cavity": mcp_server},
        system_prompt=system_prompt,
        permission_mode="bypassPermissions",
        allowed_tools=[
            "set_unit_cell",
            "design_cavity",
            "view_history",
            "compare_designs",
            "get_best_design",
        ],
    )
    if session_id:
        options.resume = session_id

    tool_id_to_name: dict[str, str] = {}
    new_session_id = session_id

    try:
        async for event in query(prompt=content, options=options):
            if isinstance(event, ResultMessage):
                if event.session_id:
                    new_session_id = event.session_id
                if event.is_error and event.result:
                    emit({"type": "error", "message": event.result})
                continue

            if isinstance(event, SystemMessage):
                continue

            if isinstance(event, AssistantMessage):
                for block in event.content:
                    if isinstance(block, TextBlock) and block.text:
                        emit({"type": "text", "delta": block.text})
                    elif isinstance(block, ToolUseBlock):
                        tool_id_to_name[block.id] = block.name
                        emit({"type": "tool_start", "name": block.name, "input": block.input or {}})
                continue

            if isinstance(event, UserMessage) and event.parent_tool_use_id is not None:
                name = tool_id_to_name.get(event.parent_tool_use_id, "unknown")
                result = _parse_tool_result(event.content)
                emit({"type": "tool_end", "name": name, "result": result})
                continue

    except Exception as e:
        emit({"type": "error", "message": str(e)})

    emit({"type": "done"})
    return new_session_id


def _parse_tool_result(content) -> dict:
    """Extract a dict from MCP tool result content (string or list of blocks)."""
    if content is None:
        return {}
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"text": content}
    if isinstance(content, list):
        texts = [
            (b.get("text") if isinstance(b, dict) else getattr(b, "text", None))
            for b in content
        ]
        combined = "\n".join(t for t in texts if t)
        try:
            return json.loads(combined)
        except json.JSONDecodeError:
            return {"text": combined}
    return {}


async def main():
    mcp_server, system_prompt = build_agent_and_server()
    emit({"type": "ready"})

    session_id: str | None = None

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            msg = json.loads(raw_line)
        except json.JSONDecodeError as e:
            emit({"type": "error", "message": f"JSON parse error: {e}"})
            continue

        if msg.get("type") != "user_message":
            continue
        content = msg.get("content", "").strip()
        if not content:
            continue

        session_id = await handle_message(content, mcp_server, system_prompt, session_id)


if __name__ == "__main__":
    asyncio.run(main())
