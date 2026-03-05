#!/usr/bin/env python
"""
agent_server.py — stdin/stdout JSON bridge using direct Anthropic SDK.

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
from anthropic import AsyncAnthropic

load_dotenv()

from core.agent import CavityAgent
from core.state import CavityDesignState
from tools.toolset import Toolset


def emit(event: dict) -> None:
    print(json.dumps(event, default=str), flush=True)


async def dispatch_tool(agent: CavityAgent, name: str, args: dict) -> dict:
    if name == "set_unit_cell":
        return agent.set_unit_cell_from_tool_params(args)
    if name == "design_cavity":
        return await agent.design_cavity(args, run=True)
    if name == "view_history":
        return agent.view_history(last_n=args.get("last_n"))
    if name == "compare_designs":
        return agent.compare_designs(iterations=args.get("iterations", []))
    if name == "get_best_design":
        return agent.get_best_design()
    if name == "analyze_sensitivity":
        return agent.analyze_sensitivity()
    if name == "suggest_next_experiment":
        return agent.suggest_next_experiment()
    return {"ok": False, "error": f"Unknown tool: {name}"}


REFLECTION_INTERVAL = 5  # Inject reflection prompt every N tool rounds

REFLECTION_PROMPT = (
    "[REFLECTION] You have completed {n} tool calls so far. "
    "Pause and reflect:\n"
    "1. Is Q/V converging or plateauing?\n"
    "2. Which parameter had the most impact recently?\n"
    "3. Should you change strategy (e.g., switch to a different parameter, "
    "try a different taper_type, explore hole chirp)?\n"
    "4. Consider calling analyze_sensitivity to reassess priorities.\n"
    "State your updated strategy before continuing."
)


async def run_agent_loop(
    content: str,
    agent: CavityAgent,
    client: AsyncAnthropic,
    model: str,
    conversation_history: list,
) -> None:
    conversation_history.append({"role": "user", "content": content})
    tool_call_count = 0

    while True:
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=agent.system_prompt,
            tools=agent.tools,
            messages=conversation_history,
        )

        conversation_history.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text" and block.text:
                emit({"type": "text", "delta": block.text})

        if response.stop_reason != "tool_use":
            break

        tool_blocks = [b for b in response.content if b.type == "tool_use"]

        async def _run_one(block):
            emit({"type": "tool_start", "name": block.name, "input": block.input})
            try:
                result = await dispatch_tool(agent, block.name, block.input)
            except Exception as e:
                result = {"ok": False, "error": str(e)}
            emit({"type": "tool_end", "name": block.name, "result": result})
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            }

        tool_results = list(await asyncio.gather(*[_run_one(b) for b in tool_blocks]))
        tool_call_count += len(tool_blocks)

        # Inject reflection prompt periodically to encourage strategic thinking
        if tool_call_count % REFLECTION_INTERVAL == 0 and tool_call_count > 0:
            reflection = REFLECTION_PROMPT.format(n=tool_call_count)
            tool_results.append({
                "type": "text",
                "text": reflection,
            })

        conversation_history.append({"role": "user", "content": tool_results})

    emit({"type": "done"})


async def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        emit({"type": "error", "message": "ANTHROPIC_API_KEY not set in .env"})
        sys.exit(1)

    model = os.getenv("MODEL_NAME", "claude-sonnet-4-6")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    client = (
        AsyncAnthropic(api_key=api_key, base_url=base_url)
        if base_url
        else AsyncAnthropic(api_key=api_key)
    )

    agent = CavityAgent(toolset=Toolset(), state=CavityDesignState())
    conversation_history: list = []

    emit({"type": "ready"})

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

        try:
            await run_agent_loop(content, agent, client, model, conversation_history)
        except Exception as e:
            emit({"type": "error", "message": str(e)})
            emit({"type": "done"})


if __name__ == "__main__":
    asyncio.run(main())
