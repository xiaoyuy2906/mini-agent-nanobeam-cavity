"""
Example: Chat using Claude Agent SDK with cavity MCP tools.

Run:
    uv run python examples/chat_with_mcp.py

Requires: ANTHROPIC_API_KEY in .env
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from core.agent import CavityAgent
from core.state import CavityDesignState
from tools.toolset import Toolset
from tools.cavity_mcp import create_cavity_mcp_server


def _build_agent():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")
    model = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    toolset = Toolset(api_key=api_key, model_name=model, base_url=base_url)
    state = CavityDesignState()
    return CavityAgent(toolset=toolset, state=state)


async def main():
    agent = _build_agent()
    mcp_server = create_cavity_mcp_server(agent)

    tools_allowed = [
        "mcp__cavity__set_unit_cell",
        "mcp__cavity__design_cavity",
        "mcp__cavity__view_history",
        "mcp__cavity__compare_designs",
        "mcp__cavity__get_best_design",
    ]

    options = ClaudeAgentOptions(
        system_prompt=agent.system_prompt,
        mcp_servers={"cavity": mcp_server},
        allowed_tools=tools_allowed,
    )

    async with ClaudeSDKClient(options=options) as client:
        prompt = "Configure a SiN cavity at 737nm. Period 294nm, wg 450x220nm, hole rx=50nm ry=100nm."
        client.query(prompt)
        async for message in client.receive_response():
            print(message)


if __name__ == "__main__":
    asyncio.run(main())
