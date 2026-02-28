"""
MCP server for cavity design tools. Wraps CavityAgent methods for use with claude-agent-sdk.

Usage:
    from tools.cavity_mcp import create_cavity_mcp_server
    from core.agent import CavityAgent
    from tools.toolset import Toolset

    agent = CavityAgent(toolset=..., state=...)
    server = create_cavity_mcp_server(agent)
    # Pass server to ClaudeAgentOptions(mcp_servers={"cavity": server})
"""

import json
import asyncio
from claude_agent_sdk import tool, create_sdk_mcp_server


def _mcp_result(data: dict) -> dict:
    """Format agent result as MCP tool response."""
    return {"content": [{"type": "text", "text": json.dumps(data, default=str)}]}


def create_cavity_mcp_server(agent):
    """
    Create an MCP server with cavity design tools. agent must implement:
    - set_unit_cell_from_tool_params(params) -> dict
    - design_cavity(design_params, run) -> dict (async)
    - view_history(last_n) -> dict
    - compare_designs(iterations) -> dict
    - get_best_design() -> dict
    """

    @tool(
        "set_unit_cell",
        "Set the unit cell parameters. MUST be called first before designing.",
        {
            "design_wavelength_nm": float,
            "period_nm": float,
            "wg_width_nm": float,
            "wg_height_nm": float,
            "hole_rx_nm": float,
            "hole_ry_nm": float,
            "wg_material": str,
            "wg_material_refractive_index": float,
            "freestanding": bool,
            "wavelength_span_nm": float,
            "substrate": str,
            "substrate_material_refractive_index": float,
        },
    )
    async def set_unit_cell(args: dict) -> dict:
        r = agent.set_unit_cell_from_tool_params(args)
        return _mcp_result(r)

    @tool(
        "design_cavity",
        "Design a cavity and run Lumerical FDTD for Q/V performance.",
        {
            "period_nm": float,
            "wg_width_nm": float,
            "hole_rx_nm": float,
            "hole_ry_nm": float,
            "num_taper_holes": int,
            "num_mirror_holes": int,
            "min_a_percent": float,
            "min_rx_percent": float,
            "min_ry_percent": float,
            "hypothesis": str,
        },
    )
    async def design_cavity(args: dict) -> dict:
        r = await agent.design_cavity(args, run=True)
        return _mcp_result(r)

    @tool(
        "view_history",
        "View the history of all designs tried so far.",
        {"last_n": int},
    )
    async def view_history(args: dict) -> dict:
        last_n = args.get("last_n")
        r = agent.view_history(last_n=last_n)
        return _mcp_result(r)

    @tool(
        "compare_designs",
        "Compare two or more designs side by side.",
        {"iterations": list},
    )
    async def compare_designs(args: dict) -> dict:
        iters = args.get("iterations", [])
        r = agent.compare_designs(iterations=iters)
        return _mcp_result(r)

    @tool(
        "get_best_design",
        "Get the current best design with highest Q/V.",
        {},
    )
    async def get_best_design(args: dict) -> dict:
        r = agent.get_best_design()
        return _mcp_result(r)

    return create_sdk_mcp_server(
        name="cavity",
        version="1.0.0",
        tools=[set_unit_cell, design_cavity, view_history, compare_designs, get_best_design],
    )
