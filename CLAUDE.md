# CLAUDE.md — mini-agent-nanobeam-cavity

## Project Overview

AI-driven nanobeam photonic crystal cavity design tool. Core goal: maximize Q/V (quality factor / mode volume).

- **Language**: Python 3.13
- **Package manager**: uv (`uv add <package>`, pyproject.toml already exists)
- **LLM**: Anthropic Claude (via `claude-agent-sdk` — `query()` + MCP)
- **Frontend**: Node.js terminal chat (`chat.js`)

## Project Structure

```
agent_server.py   # stdin/stdout JSON bridge — claude_agent_sdk.query() loop
chat.js           # Node.js terminal UI (spawns agent_server.py)
skills.md         # Agent system prompt
core/
  agent.py        # CavityAgent — system prompt, tool schemas, tool methods
  state.py        # CavityDesignState — design history, best design, log persistence
tools/
  toolset.py      # Thin wrappers: build_gds + run_simulation
  cavity_mcp.py   # MCP server exposing CavityAgent tools to claude_agent_sdk
  build_gds.py    # gdsfactory GDS layout generation
  run_lumerical.py # Lumerical FDTD simulation (Q/V extraction)
```

## Environment Variables

Must be set in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
LUMPAPI_PATH=/path/to/lumerical/api   # optional — skips FDTD if not set
MODEL_NAME=claude-sonnet-4-6           # optional — default: claude-sonnet-4-6
ANTHROPIC_BASE_URL=...                 # optional — for custom endpoints
```

## Running

```bash
node chat.js                          # Start interactive design chat
uv run python agent_server.py         # Python server only (pipe JSON manually)
```

## Architecture

```
node chat.js
  │  spawn("uv run python agent_server.py")
  │  stdin  → {"type": "user_message", "content": "..."}
  │  stdout ← {"type": "text"|"tool_start"|"tool_end"|"done"|"error"}
  ▼
agent_server.py  (claude_agent_sdk.query())
  └── ClaudeAgentOptions(mcp_servers={"cavity": mcp_server})
         ▼
  tools/cavity_mcp.py  (MCP server)
         ▼
  core/agent.py  CavityAgent
  ├── tools/build_gds.py
  └── tools/run_lumerical.py
```

## Agent Tools

| Tool | Purpose |
|------|---------|
| `set_unit_cell` | Set lattice parameters — **must be called first** |
| `design_cavity` | Build GDS + run FDTD simulation |
| `view_history` | View history of all designs |
| `compare_designs` | Compare specific design iterations |
| `get_best_design` | Retrieve current best design |

## Sweep Workflow Order

```
sweep_min_a → sweep_rx → re_sweep_min_a_1 → sweep_ry →
re_sweep_min_a_2 → sweep_taper → fine_period → complete
```

## Physics Reference

- Supported materials: SiN, Si, Diamond, GaAs — freestanding or on substrate
- Excellent metrics: Q > 1,000,000, V < 0.5 (λ/n)³
- Default initial design: 8 taper holes, 10 mirror holes, min_a=90%

## Development Rules

- **File navigation**: Always `ls` before navigating — never guess paths
- **Bug fixes**: Briefly explain why the fix works
- **Units**: Code mixes nm / μm / m — double-check unit conversions carefully
