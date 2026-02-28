# CLAUDE.md — mini-agent-nanobeam-cavity

## Project Overview

AI-driven nanobeam photonic crystal cavity design tool. Core goal: maximize Q/V (quality factor / mode volume).

- **Language**: Python 3.13
- **Package manager**: uv (`uv run` to execute, always confirm `pyproject.toml` exists before `uv add`)
- **CLI entry**: `uv run cavity` (powered by Typer)
- **LLM**: Anthropic Claude (via `anthropic` SDK)

## Project Structure

```
app/          # CLI and interactive chat interface
  cli.py      # Typer CLI entry point (cavity command)
  chat.py     # Interactive chat loop
core/         # Core agent logic
  agent.py    # CavityAgent class (state + tools + workflow)
  state.py    # CavityDesignState (design history, best design)
tools/        # Low-level tools
  build_gds.py       # gdsfactory GDS layout generation
  run_lumerical.py   # Lumerical FDTD simulation
  toolset.py         # Toolset wrapper
  cavity_mcp.py      # MCP server (experimental)
skills.md     # Agent system prompt / skill overrides
mininal_agent.py  # Early simple agent (reference only)
```

## Environment Variables (required)

Must be set in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
LUMPAPI_PATH=/path/to/lumerical/api
```

## Common Commands

```bash
uv run cavity chat          # Start interactive design chat
uv run cavity sweep         # Run full parameter sweep workflow
uv run python mininal_agent.py  # Run the early simple agent
```

## Sweep Workflow Order

```
sweep_min_a → sweep_rx → re_sweep_min_a_1 → sweep_ry →
re_sweep_min_a_2 → sweep_taper → fine_period → complete
```

## Agent Tools

| Tool | Purpose |
|------|---------|
| `set_unit_cell` | Set lattice parameters — **must be called first** |
| `design_cavity` | Build GDS + run FDTD simulation |
| `view_history` | View history of all designs |
| `compare_designs` | Compare specific design iterations |
| `get_best_design` | Retrieve current best design |

## Physics Reference

- Supported materials: SiN, Si, Diamond, GaAs — freestanding or on substrate
- Excellent metrics: Q > 1,000,000, V < 0.5 (λ/n)³
- Default initial design: 8 taper holes, 10 mirror holes, min_a=90%

## Development Rules

- **Dependencies**: Run `uv init` first if `pyproject.toml` does not exist, then `uv add <package>`
- **File navigation**: Always `ls` to confirm a directory exists before navigating — never guess paths
- **Tool commands**: Before running any setup command, list all prerequisite steps and verify each one
- **Bug fixes**: Briefly explain why the fix works
- **Units**: Code mixes nm / μm / m — double-check unit conversions carefully
