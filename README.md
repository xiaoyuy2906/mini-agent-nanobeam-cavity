# Nanobeam Cavity Design Agent

An AI agent that designs and optimizes nanobeam photonic crystal cavities. The agent systematically explores design parameters through FDTD simulation to maximize Q/V (quality factor over mode volume).

## Overview

This project implements an LLM-powered agent that:
- Designs nanobeam photonic crystal cavities targeting user-specified wavelengths
- Generates GDS layouts using gdsfactory
- Runs Lumerical FDTD simulations to compute Q-factor and mode volume
- Uses a thought-action-observation loop to systematically optimize designs
- Supports both Claude and MiniMax models via Anthropic-compatible API

## Architecture

```
node chat.js
  │  spawn("uv run python agent_server.py")
  │  stdin  → {"type": "user_message", "content": "..."}
  │  stdout ← {"type": "text"|"tool_start"|"tool_end"|"done"|"error"}
  ▼
agent_server.py  (AsyncAnthropic — tool-use loop)
  ├── client.messages.create(tools=agent.tools, ...)
  └── dispatch_tool() → CavityAgent methods
         ▼
  core/agent.py  CavityAgent
  ├── tools/build_gds.py       — gdsfactory GDS layout
  └── tools/run_lumerical.py   — Lumerical FDTD (asyncio.to_thread)
```

### Agent Tools

| Tool | Purpose |
|------|---------|
| `set_unit_cell` | Set lattice parameters — must be called first |
| `design_cavity` | Build GDS + run FDTD simulation |
| `view_history` | View history of all designs |
| `compare_designs` | Compare specific design iterations |
| `get_best_design` | Retrieve current best design |

## Requirements

- Python 3.13+
- Node.js 18+
- pnpm (`npm install -g pnpm`)
- Lumerical FDTD (with Python API) — optional, skipped if `LUMPAPI_PATH` is not set
- Anthropic API key **or** MiniMax API key

## Installation

```bash
# Clone the repository
git clone https://github.com/xiaoyuy2906/mini-agent-nanobeam-cavity.git
cd mini-agent-nanobeam-cavity

# Install Python dependencies
uv sync

# Install Node.js dependencies
pnpm install
```

## Configuration

Create a `.env` file in the project root:

```env
# ── Option A: Anthropic (Claude) ──────────────────────────
ANTHROPIC_API_KEY=sk-ant-api03-...
MODEL_NAME=claude-sonnet-4-6

# ── Option B: MiniMax (zero code changes) ─────────────────
MODEL_PROVIDER=minimax
ANTHROPIC_API_KEY=sk-api-...          # your MiniMax key
ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic
MODEL_NAME=MiniMax-M2.5

# ── Lumerical (optional) ──────────────────────────────────
LUMPAPI_PATH=C:/Program Files/ANSYS Inc/v251/Lumerical/api/python
```

> MiniMax works out of the box with the Anthropic SDK because it exposes an Anthropic-compatible endpoint — no code changes needed.

## Usage

```bash
node chat.js
```

Then describe your cavity in plain language:

```
You: Design a freestanding SiN cavity at 737nm, width 800nm, height 330nm
Agent: I'll set up the unit cell first...
```

Type `quit`, `exit`, or `q` to exit.

## Project Structure

```
chat.js               # Node.js terminal UI (spawns agent_server.py)
agent_server.py       # stdin/stdout JSON bridge — Anthropic tool-use loop
skills.md             # Agent system prompt and optimization protocol
core/
  agent.py            # CavityAgent — tools, system prompt, tool dispatch
  state.py            # CavityDesignState — design history, best design, log
tools/
  toolset.py          # Thin wrappers: build_gds + run_simulation
  build_gds.py        # gdsfactory GDS layout generation
  run_lumerical.py    # Lumerical FDTD simulation (async via asyncio.to_thread)
```

## Design Parameters

### Unit Cell
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `design_wavelength_nm` | Target resonance wavelength | 637, 737, 1550 |
| `period_nm` | Lattice period | 200–300 |
| `wg_width_nm` | Waveguide width | 400–800 |
| `wg_height_nm` | Waveguide thickness | 200–330 |
| `hole_rx_nm` | Hole radius (x-axis) | 50–100 |
| `hole_ry_nm` | Hole radius (y-axis) | 80–150 |
| `wg_material` | Waveguide material | Si, SiN, Diamond, GaAs |

### Cavity Taper
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `num_taper_holes` | Chirped holes per side | 5–15 |
| `num_mirror_holes` | Mirror holes per side | 5–10 |
| `min_a_percent` | Min period at cavity center (%) | 87–90 |

## Performance Targets

| Metric | Excellent | Description |
|--------|-----------|-------------|
| Q-factor | > 1,000,000 | Higher is better |
| Mode volume V | < 0.5 (λ/n)³ | Lower is better |
| Q/V ratio | maximized | Primary objective |

## Sweep Workflow

The agent follows a fixed optimization order:

```
sweep_min_a → sweep_rx → re_sweep_min_a_1 → sweep_ry →
re_sweep_min_a_2 → sweep_taper → fine_period → complete
```

## License

MIT
