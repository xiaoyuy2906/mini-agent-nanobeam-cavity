# Nanobeam Cavity Design Agent

An AI agent that designs and optimizes nanobeam photonic crystal cavities using the SWE-agent pattern. The agent systematically explores design parameters through FDTD simulation to maximize Q/V (quality factor over mode volume).

## Overview

This project implements an LLM-powered agent that:
- Designs nanobeam photonic crystal cavities targeting user-specified wavelengths
- Generates GDS layouts using gdsfactory
- Runs Lumerical FDTD simulations to compute Q-factor and mode volume
- Uses a thought-action-observation loop to systematically optimize designs
- Supports both Claude and MiniMax models via Anthropic-compatible API

## Architecture

The agent follows the **SWE-agent pattern** (from Princeton NLP):

```
┌─────────────────────────────────────────────────────────┐
│                    SWE Cavity Agent                     │
├─────────────────────────────────────────────────────────┤
│  1. THOUGHT   → Reason about current state              │
│  2. ACTION    → Call exactly one tool                   │
│  3. OBSERVATION → Receive result, continue              │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Available Tools                        │
├─────────────────────────────────────────────────────────┤
│  • set_unit_cell    - Configure waveguide parameters    │
│  • design_cavity    - Build GDS + run FDTD simulation   │
│  • view_history     - Inspect previous designs          │
│  • compare_designs  - Compare specific iterations       │
│  • get_best_design  - Retrieve highest Q/V design       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Persistent State Tracking                  │
├─────────────────────────────────────────────────────────┤
│  • Design history with parameters and results           │
│  • Best design tracking (highest Q/V)                   │
│  • Sweep step tracking with locked parameters           │
│  • Duplicate detection to avoid wasted simulations      │
│  • Log persistence across sessions (JSON)               │
└─────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.13+
- Lumerical FDTD (with Python API)
- Anthropic API key (or MiniMax API key)

## Installation

```bash
# Clone the repository
git clone https://github.com/xiaoyuy2906/mini-agent-nanobeam-cavity.git
cd mini-agent-nanobeam-cavity

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```env
# For Claude (default)
ANTHROPIC_API_KEY=your-anthropic-key

# For MiniMax
MODEL_PROVIDER=minimax
ANTHROPIC_API_KEY=your-minimax-key
MODEL_NAME=MiniMax-M2.5-lightning
```

## Usage

```bash
python swe_agent.py
```

### Commands

| Command | Description |
|---------|-------------|
| `Design a SiN cavity at 737nm` | Set up unit cell with natural language |
| `confirm fdtd` | Confirm parameters before first simulation |
| `auto` or `auto 15` | Run automated optimization loop (default 10 iterations) |
| `auto 15 -- <instruction>` | Auto mode with additional constraints |
| `sweep <param> from <start> to <end> [step <N>]` | Deterministic parameter sweep |
| `quit` | Exit |

### Sweep Examples

```
sweep num_taper_holes from 5 to 12
sweep min_a_percent 87 90
sweep hole_rx_nm from 80 to 100 step 5
```

### Three-Phase Optimization

The agent follows a strict optimization protocol:

1. **Phase 1 — Resonance Tuning**: Adjust `period_nm` until resonance is within ±5nm of target. No other parameters are changed.

2. **Phase 2 — Q/V Optimization**: Sweep parameters one at a time in strict order:
   - `min_a_percent` (90→89→88→87)
   - `hole_rx_nm` (±5nm steps)
   - Re-sweep `min_a_percent`
   - `hole_ry_nm` (±5nm steps)
   - Re-sweep `min_a_percent`
   - `num_taper_holes` (8, 10, 12)
   - Fine period sweep (±1–2nm)

3. **Phase 3 — Fine Tuning**: Fine period sweep around the best design once Q > 100,000.

### Sweep Enforcement

The agent includes an **ENFORCE guard** that prevents the LLM from changing parameters outside the current sweep step. For example, during `sweep_min_a`, only `min_a_percent` and `period_nm` (for resonance re-tuning) can be modified. This prevents the LLM from confounding results by changing multiple variables at once.

The `sweep` command bypasses this guard by temporarily setting the step to `manual` mode.

## Project Structure

```
.
├── swe_agent.py          # Main SWE-style agent with TAO loop
├── build_gds.py          # GDS layout generation using gdsfactory
├── run_lumerical.py      # Lumerical FDTD simulation interface
├── skills.md             # Agent skill prompt (optimization protocol)
├── cavity_design_log.json # Persistent log of all iterations (auto-generated)
├── gds_output/           # Generated GDS files
└── fdtd_output/          # FDTD simulation files (.fsp)
```

## Design Parameters

### Unit Cell
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `design_wavelength_nm` | Target resonance wavelength | e.g., 737, 1550 |
| `period_nm` | Lattice period | 200–300 |
| `wg_width_nm` | Waveguide width | 400–600 |
| `wg_height_nm` | Waveguide thickness | 200–300 |
| `hole_rx_nm` | Hole radius (x) | 50–100 |
| `hole_ry_nm` | Hole radius (y) | 80–150 |
| `material` | Waveguide material | Si3N4, Si, Diamond, GaAs |

### Cavity Taper
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `num_taper_holes` | Chirped holes per side | 5–15 |
| `num_mirror_holes` | Mirror holes per side | 5–10 |
| `min_a_percent` | Min period at center (%) | 87–90 |
| `taper_type` | Chirp profile | quadratic (fixed) |

## Performance Targets

| Metric | Target | Priority |
|--------|--------|----------|
| Resonance | Within ±5nm of target | Highest |
| Q-factor | > 1,000,000 | Second |
| Mode volume V | < 0.5 (λ/n)³ | Third |

## How It Works

1. **Unit Cell Setup**: Configure the photonic crystal unit cell (period, hole size, waveguide dimensions, material). The agent confirms all parameters before the first FDTD run.

2. **Cavity Design**: For each iteration:
   - Generates GDS layout with specified taper/mirror parameters
   - Runs Lumerical FDTD simulation
   - Extracts Q-factor from resonance peak
   - Computes mode volume normalized to (λ/n)³
   - Checks for duplicates to avoid re-running identical simulations

3. **Optimization**: The agent uses design history to systematically sweep parameters following the strict protocol in `skills.md`. An enforce guard ensures only the currently swept parameter changes between iterations.

4. **Persistence**: All results are saved to `cavity_design_log.json`. When resuming with the same unit cell configuration, previous iterations are automatically loaded.

## License

MIT
