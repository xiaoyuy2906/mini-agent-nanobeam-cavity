# Nanobeam Cavity Design Agent

An AI agent that designs and optimizes nanobeam photonic crystal cavities using the SWE-agent pattern. The agent systematically explores design parameters through FDTD simulation to maximize Q/V (quality factor over mode volume).

## Overview

This project implements an LLM-powered agent that:
- Designs nanobeam photonic crystal cavities targeting user-specified wavelengths
- Generates GDS layouts using gdsfactory
- Runs Lumerical FDTD simulations to compute Q-factor and mode volume
- Uses a thought-action-observation loop to systematically optimize designs

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
│  • Iteration count                                      │
└─────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.13+
- Lumerical FDTD (with Python API)
- Anthropic API key

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd mini-agent-nanobeam-cavity

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your-api-key
LUMPAPI_PATH=/path/to/lumerical/api/python
```

## Usage

```bash
python swe_agent.py
```

Example prompts:
- "Design a SiN cavity at 737nm"
- "Optimize Q/V with 5 iterations"
- "Compare the last 3 designs"

## Project Structure

```
.
├── swe_agent.py       # Main SWE-style agent with TAO loop
├── build_gds.py       # GDS layout generation using gdsfactory
├── run_lumerical.py   # Lumerical FDTD simulation interface
├── skills.md          # Agent skill prompt (loaded into system prompt)
├── gds_output/        # Generated GDS files
└── fdtd_output/       # FDTD simulation files (.fsp)
```

## Design Parameters

### Unit Cell
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `design_wavelength_nm` | Target resonance wavelength | e.g., 737, 1550 |
| `period_nm` | Lattice period | 200-300 |
| `wg_width_nm` | Waveguide width | 400-600 |
| `wg_height_nm` | Waveguide thickness | 200-300 |
| `hole_rx_nm` | Hole radius (x) | 50-100 |
| `hole_ry_nm` | Hole radius (y) | 80-150 |
| `material` | Waveguide material | SiN, Si, Diamond, GaAs |

### Cavity Taper
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `num_taper_holes` | Chirped holes per side | 5-15 |
| `num_mirror_holes` | Mirror holes per side | 8-15 |
| `min_a_percent` | Min period at center (%) | 85-95 |
| `taper_type` | Chirp profile | quadratic, linear, cubic |

## Performance Targets

| Metric | Good | Excellent |
|--------|------|-----------|
| Q-factor | > 10,000 | > 100,000 |
| Mode volume V | < 1.0 (λ/n)³ | < 0.5 (λ/n)³ |
| Q/V | > 50,000 | > 100,000 |

## How It Works

1. **Unit Cell Setup**: Agent configures the photonic crystal unit cell (period, hole size, waveguide dimensions)

2. **Cavity Design**: For each iteration:
   - Generates GDS layout with specified taper/mirror parameters
   - Runs Lumerical FDTD simulation
   - Extracts Q-factor from resonance peak
   - Computes mode volume normalized to (λ/n)³

3. **Optimization**: Agent uses design history to:
   - Adjust taper hole count to balance Q and V
   - Tune period if resonance drifts from target
   - Modify `min_a_percent` to adjust confinement

## License

MIT
