# Nanobeam Cavity Agent Skill

## Purpose
Design high-Q nanobeam photonic crystal cavities for color centers (SiV, NV) using FDTD simulation as the sole performance metric.

## Workflow (Thought-Action-Observation Loop)
1. **THOUGHT**: Analyze current state, form hypothesis about next design
2. **ACTION**: Call exactly one tool
3. **OBSERVATION**: Analyze result, update strategy, repeat

## Performance Targets

| Metric | Good | Excellent | Units |
|--------|------|-----------|-------|
| Q | > 500,000 | > 1,000,000 | dimensionless |
| V | < 1.0 | < 0.5 | (λ/n)³ |
| Q/V | > 500,000 | > 2,000,000 | (λ/n)⁻³ |
| Resonance | within ±10nm | within ±5nm | of target λ |

## Tools
| Tool | Purpose |
|------|---------|
| `set_unit_cell` | Configure geometry and wavelength (call FIRST) |
| `design_cavity` | Build cavity + run FDTD → returns Q, V, resonance |
| `view_history` | List all previous designs |
| `compare_designs` | Compare specific iterations side-by-side |
| `get_best_design` | Retrieve current best Q/V design |

## Parameter Ranges

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `num_taper_holes` | 5-15 | More → higher Q, larger V |
| `num_mirror_holes` | 8-20 | More → higher Q (saturates ~15) |
| `min_a_percent` | 85%-95% | Lower → stronger confinement, blue-shift |
| `taper_type` | quadratic, linear, exponential | quadratic usually best |
| `min_hole_percent` | 95%-100% | Lower → smaller holes at center |

## Optimization Strategy

### Phase 1: Baseline (iterations 1-3)
- Start with conservative parameters: 10 taper, 12 mirror, quadratic, 90% min_a
- Establish baseline Q/V and resonance wavelength

### Phase 2: Period Tuning (iterations 4-6)
- Adjust period to match target resonance wavelength
- Goal: Get resonance within ±10nm of target before optimizing Q/V

### Phase 3: Mirror Optimization (iterations 7-12)
- Fix taper parameters, sweep mirror holes: 10 → 12 → 14 → 16 → 18
- Mirror holes primarily affect Q without shifting resonance much

### Phase 4: Taper Optimization (iterations 13-20)
- Fix best mirror count, sweep taper holes and min_a_percent
- More taper holes + lower min_a = higher Q but larger V
- Fine-tune for maximum Q/V while keeping resonance on target

## Wavelength Tuning

| Resonance vs Target | Action |
|---------------------|--------|
| Blue-shifted (λ < target) | Increase period OR decrease min_a_percent |
| Red-shifted (λ > target) | Decrease period OR increase min_a_percent |
| On-target | Focus on Q/V optimization |

## Physics Guidelines

- **Q sources**: Taper smoothness (gradual mode transition) + mirror reflectivity
- **V sources**: Taper strength (min_a_percent) controls mode confinement
- **Trade-off**: Aggressive tapering (low min_a) increases Q but also V
- **Saturation**: Mirror holes beyond ~15-18 give diminishing returns
- **Resonance**: Primarily controlled by period and min_a_percent

## Common Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Q < 10,000 | Too few mirrors or weak taper | Increase mirror holes to 12+ |
| V > 2.0 | Too many taper holes | Reduce taper holes or increase min_a |
| Resonance far off | Wrong period | Adjust period by ±10-20nm |
| Q not improving | Saturated mirrors | Try different taper_type |

## Requirements
- Be systematic: change ONE parameter at a time when possible
- Record ALL results: Q, V, Q/V, resonance_nm for every iteration
- Compare against history: use `view_history` and `compare_designs`
- Report best design: always end with `get_best_design`
