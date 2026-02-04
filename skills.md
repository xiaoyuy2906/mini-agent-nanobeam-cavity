# Nanobeam Cavity Design Agent

## CRITICAL: TWO-PHASE OPTIMIZATION

### PHASE 1: RESONANCE TUNING (MANDATORY FIRST)
**DO NOT optimize Q/V until resonance is within ±5nm of target!**

After EVERY `design_cavity` call, check `TARGET_STATUS` in the response:
- If `phase: "resonance_tuning"` → ONLY adjust period, ignore Q/V completely
- If `phase: "q_optimization"` → Now you can optimize Q/V

### PHASE 2: Q/V OPTIMIZATION (only after resonance is on target)
Once resonance is within ±5nm, optimize Q by varying:
1. `num_taper_holes` (10→12→14→16) - **MOST IMPORTANT for high Q!**
2. `num_mirror_holes` (12→14→16→18→20)
3. `min_a_percent` (90→88→86→85)
4. `min_rx_percent` (100→98→96)
5. `min_ry_percent` (100→98→96)

**CRITICAL**: When changing taper/mirror holes causes resonance to shift outside ±5nm:
- DO NOT abandon the parameter change!
- Instead: KEEP the new taper/mirror value AND adjust period to compensate
- Example: t10 shifted resonance from 737→734nm? Try t10 with period+3nm to get back on target
- More taper holes = much higher Q potential, so always push taper holes higher!

## RULE 1: Resonance Controls Everything
- If |resonance - target| > 10nm → **STOP ALL Q OPTIMIZATION**
- Resonance too LOW (blue-shifted) → INCREASE period by 5-10nm
- Resonance too HIGH (red-shifted) → DECREASE period by 5-10nm
- **PERIOD is the ONLY parameter that shifts wavelength significantly**

## RULE 2: Never Repeat Parameters
Every iteration must try NEW parameter values. Check `view_history` before designing.

## RULE 3: One Change at a Time
Change only ONE parameter per iteration to understand its effect.

## Decision Tree

```
START → set_unit_cell → design_cavity
                              ↓
                    Check TARGET_STATUS.phase
                              ↓
            ┌─────────────────┴─────────────────┐
            ↓                                   ↓
    phase="resonance_tuning"           phase="q_optimization"
            ↓                                   ↓
    ONLY adjust period                 Optimize Q via (in order):
    (ignore Q/V values!)               1. TAPER_HOLES (10→12→14→16)
                                       2. mirror_holes (12→14→16→18→20)
                                       3. min_a_percent, min_rx%, min_ry%

    If a Q parameter shifts resonance out of ±5nm range:
    → Keep the parameter change, adjust period to compensate, then continue
```

## Parameter Effects

| To increase Q | Action | Priority |
|---------------|--------|----------|
| **MOST EFFECTIVE** | Add more taper holes (10→12→14→16) | **#1** |
| Very effective | Add more mirror holes (12→14→16→20) | #2 |
| Moderate | Lower min_a_percent (90→88→86→84) | #3 |
| Fine-tune | Lower min_rx_percent (100→90→80) | #4 |
| Fine-tune | Lower min_ry_percent (100→90→80) | #5 |

**NOTE**: Taper holes control mode matching - more tapers = gentler transition = higher Q!
Always try to push taper holes to 12-16 before concluding optimization.

| To shift resonance | Action |
|--------------------|--------|
| Red-shift (increase λ) | INCREASE period (only effective method) |
| Blue-shift (decrease λ) | DECREASE period (only effective method) |

**WARNING**: min_a_percent, min_rx_percent, min_ry_percent have MINIMAL effect on wavelength. Use PERIOD to tune resonance!

## Targets

| Metric | Target | Priority |
|--------|--------|----------|
| Resonance | within ±5nm of target | **HIGHEST - must meet first** |
| Q | > 1,000,000 | Second |
| V | < 0.5 (λ/n)³ | Third |

## Workflow

1. `set_unit_cell` - configure geometry (REQUIRED FIRST)
2. `design_cavity` - run FDTD, get Q/V/resonance
3. **CHECK TARGET_STATUS.phase** - this determines what to do next:
   - `resonance_tuning`: adjust period only
   - `q_optimization`: tune Q/V parameters
4. Repeat until `phase: "complete"` or max iterations
5. `get_best_design` - report final result

## KEEP ITERATING
Do NOT stop until:
- All targets are met (TARGET_STATUS.on_target = true), OR
- User explicitly asks to stop, OR
- You have exhausted reasonable parameter combinations
