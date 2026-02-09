# Nanobeam Cavity Design Agent

## CRITICAL: TWO-PHASE OPTIMIZATION

### PHASE 1: RESONANCE TUNING (MANDATORY FIRST)
**DO NOT optimize Q/V until resonance is within ±5nm of target!**

After EVERY `design_cavity` call, check `TARGET_STATUS` in the response:
- If `phase: "resonance_tuning"` → ONLY change `period_nm`, DO NOT change any other parameter!
- If `phase: "q_optimization"` → Now you can optimize Q/V

**RESONANCE TUNING RULES:**
- Use `period_nm` parameter in `design_cavity` to adjust wavelength
- Resonance too LOW → INCREASE period_nm (e.g., 225 → 250 → 275)
- Resonance too HIGH → DECREASE period_nm
- Keep ALL other parameters FIXED (taper, mirror, min_a, rx, ry)
- DO NOT touch min_a_percent during resonance tuning!

### PHASE 2: Q/V OPTIMIZATION (only after resonance is on target)
Once resonance is within ±5nm, optimize Q by varying (in order):
1. `num_taper_holes` (8→10→12)
2. `num_mirror_holes` (5→7→10)
3. `min_a_percent` (90→89→88→87, 1% steps) - **MOST IMPORTANT for Q once taper/mirror are explored**
4. `min_rx_percent` = `min_ry_percent` = 100 (DO NOT change unless user explicitly requests)
5. **IF Q is still far from 1,000,000 after trying 1-3**: STOP and ask user to check the unit cell design (period, hole sizes, waveguide dimensions may need adjustment)

**CRITICAL**: When changing taper/mirror holes causes resonance to shift outside ±5nm:
- DO NOT abandon the parameter change!
- Instead: KEEP the new taper/mirror value AND adjust period to compensate
- Example: t10 shifted resonance from 737→734nm? Try t10 with period+3nm to get back on target

## RULE 1: Resonance Controls Everything
- If |resonance - target| > 5nm → **STOP ALL Q OPTIMIZATION**
- Resonance too LOW (blue-shifted) → INCREASE `period_nm` in design_cavity
- Resonance too HIGH (red-shifted) → DECREASE `period_nm` in design_cavity
- **ONLY change period_nm during resonance tuning - DO NOT change min_a_percent or anything else!**

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
    ONLY change period_nm!             Optimize Q via (STRICT ORDER):
    (DO NOT touch min_a or             1. TAPER_HOLES (8→10→12)
     any other parameter!)             2. mirror_holes (5→7→10)
                                       3. min_a_percent (90→89→88→87, 1% steps)
                                       4. min_rx=min_ry=100 (DO NOT change)
                                       5. If Q still far from 1M after 1-3:
                                          ASK USER to check unit cell design

    If a Q parameter shifts resonance out of ±5nm range:
    → Keep the parameter change, adjust period to compensate, then continue
```

## Parameter Effects

| To increase Q | Action | Priority |
|---------------|--------|----------|
| **MOST IMPORTANT** | Lower min_a_percent (90→89→88→87, 1% steps) after taper/mirror sweeps | **#3** |
| Moderate | Add more taper holes (8→10→12) | #1 |
| Moderate | Add more mirror holes (5→7→10) | #2 |
| User-only | min_rx/ry_percent - keep at 100 unless user requests | #4 |

## CRITICAL: User-Only Parameters

**DO NOT change these parameters unless the user EXPLICITLY requests it:**

1. `min_rx_percent`, `min_ry_percent` - Keep at 100 by default
2. `wg_width_nm` - Waveguide width (only change if user requests sweep)
3. `hole_rx_nm`, `hole_ry_nm` - Mirror hole dimensions (only change if user requests sweep)

These parameters significantly affect mode confinement. The user checks E field profiles manually.

**NOTE**: Taper holes control mode matching. Use 8-12 taper holes typically.

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
