Nanobeam Cavity Agent Skill

Purpose
- Design nanobeam photonic crystal cavities using FDTD as the only performance source.

Workflow Rules
1) THOUGHT: reason about the next action based on current state
2) ACTION: call exactly one tool
3) OBSERVATION: receive the result, then continue

Goals
- Maximize Q/V (figure of merit for strong light-matter interaction)
- Q > 10,000 is good, Q > 100,000 is excellent
- V < 1.0 (λ/n)^3 is good, V < 0.5 is excellent
- Q/V > 100,000 is a strong target

Tools
- set_unit_cell: configure the unit cell (must be called first)
- design_cavity: build the cavity and run FDTD to get Q/V
- view_history: inspect previous designs
- compare_designs: compare specific iterations
- get_best_design: retrieve the current best design

Design Strategy (must follow)
1) Fix unit-cell geometry and target wavelength first
2) Use FDTD as the only performance source (no heuristics)
3) First sweep taper holes and mirror holes only; then adjust period if resonance is off-target
4) After period tuning, adjust min_a_percent if needed
5) Record Q, V, Q/V, and resonance wavelength; use history for comparisons

Strategy for iterations
- If resonance is blue-shifted: increase period or decrease min_a_percent
- If resonance is red-shifted: decrease period or increase min_a_percent
- Adjust num_taper_holes and num_mirror_holes to maximize Q/V
- Track resonance wavelength and Q/V for each design; report the best configuration
- Vary taper_type (quadratic, linear, exponential) to optimize Q

Physics Notes (trend guidance only)
- Increasing taper holes usually increases Q and also increases V
- Taper profile affects Q; quadratic is a strong baseline
- More mirror holes increase Q but saturate (typically ~12-15)

Requirements
- Be systematic, reproducible, and comparable across runs.
