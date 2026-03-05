"""Registered tools for the cavity design agent.

Each tool is defined once: schema + handler together via @tool decorator.
No separate dispatch table needed.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from core.tool_registry import tool

if TYPE_CHECKING:
    from core.agent import CavityAgent


# ---------------------------------------------------------------------------
# set_unit_cell
# ---------------------------------------------------------------------------

SUBSTRATE_LUMERICAL = {
    "SiO2": "SiO2 (Glass) - Palik",
    "Si": "Si (Silicon) - Palik",
    "Diamond": "Diamond - Palik",
    "none": None,
}

@tool(
    name="set_unit_cell",
    description="Set the unit cell parameters. MUST be called first before designing.",
    input_schema={
        "type": "object",
        "properties": {
            "design_wavelength_nm": {"type": "number"},
            "wavelength_span_nm": {"type": "number"},
            "period_nm": {"type": "number"},
            "wg_width_nm": {"type": "number"},
            "wg_height_nm": {"type": "number"},
            "hole_rx_nm": {"type": "number"},
            "hole_ry_nm": {"type": "number"},
            "initial_min_a_percent": {"type": "number"},
            "wg_material": {"type": "string"},
            "wg_material_refractive_index": {"type": "number"},
            "freestanding": {"type": "boolean"},
            "substrate": {"type": "string"},
            "substrate_material_refractive_index": {"type": "number"},
        },
        "required": [
            "design_wavelength_nm", "period_nm", "wg_width_nm", "wg_height_nm",
            "hole_rx_nm", "hole_ry_nm", "wg_material",
            "wg_material_refractive_index", "freestanding",
        ],
    },
)
async def set_unit_cell(agent: CavityAgent, params: dict) -> dict:
    required = [
        "design_wavelength_nm", "period_nm", "wg_width_nm", "wg_height_nm",
        "hole_rx_nm", "hole_ry_nm", "wg_material", "wg_material_refractive_index",
    ]
    missing = [f for f in required if f not in params or params[f] is None]
    if missing:
        return {"ok": False, "error": f"Missing required: {', '.join(missing)}"}

    freestanding = params.get("freestanding", True)
    substrate = params.get("substrate", "none")
    if freestanding:
        substrate = "none"

    nm_to_um = 1e-3
    unit_cell = {
        "design_wavelength": float(params["design_wavelength_nm"]) * 1e-9,
        "wavelength_span": float(params.get("wavelength_span_nm", 100)) * 1e-9,
        "period": float(params["period_nm"]) * nm_to_um,
        "wg_width": float(params["wg_width_nm"]) * nm_to_um,
        "wg_height": float(params["wg_height_nm"]) * nm_to_um,
        "hole_rx": float(params["hole_rx_nm"]) * nm_to_um,
        "hole_ry": float(params["hole_ry_nm"]) * nm_to_um,
        "material": str(params["wg_material"]),
        "material_refractive_index": float(params["wg_material_refractive_index"]),
        "freestanding": freestanding,
        "substrate": substrate,
        "substrate_lumerical": SUBSTRATE_LUMERICAL.get(substrate),
        "substrate_refractive_index": params.get("substrate_material_refractive_index"),
    }
    agent.state.unit_cell = unit_cell
    return {"ok": True, "message": "Unit cell configured"}


# ---------------------------------------------------------------------------
# design_cavity
# ---------------------------------------------------------------------------

@tool(
    name="design_cavity",
    description=(
        "Design a cavity and run Lumerical FDTD for Q/V performance. "
        "You MUST provide a hypothesis explaining your reasoning."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "period_nm": {"type": "number"},
            "wg_width_nm": {"type": "number"},
            "hole_rx_nm": {"type": "number"},
            "hole_ry_nm": {"type": "number"},
            "num_taper_holes": {"type": "integer"},
            "num_mirror_holes": {"type": "integer"},
            "min_a_percent": {"type": "number"},
            "min_rx_percent": {"type": "number"},
            "min_ry_percent": {"type": "number"},
            "taper_type": {"type": "string", "enum": ["linear", "quadratic", "cubic"]},
            "hypothesis": {
                "type": "string",
                "description": "REQUIRED: Explain why you chose these parameters.",
            },
        },
        "required": ["num_taper_holes", "num_mirror_holes", "min_a_percent", "hypothesis"],
    },
    required_state="unit_cell",
)
async def design_cavity(agent: CavityAgent, params: dict) -> dict:
    from tools.build_gds import build_cavity_gds
    from tools.run_lumerical import run_fdtd_simulation

    uc = agent.state.unit_cell
    nm_to_um = 1e-3

    def _get(key_nm, uc_key, default_um):
        v = params.get(key_nm)
        if v is not None:
            return float(v) * nm_to_um
        u = uc.get(uc_key)
        return float(u) if isinstance(u, (int, float)) else default_um

    period = _get("period_nm", "period", 0.2)
    wg_width = _get("wg_width_nm", "wg_width", 0.45)
    hole_rx = _get("hole_rx_nm", "hole_rx", 0.05)
    hole_ry = _get("hole_ry_nm", "hole_ry", 0.1)

    gds_kwargs = {
        "period": period,
        "hole_rx": hole_rx,
        "hole_ry": hole_ry,
        "wg_width": wg_width,
        "num_taper_holes": int(params.get("num_taper_holes", 8)),
        "num_mirror_holes": int(params.get("num_mirror_holes", 10)),
        "min_a_percent": float(params.get("min_a_percent", 90)),
        "min_rx_percent": float(params.get("min_rx_percent", 100)),
        "min_ry_percent": float(params.get("min_ry_percent", 100)),
        "taper_type": str(params.get("taper_type", "quadratic")),
    }

    # Build GDS
    try:
        cavity = build_cavity_gds(**gds_kwargs, save=True)
    except Exception as e:
        return {"ok": False, "error": f"GDS build failed: {e}"}

    config = cavity.get_config()

    # Fill config fields from state
    config.setdefault("unit_cell", {})
    config["unit_cell"]["wg_height"] = uc.get("wg_height", 0.22)
    config["wavelength"] = {
        "design_wavelength": uc.get("design_wavelength", 737e-9),
        "wavelength_span": uc.get("wavelength_span", 100e-9),
    }
    config["substrate"] = {
        "freestanding": uc.get("freestanding", True),
        "material": uc.get("substrate", "none"),
        "material_lumerical": uc.get("substrate_lumerical"),
        "refractive_index": uc.get("substrate_refractive_index"),
    }
    config.setdefault("lumerical", {})
    config["lumerical"]["refractive_index"] = uc.get("material_refractive_index", 2.4)

    # Run FDTD
    sim_result = await run_fdtd_simulation(config=config, mesh_accuracy=8, run=True)
    if isinstance(sim_result, dict) and sim_result.get("error"):
        return {"ok": False, "error": sim_result["error"]}

    # Update state
    log_params = {**params, "period": period, "wg_width": wg_width}
    agent.state.add_design(log_params, sim_result)
    agent.state.save_log()

    return {
        "ok": True,
        "iteration": agent.state.iteration,
        "result": sim_result,
        "best_qv_ratio": agent.state.best_qv_ratio,
    }


# ---------------------------------------------------------------------------
# view_history
# ---------------------------------------------------------------------------

@tool(
    name="view_history",
    description="View the history of all designs tried so far.",
    input_schema={
        "type": "object",
        "properties": {"last_n": {"type": "integer"}},
    },
)
async def view_history(agent: CavityAgent, params: dict) -> dict:
    history = agent.state.design_history
    if not history:
        return {"ok": True, "message": "No designs yet", "history": []}
    last_n = params.get("last_n")
    shown = history[-last_n:] if last_n else history
    return {"ok": True, "history": shown, "total": len(history)}


# ---------------------------------------------------------------------------
# compare_designs
# ---------------------------------------------------------------------------

@tool(
    name="compare_designs",
    description="Compare two or more designs side by side.",
    input_schema={
        "type": "object",
        "properties": {
            "iterations": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["iterations"],
    },
)
async def compare_designs(agent: CavityAgent, params: dict) -> dict:
    designs = []
    for i in params.get("iterations", []):
        entry = next(
            (d for d in agent.state.design_history if d["iteration"] == i), None
        )
        designs.append(entry if entry else {"iteration": i, "error": "Not found"})
    return {"ok": True, "designs": designs}


# ---------------------------------------------------------------------------
# get_best_design
# ---------------------------------------------------------------------------

@tool(
    name="get_best_design",
    description="Get the current best design with highest Q/V.",
    input_schema={"type": "object", "properties": {}},
)
async def get_best_design(agent: CavityAgent, _params: dict) -> dict:
    if agent.state.best_design is None:
        return {"ok": False, "message": "No design yet"}
    return {"ok": True, "best_design": agent.state.best_design}


# ---------------------------------------------------------------------------
# analyze_sensitivity
# ---------------------------------------------------------------------------

@tool(
    name="analyze_sensitivity",
    description=(
        "Analyze how sensitive Q, V, and Q/V are to each design parameter "
        "based on historical data. Returns sensitivities sorted by impact."
    ),
    input_schema={"type": "object", "properties": {}},
)
async def analyze_sensitivity(agent: CavityAgent, _params: dict) -> dict:
    return agent.state.analyze_sensitivity()


# ---------------------------------------------------------------------------
# suggest_next_experiment
# ---------------------------------------------------------------------------

@tool(
    name="suggest_next_experiment",
    description=(
        "Based on design history, suggest the most promising next experiment. "
        "Uses curve fitting to predict optimal parameter values."
    ),
    input_schema={"type": "object", "properties": {}},
)
async def suggest_next_experiment(agent: CavityAgent, _params: dict) -> dict:
    return agent.state.suggest_next_experiment()
