import os
import json
import time
import asyncio

from core.state import CavityDesignState
from tools.toolset import Toolset

SWEEP_ORDER = [
    "sweep_min_a",
    "sweep_rx",
    "re_sweep_min_a_1",
    "sweep_ry",
    "re_sweep_min_a_2",
    "sweep_taper",
    "fine_period",
    "complete",
]


class CavityAgent:
    """Orchestration layer: state + tools + workflow."""

    def __init__(self, toolset: Toolset, state: CavityDesignState | None = None):
        self.toolset = toolset
        self.state = state or CavityDesignState()
        self.model_provider = os.getenv("MODEL_PROVIDER", "claude")
        self.conversation_history = []
        self.system_prompt = self._build_system_prompt()
        self.tools = self._define_tools()

    def _build_system_prompt(self) -> str:
        """Load skills.md if available, fall back to base prompt."""
        try:
            with open("./skills.md", "r", encoding="utf-8") as f:
                skills_text = f.read().strip()
        except OSError:
            skills_text = ""

        override_rule = (
            "## CRITICAL: USER INPUT OVERRIDES EVERYTHING\n"
            "If the user gives explicit instructions, YOU MUST FOLLOW THEM EXACTLY.\n"
            "NEVER invent missing unit-cell geometry. If a required value is missing, ask the user.\n"
            "Before the FIRST FDTD run, show all unit-cell inputs and ask user confirmation.\n"
            "Only proceed after user says 'confirm fdtd'.\n\n"
        )

        base_prompt = (
            "You are an expert nanobeam photonic crystal cavity designer.\n\n"
            "## Workflow\n"
            "Each turn: THOUGHT → ACTION (one tool) → OBSERVATION\n\n"
            "## Goal\n"
            "Maximize Q/V. Q > 1,000,000 and V < 0.5 (λ/n)³ is excellent.\n\n"
            "## Tools\n"
            "- set_unit_cell: configure geometry (call first)\n"
            "- design_cavity: build GDS + run FDTD\n"
            "- view_history: inspect previous designs\n"
            "- compare_designs: compare specific iterations\n"
            "- get_best_design: retrieve current best\n"
        )

        if skills_text:
            return override_rule + skills_text
        return override_rule + base_prompt

    def _define_tools(self) -> list:
        """Return the 5 tool schemas passed to every LLM call."""
        return [
            {
                "name": "set_unit_cell",
                "description": "Set the unit cell parameters. MUST be called first before designing.",
                "input_schema": {
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
                        "design_wavelength_nm",
                        "period_nm",
                        "wg_width_nm",
                        "wg_height_nm",
                        "hole_rx_nm",
                        "hole_ry_nm",
                        "wg_material",
                        "wg_material_refractive_index",
                        "freestanding",
                    ],
                },
            },
            {
                "name": "design_cavity",
                "description": "Design a cavity and run Lumerical FDTD for Q/V performance.",
                "input_schema": {
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
                        "hypothesis": {"type": "string"},
                    },
                    "required": [
                        "num_taper_holes",
                        "num_mirror_holes",
                        "min_a_percent",
                    ],
                },
            },
            {
                "name": "view_history",
                "description": "View the history of all designs tried so far.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "last_n": {"type": "integer"},
                    },
                },
            },
            {
                "name": "compare_designs",
                "description": "Compare two or more designs side by side.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "iterations": {
                            "type": "array",
                            "items": {"type": "integer"},
                        }
                    },
                    "required": ["iterations"],
                },
            },
            {
                "name": "get_best_design",
                "description": "Get the current best design with highest Q/V.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    SUBSTRATE_LUMERICAL = {
        "SiO2": "SiO2 (Glass) - Palik",
        "Si": "Si (Silicon) - Palik",
        "Diamond": "Diamond - Palik",
        "none": None,
    }

    def set_unit_cell(self, unit_cell: dict) -> dict:
        if not isinstance(unit_cell, dict) or not unit_cell:
            return {"ok": False, "error": "unit_cell must be a non-empty dict"}

        self.state.unit_cell = unit_cell
        return {"ok": True, "message": "Unit cell configured"}

    def set_unit_cell_from_tool_params(self, params: dict) -> dict:
        """Convert flat tool params to unit_cell and call set_unit_cell."""
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

        nm_to_um = 1e-3  # 200 nm = 0.2 um
        unit_cell = {
            "design_wavelength": float(params["design_wavelength_nm"]) * 1e-9,  # nm -> m
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
            "substrate_lumerical": self.SUBSTRATE_LUMERICAL.get(substrate),
            "substrate_refractive_index": params.get("substrate_material_refractive_index"),
        }
        return self.set_unit_cell(unit_cell)

    def _design_params_to_build_gds_kwargs(self, params: dict) -> dict:
        """Merge unit_cell with design params and convert to build_gds kwargs (microns)."""
        uc = self.state.unit_cell or {}
        nm_to_um = 1e-3

        def _get(key_nm: str, uc_key: str, default_um: float) -> float:
            v = params.get(key_nm)
            if v is not None:
                return float(v) * nm_to_um
            u = uc.get(uc_key)
            if isinstance(u, (int, float)):
                return float(u)
            return default_um

        period = _get("period_nm", "period", 0.2)
        wg_width = _get("wg_width_nm", "wg_width", 0.45)
        hole_rx = _get("hole_rx_nm", "hole_rx", 0.05)
        hole_ry = _get("hole_ry_nm", "hole_ry", 0.1)

        return {
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

    async def design_cavity(self, design_params: dict, run: bool = True) -> dict:
        if self.state.unit_cell is None:
            return {"ok": False, "error": "Call set_unit_cell first"}

        kwargs = self._design_params_to_build_gds_kwargs(design_params)

        # 1) Build GDS
        gds_ret = self.toolset.build_gds(**kwargs)
        if not gds_ret["ok"]:
            return gds_ret

        cavity = gds_ret["data"]
        config = cavity.get_config()

        # 2) Fill required config fields from state
        uc = self.state.unit_cell
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
        config["lumerical"]["refractive_index"] = uc.get(
            "material_refractive_index", 2.4
        )

        # 3) Run simulation (async)
        sim_ret = await self.toolset.run_simulation(
            config=config, mesh_accuracy=8, run=run
        )
        if not sim_ret["ok"]:
            return sim_ret

        sim_result = sim_ret["data"]

        # 4) Update state (store params for display)
        log_params = {**design_params, "period": kwargs.get("period"), "wg_width": kwargs.get("wg_width")}
        self.state.add_design(log_params, sim_result)
        self.state.save_log()

        return {
            "ok": True,
            "iteration": self.state.iteration,
            "result": sim_result,
            "best_qv_ratio": self.state.best_qv_ratio,
        }

    def get_best_design(self) -> dict:
        if self.state.best_design is None:
            return {"ok": False, "message": "No design yet"}
        return {"ok": True, "best_design": self.state.best_design}

    def view_history(self, last_n: int | None = None) -> dict:
        history = self.state.design_history
        if not history:
            return {"ok": True, "message": "No designs yet", "history": []}
        if last_n:
            history = history[-last_n:]
        return {"ok": True, "history": history, "total": len(self.state.design_history)}

    def compare_designs(self, iterations: list[int]) -> dict:
        designs = []
        for i in iterations:
            entry = next(
                (d for d in self.state.design_history if d["iteration"] == i), None
            )
            if entry:
                designs.append(entry)
            else:
                designs.append({"iteration": i, "error": "Not found"})
        return {"ok": True, "designs": designs}

    def get_summary(self) -> dict:
        return {"ok": True, "summary": self.state.get_summary()}
