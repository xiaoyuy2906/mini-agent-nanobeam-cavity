from core.state import CavityDesignState
from tools.toolset import Toolset


class CavityAgent:
    """Orchestration layer: state + tools + workflow."""

    def __init__(self, toolset: Toolset, state: CavityDesignState | None = None):
        self.toolset = toolset
        self.state = state or CavityDesignState()

    def set_unit_cell(self, unit_cell: dict) -> dict:
        if not isinstance(unit_cell, dict) or not unit_cell:
            return {"ok": False, "error": "unit_cell must be a non-empty dict"}

        self.state.unit_cell = unit_cell
        return {"ok": True, "message": "Unit cell configured"}

    async def design_cavity(self, design_params: dict, run: bool = True) -> dict:
        if self.state.unit_cell is None:
            return {"ok": False, "error": "Call set_unit_cell first"}

        # 1) Build GDS
        gds_ret = self.toolset.build_gds(**design_params)
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

        # 4) Update state
        self.state.add_design(design_params, sim_result)
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

    def get_summary(self) -> dict:
        return {"ok": True, "summary": self.state.get_summary()}
