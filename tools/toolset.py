from tools.build_gds import build_cavity_gds
from tools.run_lumerical import run_fdtd_simulation


class Toolset:
    def build_gds(self, **kwargs) -> dict:
        try:
            cavity = build_cavity_gds(**kwargs)
            return {"ok": True, "data": cavity}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def run_simulation(
        self, config: dict, mesh_accuracy: int = 8, run: bool = True
    ) -> dict:
        try:
            result = await run_fdtd_simulation(
                config, mesh_accuracy=mesh_accuracy, run=run
            )
            return {"ok": True, "data": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}
