import asyncio
import os
from tools.build_gds import build_cavity_gds
from tools.run_lumerical import run_fdtd_simulation


class Toolset:
    def __init__(self):
        max_parallel = int(os.getenv("MAX_PARALLEL_SIMS", 3))
        self._sim_semaphore = asyncio.Semaphore(max_parallel)

    def build_gds(self, **kwargs) -> dict:
        try:
            cavity = build_cavity_gds(**kwargs, save=True)
            return {"ok": True, "data": cavity}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def run_simulation(
        self, config: dict, mesh_accuracy: int = 8, run: bool = True
    ) -> dict:
        async with self._sim_semaphore:
            try:
                result = await run_fdtd_simulation(
                    config, mesh_accuracy=mesh_accuracy, run=run
                )
                return {"ok": True, "data": result}
            except Exception as e:
                return {"ok": False, "error": str(e)}
