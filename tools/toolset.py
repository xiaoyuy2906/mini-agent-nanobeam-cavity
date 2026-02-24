from anthropic import Anthropic
from tools.build_gds import build_cavity_gds
from tools.run_lumerical import run_fdtd_simulation
import asyncio


class Toolset:
    def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
        self.model_name = model_name
        self.client = (
            Anthropic(api_key=api_key, base_url=base_url)
            if base_url
            else Anthropic(api_key=api_key)
        )

    def run_llm(
        self,
        system_prompt: str,
        messages: list,
        tools: list | None = None,
        max_tokens: int = 4096,
    ) -> dict:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                tools=tools or [],
                messages=messages,
            )
            return {"ok": True, "data": response}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def build_gds(self, **kwargs) -> dict:
        try:
            cavity = build_cavity_gds(**kwargs)
            return {"ok": True, "data": cavity}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def run_simulation(
        self,
        config: dict,
        mesh_accuracy: int = 8,
        run: bool = True,
    ) -> dict:
        try:
            result = await run_fdtd_simulation(
                config, mesh_accuracy=mesh_accuracy, run=run
            )
            return {"ok": True, "data": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}
