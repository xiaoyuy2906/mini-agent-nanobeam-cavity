"""
SWE-Agent style Nanobeam Cavity Designer

Core ideas from SWE-agent:
1. Thought-Action-Observation loop with explicit reasoning
2. Persistent state tracking across iterations
3. Constrained action space with well-defined tools
4. Self-correction based on observations
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic
from build_gds import build_cavity_gds
from run_lumerical import run_fdtd_simulation
import numpy as np

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")

class CavityDesignState:
    """Persistent state tracking for the agent"""

    def __init__(self):
        self.unit_cell = None
        self.design_history = []  # List of all designs tried
        self.best_design = None
        self.best_qv_ratio = 0
        self.iteration = 0

    def add_design(self, params, result):
        """Record a design attempt"""
        self.iteration += 1
        entry = {
            "iteration": self.iteration,
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "result": result,
        }
        self.design_history.append(entry)

        # Track best design
        qv_ratio = result.get("qv_ratio", 0)
        if qv_ratio > self.best_qv_ratio:
            self.best_qv_ratio = qv_ratio
            self.best_design = entry

    def get_summary(self):
        """Get state summary for agent context"""
        return {
            "unit_cell_configured": self.unit_cell is not None,
            "total_iterations": self.iteration,
            "best_qv_ratio": self.best_qv_ratio,
            "designs_tried": len(self.design_history),
        }


class SWECavityAgent:
    """SWE-Agent style cavity designer with thought-action-observation loop"""

    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        self.state = CavityDesignState()
        self.conversation_history = []

        self.system_prompt = self._build_system_prompt()
        self.tools = self._define_tools()

    def _build_system_prompt(self):
        return """You are an expert nanobeam photonic crystal cavity designer. You must follow a single-step tool call loop.

## Workflow Rules
Each turn must strictly output:
1) **THOUGHT**: Reason about the next action based on current state
2) **ACTION**: Call exactly one tool
3) **OBSERVATION**: Receive the result, then continue

## Goal
Maximize Q/V (figure of merit for strong light-matter interaction)
- Q > 10,000 is good, Q > 100,000 is excellent
- V < 1.0 (λ/n)³ is good, V < 0.5 is excellent
- Q/V > 100,000 is a strong target

## Tools
- `set_unit_cell`: Configure the unit cell (must be called first)
- `design_cavity`: Build the cavity and run FDTD to get Q/V
- `view_history`: Inspect previous designs
- `compare_designs`: Compare specific iterations
- `get_best_design`: Retrieve the current best design

## Design Strategy (must follow)
1) Fix unit-cell geometry and target wavelength first
2) Use FDTD as the only performance source (no heuristics)
3) First sweep taper holes and mirror holes only; chirp form is fixed to quadratic
4) Only adjust min_a_percent after testing taper/mirror changes
5) If the simulated resonance is far from the target, adjust the period to re-center it
6) Record Q, V, Q/V, and resonance wavelength; use history for comparisons

## Physics Notes (trend guidance only)
- Increasing taper holes usually increases Q and also increases V
- Quadratic chirp is mandatory; use min_a to adjust confinement
- More mirror holes increase Q but saturate (typically ~12-15)

Requirement: be systematic, reproducible, and comparable across runs."""

    def _define_tools(self):
        return [
            {
                "name": "set_unit_cell",
                "description": "Set the unit cell parameters. MUST be called first before designing.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "design_wavelength_nm": {
                            "type": "number",
                            "description": "Target wavelength in nm (e.g., 737 for SiV, 637 for NV)",
                        },
                        "wavelength_span_nm": {
                            "type": "number",
                            "description": "Half-span in nm (e.g., 100 for ±100nm)",
                        },
                        "period_nm": {
                            "type": "number",
                            "description": "Lattice period in nm",
                        },
                        "wg_width_nm": {
                            "type": "number",
                            "description": "Waveguide width in nm",
                        },
                        "wg_height_nm": {
                            "type": "number",
                            "description": "Waveguide height in nm",
                        },
                        "hole_rx_nm": {
                            "type": "number",
                            "description": "Hole radius x in nm",
                        },
                        "hole_ry_nm": {
                            "type": "number",
                            "description": "Hole radius y in nm",
                        },
                        "material": {
                            "type": "string",
                            "enum": ["SiN", "Si", "Diamond", "GaAs"],
                        },
                        "freestanding": {
                            "type": "boolean",
                            "description": "If true, no substrate is added in FDTD",
                        },
                        "substrate": {
                            "type": "string",
                            "description": "Substrate material name (e.g., SiO2)",
                        },
                    },
                    "required": [
                        "design_wavelength_nm",
                        "period_nm",
                        "wg_width_nm",
                        "wg_height_nm",
                        "hole_rx_nm",
                        "hole_ry_nm",
                        "material",
                    ],
                },
            },
            {
                "name": "design_cavity",
                "description": "Design a cavity and run Lumerical FDTD for Q/V performance.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "num_taper_holes": {
                            "type": "integer",
                            "description": "Number of taper/chirp holes per side (typically 5-15)",
                        },
                        "num_mirror_holes": {
                            "type": "integer",
                            "description": "Number of mirror holes per side (typically 8-15)",
                        },
                        "min_a_percent": {
                            "type": "number",
                            "description": "Minimum period at center as % of original (85-95)",
                        },
                        "min_hole_percent": {
                            "type": "number",
                            "description": "Minimum hole size at center as % (90-100)",
                        },
                        "mesh_accuracy": {
                            "type": "integer",
                            "description": "FDTD mesh accuracy (1-8)",
                        },
                        "hypothesis": {
                            "type": "string",
                            "description": "Your hypothesis for why this design might improve Q/V",
                        },
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
                        "last_n": {
                            "type": "integer",
                            "description": "Show only last N designs (optional, default all)",
                        }
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
                            "description": "List of iteration numbers to compare",
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

    def _execute_tool(self, tool_name, tool_input):
        """Execute a tool and return the observation"""
        if tool_name == "set_unit_cell":
            return self._set_unit_cell(tool_input)
        elif tool_name == "design_cavity":
            return self._design_cavity(tool_input)
        elif tool_name == "view_history":
            return self._view_history(tool_input)
        elif tool_name == "compare_designs":
            return self._compare_designs(tool_input)
        elif tool_name == "get_best_design":
            return self._get_best_design()
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _set_unit_cell(self, params):
        freestanding = params.get("freestanding", True)
        substrate = params.get("substrate", "none")
        if freestanding:
            substrate = "none"

        substrate_map = {
            "SiO2": "SiO2 (Glass) - Palik",
            "Si": "Si (Silicon) - Palik",
            "Diamond": "Diamond - Palik",
            "none": None,
        }

        wavelength_span_nm = params.get("wavelength_span_nm", 100)
        self.state.unit_cell = {
            "design_wavelength": params["design_wavelength_nm"] * 1e-9,
            "wavelength_span": wavelength_span_nm * 1e-9,
            "period": params["period_nm"] * 1e-9,
            "wg_width": params["wg_width_nm"] * 1e-9,
            "wg_height": params["wg_height_nm"] * 1e-9,
            "hole_rx": params["hole_rx_nm"] * 1e-9,
            "hole_ry": params["hole_ry_nm"] * 1e-9,
            "material": params["material"],
            "freestanding": freestanding,
            "substrate": substrate,
            "substrate_lumerical": substrate_map.get(substrate),
        }
        return {
            "status": "success",
            "message": "Unit cell configured",
            "unit_cell": {
                k: f"{v*1e9:.1f} nm" if isinstance(v, float) else v
                for k, v in self.state.unit_cell.items()
            },
            "next_step": "Now call design_cavity to create your first design",
        }

    def _design_cavity(self, params):
        if self.state.unit_cell is None:
            return {"error": "Must call set_unit_cell first!"}

        # Extract params
        design_params = {
            "num_taper_holes": params["num_taper_holes"],
            "num_mirror_holes": params["num_mirror_holes"],
            "taper_type": "quadratic",
            "min_a_percent": params["min_a_percent"],
            "min_hole_percent": params.get("min_hole_percent", 100),
        }
        hypothesis = params.get("hypothesis", "No hypothesis provided")

        # Build GDS (optional, for visualization)
        try:
            cavity = build_cavity_gds(
                period=self.state.unit_cell["period"] * 1e6,
                hole_rx=self.state.unit_cell["hole_rx"] * 1e6,
                hole_ry=self.state.unit_cell["hole_ry"] * 1e6,
                wg_width=self.state.unit_cell["wg_width"] * 1e6,
                num_taper_holes=design_params["num_taper_holes"],
                num_mirror_holes=design_params["num_mirror_holes"],
                taper_type=design_params["taper_type"],
                min_a_percent=design_params["min_a_percent"],
                min_rx_percent=design_params["min_hole_percent"],
                min_ry_percent=design_params["min_hole_percent"],
                save=True,
            )
            gds_file = cavity.gds_filepath
        except Exception as e:
            gds_file = f"Error: {e}"

        # Real FDTD simulation (required)
        mesh_accuracy = params.get("mesh_accuracy", 8)
        if not isinstance(gds_file, str) or gds_file.startswith("Error"):
            return {"error": f"GDS build failed: {gds_file}"}

        config = cavity.get_config()
        config["unit_cell"]["wg_height"] = self.state.unit_cell["wg_height"] * 1e6
        config["wavelength"] = {
            "design_wavelength": self.state.unit_cell["design_wavelength"],
            "wavelength_span": self.state.unit_cell["wavelength_span"],
        }
        config["substrate"] = {
            "freestanding": self.state.unit_cell["freestanding"],
            "material": self.state.unit_cell["substrate"],
            "material_lumerical": self.state.unit_cell["substrate_lumerical"],
        }
        try:
            fdtd_result = run_fdtd_simulation(
                config, mesh_accuracy=mesh_accuracy, run=True
            )
        except Exception as e:
            return {"error": f"FDTD failed: {e}"}

        if not fdtd_result or not fdtd_result.get("Q") or not fdtd_result.get("V"):
            return {"error": f"FDTD returned invalid result: {fdtd_result}"}

        if fdtd_result.get("resonance_nm") is not None:
            resonance_nm = float(np.round(fdtd_result["resonance_nm"], 2))
        else:
            resonance_nm = float(np.round(self.state.unit_cell["design_wavelength"] * 1e9, 2))

        performance = {
            "Q": int(fdtd_result["Q"]),
            "V": float(np.round(fdtd_result["V"], 3)),
            "qv_ratio": int(fdtd_result["Q"] / fdtd_result["V"]),
            "resonance_wavelength_nm": resonance_nm,
            "notes": [f"FDTD result at {resonance_nm} nm"],
        }

        # Record in history
        result = {
            "Q": performance["Q"],
            "V": performance["V"],
            "qv_ratio": performance["qv_ratio"],
            "resonance_nm": performance["resonance_wavelength_nm"],
            "gds_file": gds_file,
        }
        self.state.add_design(design_params, result)

        # Build observation
        is_best = result["qv_ratio"] >= self.state.best_qv_ratio

        return {
            "iteration": self.state.iteration,
            "hypothesis": hypothesis,
            "parameters": design_params,
            "results": {
                "Q": f"{performance['Q']:,}",
                "V": f"{performance['V']:.3f} (λ/n)³",
                "Q/V": f"{performance['qv_ratio']:,}",
                "resonance": f"{performance['resonance_wavelength_nm']} nm",
            },
            "analysis": performance["notes"],
            "is_new_best": is_best,
            "best_qv_so_far": f"{self.state.best_qv_ratio:,}",
            "gds_file": gds_file,
            "fdtd_result": fdtd_result,
        }

    def _view_history(self, params):
        if not self.state.design_history:
            return {"message": "No designs yet. Call design_cavity first."}

        last_n = params.get("last_n")
        history = self.state.design_history
        if last_n:
            history = history[-last_n:]

        summary = []
        for entry in history:
            summary.append(
                {
                    "iteration": entry["iteration"],
                    "params": {
                        "taper": entry["params"]["num_taper_holes"],
                        "mirror": entry["params"]["num_mirror_holes"],
                        "type": entry["params"]["taper_type"],
                        "min_a%": entry["params"]["min_a_percent"],
                        "min_hole%": entry["params"].get("min_hole_percent", 100),
                    },
                    "Q": f"{entry['result']['Q']:,}",
                    "V": f"{entry['result']['V']:.3f}",
                    "Q/V": f"{entry['result']['qv_ratio']:,}",
                }
            )

        return {
            "total_designs": len(self.state.design_history),
            "showing": len(summary),
            "history": summary,
            "best_iteration": (
                self.state.best_design["iteration"] if self.state.best_design else None
            ),
        }

    def _compare_designs(self, params):
        iterations = params["iterations"]
        designs = []

        for i in iterations:
            entry = next(
                (d for d in self.state.design_history if d["iteration"] == i), None
            )
            if entry:
                designs.append(
                    {
                        "iteration": i,
                        "params": entry["params"],
                        "Q": entry["result"]["Q"],
                        "V": entry["result"]["V"],
                        "qv_ratio": entry["result"]["qv_ratio"],
                    }
                )
            else:
                designs.append({"iteration": i, "error": "Not found"})

        # Compute differences if comparing 2 designs
        if (
            len(designs) == 2
            and "error" not in designs[0]
            and "error" not in designs[1]
        ):
            d1, d2 = designs
            diff = {
                "Q_change": f"{(d2['Q'] - d1['Q']) / d1['Q'] * 100:+.1f}%",
                "V_change": f"{(d2['V'] - d1['V']) / d1['V'] * 100:+.1f}%",
                "QV_change": f"{(d2['qv_ratio'] - d1['qv_ratio']) / d1['qv_ratio'] * 100:+.1f}%",
            }
            return {"designs": designs, "changes": diff}

        return {"designs": designs}

    def _get_best_design(self):
        if not self.state.best_design:
            return {"message": "No designs yet"}

        return {
            "best_design": self.state.best_design,
            "total_iterations": self.state.iteration,
            "improvement_potential": "Try: more taper holes, lower min_a_percent",
        }

    def chat(self, user_message):
        """Main chat loop with thought-action-observation pattern"""
        self.conversation_history.append({"role": "user", "content": user_message})

        # Add state context
        state_context = f"\n\n[Current State: {json.dumps(self.state.get_summary())}]"
        messages_with_state = self.conversation_history.copy()
        messages_with_state[-1] = {
            "role": "user",
            "content": user_message + state_context,
        }

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=self.system_prompt,
            tools=self.tools,
            messages=messages_with_state,
        )

        # Tool use loop
        while response.stop_reason == "tool_use":
            # Extract tool call
            tool_use = next(
                block for block in response.content if block.type == "tool_use"
            )
            tool_name = tool_use.name
            tool_input = tool_use.input

            # Execute tool
            print(f"\n  [ACTION: {tool_name}]")
            observation = self._execute_tool(tool_name, tool_input)
            print(f"  [OBSERVATION: {json.dumps(observation, indent=2)[:500]}...]")

            # Add to history
            self.conversation_history.append(
                {"role": "assistant", "content": response.content}
            )
            self.conversation_history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": json.dumps(observation),
                        }
                    ],
                }
            )

            # Continue conversation
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                messages=self.conversation_history,
            )

        # Extract final response
        final_response = next(
            (block.text for block in response.content if hasattr(block, "text")),
            None,
        )
        self.conversation_history.append(
            {"role": "assistant", "content": response.content}
        )

        return final_response


def main():
    print("=" * 60)
    print("SWE-Agent Style Nanobeam Cavity Designer")
    print("=" * 60)
    print("\nThis agent uses a thought-action-observation loop to")
    print("systematically design and optimize photonic cavities.")
    print("\nExample: 'Design a SiN cavity for SiV centers at 737nm'")
    print("         'Optimize Q/V with 5 iterations'")
    print("\nType 'quit' to exit.\n")

    agent = SWECavityAgent(api_key=ANTHROPIC_API_KEY)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if not user_input:
            continue

        print("\n" + "-" * 40)
        response = agent.chat(user_input)
        print("-" * 40)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
