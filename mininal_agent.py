import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from build_gds import build_cavity_gds
from run_lumerical import run_fdtd_simulation

load_dotenv()

# Check API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LUMPAPI_PATH = os.getenv("LUMPAPI_PATH")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")

print(f"API Key loaded: {ANTHROPIC_API_KEY[:20]}...")
if LUMPAPI_PATH:
    print(f"Lumerical path: {LUMPAPI_PATH}")
else:
    print("Lumerical not configured - simulation will be skipped")


class CavityAgent:
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        self.conversation_history = []
        self.current_cavity = None

        # SiN nanobeam freestanding cavity unit cell parameters
        # self.unit_cell = {
        #     "width": 0.45e-6,  # 0.45 µm - SiN waveguide width
        #     "height": 0.2e-6,  # 0.2 µm - SiN waveguide height
        #     "a": 200e-9,  # 200 nm - Lattice period
        #     "rx": 50e-9,  # 50 nm - Air hole radius (x-direction)
        #     "ry": 100e-9,  # 100 nm - Air hole radius (y-direction)
        # }

        self.system_prompt = (
            "You are an expert nanobeam photonic cavity designer.\n"
            "\n"
            "MAIN GOAL: Maximize Q/V (quality factor / mode volume) for strong light-matter interaction.\n"
            "\n"
            "WORKFLOW:\n"
            "1. FIRST: Get unit cell and DESIGN WAVELENGTH from user:\n"
            "   - design_wavelength_nm: CRITICAL! The target wavelength (e.g., 737 for SiV, 637 for NV)\n"
            "   - wavelength_span_nm: simulation range (default ±100nm)\n"
            "   - period (nm), wg_width (nm), wg_height (nm)\n"
            "   - hole_rx (nm), hole_ry (nm) - elliptical holes (rx=ry for round)\n"
            "   - material (SiN, Si, Diamond, GaAs)\n"
            "   - freestanding or on substrate?\n"
            "   Common platforms:\n"
            "   - SiN freestanding for SiV (737nm)\n"
            "   - Diamond freestanding for NV (637nm)\n"
            "   - GaAs on Diamond\n"
            "   If not freestanding, ask about substrate material.\n"
            "   Call set_unit_cell tool.\n"
            "\n"
            "2. THEN: Design cavity to maximize Q/V.\n"
            "   Call design_and_simulate with YOUR chosen parameters.\n"
            "\n"
            "KEY PHYSICS:\n"
            "- TAPER HOLES (chirp holes): More taper holes increases BOTH Q and V,\n"
            "  but Q/V ratio still improves. No hard limit on range.\n"
            "- MIRROR HOLES: Sufficient mirrors needed, but too many has diminishing returns.\n"
            "- TAPER PROFILE: Quadratic is standard.\n"
            "- ONLY CHIRP THE PERIOD initially, do NOT chirp hole size (min_hole_percent=100).\n"
            "\n"
            "INITIAL DESIGN (first attempt):\n"
            "- num_taper_holes: 8\n"
            "- num_mirror_holes: 10\n"
            "- taper_type: quadratic\n"
            "- min_a_percent: 90\n"
            "- min_hole_percent: 100 (no hole chirp initially)\n"
            "\n"
            "OPTIMIZATION STRATEGY (if Q/V is not ideal):\n"
            "1. Increase num_taper_holes (e.g., 10, 12, 15...)\n"
            "2. Decrease min_a_percent (e.g., 85, 80, 75...)\n"
            "3. Only add hole chirp (min_hole_percent < 100) as a last resort\n"
            "4. These are starting points, not limits. Explore freely.\n"
        )

        self.tools = [
            {
                "name": "set_unit_cell",
                "description": "Set unit cell parameters from user design. Call this first before designing cavity.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "design_wavelength_nm": {
                            "type": "number",
                            "description": "Target design wavelength in nanometers (e.g., 737 for SiV, 637 for NV center)",
                        },
                        "wavelength_span_nm": {
                            "type": "number",
                            "description": "Wavelength span for simulation in nm (default 100, meaning ±100nm around design wavelength)",
                        },
                        "period_nm": {
                            "type": "number",
                            "description": "Lattice period in nanometers",
                        },
                        "wg_width_nm": {
                            "type": "number",
                            "description": "Waveguide width in nanometers",
                        },
                        "wg_height_nm": {
                            "type": "number",
                            "description": "Waveguide height/thickness in nanometers",
                        },
                        "hole_rx_nm": {
                            "type": "number",
                            "description": "Hole radius in x-direction in nanometers",
                        },
                        "hole_ry_nm": {
                            "type": "number",
                            "description": "Hole radius in y-direction in nanometers (same as rx for round holes)",
                        },
                        "material": {
                            "type": "string",
                            "enum": ["SiN", "Si", "Diamond", "GaAs"],
                            "description": "Waveguide material",
                        },
                        "freestanding": {
                            "type": "boolean",
                            "description": "True if freestanding (air clad), False if on substrate",
                        },
                        "substrate": {
                            "type": "string",
                            "enum": ["none", "Si", "SiO2", "Diamond"],
                            "description": "Substrate material. Use none if freestanding.",
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
                        "freestanding",
                    ],
                },
            },
            {
                "name": "design_and_simulate",
                "description": "Design cavity taper to maximize Q/V and run FDTD simulation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "num_taper_holes": {
                            "type": "integer",
                            "description": "Number of taper/chirp holes per side (5-10). More = higher Q/V.",
                        },
                        "num_mirror_holes": {
                            "type": "integer",
                            "description": "Number of mirror holes per side (8-12). More than 12 has limited benefit.",
                        },
                        "taper_type": {
                            "type": "string",
                            "enum": ["linear", "quadratic", "cubic"],
                            "description": "Taper profile. Quadratic is standard.",
                        },
                        "min_a_percent": {
                            "type": "number",
                            "description": "Min period at cavity center as % of original (70-90)",
                        },
                        "min_hole_percent": {
                            "type": "number",
                            "description": "Min hole size (rx, ry) at cavity center as % (70-100). Same scaling for both.",
                        },
                        "run_simulation": {
                            "type": "boolean",
                            "description": "If true, run FDTD simulation",
                        },
                    },
                    "required": [
                        "num_taper_holes",
                        "num_mirror_holes",
                        "taper_type",
                        "min_a_percent",
                    ],
                },
            },
        ]

    def chat(self, user_message):
        self.conversation_history.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=self.system_prompt,
            tools=self.tools,
            messages=self.conversation_history,
        )

        while response.stop_reason == "tool_use":
            tool_use = next(
                block for block in response.content if block.type == "tool_use"
            )
            tool_result = self._execute_tool(tool_use.name, tool_use.input)

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
                            "content": str(tool_result),
                        }
                    ],
                }
            )

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                messages=self.conversation_history,
            )

        final_response = next(
            (block.text for block in response.content if hasattr(block, "text")),
            None,
        )
        self.conversation_history.append(
            {"role": "assistant", "content": response.content}
        )
        return final_response

    def _execute_tool(self, tool_name, tool_input):
        if tool_name == "set_unit_cell":
            return self._set_unit_cell(tool_input)
        elif tool_name == "design_and_simulate":
            return self._design_and_simulate(tool_input)
        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}

    def _set_unit_cell(self, params):
        """Set unit cell parameters from user input"""
        # Material mapping for Lumerical
        material_map = {
            "SiN": "Si3N4 (Silicon Nitride) - Phillip",
            "Si": "Si (Silicon) - Palik",
            "Diamond": "C (Diamond) - Phillip",
            "GaAs": "GaAs (Gallium Arsenide) - Palik",
        }

        substrate_map = {
            "Si": "Si (Silicon) - Palik",
            "SiO2": "SiO2 (Glass) - Palik",
            "Diamond": "C (Diamond) - Phillip",
            "none": None,
        }

        freestanding = params["freestanding"]
        substrate = params.get("substrate", "none")
        if freestanding:
            substrate = "none"

        # Design wavelength (critical parameter)
        design_wavelength_nm = params["design_wavelength_nm"]
        wavelength_span_nm = params.get("wavelength_span_nm", 100)  # default ±100nm

        self.unit_cell = {
            "a": params["period_nm"] * 1e-9,
            "width": params["wg_width_nm"] * 1e-9,
            "height": params["wg_height_nm"] * 1e-9,
            "rx": params["hole_rx_nm"] * 1e-9,
            "ry": params["hole_ry_nm"] * 1e-9,
            "material": params["material"],
            "material_lumerical": material_map.get(
                params["material"], "Si3N4 (Silicon Nitride) - Phillip"
            ),
            "freestanding": freestanding,
            "substrate": substrate,
            "substrate_lumerical": substrate_map.get(substrate, None),
            # Wavelength parameters
            "design_wavelength": design_wavelength_nm * 1e-9,  # convert to meters
            "wavelength_span": wavelength_span_nm * 1e-9,  # convert to meters
        }

        return {
            "status": "success",
            "message": "Unit cell configured",
            "unit_cell": {
                "design_wavelength": f"{design_wavelength_nm} nm",
                "wavelength_span": f"±{wavelength_span_nm} nm",
                "period": f"{params['period_nm']} nm",
                "wg_width": f"{params['wg_width_nm']} nm",
                "wg_height": f"{params['wg_height_nm']} nm",
                "hole_rx": f"{params['hole_rx_nm']} nm",
                "hole_ry": f"{params['hole_ry_nm']} nm",
                "material": params["material"],
                "freestanding": freestanding,
                "substrate": substrate,
            },
        }

    def _design_and_simulate(self, params):
        # Get params from LLM
        num_taper_holes = params.get("num_taper_holes", 6)
        num_mirror_holes = params.get("num_mirror_holes", 10)
        taper_type = params.get("taper_type", "quadratic")
        min_a_percent = params.get("min_a_percent", 80)
        min_hole_percent = params.get("min_hole_percent", 100)  # same for rx and ry
        run_simulation = params.get("run_simulation", False)

        # Unit cell params (in microns for gdsfactory)
        period = self.unit_cell["a"] * 1e6  # 200nm -> 0.2um
        hole_rx = self.unit_cell["rx"] * 1e6  # 50nm -> 0.05um
        hole_ry = self.unit_cell["ry"] * 1e6  # 100nm -> 0.1um
        wg_width = self.unit_cell["width"] * 1e6  # 450nm -> 0.45um

        # Step 1: Build cavity GDS (2D - no height needed)
        print(
            f"Building cavity: taper={num_taper_holes}, mirror={num_mirror_holes}, type={taper_type}, min_a={min_a_percent}%, min_hole={min_hole_percent}%"
        )
        cavity = build_cavity_gds(
            period=period,
            hole_rx=hole_rx,
            hole_ry=hole_ry,
            wg_width=wg_width,
            num_taper_holes=num_taper_holes,
            num_mirror_holes=num_mirror_holes,
            taper_type=taper_type,
            min_a_percent=min_a_percent,
            min_rx_percent=min_hole_percent,
            min_ry_percent=min_hole_percent,
            save=True,
        )

        config = cavity.get_config()
        print(f'GDS saved: {config["lumerical"]["gds_file"]}')

        # Add wavelength to config for run_lumerical
        config["wavelength"] = {
            "design_wavelength": self.unit_cell["design_wavelength"],
            "wavelength_span": self.unit_cell["wavelength_span"],
        }

        # Add substrate info to config for run_lumerical
        config["substrate"] = {
            "freestanding": self.unit_cell["freestanding"],
            "material": self.unit_cell["substrate"],
            "material_lumerical": self.unit_cell["substrate_lumerical"],
        }

        # Add wg_height for 3D simulation (not in GDS which is 2D)
        config["unit_cell"]["wg_height"] = self.unit_cell["height"] * 1e6  # in microns

        # Step 2: Run FDTD simulation
        print(f"Setting up FDTD simulation...")
        try:
            result = run_fdtd_simulation(config, mesh_accuracy=4, run=run_simulation)
            config["simulation"] = result
        except Exception as e:
            config["simulation"] = {"status": "error", "message": str(e)}

        return config


# Usage
if __name__ == "__main__":
    agent = CavityAgent(api_key=ANTHROPIC_API_KEY)

    print("\n=== Nanobeam Cavity Design Agent ===")
    print('Type your request or "quit" to exit.\n')

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if not user_input:
            continue

        response = agent.chat(user_input)
        print(f"\nAgent: {response}\n")
