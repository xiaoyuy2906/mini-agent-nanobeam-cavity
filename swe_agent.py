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

# Model provider selection: "claude" or "minimax"
# Set MODEL_PROVIDER=minimax in .env to use MiniMax
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "claude")

if MODEL_PROVIDER == "minimax":
    # MiniMax uses Anthropic-compatible API
    ANTHROPIC_BASE_URL = os.getenv(
        "ANTHROPIC_BASE_URL", "https://api.minimax.io/anthropic"
    )
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "MiniMax-M2.1")
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in .env (use your MiniMax API key)")
else:
    # Default: Claude
    ANTHROPIC_BASE_URL = None  # Use default Anthropic URL
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")

# Log file for storing iteration results
LOG_FILE = "cavity_design_log.json"


def _generate_config_key(unit_cell):
    """Generate a unique key from unit_cell parameters for log matching"""
    if not unit_cell:
        return None
    # Key fields that define a unique cavity configuration
    key_fields = [
        unit_cell.get("design_wavelength_nm", 0),
        unit_cell.get("period", 0),
        unit_cell.get("wg_width", 0),
        unit_cell.get("wg_height", 0),
        unit_cell.get("hole_rx", 0),
        unit_cell.get("hole_ry", 0),
        unit_cell.get("material", ""),
        unit_cell.get("freestanding", True),
    ]
    return "_".join(str(v) for v in key_fields)


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

    def save_log(self, filepath=LOG_FILE):
        """Save all iteration results to JSON log file"""
        if not self.unit_cell:
            return

        config_key = _generate_config_key(self.unit_cell)
        log_data = {
            "config_key": config_key,
            "unit_cell": self.unit_cell,
            "best_qv_ratio": self.best_qv_ratio,
            "best_design": self.best_design,
            "iteration": self.iteration,
            "design_history": self.design_history,
            "last_updated": datetime.now().isoformat(),
        }

        # Load existing logs and update/append
        all_logs = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    all_logs = json.load(f)
            except (json.JSONDecodeError, OSError):
                all_logs = {}

        # Store by config_key
        all_logs[config_key] = log_data

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_logs, f, indent=2, default=str)

        print(f"[LOG] Saved {self.iteration} iterations to {filepath}")

    def load_log(self, unit_cell, filepath=LOG_FILE):
        """Load previous results if same configuration exists"""
        if not os.path.exists(filepath):
            return False

        config_key = _generate_config_key(unit_cell)
        if not config_key:
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                all_logs = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        if config_key not in all_logs:
            return False

        # Restore state from log
        log_data = all_logs[config_key]
        self.unit_cell = log_data.get("unit_cell")
        self.best_qv_ratio = log_data.get("best_qv_ratio", 0)
        self.best_design = log_data.get("best_design")
        self.iteration = log_data.get("iteration", 0)
        self.design_history = log_data.get("design_history", [])

        print(f"[LOG] Loaded {self.iteration} previous iterations from {filepath}")
        print(f"[LOG] Best Q/V so far: {self.best_qv_ratio:,.0f}")
        return True


class SWECavityAgent:
    """SWE-Agent style cavity designer with thought-action-observation loop"""

    def __init__(self, api_key, base_url=None, model_name=None):
        """
        Initialize agent with Anthropic-compatible API.

        Args:
            api_key: API key (works for Claude or MiniMax)
            base_url: API base URL (None for Claude, "https://api.minimax.io/anthropic" for MiniMax)
            model_name: Model to use (e.g., "claude-sonnet-4-20250514" or "MiniMax-M2.1")
        """
        self.state = CavityDesignState()
        self.conversation_history = []
        self.model_name = model_name or MODEL_NAME

        # Initialize Anthropic client (works with MiniMax via base_url)
        if base_url:
            self.client = Anthropic(api_key=api_key, base_url=base_url)
            print(f"[INIT] Using API: {base_url}")
        else:
            self.client = Anthropic(api_key=api_key)
            print("[INIT] Using Anthropic API")

        print(f"[INIT] Model: {self.model_name}")

        self.system_prompt = self._build_system_prompt()
        self.tools = self._define_tools()

    def _build_system_prompt(self):
        try:
            with open("./skills.md", "r", encoding="utf-8") as f:
                skills_text = f.read().strip()
        except OSError:
            skills_text = ""

        # Critical override rule - user input always takes priority
        override_rule = (
            "## CRITICAL: USER INPUT OVERRIDES EVERYTHING\n"
            "If the user gives explicit instructions (e.g., specific parameter values, "
            "specific sweeps, specific ranges), YOU MUST FOLLOW THEM EXACTLY.\n"
            "The guidelines below are DEFAULTS only. User instructions ALWAYS take priority.\n"
            "DO NOT substitute your own parameter choices when the user specifies what to do.\n\n"
        )

        base_prompt = (
            "You are an expert nanobeam photonic crystal cavity designer. You must follow a single-step tool call loop.\n"
            "\n"
            "## Workflow Rules\n"
            "Each turn must strictly output:\n"
            "1) **THOUGHT**: Reason about the next action based on current state\n"
            "2) **ACTION**: Call exactly one tool\n"
            "3) **OBSERVATION**: Receive the result, then continue\n"
            "\n"
            "## Goal\n"
            "Maximize Q/V (figure of merit for strong light-matter interaction)\n"
            "- Q > 10,000 is good, Q > 100,000 is excellent\n"
            "- V < 1.0 (λ/n)³ is good, V < 0.5 is excellent\n"
            "- Q/V > 100,000 is a strong target\n"
            "\n"
            "## Tools\n"
            "- `set_unit_cell`: Configure the unit cell (must be called first)\n"
            "- `design_cavity`: Build the cavity and run FDTD to get Q/V\n"
            "- `view_history`: Inspect previous designs\n"
            "- `compare_designs`: Compare specific iterations\n"
            "- `get_best_design`: Retrieve the current best design\n"
            "\n"
            "## Design Strategy (must follow)\n"
            "1) Fix unit-cell geometry and target wavelength first\n"
            "2) Use FDTD as the only performance source (no heuristics)\n"
            "3) First sweep taper holes and mirror holes only; chirp form is fixed to quadratic\n"
            "4) If the simulated resonance is far from the target, adjust the period to re-center it\n"
            "5) Only adjust min_a_percent after testing taper/mirror changes\n"
            "6) Record Q, V, Q/V, and resonance wavelength; use history for comparisons\n"
            "\n"
            "## Physics Notes (trend guidance only)\n"
            "- Increasing taper holes usually increases Q and also increases V\n"
            "- Quadratic chirp is mandatory; use min_a to adjust confinement\n"
            "- More mirror holes increase Q but saturate (typically ~12-15)\n"
            "\n"
            "Requirement: be systematic, reproducible, and comparable across runs."
        )
        if skills_text:
            return override_rule + skills_text
        return override_rule + base_prompt

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
                            "description": "Target resonance wavelength in nm",
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
                        "period_nm": {
                            "type": "number",
                            "description": "Override period in nm. Use to tune resonance wavelength. Increase to red-shift, decrease to blue-shift.",
                        },
                        "wg_width_nm": {
                            "type": "number",
                            "description": "Override waveguide width in nm. Only change if user requests.",
                        },
                        "hole_rx_nm": {
                            "type": "number",
                            "description": "Override hole x-radius in nm (for mirror holes). Only change if user requests.",
                        },
                        "hole_ry_nm": {
                            "type": "number",
                            "description": "Override hole y-radius in nm (for mirror holes). Only change if user requests.",
                        },
                        "num_taper_holes": {
                            "type": "integer",
                            "description": "Number of taper/chirp holes per side (typically 8-12)",
                        },
                        "num_mirror_holes": {
                            "type": "integer",
                            "description": "Number of mirror holes per side (typically 5-10)",
                        },
                        "min_a_percent": {
                            "type": "number",
                            "description": "Minimum period at center as % of original (87-90)",
                        },
                        "min_rx_percent": {
                            "type": "number",
                            "description": "Minimum hole x-radius at center as % of original for taper (keep at 100 unless user requests)",
                        },
                        "min_ry_percent": {
                            "type": "number",
                            "description": "Minimum hole y-radius at center as % of original for taper (keep at 100 unless user requests)",
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
        new_unit_cell = {
            "design_wavelength_nm": params["design_wavelength_nm"],  # For log matching
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

        # Check if we have previous results for this configuration
        loaded = self.state.load_log(new_unit_cell)
        if not loaded:
            self.state.unit_cell = new_unit_cell

        message = "Unit cell configured"
        if loaded:
            message = f"Unit cell configured. Loaded {self.state.iteration} previous iterations (best Q/V: {self.state.best_qv_ratio:,.0f})"

        return {
            "status": "success",
            "message": message,
            "previous_iterations": self.state.iteration if loaded else 0,
            "unit_cell": {
                k: (
                    f"{v*1e9:.1f} nm"
                    if isinstance(v, float) and k != "design_wavelength_nm"
                    else v
                )
                for k, v in self.state.unit_cell.items()
            },
            "next_step": (
                "Now call design_cavity to create your first design"
                if not loaded
                else "Continue optimizing or call get_best_design to see previous best"
            ),
        }

    def _design_cavity(self, params):
        if self.state.unit_cell is None:
            return {"error": "Must call set_unit_cell first!"}

        # Create working copy with overrides (don't modify state.unit_cell to preserve config_key)
        current_cell = {
            "period": (
                params.get("period_nm", self.state.unit_cell["period"] * 1e9) * 1e-9
                if "period_nm" in params and params["period_nm"]
                else self.state.unit_cell["period"]
            ),
            "wg_width": (
                params.get("wg_width_nm", self.state.unit_cell["wg_width"] * 1e9) * 1e-9
                if "wg_width_nm" in params and params["wg_width_nm"]
                else self.state.unit_cell["wg_width"]
            ),
            "hole_rx": (
                params.get("hole_rx_nm", self.state.unit_cell["hole_rx"] * 1e9) * 1e-9
                if "hole_rx_nm" in params and params["hole_rx_nm"]
                else self.state.unit_cell["hole_rx"]
            ),
            "hole_ry": (
                params.get("hole_ry_nm", self.state.unit_cell["hole_ry"] * 1e9) * 1e-9
                if "hole_ry_nm" in params and params["hole_ry_nm"]
                else self.state.unit_cell["hole_ry"]
            ),
            "wg_height": self.state.unit_cell["wg_height"],
            "design_wavelength": self.state.unit_cell["design_wavelength"],
            "wavelength_span": self.state.unit_cell["wavelength_span"],
            "freestanding": self.state.unit_cell["freestanding"],
            "substrate": self.state.unit_cell["substrate"],
            "substrate_lumerical": self.state.unit_cell["substrate_lumerical"],
        }

        # Extract params for logging
        design_params = {
            "period_nm": current_cell["period"] * 1e9,
            "wg_width_nm": current_cell["wg_width"] * 1e9,
            "hole_rx_nm": current_cell["hole_rx"] * 1e9,
            "hole_ry_nm": current_cell["hole_ry"] * 1e9,
            "num_taper_holes": params["num_taper_holes"],
            "num_mirror_holes": params["num_mirror_holes"],
            "taper_type": "quadratic",
            "min_a_percent": params["min_a_percent"],
            "min_rx_percent": params.get("min_rx_percent", 100),
            "min_ry_percent": params.get("min_ry_percent", 100),
        }
        hypothesis = params.get("hypothesis", "No hypothesis provided")

        # Build GDS (optional, for visualization)
        try:
            cavity = build_cavity_gds(
                period=current_cell["period"] * 1e6,
                hole_rx=current_cell["hole_rx"] * 1e6,
                hole_ry=current_cell["hole_ry"] * 1e6,
                wg_width=current_cell["wg_width"] * 1e6,
                num_taper_holes=design_params["num_taper_holes"],
                num_mirror_holes=design_params["num_mirror_holes"],
                taper_type=design_params["taper_type"],
                min_a_percent=design_params["min_a_percent"],
                min_rx_percent=design_params["min_rx_percent"],
                min_ry_percent=design_params["min_ry_percent"],
                save=True,
            )
            gds_file = cavity.gds_filepath
        except Exception as e:
            gds_file = f"Error: {e}"

        # Real FDTD simulation (required)
        mesh_accuracy = 8  # Always use highest accuracy for reliable Q values
        if not isinstance(gds_file, str) or gds_file.startswith("Error"):
            return {"error": f"GDS build failed: {gds_file}"}

        config = cavity.get_config()
        config["unit_cell"]["wg_height"] = current_cell["wg_height"] * 1e6
        config["wavelength"] = {
            "design_wavelength": current_cell["design_wavelength"],
            "wavelength_span": current_cell["wavelength_span"],
        }
        config["substrate"] = {
            "freestanding": current_cell["freestanding"],
            "material": current_cell["substrate"],
            "material_lumerical": current_cell["substrate_lumerical"],
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
            resonance_nm = float(np.round(current_cell["design_wavelength"] * 1e9, 2))

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

        # Save log after each iteration
        self.state.save_log()

        # Build observation
        is_best = result["qv_ratio"] >= self.state.best_qv_ratio

        # Check targets - this tells the agent what phase we're in
        target_status = self.check_targets(result)

        return {
            "iteration": self.state.iteration,
            "hypothesis": hypothesis,
            "parameters": design_params,
            "results": {
                "Q": f"{performance['Q']:,}",
                "V": f"{performance['V']:.3f} (λ/n)³",
                "Q/V": f"{performance['qv_ratio']:,}",
                "resonance": f"{performance['resonance_wavelength_nm']} nm",
                "target_wavelength": f"{self.state.unit_cell['design_wavelength'] * 1e9:.1f} nm",
            },
            "analysis": performance["notes"],
            "is_new_best": is_best,
            "best_qv_so_far": f"{self.state.best_qv_ratio:,}",
            "gds_file": gds_file,
            "fdtd_result": fdtd_result,
            "TARGET_STATUS": target_status,
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
                        "period_nm": entry["params"].get("period_nm", "N/A"),
                        "wg_width_nm": entry["params"].get("wg_width_nm", "N/A"),
                        "hole_rx_nm": entry["params"].get("hole_rx_nm", "N/A"),
                        "hole_ry_nm": entry["params"].get("hole_ry_nm", "N/A"),
                        "taper": entry["params"]["num_taper_holes"],
                        "mirror": entry["params"]["num_mirror_holes"],
                        "min_a%": entry["params"]["min_a_percent"],
                        "min_rx%": entry["params"].get("min_rx_percent", 100),
                        "min_ry%": entry["params"].get("min_ry_percent", 100),
                    },
                    "Q": f"{entry['result']['Q']:,}",
                    "V": f"{entry['result']['V']:.3f}",
                    "Q/V": f"{entry['result']['qv_ratio']:,}",
                    "resonance_nm": entry["result"].get("resonance_nm", "N/A"),
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
            "improvement_potential": "Try: lower min_a_percent (90→89→88→87), more taper holes (8→12), more mirror holes (5→10)",
        }

    def _get_history_summary(self):
        """Get a concise summary of design history for agent context"""
        if not self.state.design_history:
            return "No previous designs."

        lines = []
        lines.append(f"Total iterations: {self.state.iteration}")
        lines.append(
            f"Best Q/V: {self.state.best_qv_ratio:,} (iteration {self.state.best_design['iteration']})"
        )

        # Summarize parameter ranges tried
        periods = set()
        tapers = set()
        mirrors = set()
        min_a_vals = set()
        rx_ry_pairs = set()

        for entry in self.state.design_history:
            p = entry["params"]
            periods.add(p.get("period_nm", 0))
            tapers.add(p.get("num_taper_holes", 0))
            mirrors.add(p.get("num_mirror_holes", 0))
            min_a_vals.add(p.get("min_a_percent", 0))
            rx = p.get("min_rx_percent", 100)
            ry = p.get("min_ry_percent", 100)
            rx_ry_pairs.add((rx, ry))

        lines.append(f"Periods tried (nm): {sorted(periods)}")
        lines.append(f"Taper holes tried: {sorted(tapers)}")
        lines.append(f"Mirror holes tried: {sorted(mirrors)}")
        lines.append(f"min_a% tried: {sorted(min_a_vals)}")
        lines.append(f"(rx%, ry%) pairs tried: {sorted(rx_ry_pairs)}")

        # Last 3 results
        lines.append("\nLast 3 designs:")
        for entry in self.state.design_history[-3:]:
            p = entry["params"]
            r = entry["result"]
            lines.append(
                f"  #{entry['iteration']}: t={p['num_taper_holes']}, m={p['num_mirror_holes']}, "
                f"a={p['min_a_percent']}%, rx={p.get('min_rx_percent', 100)}%, ry={p.get('min_ry_percent', 100)}% "
                f"→ Q={r['Q']:,}, V={r['V']:.3f}, Q/V={r['qv_ratio']:,}"
            )

        return "\n".join(lines)

    def check_targets(self, result):
        """
        Check if design meets targets. Returns status dict.
        PRIORITY: Resonance must be on-target before Q/V matters.
        """
        if not result or "resonance_nm" not in result:
            return {"on_target": False, "phase": "no_result"}

        target_nm = self.state.unit_cell["design_wavelength"] * 1e9
        resonance_nm = result["resonance_nm"]
        wavelength_diff = abs(resonance_nm - target_nm)

        q_value = result.get("Q", 0)
        v_value = result.get("V", 0)

        # Phase 1: Resonance tuning (TOP PRIORITY)
        if wavelength_diff > 5:
            return {
                "on_target": False,
                "phase": "resonance_tuning",
                "wavelength_diff_nm": wavelength_diff,
                "target_nm": target_nm,
                "resonance_nm": resonance_nm,
                "direction": (
                    "increase_period" if resonance_nm < target_nm else "decrease_period"
                ),
                "message": f"RESONANCE OFF BY {wavelength_diff:.1f}nm - IGNORE Q/V, FIX PERIOD FIRST",
            }

        # Phase 2: Q optimization (only when resonance is on target)
        q_target = 1_000_000
        if q_value < q_target:
            return {
                "on_target": False,
                "phase": "q_optimization",
                "wavelength_diff_nm": wavelength_diff,
                "Q": q_value,
                "Q_target": q_target,
                "V": v_value,
                "message": f"Resonance OK ({wavelength_diff:.1f}nm off). Q={q_value:,} < {q_target:,} target",
            }

        # All targets met
        return {
            "on_target": True,
            "phase": "complete",
            "Q": q_value,
            "V": v_value,
            "resonance_nm": resonance_nm,
            "message": "ALL TARGETS MET!",
        }

    def run_optimization_loop(self, max_iterations=10):
        """
        Automated optimization loop that keeps iterating until targets are met.
        After max_iterations, shows best result and asks user to continue.
        """
        if not self.state.unit_cell:
            return {"error": "Must call set_unit_cell first via chat()"}

        print(f"\n{'='*60}")
        print(f"Starting automated optimization ({max_iterations} iterations)")
        print(f"Target: resonance within ±5nm, Q > 1,000,000")
        print(f"{'='*60}\n")

        last_best_qv = self.state.best_qv_ratio

        # If resuming from existing history, review it first
        if self.state.iteration > 0 and self.state.design_history:
            print(
                f"[RESUME] Found {self.state.iteration} previous iterations. Reviewing history..."
            )
            history_summary = self._get_history_summary()
            review_prompt = (
                f"RESUMING OPTIMIZATION - Review the previous {self.state.iteration} iterations before continuing.\n\n"
                f"HISTORY SUMMARY:\n{history_summary}\n\n"
                f"Call view_history to see full details, then continue optimization. "
                f"DO NOT repeat parameters that have already been tried!"
            )
            self.chat(review_prompt)
            print("[RESUME] History reviewed. Continuing optimization...\n")

        for i in range(max_iterations):
            # Build prompt based on current state
            if self.state.iteration == 0:
                prompt = "Run the first baseline design with taper=8, mirror=10, min_a=90, min_rx=100, min_ry=100"
            else:
                # Get last result to inform next action
                last_result = (
                    self.state.design_history[-1]["result"]
                    if self.state.design_history
                    else None
                )
                target_status = (
                    self.check_targets(last_result)
                    if last_result
                    else {"phase": "no_result"}
                )

                if target_status.get("on_target"):
                    print(f"\n{'='*60}")
                    print("ALL TARGETS MET!")
                    print(f"{'='*60}\n")
                    return {"status": "complete", "result": self._get_best_design()}

                # Guide the agent based on phase
                if target_status.get("phase") == "resonance_tuning":
                    direction = target_status.get("direction", "adjust")
                    diff = target_status.get("wavelength_diff_nm", 0)
                    prompt = (
                        f"RESONANCE OFF TARGET by {diff:.1f}nm. "
                        f"You MUST {direction} to fix wavelength. "
                        f"Adjust period by ~5-10nm in the right direction. "
                        f"DO NOT change other parameters - only period matters now."
                    )
                else:
                    # Check current taper holes from last design
                    last_params = (
                        self.state.design_history[-1]["params"]
                        if self.state.design_history
                        else {}
                    )
                    current_taper = last_params.get("num_taper_holes", 8)

                    if current_taper < 12:
                        prompt = (
                            f"Resonance is on target. Q={target_status.get('Q', 0):,} (target: 1,000,000). "
                            f"Try increasing num_taper_holes (currently {current_taper}, try {current_taper + 2}). "
                            f"If resonance shifts, adjust period to compensate."
                        )
                    else:
                        prompt = (
                            f"Resonance is on target. Q={target_status.get('Q', 0):,} (target: 1,000,000). "
                            f"Taper holes at {current_taper}. Try: lower min_a_percent (90→89→88→87, 1% steps) or more mirror holes. "
                            f"Keep min_rx=min_ry=100. Change only ONE parameter. Check view_history to avoid repeating."
                        )

            print(f"\n--- Optimization iteration {i+1}/{max_iterations} ---")
            print(f"Prompt: {prompt[:100]}...")

            # Run one chat turn
            response = self.chat(prompt)
            print(f"Agent: {response[:200] if response else 'No response'}...")

            # Check for improvement
            if self.state.best_qv_ratio > last_best_qv:
                last_best_qv = self.state.best_qv_ratio
                print(f"[NEW BEST] Q/V = {last_best_qv:,}")

        # Return status for the main loop to handle continuation
        return {"status": "paused", "result": self._get_best_design()}

    def _summarize_tool_result(self, tool_name, observation):
        """Create a brief summary of tool result to save context space"""
        if tool_name == "design_cavity":
            if "error" in observation:
                return f"Error: {observation['error']}"
            r = observation.get("results", {})
            p = observation.get("parameters", {})
            ts = observation.get("TARGET_STATUS", {})
            return (
                f"Iter {observation.get('iteration')}: "
                f"p={p.get('period_nm', '?'):.0f}nm, wg={p.get('wg_width_nm', '?'):.0f}nm, "
                f"rx={p.get('hole_rx_nm', '?'):.0f}nm, ry={p.get('hole_ry_nm', '?'):.0f}nm, "
                f"t={p.get('num_taper_holes')}, m={p.get('num_mirror_holes')}, a={p.get('min_a_percent')}% → "
                f"Q={r.get('Q')}, V={r.get('V')}, resonance={r.get('resonance')}. "
                f"Phase: {ts.get('phase', 'unknown')}. "
                f"{'NEW BEST!' if observation.get('is_new_best') else ''}"
            )
        elif tool_name == "set_unit_cell":
            return f"Unit cell configured. {observation.get('message', '')}"
        elif tool_name == "view_history":
            return f"History: {observation.get('total_designs', 0)} designs. Best iter: {observation.get('best_iteration')}"
        elif tool_name == "get_best_design":
            bd = observation.get("best_design", {})
            r = bd.get("result", {})
            return f"Best: iter {bd.get('iteration')}, Q={r.get('Q')}, V={r.get('V')}, Q/V={r.get('qv_ratio')}"
        else:
            # For other tools, truncate to 200 chars
            return json.dumps(observation)[:200]

    def chat(self, user_message, max_retries=3):
        """Main chat loop with thought-action-observation pattern (works with Claude or MiniMax)"""
        # Limit conversation history to last 10 messages to avoid context overflow
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        self.conversation_history.append({"role": "user", "content": user_message})

        # Add state context
        state_context = f"\n\n[Current State: {json.dumps(self.state.get_summary())}]"
        messages_with_state = self.conversation_history.copy()
        messages_with_state[-1] = {
            "role": "user",
            "content": user_message + state_context,
        }

        # Retry logic for transient API errors
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2048,
                    system=self.system_prompt,
                    tools=self.tools,
                    messages=messages_with_state,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(
                        f"[RETRY] API error (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    import time

                    time.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s
                else:
                    raise

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

            # Create brief summary for conversation history (saves context)
            summary = self._summarize_tool_result(tool_name, observation)

            # Add assistant message to history
            self.conversation_history.append(
                {"role": "assistant", "content": response.content}
            )

            # Format tool result content - MiniMax may need string, Claude accepts JSON string
            if MODEL_PROVIDER == "minimax":
                # MiniMax: use string format, ensure it's not double-encoded
                if isinstance(observation, dict):
                    tool_result_content = json.dumps(observation, ensure_ascii=False)
                else:
                    tool_result_content = str(observation)
            else:
                # Claude: JSON string is fine
                tool_result_content = json.dumps(observation)

            # For the CURRENT API call, use full result so agent can make decisions
            # Build messages correctly: assistant message + tool result
            current_messages = self.conversation_history + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": tool_result_content,
                        }
                    ],
                }
            ]

            # Store summary in history for future turns (after current call succeeds)
            # We'll add this after the API call succeeds to avoid duplication

            # Continue conversation with full result for current decision
            for attempt in range(max_retries):
                try:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=2048,
                        system=self.system_prompt,
                        tools=self.tools,
                        messages=current_messages,  # Use full result for this call
                    )
                    # Only add summary to history after successful API call
                    self.conversation_history.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use.id,
                                    "content": summary,  # Summary for future context
                                }
                            ],
                        }
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(
                            f"[RETRY] API error (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        import time

                        time.sleep(2**attempt)
                    else:
                        raise

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
    print("\nCommands:")
    print("  'Design a SiN cavity at 737nm' - setup unit cell")
    print("  'auto' or 'auto 15' - run automated optimization (default 10 iterations)")
    print("  'quit' - exit")
    print(f"\nResults are saved to {LOG_FILE}")
    print("\nOptimization Priority:")
    print("  1. Resonance within ±5nm of target (MUST meet first)")
    print("  2. Q > 1,000,000")
    print("  3. V < 0.5 (λ/n)³")
    print(
        f"\nModel: {MODEL_NAME} ({'MiniMax' if MODEL_PROVIDER == 'minimax' else 'Claude'})"
    )
    print("  (Set MODEL_PROVIDER=minimax in .env to use MiniMax)")
    print()

    # Initialize agent (works with Claude or MiniMax via Anthropic-compatible API)
    agent = SWECavityAgent(
        api_key=ANTHROPIC_API_KEY,
        base_url=ANTHROPIC_BASE_URL,
        model_name=MODEL_NAME,
    )

    def show_best_result():
        """Display the current best design nicely"""
        best = agent.state.best_design
        if not best:
            print("\nNo designs yet.")
            return

        result = best["result"]
        params = best["params"]
        target_nm = agent.state.unit_cell["design_wavelength"] * 1e9
        resonance_nm = result.get("resonance_nm", 0)
        wavelength_diff = abs(resonance_nm - target_nm)

        print(f"\n{'='*60}")
        print("BEST DESIGN SO FAR")
        print(f"{'='*60}")
        print(f"  Iteration:    {best['iteration']}")
        print(f"  Q:            {result['Q']:,}")
        print(f"  V:            {result['V']:.3f} (λ/n)³")
        print(f"  Q/V:          {result['qv_ratio']:,}")
        print(
            f"  Resonance:    {resonance_nm:.2f} nm (target: {target_nm:.1f} nm, diff: {wavelength_diff:.1f} nm)"
        )
        print(
            f"  Parameters:   taper={params['num_taper_holes']}, mirror={params['num_mirror_holes']}, "
            f"min_a={params['min_a_percent']}%"
        )
        print(
            f"                min_rx={params.get('min_rx_percent', 100)}%, "
            f"min_ry={params.get('min_ry_percent', 100)}%"
        )
        print(f"  GDS file:     {result.get('gds_file', 'N/A')}")
        print(f"{'='*60}")

        # Show target status
        target_status = agent.check_targets(result)
        if target_status.get("on_target"):
            print("STATUS: ALL TARGETS MET!")
        elif target_status.get("phase") == "resonance_tuning":
            print(
                f"STATUS: Resonance off by {wavelength_diff:.1f}nm - need to tune period"
            )
        else:
            print(
                f"STATUS: Resonance OK, Q needs improvement ({result['Q']:,} / 1,000,000)"
            )
        print()

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            if not user_input:
                continue

            # Check for auto mode
            if user_input.lower().startswith("auto"):
                parts = user_input.split()
                max_iter = int(parts[1]) if len(parts) > 1 else 5

                while True:
                    result = agent.run_optimization_loop(max_iterations=max_iter)

                    # Show best result
                    show_best_result()

                    # Check if targets met
                    if result.get("status") == "complete":
                        print("Optimization complete - all targets met!")
                        break

                    # Ask to continue
                    print(f"Total iterations so far: {agent.state.iteration}")
                    cont = (
                        input("Continue optimization? [y/N/number]: ").strip().lower()
                    )

                    if cont in ["", "n", "no"]:
                        print("Stopping optimization.")
                        break
                    elif cont in ["y", "yes"]:
                        max_iter = 10  # Default batch size
                    elif cont.isdigit():
                        max_iter = int(cont)
                    else:
                        print("Stopping optimization.")
                        break

                continue

            print("\n" + "-" * 40)
            response = agent.chat(user_input)
            print("-" * 40)
            print(f"\nAgent: {response}\n")
    finally:
        # Save log on exit
        if agent.state.unit_cell and agent.state.iteration > 0:
            agent.state.save_log()
            print(
                f"\n[LOG] Final save: {agent.state.iteration} iterations saved to {LOG_FILE}"
            )


if __name__ == "__main__":
    main()
