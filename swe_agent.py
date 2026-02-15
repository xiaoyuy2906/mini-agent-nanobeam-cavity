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

# Optimization sweep order from skills.md
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
    MODEL_NAME = os.getenv("MODEL_NAME", "MiniMax-M2.5")
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
        self.last_params = None  # Last-used override values (period, rx, ry, etc.)
        self.design_history = []  # List of all designs tried
        self.best_design = None
        self.best_qv_ratio = 0
        self.iteration = 0
        self.fdtd_confirmed = False
        self.sweep_step = "initial"
        self.step_start_iter = 0
        self.locked_params = {}

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
            "fdtd_confirmed": self.fdtd_confirmed,
        }

    def find_duplicate(self, params):
        """Check if exact params were already tried. Returns entry or None."""
        for entry in self.design_history:
            p = entry["params"]
            if (
                abs(p.get("period_nm", 0) - params.get("period_nm", 0)) < 0.5
                and abs(p.get("hole_rx_nm", 0) - params.get("hole_rx_nm", 0)) < 0.5
                and abs(p.get("hole_ry_nm", 0) - params.get("hole_ry_nm", 0)) < 0.5
                and p.get("num_taper_holes") == params.get("num_taper_holes")
                and p.get("num_mirror_holes") == params.get("num_mirror_holes")
                and abs(p.get("min_a_percent", 0) - params.get("min_a_percent", 0))
                < 0.5
                and abs(
                    p.get("min_rx_percent", 100) - params.get("min_rx_percent", 100)
                )
                < 0.5
                and abs(
                    p.get("min_ry_percent", 100) - params.get("min_ry_percent", 100)
                )
                < 0.5
                and abs(p.get("wg_width_nm", 0) - params.get("wg_width_nm", 0)) < 0.5
            ):
                return entry
        return None

    def get_step_history(self):
        """Get design history entries from current step only."""
        return [e for e in self.design_history if e["iteration"] > self.step_start_iter]

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
            "fdtd_confirmed": self.fdtd_confirmed,
            "design_history": self.design_history,
            "sweep_step": self.sweep_step,
            "step_start_iter": self.step_start_iter,
            "locked_params": self.locked_params,
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
        self.fdtd_confirmed = log_data.get("fdtd_confirmed", self.iteration > 0)
        self.design_history = log_data.get("design_history", [])
        self.sweep_step = log_data.get("sweep_step", "initial")
        self.step_start_iter = log_data.get("step_start_iter", 0)
        self.locked_params = log_data.get("locked_params", {})

        print(f"[LOG] Loaded {self.iteration} previous iterations from {filepath}")
        print(f"[LOG] Sweep step: {self.sweep_step}, locked: {self.locked_params}")
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
            "DO NOT substitute your own parameter choices when the user specifies what to do.\n"
            "NEVER invent missing unit-cell geometry (period/width/height/rx/ry). "
            "If a required value is missing, ask the user instead of calling set_unit_cell.\n"
            "Before the FIRST FDTD run, show all unit-cell inputs and ask user confirmation. "
            "Only proceed after user says 'confirm fdtd'.\n\n"
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
            "- Q > 10,000 is good, Q > 1,000,000 is excellent\n"
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
                        "initial_min_a_percent": {
                            "type": "number",
                            "description": "Initial min_a_percent for the first baseline run (optional, defaults to 90)",
                        },
                        "material": {
                            "type": "string",
                            "description": "Core material name for Lumerical (exact DB name)",
                        },
                        "material_refractive_index": {
                            "type": "number",
                            "description": "Core material refractive index (n) at the design wavelength",
                        },
                        "freestanding": {
                            "type": "boolean",
                            "description": "If true, no substrate is added in FDTD",
                        },
                        "substrate": {
                            "type": "string",
                            "description": "Substrate material name for Lumerical (exact DB name). Use 'none' when freestanding.",
                        },
                        "substrate_refractive_index": {
                            "type": "number",
                            "description": "Substrate refractive index (n) at the design wavelength (required if substrate is used)",
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
                        "material_refractive_index",
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

    def confirm_fdtd_inputs(self):
        """Mark the first FDTD input review as confirmed by user."""
        if not self.state.unit_cell:
            return {"error": "Set unit cell first before confirming FDTD inputs."}
        self.state.fdtd_confirmed = True
        return {"status": "success", "message": "FDTD input confirmed by user."}

    def _set_unit_cell(self, params):
        field_labels = {
            "design_wavelength_nm": "design_wavelength_nm (target wavelength, nm)",
            "period_nm": "period_nm (lattice period, nm)",
            "wg_width_nm": "wg_width_nm (waveguide width, nm)",
            "wg_height_nm": "wg_height_nm (waveguide height, nm)",
            "hole_rx_nm": "hole_rx_nm (ellipse hole rx, nm)",
            "hole_ry_nm": "hole_ry_nm (ellipse hole ry, nm)",
            "material": "material (core material name)",
            "material_refractive_index": "material_refractive_index (core refractive index n)",
            "substrate_refractive_index": "substrate_refractive_index (substrate refractive index n)",
        }
        required_fields = [
            "design_wavelength_nm",
            "period_nm",
            "wg_width_nm",
            "wg_height_nm",
            "hole_rx_nm",
            "hole_ry_nm",
            "material",
            "material_refractive_index",
        ]
        missing_fields = [
            field
            for field in required_fields
            if field not in params
            or params[field] is None
            or (isinstance(params[field], str) and not params[field].strip())
        ]
        if missing_fields:
            missing_detail = ", ".join(field_labels.get(f, f) for f in missing_fields)
            return {
                "error": "Missing required tool input fields: " + missing_detail,
                "requires_user_input": True,
                "user_prompt": (
                    "Please provide the missing fields above, then retry set_unit_cell. "
                    "FDTD will not run with missing values."
                ),
            }

        numeric_positive_fields = [
            "design_wavelength_nm",
            "period_nm",
            "wg_width_nm",
            "wg_height_nm",
            "hole_rx_nm",
            "hole_ry_nm",
            "material_refractive_index",
        ]
        invalid_fields = []
        for field in numeric_positive_fields:
            try:
                if float(params[field]) <= 0:
                    invalid_fields.append(field)
            except (TypeError, ValueError):
                invalid_fields.append(field)
        if invalid_fields:
            invalid_detail = ", ".join(field_labels.get(f, f) for f in invalid_fields)
            return {
                "error": (
                    "Invalid tool input fields (must be positive numbers): "
                    + invalid_detail
                ),
                "requires_user_input": True,
                "user_prompt": (
                    "Please provide valid numeric values before running FDTD."
                ),
            }

        freestanding = params.get("freestanding", True)
        substrate = params.get("substrate", "none")
        if freestanding:
            substrate = "none"

        material = params["material"]
        material_n = float(params["material_refractive_index"])

        substrate_n = params.get("substrate_refractive_index")
        if not freestanding and substrate.lower() != "none":
            if substrate_n is None:
                return {
                    "error": (
                        "Missing required tool input fields: "
                        + field_labels["substrate_refractive_index"]
                    ),
                    "requires_user_input": True,
                    "user_prompt": "Please provide substrate_refractive_index for non-freestanding design.",
                }
            try:
                substrate_n = float(substrate_n)
            except (TypeError, ValueError):
                return {
                    "error": "Invalid tool input field: substrate_refractive_index (must be a positive number)",
                    "requires_user_input": True,
                    "user_prompt": "Please provide a valid substrate_refractive_index value.",
                }
            if substrate_n <= 0:
                return {
                    "error": "Invalid tool input field: substrate_refractive_index (must be > 0)",
                    "requires_user_input": True,
                    "user_prompt": "Please provide a positive substrate_refractive_index value.",
                }
        else:
            substrate_n = None

        wavelength_span_nm = params.get("wavelength_span_nm", 100)
        initial_min_a_percent = params.get("initial_min_a_percent", 90)
        try:
            initial_min_a_percent = float(initial_min_a_percent)
        except (TypeError, ValueError):
            return {
                "error": "Invalid tool input field: initial_min_a_percent (must be a number)",
                "requires_user_input": True,
                "user_prompt": "Please provide a numeric initial_min_a_percent value.",
            }
        new_unit_cell = {
            "design_wavelength_nm": params["design_wavelength_nm"],  # For log matching
            "design_wavelength": params["design_wavelength_nm"] * 1e-9,
            "wavelength_span": wavelength_span_nm * 1e-9,
            "period": params["period_nm"] * 1e-9,
            "wg_width": params["wg_width_nm"] * 1e-9,
            "wg_height": params["wg_height_nm"] * 1e-9,
            "hole_rx": params["hole_rx_nm"] * 1e-9,
            "hole_ry": params["hole_ry_nm"] * 1e-9,
            "initial_min_a_percent": initial_min_a_percent,
            "material": material,
            "material_lumerical": material,
            "material_refractive_index": material_n,
            "freestanding": freestanding,
            "substrate": substrate,
            "substrate_lumerical": None if substrate.lower() == "none" else substrate,
            "substrate_refractive_index": substrate_n,
        }

        # Check if we have previous results for this configuration
        loaded = self.state.load_log(new_unit_cell)
        # Always use fresh unit_cell (old logs may miss new keys like material_lumerical)
        self.state.unit_cell = new_unit_cell
        self.state.fdtd_confirmed = loaded and self.state.iteration > 0

        message = "Unit cell configured"
        if loaded:
            message = f"Unit cell configured. Loaded {self.state.iteration} previous iterations (best Q/V: {self.state.best_qv_ratio:,.0f})"

        return {
            "status": "success",
            "message": (
                message
                + ". Before first FDTD run, user must confirm inputs with: confirm fdtd"
            ),
            "previous_iterations": self.state.iteration if loaded else 0,
            "unit_cell": {
                k: (
                    f"{v*1e9:.1f} nm"
                    if isinstance(v, float)
                    and k
                    in {
                        "design_wavelength",
                        "wavelength_span",
                        "period",
                        "wg_width",
                        "wg_height",
                        "hole_rx",
                        "hole_ry",
                    }
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
        if self.state.iteration == 0 and not self.state.fdtd_confirmed:
            uc = self.state.unit_cell
            preview = {
                "design_wavelength_nm": uc["design_wavelength_nm"],
                "wavelength_span_nm": uc["wavelength_span"] * 1e9,
                "period_nm": uc["period"] * 1e9,
                "wg_width_nm": uc["wg_width"] * 1e9,
                "wg_height_nm": uc["wg_height"] * 1e9,
                "hole_rx_nm": uc["hole_rx"] * 1e9,
                "hole_ry_nm": uc["hole_ry"] * 1e9,
                "material": uc["material"],
                "material_refractive_index": uc["material_refractive_index"],
                "freestanding": uc["freestanding"],
                "substrate": uc["substrate"],
                "substrate_refractive_index": uc["substrate_refractive_index"],
            }
            return {
                "error": "First FDTD run requires user confirmation.",
                "requires_user_input": True,
                "user_prompt": (
                    "Please check these inputs and reply `confirm fdtd` to proceed, "
                    "or provide corrected parameters."
                ),
                "pending_fdtd_input": preview,
            }

        # Fallback: use last-used params if available, otherwise unit_cell initial values
        fallback = (
            self.state.last_params if self.state.last_params else self.state.unit_cell
        )

        # Create working copy with overrides
        current_cell = {
            "period": (
                params["period_nm"] * 1e-9
                if "period_nm" in params and params["period_nm"]
                else fallback["period"]
            ),
            "wg_width": (
                params["wg_width_nm"] * 1e-9
                if "wg_width_nm" in params and params["wg_width_nm"]
                else fallback["wg_width"]
            ),
            "hole_rx": (
                params["hole_rx_nm"] * 1e-9
                if "hole_rx_nm" in params and params["hole_rx_nm"]
                else fallback["hole_rx"]
            ),
            "hole_ry": (
                params["hole_ry_nm"] * 1e-9
                if "hole_ry_nm" in params and params["hole_ry_nm"]
                else fallback["hole_ry"]
            ),
            "wg_height": self.state.unit_cell["wg_height"],
            "design_wavelength": self.state.unit_cell["design_wavelength"],
            "wavelength_span": self.state.unit_cell["wavelength_span"],
            "freestanding": self.state.unit_cell["freestanding"],
            "substrate": self.state.unit_cell["substrate"],
            "substrate_lumerical": self.state.unit_cell["substrate_lumerical"],
            "material_refractive_index": self.state.unit_cell[
                "material_refractive_index"
            ],
            "substrate_refractive_index": self.state.unit_cell[
                "substrate_refractive_index"
            ],
        }

        # Extract params with safe defaults (LLM tool calls can omit fields)
        default_taper = self.state.locked_params.get("num_taper_holes", 10)
        default_mirror = 7
        default_min_a = self.state.locked_params.get(
            "min_a_percent", self.state.unit_cell.get("initial_min_a_percent", 90)
        )
        try:
            num_taper_holes = int(params.get("num_taper_holes", default_taper))
        except (TypeError, ValueError):
            num_taper_holes = int(default_taper)
        try:
            num_mirror_holes = int(params.get("num_mirror_holes", default_mirror))
        except (TypeError, ValueError):
            num_mirror_holes = int(default_mirror)
        try:
            min_a_percent = float(params.get("min_a_percent", default_min_a))
        except (TypeError, ValueError):
            min_a_percent = float(default_min_a)

        # Extract params for logging
        design_params = {
            "period_nm": current_cell["period"] * 1e9,
            "wg_width_nm": current_cell["wg_width"] * 1e9,
            "hole_rx_nm": current_cell["hole_rx"] * 1e9,
            "hole_ry_nm": current_cell["hole_ry"] * 1e9,
            "num_taper_holes": num_taper_holes,
            "num_mirror_holes": num_mirror_holes,
            "taper_type": "quadratic",
            "min_a_percent": min_a_percent,
            "min_rx_percent": params.get("min_rx_percent", 100),
            "min_ry_percent": params.get("min_ry_percent", 100),
        }

        # For the very first iteration, force exact unit cell values
        if self.state.sweep_step == "initial" and not self.state.design_history:
            uc = self.state.unit_cell
            forced = {
                "period_nm": round(uc["period"] * 1e9),
                "hole_rx_nm": round(uc["hole_rx"] * 1e9),
                "hole_ry_nm": round(uc["hole_ry"] * 1e9),
            }
            enforced_first = []
            for key, val in forced.items():
                if design_params.get(key) != val:
                    enforced_first.append(f"{key}: {design_params[key]}->{val}")
                    design_params[key] = val
            current_cell["period"] = uc["period"]
            current_cell["hole_rx"] = uc["hole_rx"]
            current_cell["hole_ry"] = uc["hole_ry"]
            if enforced_first:
                print(
                    f"  [ENFORCE] First iteration: forced unit cell values: {', '.join(enforced_first)}"
                )

        # Enforce locked params during sweep steps (prevent LLM from
        # changing non-swept params, which confounds results)
        if self.state.sweep_step not in ("initial", "complete", "manual"):
            locked = self.state.locked_params
            swept_param_map = {
                "sweep_min_a": "min_a_percent",
                "re_sweep_min_a_1": "min_a_percent",
                "re_sweep_min_a_2": "min_a_percent",
                "sweep_rx": "hole_rx_nm",
                "sweep_ry": "hole_ry_nm",
                "sweep_taper": "num_taper_holes",
                "fine_period": "period_nm",
            }
            swept = swept_param_map.get(self.state.sweep_step)
            # Allow the swept param and period (for resonance tuning) to change
            allowed_to_change = {swept}
            if swept != "period_nm":
                allowed_to_change.add("period_nm")

            enforced = []
            for key, val in locked.items():
                if key in allowed_to_change:
                    continue
                if key in design_params and design_params[key] != val:
                    enforced.append(f"{key}: {design_params[key]}->{val}")
                    design_params[key] = val

            # Sync current_cell with enforced values
            cell_mapping = {
                "period_nm": ("period", 1e-9),
                "hole_rx_nm": ("hole_rx", 1e-9),
                "hole_ry_nm": ("hole_ry", 1e-9),
            }
            for param_key, (cell_key, scale) in cell_mapping.items():
                if param_key in locked and param_key not in allowed_to_change:
                    current_cell[cell_key] = locked[param_key] * scale

            if enforced:
                print(f"  [ENFORCE] Overrode non-swept params: {', '.join(enforced)}")

            # Clamp period to ±15nm from locked value during non-period sweeps
            # to prevent wild LLM drift (e.g. 251-320nm when locked=294)
            if swept != "period_nm" and "period_nm" in locked:
                locked_period = locked["period_nm"]
                max_drift = 15
                current_period = design_params["period_nm"]
                clamped = max(
                    locked_period - max_drift,
                    min(locked_period + max_drift, current_period),
                )
                if abs(clamped - current_period) > 0.5:
                    print(
                        f"  [CLAMP] Period {current_period:.0f}nm -> {clamped:.0f}nm (locked={locked_period}±{max_drift}nm)"
                    )
                    design_params["period_nm"] = clamped
                    current_cell["period"] = clamped * 1e-9

        # Block duplicate simulations before running expensive FDTD
        duplicate = self.state.find_duplicate(design_params)
        if duplicate:
            return {
                "error": (
                    f"DUPLICATE: iteration {duplicate['iteration']} used identical parameters. "
                    f"Result was Q={duplicate['result']['Q']:,}, V={duplicate['result']['V']:.3f}, "
                    f"Q/V={duplicate['result']['qv_ratio']:,}. Skipping."
                )
            }

        hypothesis = params.get("hypothesis", "No hypothesis provided")

        # Save current values as last_params so next call keeps them as defaults
        self.state.last_params = {
            "period": current_cell["period"],
            "wg_width": current_cell["wg_width"],
            "hole_rx": current_cell["hole_rx"],
            "hole_ry": current_cell["hole_ry"],
        }

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
        config["lumerical"]["material"] = self.state.unit_cell["material_lumerical"]
        config["lumerical"]["refractive_index"] = current_cell[
            "material_refractive_index"
        ]
        config["wavelength"] = {
            "design_wavelength": current_cell["design_wavelength"],
            "wavelength_span": current_cell["wavelength_span"],
        }
        config["substrate"] = {
            "freestanding": current_cell["freestanding"],
            "material": current_cell["substrate"],
            "material_lumerical": current_cell["substrate_lumerical"],
            "refractive_index": current_cell["substrate_refractive_index"],
        }
        try:
            fdtd_result = run_fdtd_simulation(
                config, mesh_accuracy=mesh_accuracy, run=True
            )
        except Exception as e:
            err = str(e)
            if "refractive index" in err.lower() or "index" in err.lower():
                return {
                    "error": f"FDTD failed: {err}",
                    "requires_user_input": True,
                    "user_prompt": (
                        "FDTD could not apply refractive-index settings. "
                        "Please provide/confirm:\n"
                        "1) core material refractive index at target wavelength (material_refractive_index)\n"
                        "2) substrate refractive index if using substrate (substrate_refractive_index)\n"
                        "3) whether to use freestanding=true instead"
                    ),
                }
            return {
                "error": f"FDTD failed: {err}",
                "requires_user_input": True,
                "user_prompt": (
                    "FDTD failed. Please provide any missing setup details "
                    "(material names, refractive indices, target wavelength span, "
                    "or substrate/freestanding choice) so I can retry."
                ),
            }

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

        # Check and advance sweep step immediately after each design
        try:
            if self._is_step_complete():
                print(
                    f"  [STEP-CHECK] Step '{self.state.sweep_step}' complete after iter {self.state.iteration}"
                )
                self._advance_step()
            else:
                print(
                    f"  [STEP-CHECK] Step '{self.state.sweep_step}' not yet complete (iter {self.state.iteration})"
                )
        except Exception as e:
            print(f"  [STEP-CHECK ERROR] {e}")

        # Save log after each iteration (includes updated sweep state)
        self.state.save_log()

        # Build observation
        is_best = result["qv_ratio"] >= self.state.best_qv_ratio

        # Check targets - this tells the agent what phase we're in
        target_status = self.check_targets(result)

        observation = {
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

        # Every 5 iterations: inject full history + skills review
        if self.state.iteration % 5 == 0:
            review = self._build_periodic_review()
            observation["PERIODIC_REVIEW"] = review

        return observation

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

    def _build_periodic_review(self):
        """Build a full review every 5 iterations: history + skills + sweep progress."""
        target_nm = self.state.unit_cell["design_wavelength"] * 1e9

        # Load skills.md
        skills_text = ""
        try:
            with open("./skills.md", "r", encoding="utf-8") as f:
                skills_text = f.read().strip()
        except OSError:
            pass

        # Full history table
        history_lines = []
        for e in self.state.design_history:
            p = e["params"]
            r = e["result"]
            on_target = abs(r.get("resonance_nm", 0) - target_nm) <= 5
            history_lines.append(
                f"  #{e['iteration']}: p={p['period_nm']:.0f}, rx={p['hole_rx_nm']:.0f}, "
                f"ry={p['hole_ry_nm']:.0f}, a={p['min_a_percent']}%, "
                f"t={p['num_taper_holes']}, m={p['num_mirror_holes']} → "
                f"Q={r['Q']:,}, V={r['V']:.3f}, Q/V={r['qv_ratio']:,}, "
                f"res={r.get('resonance_nm', 0):.1f}nm "
                f"{'[ON TARGET]' if on_target else '[OFF TARGET]'}"
            )

        # Detect which parameters have been varied
        rx_values = sorted(
            set(round(e["params"]["hole_rx_nm"]) for e in self.state.design_history)
        )
        ry_values = sorted(
            set(round(e["params"]["hole_ry_nm"]) for e in self.state.design_history)
        )
        min_a_values = sorted(
            set(e["params"]["min_a_percent"] for e in self.state.design_history)
        )
        taper_values = sorted(
            set(e["params"]["num_taper_holes"] for e in self.state.design_history)
        )

        # Find best on-target result
        on_target_entries = [
            e
            for e in self.state.design_history
            if abs(e["result"].get("resonance_nm", 0) - target_nm) <= 5
        ]
        best_on_target = max(
            on_target_entries, key=lambda e: e["result"]["qv_ratio"], default=None
        )

        sweep_status = (
            f"SWEEP PROGRESS:\n"
            f"  rx values tried: {rx_values} ({'ONLY 1 VALUE - NOT SWEPT YET' if len(rx_values) == 1 else 'swept'})\n"
            f"  ry values tried: {ry_values} ({'ONLY 1 VALUE - NOT SWEPT YET' if len(ry_values) == 1 else 'swept'})\n"
            f"  min_a values tried: {min_a_values}\n"
            f"  taper values tried: {taper_values}\n"
        )

        if best_on_target:
            bp = best_on_target["params"]
            br = best_on_target["result"]
            sweep_status += (
                f"\n  BEST ON-TARGET: iter #{best_on_target['iteration']}: "
                f"rx={bp['hole_rx_nm']:.0f}, ry={bp['hole_ry_nm']:.0f}, "
                f"a={bp['min_a_percent']}%, p={bp['period_nm']:.0f} → "
                f"Q={br['Q']:,}, Q/V={br['qv_ratio']:,}\n"
            )

        # Determine what the LLM should do next per skills.md order
        next_action = ""
        min_a_done = set(min_a_values).issuperset({90, 89, 88, 87})
        best_min_a = best_on_target["params"]["min_a_percent"] if best_on_target else 90
        best_period = best_on_target["params"]["period_nm"] if best_on_target else 0
        best_rx = (
            best_on_target["params"]["hole_rx_nm"] if best_on_target else rx_values[0]
        )
        best_ry = (
            best_on_target["params"]["hole_ry_nm"] if best_on_target else ry_values[0]
        )

        if len(ry_values) == 1 and len(rx_values) > 1:
            next_action = (
                f"ACTION REQUIRED: rx has been swept but ry has NEVER been changed! "
                f"Per skills.md Step 3, you MUST now sweep hole_ry_nm in ±5nm steps. "
                f"Lock rx={best_rx:.0f}nm, min_a={best_min_a}%, period={best_period:.0f}nm. "
                f"Start trying ry={ry_values[0]+5}nm."
            )
        elif len(rx_values) == 1 and min_a_done:
            next_action = (
                f"ACTION REQUIRED: min_a sweep is COMPLETE (tried {min_a_values}). "
                f"Best on-target min_a={best_min_a}%. LOCK IT and move to rx sweep NOW. "
                f"Per skills.md Step 2, sweep hole_rx_nm in +5nm steps. "
                f"Lock min_a={best_min_a}%, period={best_period:.0f}nm. "
                f"Try rx={rx_values[0]+5}nm next. Do NOT re-sweep min_a again until rx is done."
            )
        elif len(rx_values) == 1 and not min_a_done:
            tried_str = ", ".join(str(v) for v in min_a_values)
            remaining = sorted(set([90, 89, 88, 87]) - set(min_a_values))
            next_action = (
                f"min_a sweep in progress (tried: {tried_str}, remaining: {remaining}). "
                f"Finish min_a sweep, then move to rx sweep per skills.md Step 2."
            )

        review = (
            f"{'='*60}\n"
            f"5-ITERATION REVIEW (iteration {self.state.iteration})\n"
            f"{'='*60}\n\n"
            f"FULL HISTORY:\n" + "\n".join(history_lines) + "\n\n"
            f"{sweep_status}\n"
            f"{next_action}\n\n"
            f"SKILLS REMINDER:\n{skills_text}\n"
            f"{'='*60}"
        )
        return review

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

        # Phase 3: V optimization (only when resonance and Q are met)
        v_target = 0.5
        if v_value > v_target:
            return {
                "on_target": False,
                "phase": "v_optimization",
                "wavelength_diff_nm": wavelength_diff,
                "Q": q_value,
                "V": v_value,
                "V_target": v_target,
                "message": f"Resonance OK, Q={q_value:,} OK. V={v_value:.3f} > {v_target} target — continue sweeping to reduce V",
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

    # ------------------------------------------------------------------
    # Deterministic optimization helpers (used by run_optimization_loop)
    # ------------------------------------------------------------------

    def _current_base_params(self):
        """Get base params from locked values, falling back to unit cell defaults."""
        locked = self.state.locked_params
        return {
            "period_nm": locked.get(
                "period_nm", round(self.state.unit_cell["period"] * 1e9)
            ),
            "hole_rx_nm": locked.get(
                "hole_rx_nm", round(self.state.unit_cell["hole_rx"] * 1e9)
            ),
            "hole_ry_nm": locked.get(
                "hole_ry_nm", round(self.state.unit_cell["hole_ry"] * 1e9)
            ),
            "min_a_percent": locked.get("min_a_percent", 90),
            "num_taper_holes": locked.get("num_taper_holes", 10),
            "num_mirror_holes": 7,
            "min_rx_percent": 100,
            "min_ry_percent": 100,
        }

    def _infer_state_from_history(self):
        """Infer sweep_step and locked_params from existing history when resuming old logs."""
        if self.state.locked_params or not self.state.design_history:
            return  # Already have state or no history

        target_nm = self.state.unit_cell["design_wavelength"] * 1e9

        # Find overall best on-target result
        best = None
        best_qv = 0
        for e in self.state.design_history:
            res = e["result"]
            if abs(res.get("resonance_nm", 0) - target_nm) <= 5:
                if res["qv_ratio"] > best_qv:
                    best_qv = res["qv_ratio"]
                    best = e

        if not best:
            # No on-target results, start from scratch with resonance tuning
            self.state.sweep_step = "initial"
            return

        p = best["params"]
        self.state.locked_params = {
            "period_nm": round(p["period_nm"]),
            "min_a_percent": p["min_a_percent"],
            "hole_rx_nm": round(p["hole_rx_nm"]),
            "hole_ry_nm": round(p["hole_ry_nm"]),
            "num_taper_holes": p["num_taper_holes"],
        }

        # Infer step from what was varied in history
        initial_rx = round(self.state.unit_cell["hole_rx"] * 1e9)
        initial_ry = round(self.state.unit_cell["hole_ry"] * 1e9)
        rx_values = set(
            round(e["params"]["hole_rx_nm"]) for e in self.state.design_history
        )
        ry_values = set(
            round(e["params"]["hole_ry_nm"]) for e in self.state.design_history
        )
        taper_values = set(
            e["params"]["num_taper_holes"] for e in self.state.design_history
        )

        if len(taper_values) > 1:
            self.state.sweep_step = "fine_period"
        elif len(ry_values) > 1:
            self.state.sweep_step = "re_sweep_min_a_2"
        elif len(rx_values) > 1:
            # rx was varied — check if it peaked (Q dropped on last tried rx)
            # Move to re_sweep_min_a_1 (re-sweep min_a with best rx locked)
            self.state.sweep_step = "re_sweep_min_a_1"
        else:
            # Only min_a was varied (or nothing), start rx sweep
            self.state.sweep_step = "sweep_rx"

        self.state.step_start_iter = 0
        self.state.save_log()

        print(f"[INFER] Inferred from {self.state.iteration} previous iterations:")
        print(f"[INFER] Locked params: {self.state.locked_params}")
        print(f"[INFER] Starting at step: {self.state.sweep_step}")

    def _estimate_target_period(self):
        """Estimate period needed to hit target resonance from history data."""
        if len(self.state.design_history) < 2:
            return None

        target_nm = self.state.unit_cell["design_wavelength"] * 1e9
        pairs = [
            (e["params"]["period_nm"], e["result"].get("resonance_nm", 0))
            for e in self.state.design_history
            if e["result"].get("resonance_nm", 0) > 0
        ]
        if len(pairs) < 2:
            return None

        # Compute average d(resonance)/d(period) from meaningful period changes
        ratios = []
        for i in range(1, len(pairs)):
            dp = pairs[i][0] - pairs[i - 1][0]
            dr = pairs[i][1] - pairs[i - 1][1]
            if abs(dp) > 0.5:
                ratios.append(dr / dp)

        if not ratios:
            return None
        avg_ratio = sum(ratios) / len(ratios)
        if abs(avg_ratio) < 0.1:
            return None

        last = self.state.design_history[-1]
        current_period = last["params"]["period_nm"]
        current_resonance = last["result"]["resonance_nm"]
        needed_shift = target_nm - current_resonance
        period_adjust = needed_shift / avg_ratio

        return round(current_period + period_adjust)

    def _retune_period_and_run(self, base_params, max_retune=5):
        """Run design_cavity with automatic period retuning to stay on target.

        Runs the design, checks resonance, and adjusts period if off-target
        using linear interpolation. Returns the observation from the final run.
        """
        target_nm = self.state.unit_cell["design_wavelength"] * 1e9
        period = base_params.get(
            "period_nm",
            self.state.locked_params.get(
                "period_nm", round(self.state.unit_cell["period"] * 1e9)
            ),
        )

        last_obs = None
        period_resonance_pairs = []  # (period, resonance) for interpolation

        for attempt in range(max_retune + 1):
            params = dict(base_params)
            params["period_nm"] = round(period)
            if attempt > 0:
                params["hypothesis"] = (
                    f"Period retune #{attempt}: period={round(period)}nm"
                )

            obs = self._design_cavity(params)
            last_obs = obs

            # Handle errors
            if isinstance(obs, dict) and "error" in obs:
                err_msg = str(obs["error"])
                if "DUPLICATE" in err_msg:
                    dup = self.state.find_duplicate(params)
                    if dup:
                        res_nm = dup["result"]["resonance_nm"]
                        period_resonance_pairs.append((round(period), res_nm))
                        if abs(res_nm - target_nm) <= 5:
                            return obs  # Duplicate was on target, good enough
                        # Use data to estimate next period
                        if len(period_resonance_pairs) >= 2:
                            p1, r1 = period_resonance_pairs[-2]
                            p2, r2 = period_resonance_pairs[-1]
                            if abs(r2 - r1) > 0.5:
                                ratio = (p2 - p1) / (r2 - r1)
                                period = round(p2 + (target_nm - r2) * ratio)
                            else:
                                period += 1 if res_nm < target_nm else -1
                        else:
                            period += 2 if res_nm < target_nm else -2
                        continue
                    else:
                        return obs
                else:
                    return obs  # Real error, bail out

            # Get resonance from history
            last_entry = self.state.design_history[-1]
            res_nm = last_entry["result"]["resonance_nm"]
            period_resonance_pairs.append((round(period), res_nm))

            if abs(res_nm - target_nm) <= 5:
                return obs  # On target!

            # Estimate new period
            if len(period_resonance_pairs) >= 2:
                p1, r1 = period_resonance_pairs[-2]
                p2, r2 = period_resonance_pairs[-1]
                if abs(r2 - r1) > 0.5 and abs(p2 - p1) >= 1:
                    ratio = (p2 - p1) / (r2 - r1)
                    period = round(p2 + (target_nm - r2) * ratio)
                else:
                    period = round(period + (target_nm - res_nm) / 2.5)
            else:
                # First attempt fallback: ~2.5nm resonance per 1nm period
                period = round(period + (target_nm - res_nm) / 2.5)

            print(
                f"  [RETUNE] Resonance {res_nm:.1f}nm (target {target_nm:.1f}nm) "
                f"-> trying period {round(period)}nm"
            )

        return last_obs

    def run_manual_sweep(self, param_name, start, end, step=None):
        """Run a deterministic parameter sweep with automatic period retuning.

        Args:
            param_name: Parameter to sweep (e.g. 'num_taper_holes', 'min_a_percent')
            start: Start value (inclusive)
            end: End value (inclusive)
            step: Step size (auto-detected if None)

        Returns:
            dict with 'results' list and 'best' on-target result
        """
        if not self.state.unit_cell:
            return {"error": "Must call set_unit_cell first"}
        if not self.state.fdtd_confirmed:
            return {"error": "Must confirm fdtd first"}

        # Validate param name
        valid_params = {
            "num_taper_holes",
            "num_mirror_holes",
            "min_a_percent",
            "hole_rx_nm",
            "hole_ry_nm",
            "period_nm",
            "min_rx_percent",
            "min_ry_percent",
        }
        if param_name not in valid_params:
            return {
                "error": f"Unknown parameter '{param_name}'. Valid: {sorted(valid_params)}"
            }

        # Auto-detect step
        if step is None:
            if param_name in ("hole_rx_nm", "hole_ry_nm"):
                step = 5
            else:
                step = 1

        # Generate sweep values
        int_params = {"num_taper_holes", "num_mirror_holes"}
        if param_name in int_params:
            start, end, step = int(start), int(end), int(step)
            if start <= end:
                values = list(range(start, end + 1, step))
            else:
                values = list(range(start, end - 1, -step))
        else:
            start, end, step = float(start), float(end), float(step)
            values = []
            v = start
            if start <= end:
                while v <= end + step * 0.01:
                    values.append(round(v, 2))
                    v += step
            else:
                while v >= end - step * 0.01:
                    values.append(round(v, 2))
                    v -= step

        target_nm = self.state.unit_cell["design_wavelength"] * 1e9

        # Temporarily set sweep_step to "manual" to disable enforce logic
        old_step = self.state.sweep_step
        old_start_iter = self.state.step_start_iter
        self.state.sweep_step = "manual"

        print(f"\n{'='*60}")
        print(f"MANUAL SWEEP: {param_name} = {values}")
        print(f"Locked params: {self.state.locked_params}")
        print(f"Target wavelength: {target_nm:.1f}nm")
        print(f"{'='*60}\n")

        results = []

        try:
            for i, val in enumerate(values):
                print(f"\n--- Sweep point {i+1}/{len(values)}: {param_name}={val} ---")

                # Build params from locked/base values
                base = self._current_base_params()
                base[param_name] = val

                obs = self._retune_period_and_run(base)

                if isinstance(obs, dict) and "error" in obs:
                    err = obs["error"]
                    # For duplicates, try to find the existing result
                    if "DUPLICATE" in str(err):
                        dup = self.state.find_duplicate(base)
                        if dup:
                            r = dup["result"]
                            on_target = abs(r["resonance_nm"] - target_nm) <= 5
                            results.append(
                                {
                                    "value": val,
                                    "iteration": dup["iteration"],
                                    "Q": r["Q"],
                                    "V": r["V"],
                                    "qv_ratio": r["qv_ratio"],
                                    "resonance_nm": r["resonance_nm"],
                                    "on_target": on_target,
                                    "period_nm": dup["params"]["period_nm"],
                                    "note": "from_cache",
                                }
                            )
                            print(f"  Cached: Q={r['Q']:,}, Q/V={r['qv_ratio']:,}")
                            continue
                    print(f"  Error: {err}")
                    results.append({"value": val, "error": str(err)})
                    continue

                # Get actual result from history
                last = self.state.design_history[-1]
                r = last["result"]
                on_target = abs(r["resonance_nm"] - target_nm) <= 5

                results.append(
                    {
                        "value": val,
                        "iteration": last["iteration"],
                        "Q": r["Q"],
                        "V": r["V"],
                        "qv_ratio": r["qv_ratio"],
                        "resonance_nm": r["resonance_nm"],
                        "on_target": on_target,
                        "period_nm": last["params"]["period_nm"],
                    }
                )

                print(
                    f"  Result: Q={r['Q']:,}, V={r['V']:.3f}, Q/V={r['qv_ratio']:,}, "
                    f"res={r['resonance_nm']:.1f}nm "
                    f"{'[ON TARGET]' if on_target else '[OFF TARGET]'}"
                )
        finally:
            self.state.sweep_step = old_step
            self.state.step_start_iter = old_start_iter

        # Find best on-target result
        on_target_results = [r for r in results if r.get("on_target")]
        best = (
            max(on_target_results, key=lambda r: r["qv_ratio"])
            if on_target_results
            else None
        )

        # Update locked params with best value
        if best:
            self.state.locked_params[param_name] = best["value"]
            self.state.locked_params["period_nm"] = round(best["period_nm"])
            self.state.save_log()

        # Print summary table
        print(f"\n{'='*60}")
        print(f"SWEEP SUMMARY: {param_name}")
        print(f"{'='*60}")
        print(
            f"{'Value':>8} {'Period':>8} {'Q':>12} {'V':>8} "
            f"{'Q/V':>12} {'Res(nm)':>8} {'Status':>12}"
        )
        print("-" * 76)
        for r in results:
            if "error" in r:
                print(
                    f"{r['value']:>8} {'':>8} {'':>12} {'':>8} {'':>12} {'':>8} {'ERROR':>12}"
                )
            else:
                status = "ON TARGET" if r["on_target"] else "OFF"
                mark = " <-- BEST" if best and r["value"] == best["value"] else ""
                print(
                    f"{r['value']:>8} {r['period_nm']:>8.0f} {r['Q']:>12,} "
                    f"{r['V']:>8.3f} {r['qv_ratio']:>12,} {r['resonance_nm']:>8.1f} "
                    f"{status:>12}{mark}"
                )
        if best:
            print(
                f"\nBEST: {param_name}={best['value']}, Q/V={best['qv_ratio']:,}, "
                f"period={round(best['period_nm'])}nm"
            )
        else:
            print(f"\nNo on-target results found!")
        print(f"{'='*60}\n")

        return {"results": results, "best": best}

    def _is_step_complete(self):
        """Check if current sweep step is complete based on history."""
        step = self.state.sweep_step
        step_history = self.state.get_step_history()
        target_nm = self.state.unit_cell["design_wavelength"] * 1e9

        on_target = [
            e
            for e in step_history
            if abs(e["result"].get("resonance_nm", 0) - target_nm) <= 5
        ]

        if step == "initial":
            return len(on_target) >= 1

        if step in ("sweep_min_a", "re_sweep_min_a_1", "re_sweep_min_a_2"):
            tried = set(e["params"]["min_a_percent"] for e in on_target)
            return tried.issuperset({90, 89, 88, 87})

        if step == "sweep_rx":
            rx_best = {}
            for e in on_target:
                rx = round(e["params"]["hole_rx_nm"])
                qv = e["result"]["qv_ratio"]
                if rx not in rx_best or qv > rx_best[rx]:
                    rx_best[rx] = qv
            if len(rx_best) < 2:
                return False
            best_rx = max(rx_best, key=rx_best.get)
            # Done when Q/V dropped on both sides of the peak
            has_worse_above = any(rx > best_rx for rx in rx_best)
            has_worse_below = any(rx < best_rx for rx in rx_best)
            return has_worse_above and has_worse_below

        if step == "sweep_ry":
            ry_best = {}
            for e in on_target:
                ry = round(e["params"]["hole_ry_nm"])
                qv = e["result"]["qv_ratio"]
                if ry not in ry_best or qv > ry_best[ry]:
                    ry_best[ry] = qv
            if len(ry_best) < 2:
                return False
            best_ry = max(ry_best, key=ry_best.get)
            has_worse_above = any(ry > best_ry for ry in ry_best)
            has_worse_below = any(ry < best_ry for ry in ry_best)
            return has_worse_above and has_worse_below

        if step == "sweep_taper":
            tried = set(e["params"]["num_taper_holes"] for e in on_target)
            return tried.issuperset({8, 10, 12})

        if step == "fine_period":
            center = round(
                self.state.locked_params.get(
                    "period_nm", self.state.unit_cell["period"] * 1e9
                )
            )
            tried = set(round(e["params"]["period_nm"]) for e in on_target)
            needed = {center + o for o in [-2, -1, 0, 1, 2]}
            return needed.issubset(tried)

        return False

    def _advance_step(self):
        """Lock in best value from current step and advance to next."""
        step = self.state.sweep_step
        step_history = self.state.get_step_history()
        target_nm = self.state.unit_cell["design_wavelength"] * 1e9

        on_target = [
            e
            for e in step_history
            if abs(e["result"].get("resonance_nm", 0) - target_nm) <= 5
        ]
        best_entry = max(on_target, key=lambda e: e["result"]["qv_ratio"], default=None)

        if best_entry:
            p = best_entry["params"]
            # Always lock ALL base params from best entry
            self.state.locked_params["period_nm"] = round(p["period_nm"])
            self.state.locked_params["hole_rx_nm"] = round(p["hole_rx_nm"])
            self.state.locked_params["hole_ry_nm"] = round(p["hole_ry_nm"])
            self.state.locked_params["min_a_percent"] = p["min_a_percent"]
            self.state.locked_params["num_taper_holes"] = p["num_taper_holes"]
            print(
                f"[LOCK] All params from best: period={round(p['period_nm'])}, "
                f"rx={round(p['hole_rx_nm'])}, ry={round(p['hole_ry_nm'])}, "
                f"min_a={p['min_a_percent']}%, taper={p['num_taper_holes']} "
                f"(Q/V={best_entry['result']['qv_ratio']:,})"
            )

        if step == "initial":
            next_step = "sweep_min_a"
        else:
            try:
                idx = SWEEP_ORDER.index(step)
                next_step = (
                    SWEEP_ORDER[idx + 1] if idx + 1 < len(SWEEP_ORDER) else "complete"
                )
            except ValueError:
                next_step = "complete"

        self.state.sweep_step = next_step
        # Include the best entry from the previous step in the new step's
        # history so its parameter value counts as "already tried".
        if best_entry:
            self.state.step_start_iter = best_entry["iteration"] - 1
        else:
            self.state.step_start_iter = self.state.iteration
        self.state.save_log()
        print(f"[STEP] {step} -> {next_step}")

    def _build_step_prompt(self):
        """Build a step-aware prompt that tells the LLM exactly what to focus on."""
        step = self.state.sweep_step
        locked = self.state.locked_params
        target_nm = self.state.unit_cell["design_wavelength"] * 1e9
        step_history = self.state.get_step_history()

        # Collect on-target results from this step
        on_target = [
            e
            for e in step_history
            if abs(e["result"].get("resonance_nm", 0) - target_nm) <= 5
        ]

        header = (
            f"OPTIMIZATION STEP: {step}\n"
            f"Locked params (use these as base, do NOT change unless told): {locked}\n"
            f"Target wavelength: {target_nm:.1f}nm\n"
        )

        # Check resonance first
        if self.state.design_history:
            last = self.state.design_history[-1]
            res_nm = last["result"].get("resonance_nm", 0)
            if abs(res_nm - target_nm) > 5:
                diff = target_nm - res_nm
                return (
                    f"{header}\n"
                    f"PRIORITY: Resonance is at {res_nm:.1f}nm, off target by {abs(diff):.1f}nm.\n"
                    f"{'INCREASE' if diff > 0 else 'DECREASE'} period_nm to fix. "
                    f"Change ONLY period. Keep everything else at locked values.\n"
                    f"Call view_history first, then design_cavity."
                )

        if step == "initial":
            period_nm = round(self.state.unit_cell["period"] * 1e9)
            rx_nm = round(self.state.unit_cell["hole_rx"] * 1e9)
            ry_nm = round(self.state.unit_cell["hole_ry"] * 1e9)
            initial_min_a = self.state.unit_cell.get("initial_min_a_percent", 90)
            return (
                f"{header}\n"
                f"Run the first baseline design using the EXACT unit cell values — "
                f"DO NOT change period, hole_rx, or hole_ry:\n"
                f"  period_nm={period_nm}, hole_rx_nm={rx_nm}, hole_ry_nm={ry_nm},\n"
                f"  taper=10, mirror=7, min_a={initial_min_a:g}, min_rx=100, min_ry=100.\n"
                f"Call design_cavity now with these exact values."
            )

        if step in ("sweep_min_a", "re_sweep_min_a_1", "re_sweep_min_a_2"):
            tried = set(e["params"]["min_a_percent"] for e in on_target)
            results_str = ", ".join(
                f"a={e['params']['min_a_percent']}%->Q/V={e['result']['qv_ratio']:,}"
                for e in on_target
            )
            return (
                f"{header}\n"
                f"SWEEP min_a_percent through [90, 89, 88, 87].\n"
                f"Already tried (on-target): {sorted(tried)} -> {results_str}\n"
                f"Try the next untried value. Keep ALL other params at locked values.\n"
                f"Call view_history first to check for duplicates, then design_cavity."
            )

        if step == "sweep_rx":
            tried = sorted(set(round(e["params"]["hole_rx_nm"]) for e in on_target))
            results_str = ", ".join(
                f"rx={round(e['params']['hole_rx_nm'])}->Q/V={e['result']['qv_ratio']:,}"
                for e in on_target
            )
            return (
                f"{header}\n"
                f"SWEEP hole_rx_nm in ±5nm steps to find the peak Q/V.\n"
                f"Already tried (on-target): {tried} -> {results_str}\n"
                f"Strategy: try +5nm first. If Q/V drops, try -5nm from initial instead.\n"
                f"STOP when Q/V has dropped on BOTH sides of the best rx.\n"
                f"Keep ALL other params at locked values.\n"
                f"If resonance shifts >5nm, retune period first.\n"
                f"Call view_history first, then design_cavity."
            )

        if step == "sweep_ry":
            tried = sorted(set(round(e["params"]["hole_ry_nm"]) for e in on_target))
            results_str = ", ".join(
                f"ry={round(e['params']['hole_ry_nm'])}->Q/V={e['result']['qv_ratio']:,}"
                for e in on_target
            )
            return (
                f"{header}\n"
                f"SWEEP hole_ry_nm in ±5nm steps to find the peak Q/V.\n"
                f"Already tried (on-target): {tried} -> {results_str}\n"
                f"Strategy: try +5nm first. If Q/V drops, try -5nm from initial instead.\n"
                f"STOP when Q/V has dropped on BOTH sides of the best ry.\n"
                f"Keep ALL other params at locked values.\n"
                f"If resonance shifts >5nm, retune period first.\n"
                f"Call view_history first, then design_cavity."
            )

        if step == "sweep_taper":
            tried = set(e["params"]["num_taper_holes"] for e in on_target)
            results_str = ", ".join(
                f"t={e['params']['num_taper_holes']}->Q/V={e['result']['qv_ratio']:,}"
                for e in on_target
            )
            return (
                f"{header}\n"
                f"SWEEP num_taper_holes through [8, 10, 12].\n"
                f"Already tried (on-target): {sorted(tried)} -> {results_str}\n"
                f"Try the next untried value. Keep ALL other params at locked values.\n"
                f"Call view_history first, then design_cavity."
            )

        if step == "fine_period":
            center = round(
                locked.get("period_nm", self.state.unit_cell["period"] * 1e9)
            )
            tried = set(round(e["params"]["period_nm"]) for e in on_target)
            results_str = ", ".join(
                f"p={round(e['params']['period_nm'])}->Q/V={e['result']['qv_ratio']:,}"
                for e in on_target
            )
            return (
                f"{header}\n"
                f"FINE PERIOD SWEEP: try {center-2}, {center-1}, {center}, {center+1}, {center+2}nm.\n"
                f"Already tried (on-target): {sorted(tried)} -> {results_str}\n"
                f"Try the next untried value. Keep ALL other params at locked values.\n"
                f"Call view_history first, then design_cavity."
            )

        return f"{header}\nOptimization complete. Call get_best_design to see the final result."

    def run_optimization_loop(self, max_iterations=10, extra_instruction=None):
        """
        LLM-driven optimization loop with step-aware prompts.
        The LLM reasons about results; the code tracks which step we're on
        and refreshes skills.md every 5 iterations to prevent context loss.
        """
        if not self.state.unit_cell:
            return {"error": "Must call set_unit_cell first via chat()"}
        if self.state.iteration == 0 and not self.state.fdtd_confirmed:
            return {"error": "First FDTD run not confirmed. Use command: confirm fdtd"}

        target_nm = self.state.unit_cell["design_wavelength"] * 1e9

        # Bootstrap state from old logs that lack step tracking
        self._infer_state_from_history()

        # Load skills.md once for periodic injection
        skills_text = ""
        try:
            with open("./skills.md", "r", encoding="utf-8") as f:
                skills_text = f.read().strip()
        except OSError:
            pass

        print(f"\n{'='*60}")
        print(f"LLM optimization ({max_iterations} iterations)")
        print(f"Step: {self.state.sweep_step} | Locked: {self.state.locked_params}")
        print(f"Best Q/V so far: {self.state.best_qv_ratio:,}")
        if extra_instruction:
            print(f"Extra instruction: {extra_instruction}")
        print(f"{'='*60}\n")

        last_best_qv = self.state.best_qv_ratio

        # Resume context if continuing from previous run
        if self.state.iteration > 0 and self.state.design_history:
            history_summary = self._get_history_summary()
            review_prompt = (
                f"RESUMING OPTIMIZATION\n"
                f"Step: {self.state.sweep_step} | Locked: {self.state.locked_params}\n\n"
                f"HISTORY:\n{history_summary}\n\n"
                f"Continue from step: {self.state.sweep_step}."
            )
            self.chat(review_prompt)

        for i in range(max_iterations):
            # Check if current step is complete -> advance
            step_complete = self._is_step_complete()
            print(
                f"  [DEBUG] step={self.state.sweep_step}, step_complete={step_complete}, "
                f"history_len={len(self.state.design_history)}, "
                f"step_start_iter={self.state.step_start_iter}"
            )
            if step_complete:
                self._advance_step()
                if self.state.sweep_step == "complete":
                    print("\n[DONE] All optimization steps complete!")
                    return {"status": "complete", "result": self._get_best_design()}

            # Build step-aware prompt
            prompt = self._build_step_prompt()
            if extra_instruction:
                prompt = (
                    f"ADDITIONAL USER INSTRUCTION (MUST FOLLOW): {extra_instruction}\n\n"
                    f"{prompt}"
                )

            # Re-inject skills.md every 5 iterations to combat context loss
            if self.state.iteration % 5 == 0 and skills_text:
                prompt = (
                    f"RULES REFRESH (re-read carefully):\n\n{skills_text}\n\n"
                    f"---\n\n{prompt}"
                )

            print(
                f"\n--- Iteration {i+1}/{max_iterations} | Step: {self.state.sweep_step} ---"
            )
            print(f"Prompt: {prompt[:120]}...")

            response = self.chat(prompt)
            print(f"Agent: {response[:200] if response else 'No response'}...")

            # Check for improvement
            if self.state.best_qv_ratio > last_best_qv:
                last_best_qv = self.state.best_qv_ratio
                print(f"[NEW BEST] Q/V = {last_best_qv:,}")

            # Check targets
            if self.state.design_history:
                last_result = self.state.design_history[-1]["result"]
                if self.check_targets(last_result).get("on_target"):
                    print(f"\n{'='*60}")
                    print("ALL TARGETS MET!")
                    print(f"{'='*60}\n")
                    return {"status": "complete", "result": self._get_best_design()}

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
        # Limit conversation history to avoid context overflow.
        # Must keep tool_use/tool_result pairs intact — find a safe cut point
        # where we don't orphan a tool_result without its tool_use.
        if len(self.conversation_history) > 10:
            # Start from position -10 and scan forward to find a "user" role
            # message that is NOT a tool_result (safe start point)
            cut = len(self.conversation_history) - 10
            while cut < len(self.conversation_history):
                msg = self.conversation_history[cut]
                content = msg.get("content", "")
                # tool_result messages have list content with type "tool_result"
                is_tool_result = isinstance(content, list) and any(
                    isinstance(c, dict) and c.get("type") == "tool_result"
                    for c in content
                )
                if not is_tool_result:
                    break
                cut += 1
            self.conversation_history = self.conversation_history[cut:]

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
                    max_tokens=4096,
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

            # If a tool explicitly needs user clarification, stop looping and ask.
            if isinstance(observation, dict) and observation.get("requires_user_input"):
                prompt = observation.get(
                    "user_prompt",
                    "I need a bit more input before continuing. Please provide missing setup details.",
                )
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": f"{observation.get('error', 'Tool error')}\n{prompt}",
                    }
                )
                return f"{observation.get('error', 'Tool error')}\n{prompt}"

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
                        max_tokens=4096,
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


def _parse_sweep_command(text):
    """Parse flexible sweep syntax. Returns (param, start, end, step) or None.

    Accepted formats:
        sweep num_taper_holes from 5 to 10
        sweep num_taper_holes from 5 to 10 step 1
        sweep num_taper_holes 5 to 10
        sweep num_taper_holes 5 10
        sweep num_taper_holes 5 10 1
    """
    tokens = text.split()
    if len(tokens) < 3 or tokens[0].lower() != "sweep":
        return None

    param_name = tokens[1]

    # Remove filler words: "from", "to", "step", "="
    numbers = []
    labels = []
    i = 2
    while i < len(tokens):
        word = tokens[i].lower().strip(",;:")
        if word in ("from", "to", "="):
            i += 1
            continue
        if word == "step":
            labels.append("step")
            i += 1
            continue
        # Try to parse as number
        try:
            numbers.append(float(word))
            labels.append("num")
        except ValueError:
            pass  # skip unknown words
        i += 1

    if len(numbers) < 2:
        return None

    start = numbers[0]
    end = numbers[1]
    step = numbers[2] if len(numbers) >= 3 else None

    return param_name, start, end, step


def main():
    print("=" * 60)
    print("SWE-Agent Style Nanobeam Cavity Designer")
    print("=" * 60)
    print("\nThis agent uses a thought-action-observation loop to")
    print("systematically design and optimize photonic cavities.")
    print("\nCommands:")
    print("  'Design a SiN cavity at 737nm' - setup unit cell")
    print("  'confirm fdtd' - confirm all inputs before first FDTD run")
    print("  'auto' or 'auto 15' - run automated optimization (default 10 iterations)")
    print("  'auto 15 -- <instruction>' - run auto with additional constraint")
    print("  'sweep <param> from <start> to <end> [step <N>]' - deterministic sweep")
    print("    e.g. sweep num_taper_holes from 5 to 10")
    print("         sweep min_a_percent 87 90")
    print("         sweep hole_rx_nm from 80 to 100 step 5")
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

            if user_input.lower() in ["confirm fdtd", "confirm"]:
                result = agent.confirm_fdtd_inputs()
                if "error" in result:
                    print(f"\nAgent: {result['error']}\n")
                else:
                    print(f"\nAgent: {result['message']}\n")
                continue

            # Check for sweep command
            if user_input.lower().startswith("sweep"):
                parsed = _parse_sweep_command(user_input)
                if parsed is None:
                    print(
                        "\nInvalid sweep syntax. Examples:\n"
                        "  sweep num_taper_holes from 5 to 10\n"
                        "  sweep min_a_percent 87 90\n"
                        "  sweep hole_rx_nm from 80 to 100 step 5\n"
                    )
                    continue

                param_name, start, end, step = parsed
                print(
                    f"\nStarting sweep: {param_name} from {start} to {end}"
                    + (f" step {step}" if step else "")
                )

                result = agent.run_manual_sweep(param_name, start, end, step)

                if "error" in result:
                    print(f"\nError: {result['error']}\n")
                else:
                    show_best_result()

                continue

            # Check for auto mode
            if user_input.lower().startswith("auto"):
                parts = user_input.split()
                max_iter = 10
                extra_instruction = None

                # Preferred syntax: auto [N] -- <instruction>
                if "--" in user_input:
                    left, right = user_input.split("--", 1)
                    extra_instruction = right.strip() or None
                    parts = left.split()
                elif "," in user_input:
                    # Backward-compatible free-text suffix: auto, <instruction>
                    left, right = user_input.split(",", 1)
                    if left.strip().lower().startswith("auto"):
                        extra_instruction = right.strip() or None
                        parts = left.split()

                if len(parts) > 1:
                    for token in parts[1:]:
                        cleaned = token.strip(",.;:!?")
                        if cleaned.isdigit():
                            max_iter = int(cleaned)
                            break

                while True:
                    result = agent.run_optimization_loop(
                        max_iterations=max_iter, extra_instruction=extra_instruction
                    )

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
