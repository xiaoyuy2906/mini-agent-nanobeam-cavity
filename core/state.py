import os
import json
from datetime import datetime

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
        unit_cell.get("wg_material", unit_cell.get("material", "")),
        unit_cell.get("substrate_material", unit_cell.get("substrate", "")),
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
