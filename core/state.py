import os
import sys
import json
from datetime import datetime

_log = lambda *a, **kw: print(*a, file=sys.stderr, **kw)

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

        _log(f"[LOG] Saved {self.iteration} iterations to {filepath}")

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

        _log(f"[LOG] Loaded {self.iteration} previous iterations from {filepath}")
        _log(f"[LOG] Sweep step: {self.sweep_step}, locked: {self.locked_params}")
        _log(f"[LOG] Best Q/V so far: {self.best_qv_ratio:,.0f}")
        return True

    # ------------------------------------------------------------------
    # Sensitivity analysis & experiment suggestion
    # ------------------------------------------------------------------

    # Parameters that can be swept (keys as they appear in design params)
    SWEEP_PARAMS = [
        "period_nm", "min_a_percent", "hole_rx_nm", "hole_ry_nm",
        "num_taper_holes", "min_rx_percent", "min_ry_percent",
    ]

    def analyze_sensitivity(self) -> dict:
        """Compute finite-difference sensitivity of Q, V, Q/V to each parameter.

        For each swept parameter, groups designs that differ only in that
        parameter (all other params within tolerance), then computes
        ΔQ/Δparam, ΔV/Δparam, Δ(Q/V)/Δparam.
        """
        if len(self.design_history) < 2:
            return {
                "ok": True,
                "message": "Need at least 2 designs to compute sensitivity.",
                "sensitivities": [],
            }

        sensitivities = []
        for param in self.SWEEP_PARAMS:
            pairs = self._find_param_variation_pairs(param)
            if not pairs:
                continue

            dq_list, dv_list, dqv_list = [], [], []
            for e1, e2 in pairs:
                dp = self._param_val(e2, param) - self._param_val(e1, param)
                if abs(dp) < 1e-9:
                    continue
                r1, r2 = e1["result"], e2["result"]
                dq_list.append((r2.get("Q", 0) - r1.get("Q", 0)) / dp)
                dv_list.append((r2.get("V", 0) - r1.get("V", 0)) / dp)
                dqv_list.append((r2.get("qv_ratio", 0) - r1.get("qv_ratio", 0)) / dp)

            if not dqv_list:
                continue

            avg = lambda lst: sum(lst) / len(lst) if lst else 0
            sensitivities.append({
                "param": param,
                "dQ_dparam": round(avg(dq_list), 2),
                "dV_dparam": round(avg(dv_list), 6),
                "dQV_dparam": round(avg(dqv_list), 2),
                "abs_dQV_dparam": round(abs(avg(dqv_list)), 2),
                "num_pairs": len(dqv_list),
            })

        # Sort by absolute Q/V sensitivity (highest first)
        sensitivities.sort(key=lambda s: s["abs_dQV_dparam"], reverse=True)

        suggestion = ""
        if sensitivities:
            top = sensitivities[0]
            suggestion = (
                f"Parameter '{top['param']}' has the highest Q/V sensitivity "
                f"({top['dQV_dparam']:+.2f} per unit). Consider prioritizing it."
            )

        return {"ok": True, "sensitivities": sensitivities, "suggestion": suggestion}

    def suggest_next_experiment(self) -> dict:
        """Suggest the most promising next experiment based on history.

        Uses simple quadratic fitting per parameter to predict the value
        that maximizes Q/V, and identifies under-explored parameters.
        """
        if len(self.design_history) < 2:
            return {
                "ok": True,
                "message": "Need at least 2 designs. Try the recommended starting order.",
                "suggestions": [],
            }

        suggestions = []
        for param in self.SWEEP_PARAMS:
            points = self._get_param_qv_points(param)
            if len(points) < 2:
                suggestions.append({
                    "param": param,
                    "status": "unexplored",
                    "message": f"Only {len(points)} data point(s). Consider exploring this parameter.",
                    "priority": "medium" if len(points) == 0 else "low",
                })
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            best_idx = max(range(len(ys)), key=lambda i: ys[i])
            best_x, best_y = xs[best_idx], ys[best_idx]

            # Try quadratic fit if enough points
            predicted_optimum = None
            if len(points) >= 3:
                predicted_optimum = self._quadratic_peak(xs, ys)

            # Check if the best is at an edge (suggesting further exploration)
            at_edge = best_idx == 0 or best_idx == len(xs) - 1
            x_range = max(xs) - min(xs)

            if predicted_optimum is not None and abs(predicted_optimum - best_x) > 0.1:
                suggestions.append({
                    "param": param,
                    "status": "predicted_optimum",
                    "current_best_value": round(best_x, 2),
                    "current_best_qv": round(best_y, 2),
                    "predicted_optimal_value": round(predicted_optimum, 2),
                    "message": f"Quadratic fit predicts optimum at {predicted_optimum:.1f}. Try it.",
                    "priority": "high",
                })
            elif at_edge and x_range > 0:
                direction = "higher" if best_idx == len(xs) - 1 else "lower"
                suggestions.append({
                    "param": param,
                    "status": "edge_best",
                    "current_best_value": round(best_x, 2),
                    "current_best_qv": round(best_y, 2),
                    "message": f"Best Q/V at {direction} edge. Extend sweep in that direction.",
                    "priority": "high",
                })
            else:
                suggestions.append({
                    "param": param,
                    "status": "converged",
                    "current_best_value": round(best_x, 2),
                    "current_best_qv": round(best_y, 2),
                    "message": "Appears converged around current best.",
                    "priority": "low",
                })

        # Sort: high priority first
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: priority_order.get(s.get("priority", "low"), 2))

        return {"ok": True, "suggestions": suggestions}

    # --- helpers ---

    @staticmethod
    def _param_val(entry: dict, param: str) -> float:
        """Extract a numeric parameter value from a design history entry."""
        return float(entry["params"].get(param, 0))

    def _find_param_variation_pairs(self, target_param: str) -> list[tuple[dict, dict]]:
        """Find pairs of designs that differ primarily in *target_param*.

        Two designs are a valid pair if all other sweep parameters are
        within tolerance and the target parameter actually differs.
        """
        tol = {
            "period_nm": 0.5, "min_a_percent": 0.5, "hole_rx_nm": 0.5,
            "hole_ry_nm": 0.5, "num_taper_holes": 0, "min_rx_percent": 0.5,
            "min_ry_percent": 0.5,
        }
        pairs = []
        n = len(self.design_history)
        for i in range(n):
            for j in range(i + 1, n):
                ei, ej = self.design_history[i], self.design_history[j]
                pi, pj = ei["params"], ej["params"]
                # target param must differ
                vi = float(pi.get(target_param, 0))
                vj = float(pj.get(target_param, 0))
                if abs(vi - vj) < 1e-9:
                    continue
                # all other params must be close
                others_match = True
                for other in self.SWEEP_PARAMS:
                    if other == target_param:
                        continue
                    oi = float(pi.get(other, 0 if other != "min_rx_percent" and other != "min_ry_percent" else 100))
                    oj = float(pj.get(other, 0 if other != "min_rx_percent" and other != "min_ry_percent" else 100))
                    if abs(oi - oj) > tol.get(other, 0.5):
                        others_match = False
                        break
                if others_match:
                    # Order by target param ascending
                    if vi <= vj:
                        pairs.append((ei, ej))
                    else:
                        pairs.append((ej, ei))
        return pairs

    def _get_param_qv_points(self, param: str) -> list[tuple[float, float]]:
        """Get (param_value, qv_ratio) points for a parameter across all designs."""
        points = []
        for entry in self.design_history:
            val = entry["params"].get(param)
            if val is None:
                continue
            qv = entry["result"].get("qv_ratio", 0)
            if qv > 0:
                points.append((float(val), float(qv)))
        # Sort by param value
        points.sort(key=lambda p: p[0])
        return points

    @staticmethod
    def _quadratic_peak(xs: list[float], ys: list[float]) -> float | None:
        """Fit y = ax² + bx + c and return x at peak (if a < 0).

        Uses simple least-squares without numpy dependency.
        """
        n = len(xs)
        if n < 3:
            return None

        # Build normal equations for y = a*x^2 + b*x + c
        sx = sum(xs)
        sx2 = sum(x ** 2 for x in xs)
        sx3 = sum(x ** 3 for x in xs)
        sx4 = sum(x ** 4 for x in xs)
        sy = sum(ys)
        sxy = sum(x * y for x, y in zip(xs, ys))
        sx2y = sum(x ** 2 * y for x, y in zip(xs, ys))

        # Solve 3x3 system using Cramer's rule
        # [sx4 sx3 sx2] [a]   [sx2y]
        # [sx3 sx2 sx ] [b] = [sxy ]
        # [sx2 sx  n  ] [c]   [sy  ]
        det = (sx4 * (sx2 * n - sx * sx)
               - sx3 * (sx3 * n - sx * sx2)
               + sx2 * (sx3 * sx - sx2 * sx2))

        if abs(det) < 1e-20:
            return None

        a = (sx2y * (sx2 * n - sx * sx)
             - sx3 * (sxy * n - sx * sy)
             + sx2 * (sxy * sx - sx2 * sy)) / det

        b = (sx4 * (sxy * n - sx * sy)
             - sx2y * (sx3 * n - sx * sx2)
             + sx2 * (sx3 * sy - sxy * sx2)) / det

        # Peak at x = -b / (2a), only if a < 0 (concave down)
        if a >= 0:
            return None

        peak_x = -b / (2 * a)

        # Only return if peak is within a reasonable extrapolation range
        x_min, x_max = min(xs), max(xs)
        x_range = x_max - x_min
        if x_range > 0 and (peak_x < x_min - x_range or peak_x > x_max + x_range):
            return None

        return peak_x
