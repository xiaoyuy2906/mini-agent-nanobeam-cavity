import os
import numpy as np
import gdsfactory as gf

# Activate PDK (required for gdsfactory components)
gf.config.rich_output = False
gf.gpdk.PDK.activate()

# Default output folder for GDS files
GDS_OUTPUT_FOLDER = "gds_output"

# GDS cell name
CELL_NAME = "cavity_design"

# Material for Lumerical
MATERIAL = "Si3N4 (Silicon Nitride) - Phillip"


def normalize_percent(value, threshold=1.5):
    """
    Normalize percent values - accepts both ratio (0.7) and percent (70) formats.
    If value <= threshold, treat as ratio and convert to percent (multiply by 100).
    If value > threshold, treat as percent directly.
    """
    value = float(value)
    if value <= threshold:
        return value * 100.0
    return value


class build_cavity_gds:

    def __init__(
        self,
        # Unit cell parameters (in microns)
        period=0.2,
        hole_rx=0.05,
        hole_ry=0.1,
        wg_width=0.45,
        # Hole counts
        num_taper_holes=8,
        num_mirror_holes=12,
        # Taper parameters
        taper_type="quadratic",
        min_a_percent=92,
        min_rx_percent=100,
        min_ry_percent=100,
        # GDS layer
        layer=(1, 0),
        # Auto-save: True = save with auto-generated name, or provide custom filename
        save=False,
    ) -> None:
        self.period = float(period)
        self.hole_rx = float(hole_rx)
        self.hole_ry = float(hole_ry)
        self.wg_width = float(wg_width)

        self.num_taper_holes = int(num_taper_holes)
        self.num_mirror_holes = int(num_mirror_holes)

        self.taper_type = taper_type
        self.min_a_percent = normalize_percent(min_a_percent)
        self.min_rx_percent = normalize_percent(min_rx_percent)
        self.min_ry_percent = normalize_percent(min_ry_percent)

        self.layer = layer
        self.gds_filepath = None
        self.cell_name = None

        self._precompute_cavity_params()
        self._precompute_geometry()

        # Auto-save if requested
        if save:
            self.gds_filepath = self.save_gds()

    def _calculate_taper_scale(self, i, n, min_percent):
        """Calculate taper scale factor based on taper_type"""
        if n <= 1:
            return min_percent / 100.0

        t = i / (n - 1)  # normalized position 0 to 1
        min_scale = min_percent / 100.0

        if self.taper_type == "linear":
            scale = min_scale + (1.0 - min_scale) * t
        elif self.taper_type == "quadratic":
            scale = min_scale + (1.0 - min_scale) * (t**2)
        elif self.taper_type == "cubic":
            scale = min_scale + (1.0 - min_scale) * (3 * t**2 - 2 * t**3)
        else:
            scale = min_scale + (1.0 - min_scale) * (t**2)  # default quadratic

        return scale

    def _precompute_cavity_params(self):
        # Calculate period for each taper hole
        self.a_list = np.array(
            [
                self.period
                * self._calculate_taper_scale(
                    idx, self.num_taper_holes, self.min_a_percent
                )
                for idx in range(self.num_taper_holes)
            ]
        )
        self.a_cumsum = np.cumsum(self.a_list)
        self.a_total = float(self.a_cumsum[-1]) if len(self.a_cumsum) > 0 else 0.0

        # Total cavity length
        self.cav_len = 2.0 * self.a_total + 2.0 * self.num_mirror_holes * self.period

    def _precompute_geometry(self):
        """Pre-compute hole positions and sizes"""
        # Clear gdsfactory cache to avoid name conflicts between iterations
        gf.clear_cache()

        period = self.period
        rx = self.hole_rx
        ry = self.hole_ry

        # ---------- 1. Taper region ----------
        a_list = self.a_list
        a_cumsum = self.a_cumsum

        # Taper hole centers
        taper_circle_x_r = a_cumsum - a_list / 2.0
        taper_circle_x_l = -a_cumsum + a_list / 2.0

        # Taper segment origins
        taper_origin_x_r = a_cumsum - a_list

        # Taper hole radii
        taper_rx = np.array(
            [
                rx
                * self._calculate_taper_scale(
                    idx, self.num_taper_holes, self.min_rx_percent
                )
                for idx in range(self.num_taper_holes)
            ]
        )
        taper_ry = np.array(
            [
                ry
                * self._calculate_taper_scale(
                    idx, self.num_taper_holes, self.min_ry_percent
                )
                for idx in range(self.num_taper_holes)
            ]
        )

        # ---------- 2. Mirror region ----------
        if self.num_mirror_holes > 0:
            j = np.arange(self.num_mirror_holes, dtype=float)
        else:
            j = np.array([], dtype=float)

        taper_hole_end_r = taper_circle_x_r[-1] if len(taper_circle_x_r) > 0 else 0.0
        taper_hole_end_l = taper_circle_x_l[-1] if len(taper_circle_x_l) > 0 else 0.0

        mirror_circle_x_r = taper_hole_end_r + (j + 1.0) * period
        mirror_circle_x_l = taper_hole_end_l - (j + 1.0) * period

        mirror_origin_x_r = self.a_total + j * period

        mirror_rx = np.full_like(j, rx, dtype=float)
        mirror_ry = np.full_like(j, ry, dtype=float)

        # ---------- 3. Assemble templates ----------
        right_origin = np.concatenate([taper_origin_x_r, mirror_origin_x_r])

        right_circle_x = np.concatenate([taper_circle_x_r, mirror_circle_x_r])
        left_circle_x = np.concatenate(
            [mirror_circle_x_l[::-1], taper_circle_x_l[::-1]]
        )

        right_rx = np.concatenate([taper_rx, mirror_rx])
        right_ry = np.concatenate([taper_ry, mirror_ry])

        left_rx = np.concatenate([mirror_rx[::-1], taper_rx[::-1]])
        left_ry = np.concatenate([mirror_ry[::-1], taper_ry[::-1]])

        # Combine all holes
        self.hole_x = np.concatenate([left_circle_x, right_circle_x])
        self.hole_rx_list = np.concatenate([left_rx, right_rx])
        self.hole_ry_list = np.concatenate([left_ry, right_ry])

        self.segment_origin_x = np.concatenate([-right_origin[::-1], right_origin])

        # ---------- 4. Build GDS geometry ----------
        wg_length = (
            (float(right_origin[-1]) + self.period) * 2.0 + 4.0
            if len(right_origin) > 0
            else 4.0
        )

        wg_component = gf.Component()
        xs = gf.cross_section.cross_section(width=self.wg_width, layer=self.layer)
        straight = gf.components.straight(length=wg_length, cross_section=xs)
        straight_ref = wg_component.add_ref(straight)
        straight_ref.center = (0.0, 0.0)

        holes_component = gf.Component()
        hole_count = self.hole_x.size

        hole_added = False
        for i in range(hole_count):
            hx = self.hole_x[i]
            hrx = self.hole_rx_list[i]
            hry = self.hole_ry_list[i]

            if hrx <= 0.001 or hry <= 0.001:
                continue

            ellipse = gf.components.ellipse(radii=(hrx, hry), layer=self.layer)
            h_ref = holes_component.add_ref(ellipse)
            h_ref.center = (hx, 0.0)
            hole_added = True

        if hole_added:
            self.cavity_template = gf.boolean(
                A=wg_component, B=holes_component, operation="A-B", layer=self.layer
            )
        else:
            self.cavity_template = wg_component

    def get_taper_equation(self):
        """Return taper equation as string"""
        equations = {
            "linear": "scale = min/100 + (1 - min/100) * (i / (n-1))",
            "quadratic": "scale = min/100 + (1 - min/100) * (i / (n-1))^2",
            "cubic": "scale = min/100 + (1 - min/100) * (3t^2 - 2t^3), t = i/(n-1)",
        }
        return equations.get(self.taper_type, "quadratic")

    def get_config(self):
        """Return cavity config as dict for LLM and run_lumerical"""
        config = {
            "unit_cell": {
                "period": self.period,
                "hole_rx": self.hole_rx,
                "hole_ry": self.hole_ry,
                "wg_width": self.wg_width,
            },
            "taper_params": {
                "num_taper_holes": self.num_taper_holes,
                "num_mirror_holes": self.num_mirror_holes,
                "taper_type": self.taper_type,
                "min_a_percent": self.min_a_percent,
                "min_rx_percent": self.min_rx_percent,
                "min_ry_percent": self.min_ry_percent,
            },
            "taper_equation": self.get_taper_equation(),
            "geometry": {
                "total_holes": int(self.hole_x.size),
                "cavity_length": float(self.cav_len),
                "period_values": self.a_list.tolist(),
                # Distance from cavity center (x=0) to first hole center
                # First hole is at a_list[0]/2 (half of first period)
                "first_hole_distance": (
                    float(self.a_list[0] / 2.0) if len(self.a_list) > 0 else 0.0
                ),
                # First hole radii (for mode volume region)
                "first_hole_rx": float(
                    self.hole_rx
                    * self._calculate_taper_scale(
                        0, self.num_taper_holes, self.min_rx_percent
                    )
                ),
                "first_hole_ry": float(
                    self.hole_ry
                    * self._calculate_taper_scale(
                        0, self.num_taper_holes, self.min_ry_percent
                    )
                ),
            },
            # For run_lumerical
            "lumerical": {
                "gds_file": self.gds_filepath,
                "cell_name": self.cell_name,
                "layer": self.layer,
                "material": MATERIAL,
            },
        }
        return config

    def _generate_filename(self):
        """Generate filename based on key parameters"""
        # Convert to nm for filename (values are in microns)
        period_nm = int(self.period * 1000)
        wg_width_nm = int(self.wg_width * 1000)
        hole_rx_nm = int(self.hole_rx * 1000)
        hole_ry_nm = int(self.hole_ry * 1000)

        name = (
            f"cavity_"
            f"p{period_nm}_"
            f"w{wg_width_nm}_"
            f"rx{hole_rx_nm}_"
            f"ry{hole_ry_nm}_"
            f"t{self.num_taper_holes}_"
            f"m{self.num_mirror_holes}_"
            f"a{int(self.min_a_percent)}"
        )
        # Only add taper rx/ry if not 100%
        if int(self.min_rx_percent) != 100:
            name += f"_trx{int(self.min_rx_percent)}"
        if int(self.min_ry_percent) != 100:
            name += f"_try{int(self.min_ry_percent)}"
        name += ".gds"
        return name

    def save_gds(self, filename=None, folder=None):
        """Save cavity to GDS file, returns relative path"""
        # Use default folder if not specified
        if folder is None:
            folder = GDS_OUTPUT_FOLDER

        # Create folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Generate filename if not specified
        if filename is None:
            filename = self._generate_filename()

        # Relative path (use forward slashes for cross-platform)
        rel_path = f"{folder}/{filename}"

        # Use unique cell name from filename (without .gds)
        cell_name = filename.replace(".gds", "")

        # Save GDS with unique cell name
        c = gf.Component(cell_name)
        c.add_ref(self.cavity_template)
        c.write_gds(rel_path)

        self.gds_filepath = rel_path
        self.cell_name = cell_name
        return rel_path
