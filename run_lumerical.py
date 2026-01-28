import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Lumerical API path
LUMPAPI_PATH = os.getenv("LUMPAPI_PATH")
if not LUMPAPI_PATH:
    raise ValueError("LUMPAPI_PATH not set in .env")

sys.path.append(LUMPAPI_PATH)
import lumapi

# Output folder for FDTD files
FDTD_OUTPUT_FOLDER = "fdtd_output"

# Default wavelength (SiV) - used if not specified in config
DEFAULT_WAVELENGTH = 737e-9  # 737 nm
DEFAULT_SPAN = 100e-9  # ±100 nm


def run_fdtd_simulation(config, mesh_accuracy=4, run=False):
    """
    Run Lumerical FDTD simulation for nanobeam cavity
    Goal: Find high Q-factor and small mode volume

    Args:
        config: dict from build_cavity_gds.get_config() + wavelength info
        mesh_accuracy: FDTD mesh accuracy (1-8)
        run: if True, run simulation; if False, just save project

    Returns:
        dict with simulation results
    """
    # Extract lumerical config
    lum_config = config["lumerical"]
    gds_file = lum_config["gds_file"]
    cell_name = lum_config["cell_name"]
    layer = lum_config["layer"]
    material = lum_config["material"]

    # Extract wavelength from config (set by agent from user input)
    wavelength_config = config.get("wavelength", {})
    design_wavelength = wavelength_config.get("design_wavelength", DEFAULT_WAVELENGTH)
    wavelength_span = wavelength_config.get("wavelength_span", DEFAULT_SPAN)

    # Extract substrate info
    substrate_config = config.get("substrate", {})
    freestanding = substrate_config.get("freestanding", True)
    substrate_material = substrate_config.get("material_lumerical", None)

    # Extract geometry for simulation region
    geometry = config["geometry"]
    cavity_length = geometry["cavity_length"]  # in microns
    first_hole_distance = geometry.get("first_hole_distance", 0.1)  # in microns
    first_hole_rx = geometry.get("first_hole_rx", 0.05)  # in microns
    first_hole_ry = geometry.get("first_hole_ry", 0.1)  # in microns

    unit_cell = config["unit_cell"]
    wg_width = unit_cell["wg_width"]  # in microns
    wg_height = unit_cell["wg_height"]  # in microns (from build_gds)

    # Create output folder
    os.makedirs(FDTD_OUTPUT_FOLDER, exist_ok=True)

    # Output file path
    output_name = Path(gds_file).stem
    fdtd_file = f"{FDTD_OUTPUT_FOLDER}/{output_name}.fsp"

    # Start Lumerical FDTD
    fdtd = lumapi.FDTD()
    fdtd.newproject()

    # Waveguide thickness in meters
    thickness = wg_height * 1e-6

    # Import GDS (cavity structure)
    # gdsimport(filename, cellname, layer, material, z_min, z_max)
    fdtd.gdsimport(
        gds_file, cell_name, layer[0], material, -thickness / 2, thickness / 2
    )

    # Create substrate if NOT freestanding
    if not freestanding and substrate_material:
        substrate_thickness = 2e-6  # 2um substrate thickness
        fdtd.addrect()
        fdtd.set("name", "substrate")
        fdtd.set("x", 0)
        fdtd.set("x span", (cavity_length + 4) * 1e-6)  # wider than cavity
        fdtd.set("y", 0)
        fdtd.set("y span", (wg_width * 6) * 1e-6)  # wider than waveguide
        fdtd.set("z min", -thickness / 2 - substrate_thickness)  # below waveguide
        fdtd.set("z max", -thickness / 2)  # top at waveguide bottom
        fdtd.set("material", substrate_material)
        print(f"Added substrate: {substrate_material}")

    # FDTD simulation region
    fdtd.addfdtd()
    fdtd.set("x", 0)
    fdtd.set("x span", (cavity_length + 2) * 1e-6)  # cavity length + margin
    fdtd.set("y", 0)
    fdtd.set("y span", (wg_width * 4) * 1e-6)  # 4x waveguide width

    # Z span depends on freestanding or with substrate
    if freestanding:
        fdtd.set("z", 0)
        fdtd.set("z span", thickness * 4)
    else:
        # With substrate: extend z span to include substrate
        substrate_thickness = 2e-6
        z_min = -thickness / 2 - substrate_thickness - 0.5e-6  # substrate + margin
        z_max = thickness / 2 + 1e-6  # above waveguide
        fdtd.set("z min", z_min)
        fdtd.set("z max", z_max)

    fdtd.set("mesh accuracy", mesh_accuracy)
    # simulation time uses default value

    # Boundary conditions for nanobeam cavity symmetry
    # x min: Symmetric (cavity is symmetric along x-axis)
    # x max: PML
    # y min: Anti-Symmetric (for TE-like mode)
    # y max: PML
    # z min/max: PML
    fdtd.set("x min bc", "Symmetric")
    fdtd.set("x max bc", "PML")
    fdtd.set("y min bc", "Anti-Symmetric")
    fdtd.set("y max bc", "PML")
    fdtd.set("z min bc", "PML")
    fdtd.set("z max bc", "PML")

    # Magnetic dipole source at cavity center
    # Convert wavelength to frequency (THz)
    c = 299792458  # speed of light in m/s
    wavelength_min = design_wavelength - wavelength_span
    wavelength_max = design_wavelength + wavelength_span
    freq_min_THz = c / wavelength_max / 1e12  # longer wavelength = lower frequency
    freq_max_THz = c / wavelength_min / 1e12  # shorter wavelength = higher frequency

    fdtd.adddipole()
    fdtd.set("name", "magnetic_dipole")
    fdtd.set("dipole type", 2)  # 2 = magnetic dipole
    fdtd.set("x", 0)
    fdtd.set("y", 0)
    fdtd.set("z", 0)
    fdtd.set("wavelength start", design_wavelength - wavelength_span)
    fdtd.set("wavelength stop", design_wavelength + wavelength_span)

    # Q factor analysis
    fdtd.addobject("Qanalysis")
    fdtd.set("name", "Q_analysis")
    fdtd.set("x span", first_hole_distance / 3 * 1e-6)
    fdtd.set("x", first_hole_distance / 3 * 1e-6 / 2)
    fdtd.set("y span", wg_width / 8 * 1e-6)
    fdtd.set("y", wg_width / 8 * 1e-6 / 2)
    fdtd.set("z span", wg_height * 1e-6)
    fdtd.set("z", wg_height * 1e-6 / 2)
    fdtd.set("nx", 3)
    fdtd.set("ny", 3)
    fdtd.set("nz", 3)
    # fdtd.set("make plot", 1)
    fdtd.set("f min", freq_min_THz)
    fdtd.set("f max", freq_max_THz)

    # Mode volume analysis
    fdtd.addobject("mode_volume")
    fdtd.set("name", "mode_volume")

    # Save project
    fdtd.save(fdtd_file)
    print(f"Saved: {fdtd_file}")

    result = {
        "status": "success",
        "fdtd_file": fdtd_file,
        "gds_file": gds_file,
    }

    # Run simulation if requested
    if run:
        print("Running simulation...")
        fdtd.run()

        # Extract results
        # t_data = fdtd.getresult('time_monitor', 't')
        # E_data = fdtd.getresult('time_monitor', 'E')

        result["simulation_completed"] = True
        # TODO: Add Q factor and mode volume extraction

    fdtd.close()
    return result


if __name__ == "__main__":
    # Example: run with build_gds config
    from build_gds import build_cavity_gds

    # Build cavity and get config (GDS is 2D, no wg_height)
    cavity = build_cavity_gds(
        period=0.2,  # 200nm
        hole_rx=0.05,  # 50nm
        hole_ry=0.1,  # 100nm
        wg_width=0.45,  # 450nm
        num_taper_holes=8,
        num_mirror_holes=10,
        taper_type="quadratic",
        min_a_percent=90,
        save=True,
    )
    config = cavity.get_config()

    # Add parameters that agent normally provides (not from build_gds)
    config["unit_cell"]["wg_height"] = 0.2  # 200nm in microns (for 3D simulation)
    config["wavelength"] = {
        "design_wavelength": 737e-9,  # 737nm for SiV
        "wavelength_span": 100e-9,  # ±100nm
    }
    config["substrate"] = {
        "freestanding": True,
        "material": "none",
        "material_lumerical": None,
    }

    # Run FDTD simulation
    result = run_fdtd_simulation(config, run=False)
    print(result)
