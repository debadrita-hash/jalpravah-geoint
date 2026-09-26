"""
Scenario builder: turns user inputs (location, dam, data sources, model settings) into a scenario config,
fetches open data when requested and builds the scenario geometry from the DEM.

    python -m hydrosim.builder <params.json>
"""
import json
import re
import shutil
import sys
from pathlib import Path

from .config import HYDROSIM_ROOT, Scenario
from . import opendata

CONFIG_DIR = HYDROSIM_ROOT / "config"

DAM_TYPES = {
    # constructed embankment / concrete dam: narrow crest, steep faces
    "dam": {"crest_width_m": 20.0, "upstream_slope_h_per_v": 2.0, "downstream_slope_h_per_v": 2.0, "failure_mode": "overtopping",
            "label": "Constructed dam"},
    # landslide / debris dam blocking the river: wide crest, flat faces (typical of Himalayan blockages)
    "blockage": {"crest_width_m": 150.0, "upstream_slope_h_per_v": 3.0, "downstream_slope_h_per_v": 4.0, "failure_mode": "overtopping",
                 "label": "Landslide dam / river blockage"},
}


def slug(name):
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s[:48] or "scenario"


def make_config(p):
    name = slug(p["name"])
    lat, lon = float(p["lat"]), float(p["lon"])
    kind = DAM_TYPES[p.get("dam_type", "dam")]
    reach = float(p.get("reach_km", 30))
    run_dir = HYDROSIM_ROOT / "runs" / name
    inputs = {}
    return name, {
        "name": name,
        "title": p["name"].strip(),
        "description": p.get("description") or f"{kind['label']} failure at {p['name']} "
                                               f"({lat:.4f} N, {lon:.4f} E), {float(p['height_m']):g} m high, routed {reach:g} km downstream.",
        "inputs": inputs,
        "crs": opendata.utm_crs(lon, lat),
        "dam": {"lon": lon, "lat": lat, "height_m": float(p["height_m"]), "crest_width_m": kind["crest_width_m"],
                "upstream_slope_h_per_v": kind["upstream_slope_h_per_v"], "downstream_slope_h_per_v": kind["downstream_slope_h_per_v"],
                "siting_search_m": float(p.get("siting_search_m", 1500)), "type": p.get("dam_type", "dam"), "label": p["name"]},
        "breach": {"method": "froehlich2008", "failure_mode": kind["failure_mode"], "side_slope_h_per_v": 1.0, "timing": "instantaneous"},
        "domain": {"reach_length_km": reach, "upstream_buffer_km": 1.0, "margin_above_thalweg_m": float(p.get("margin_m", 45)),
                   "max_lateral_distance_m": 1500.0, "terrain_resolution_m": 10.0},
        "simulation": {"duration_s": float(p.get("duration_h", 3)) * 3600.0, "output_interval_s": 30.0, "wet_threshold_m": 0.1,
                       "arrival_threshold_m": 0.5,
                       "stations_km": [k for k in (0.5, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80) if k < reach - 1]},
        "delft3d": {"cell_size_m": float(p.get("cell_m", 30)), "time_step_s": 0.75 if float(p.get("cell_m", 30)) >= 30 else 0.6,
                    "manning_n": float(p.get("manning_n", 0.045)), "horizontal_eddy_viscosity": 1.0, "dry_threshold_m": 0.05,
                    "advection_scheme": "Flood"},
        "sph": {"dp_m": float(p.get("sph_dp_m", 12)), "coefh": 1.0, "coefsound": 10.0, "cfl": 0.2, "kernel": "wendland",
                "viscosity_artificial": 0.01, "visco_bound_factor": 1.0, "density_diffusion": "fourtakas", "density_diffusion_value": 0.1,
                "boundary_layers": 2, "gate_lift_speed_ms": 40.0, "threads": 10, "duration_s": 5400.0, "output_interval_s": 30.0,
                "max_duration_s": float(p.get("duration_h", 3)) * 3600.0 * 2},
    }, run_dir


def build(p, log=print):
    name, cfg, run_dir = make_config(p)
    inp = run_dir / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    lat, lon = cfg["dam"]["lat"], cfg["dam"]["lon"]
    r_km = cfg["domain"]["reach_length_km"] + 6.0
    dlat, dlon = r_km / 111.0, r_km / (111.0 * max(0.2, abs(__import__("math").cos(__import__("math").radians(lat)))))
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    prov = {}
    if p.get("dem_source", "copernicus") == "copernicus":
        prov["dem"] = opendata.download_dem(bbox, inp / "dem.tif", log)
        cfg["inputs"]["dem"] = str(inp / "dem.tif")
    else:
        src = Path(p["dem_path"])
        if not src.exists():
            raise FileNotFoundError(f"DEM file not found: {src}")
        cfg["inputs"]["dem"] = str(src)
        prov["dem"] = {"source": f"user file {src}"}
    img = p.get("imagery_source", "sentinel2")
    if img == "sentinel2":
        try:
            prov["imagery"] = opendata.download_sentinel2(bbox, inp / "sentinel2_rgb.tif", log=log)
            cfg["inputs"]["satellite"] = str(inp / "sentinel2_rgb.tif")
        except Exception as ex:
            log(f"[s2] imagery not available ({ex}); maps will use shaded relief")
    elif img == "file" and p.get("imagery_path"):
        cfg["inputs"]["satellite"] = str(Path(p["imagery_path"]))
    cfg["provenance"] = prov
    cfg_path = CONFIG_DIR / f"{name}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    log(f"[builder] config written: {cfg_path}")
    from .terrain import build_scenario
    sc = Scenario(cfg_path)
    meta = build_scenario(sc, log=log)
    from .viewer import build as build_viewer
    build_viewer(sc)
    summary = {"scenario": name, "config": str(cfg_path), "reservoir_Mm3": round(meta["reservoir"]["volume_m3"] / 1e6, 3),
               "crest_m": round(meta["dam"]["crest_m"], 1), "breach_bottom_m": round(meta["breach"]["bottom_width_m"], 1),
               "breach_top_m": round(meta["breach"]["top_width_m"], 1), "domain_km2": round(meta["domain_area_m2"] / 1e6, 1)}
    (run_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    log("[builder] done " + json.dumps(summary))
    return summary


if __name__ == "__main__":
    params = json.loads(Path(sys.argv[1]).read_text())
    build(params)
