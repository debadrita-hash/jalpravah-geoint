"""
hydrosim pipeline: DEM -> scenario -> independent Delft3D-FLOW and DualSPHysics runs -> analysis -> viewer.

    python pipeline.py scenario            build terrain/dam/reservoir/breach/domain from the DEM
    python pipeline.py delft3d             build + run Delft3D-FLOW (Docker)
    python pipeline.py sph                 build + run DualSPHysics (Docker)
    python pipeline.py analyze [model..]   extract results, metrics, GIS (.tif/.shp/.kml), OSM exposure, comparison
    python pipeline.py viewer              write runs/<scenario>/viewer/*.html
    python pipeline.py all                 every step in order

Options: --config config/<scenario>.json (default: the Trishuli scenario)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from hydrosim.config import Scenario

HERE = Path(__file__).resolve().parent


def step_scenario(sc):
    from hydrosim.terrain import build_scenario
    build_scenario(sc)


def step_run(sc, model):
    script = HERE / ("run_delft3d.py" if model == "delft3d" else "run_sph.py")
    subprocess.run([sys.executable, str(script), str(sc.path)], check=True, cwd=HERE)


def write_impacts(sc, store, fields):
    """Exposure / damage from OpenStreetMap; if OSM cannot be reached the reason is recorded instead of numbers."""
    from hydrosim import exposure
    try:
        summary, hits = exposure.impacts(sc, store, fields)
        out = {"summary": summary, "assets": hits}
    except Exception as ex:
        out = {"summary": None, "assets": [], "error": str(ex)}
    (store.dir / "impacts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def analyze_model(sc, model):
    from hydrosim import delft3d_model, exposure, sph_model
    from hydrosim.results import ResultStore, compute_metrics, export_gis
    t0 = time.time()
    store = delft3d_model.extract(sc) if model == "delft3d" else sph_model.extract(sc)
    metrics, fields = compute_metrics(store)
    export_gis(store, fields)
    imp = write_impacts(sc, store, fields)
    print(f"[{model}] analysed in {time.time() - t0:.0f} s: flooded {metrics['flooded_area_km2']:.2f} km2, "
          f"breach peak {metrics['breach_peak_discharge_m3s']:.0f} m3/s, exposure {imp['summary'] or imp.get('error')}", flush=True)


def refresh_metrics(sc, model):
    """Recompute metrics, GIS and exposure from already-extracted frames (no solver output needed)."""
    from hydrosim import exposure
    from hydrosim.results import ResultStore, compute_metrics, export_gis
    store = ResultStore(sc, model)
    metrics, fields = compute_metrics(store)
    export_gis(store, fields)
    write_impacts(sc, store, fields)
    print(f"[{model}] metrics refreshed: max depth {metrics['max_depth_m']:.1f} m", flush=True)


def step_compare(sc):
    from hydrosim.compare import compare
    r = compare(sc)
    print("[compare] CSI %.3f, hit rate %.3f, false alarm %.3f" % (
        r["agreement"]["critical_success_index"], r["agreement"]["hit_rate"], r["agreement"]["false_alarm_ratio"]))


def step_viewer(sc):
    from hydrosim.viewer import build
    out = build(sc)
    print("[viewer]", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["scenario", "delft3d", "sph", "analyze", "compare", "viewer", "all"])
    ap.add_argument("models", nargs="*", default=["delft3d", "sph"])
    ap.add_argument("--config", default=str(HERE / "config" / "trishuli_scenario.json"))
    a = ap.parse_args()
    sc = Scenario(a.config)
    if a.step in ("scenario", "all"):
        step_scenario(sc)
    if a.step == "delft3d" or a.step == "all":
        step_run(sc, "delft3d")
    if a.step == "sph" or a.step == "all":
        step_run(sc, "sph")
    if a.step in ("analyze", "all"):
        for m in a.models:
            analyze_model(sc, m)
    if a.step in ("analyze", "compare", "all"):
        from hydrosim.results import ResultStore
        if all((ResultStore(sc, m).dir / "metrics.json").exists() for m in ("sph", "delft3d")):
            step_compare(sc)
    if a.step in ("analyze", "compare", "viewer", "all"):
        step_viewer(sc)


if __name__ == "__main__":
    main()
