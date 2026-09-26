"""
Build and run the full Delft3D-FLOW simulation for a scenario (independent of SPH).

    python run_delft3d.py [config] [--cpus 10] [--analyze]

--cpus     CPU(s) the solver container may use (default: the last two logical CPUs)
--analyze  when the solver finishes: extract results, metrics, GIS, exposure, comparison (if SPH exists),
           viewer pages and the raw-output archive
"""
import argparse
import json
import time

from hydrosim.config import Scenario
from hydrosim import delft3d_model as D

ap = argparse.ArgumentParser()
ap.add_argument("config", nargs="?", default="config/trishuli_scenario.json")
ap.add_argument("--cpus", default=None)
ap.add_argument("--analyze", action="store_true")
a = ap.parse_args()
sc = Scenario(a.config)
meta = sc.load_meta()
case_dir, info = D.build_case(sc, meta)
print("[delft3d] case", json.dumps(info), flush=True)
t0 = time.time()
rc = D.run_solver(sc, case_dir, nproc=sc["delft3d"].get("mpi_processes_run", 1), cpus=a.cpus)
wall = time.time() - t0
(case_dir / "wallclock.json").write_text(json.dumps({"return_code": rc, "wall_clock_s": wall}))
print(f"[delft3d] solver finished rc={rc} in {wall / 60:.1f} min", flush=True)
if rc == 0 and a.analyze:
    import pipeline
    from hydrosim import archive
    from hydrosim.results import ResultStore
    pipeline.analyze_model(sc, "delft3d")
    if (ResultStore(sc, "sph").dir / "metrics.json").exists():
        pipeline.step_compare(sc)
    pipeline.step_viewer(sc)
    archive.archive_solver_outputs(sc, "delft3d")
    print("[delft3d] archived, manifest", archive.write_manifest(sc)["total_gb"], "GB", flush=True)
