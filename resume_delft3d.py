"""
Finish a Delft3D run whose driver process ended early (e.g. the laptop or the app was restarted):
waits for the scenario's Delft3D container to stop, then runs the same analysis as run_delft3d.py --analyze.

    python resume_delft3d.py config/<scenario>.json
"""
import os
import subprocess
import sys
import time

from hydrosim.config import Scenario

sc = Scenario(sys.argv[1])
env = dict(os.environ, MSYS_NO_PATHCONV="1")
name = f"hydrosim_d3d_solver_{sc.name}"
print(f"[resume] waiting for {name} to finish", flush=True)
while subprocess.run(["docker", "ps", "-q", "-f", f"name={name}"], capture_output=True, text=True, env=env).stdout.strip():
    time.sleep(30)
from hydrosim import delft3d_model as D  # noqa: E402

D.fetch(sc, f"/work/case/tri-diag.{D.RUNID}", sc.delft3d_dir / "case")
log = (sc.delft3d_dir / "case" / "run.log").read_text(errors="ignore")
if "shutting down normally" not in log[-3000:]:
    print("[resume] the solver did not finish normally; see run.log", flush=True)
    sys.exit(1)
import pipeline  # noqa: E402
from hydrosim import archive  # noqa: E402
from hydrosim.results import ResultStore  # noqa: E402

pipeline.analyze_model(sc, "delft3d")
if (ResultStore(sc, "sph").dir / "metrics.json").exists():
    pipeline.step_compare(sc)
pipeline.step_viewer(sc)
archive.archive_solver_outputs(sc, "delft3d")
print("[delft3d] archived, manifest", archive.write_manifest(sc)["total_gb"], "GB", flush=True)
