"""Build and run the full DualSPHysics simulation for a scenario (independent of Delft3D)."""
import json
import re
import sys
import time

from hydrosim.config import Scenario
from hydrosim import sph_model as S

sc = Scenario(sys.argv[1] if len(sys.argv) > 1 else "config/trishuli_scenario.json")
cfg = sc["sph"]
meta = sc.load_meta()
case_dir, info = S.build_case(sc, meta, tmax=cfg["duration_s"], tout=cfg["output_interval_s"], dp=cfg["dp_m"])
rc, log = S.gencase(sc, case_dir)
info["gencase_rc"] = rc
info.update({k: int(v.replace(",", "")) for k, v in re.findall(r"(Fixed|Moving|Fluid)\.+: ([\d,]+)", log)})
print("[sph] case", json.dumps(info), flush=True)
t0 = time.time()
rc = S.run_solver(sc, case_dir, threads=cfg["threads"])
wall = time.time() - t0
(case_dir / "wallclock.json").write_text(json.dumps({"return_code": rc, "wall_clock_s": wall, **info}))
print(f"[sph] solver finished rc={rc} in {wall / 3600:.2f} h", flush=True)
