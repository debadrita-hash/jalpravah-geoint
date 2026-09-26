"""
Continue a finished DualSPHysics run in 30-minute chunks (restart from its last snapshot) until the SPH
flood itself has run its course: the maximum flood extent grows by < 1 % over the last 30 simulated minutes
and the front has either left the domain or stopped advancing (< 0.3 km). Hard cap: sph.max_duration_s.

    python run_sph_extend.py [config]

After every chunk the results, comparison, viewer and archive are refreshed, so JalPravah always shows the
latest state. The stopping decision uses only the SPH run's own output.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from hydrosim.config import Scenario
from hydrosim import sph_model as S

CHUNK_S = 1800.0


def solver_running():
    r = S._run(["docker", "ps", "-q", "-f", "name=hydrosim_sph_solver"])
    return bool(r.stdout.strip())


def out_dirs(sc):
    r = S._run(["docker", "run", "--rm", "-v", f"{S.volume_name(sc)}:/work:ro", "alpine", "sh", "-c",
                "cd /work/case && ls -d out out_ext* 2>/dev/null"])
    ds = r.stdout.split()
    return sorted(ds, key=lambda d: 0 if d == "out" else int(d.replace("out_ext", "")))


def last_part(sc, d):
    r = S._run(["docker", "run", "--rm", "-v", f"{S.volume_name(sc)}:/work:ro", "alpine", "sh", "-c",
                f"ls /work/case/{d}/data | grep -E '^Part_[0-9]+\\.bi4$' | tail -1"])
    m = re.search(r"Part_(\d+)", r.stdout)
    return int(m.group(1)) if m else None


def analyse(sc, log):
    import pipeline
    from hydrosim import archive
    pipeline.analyze_model(sc, "sph")
    from hydrosim.results import ResultStore
    if (ResultStore(sc, "delft3d").dir / "metrics.json").exists():
        pipeline.step_compare(sc)
    pipeline.step_viewer(sc)
    archive.archive_solver_outputs(sc, "sph")
    archive.write_manifest(sc)
    fr = np.load(sc.results_dir / "sph" / "frames.npz")
    T = fr["times"].astype(float)
    wet = fr["depth"] >= sc["simulation"]["wet_threshold_m"]
    resv = np.load(sc.terrain_file)["cell_reservoir"].ravel()[np.nonzero(np.load(sc.terrain_file)["cell_domain"].ravel())[0]]
    extent = np.maximum.accumulate((wet & ~resv[None]).sum(1)) * sc.load_meta()["analysis_grid"]["dx"] ** 2 / 1e6
    front = np.asarray(json.loads((sc.results_dir / "sph" / "metrics.json").read_text())["timeseries"]["front_km"])
    k = int(np.searchsorted(T, T[-1] - CHUNK_S))
    grow = (extent[-1] - extent[k]) / max(extent[k], 1e-9)
    adv = float(front.max() - front[: k + 1].max())
    reach = sc.load_meta()["reach_length_m"] / 1000.0
    log(f"[extend] t={T[-1] / 60:.0f} min  max extent {extent[-1]:.2f} km2 (+{100 * grow:.1f} % in 30 min)  "
        f"front {front.max():.1f} km (+{adv:.2f} km)")
    return T[-1], grow < 0.01 and (front.max() >= reach - 1.0 or adv < 0.3)


def main():
    sc = Scenario(sys.argv[1] if len(sys.argv) > 1 else "config/trishuli_scenario.json")
    cap = float(sc["sph"].get("max_duration_s", 21600.0))
    logf = open(sc.sph_dir / "extend.log", "a", encoding="utf-8")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    while solver_running():
        time.sleep(60)
    case_dir = sc.sph_dir / "case"

    def set_duration(v):  # the pages show progress against this value
        c = json.loads(Path(sc.path).read_text())
        c["sph"]["duration_s"] = float(v)
        Path(sc.path).write_text(json.dumps(c, indent=2))
        sc.cfg["sph"]["duration_s"] = float(v)

    set_duration(cap)
    while True:
        t_end, done = analyse(sc, log)
        if done:
            log(f"[extend] SPH flood has run its course at {t_end / 60:.0f} min: stopping.")
            set_duration(t_end)
            import pipeline
            pipeline.step_viewer(sc)
            break
        if t_end >= cap - 1:
            log(f"[extend] reached the {cap / 3600:.1f} h cap.")
            set_duration(t_end)
            break
        dirs = out_dirs(sc)
        src = dirs[-1]
        part = last_part(sc, src)
        new = f"out_ext{len(dirs)}"
        tmax = min(cap, t_end + CHUNK_S)
        log(f"[extend] restarting from {src}/data Part_{part:04d} (t={t_end:.0f} s) to {tmax:.0f} s in {new}")
        cmd = (f"DualSPHysics5.4CPU_linux64 -cpu -ompthreads:{sc['sph']['threads']} out/Dambreak {new} -dirdataout data "
               f"-partbegin:{part} {src}/data -tmax:{tmax} -svres -cellmode:full")
        t0 = time.time()
        rc = S.in_volume(sc, cmd, case_dir / f"run_{new}.log", name="hydrosim_sph_solver", threads=sc["sph"]["threads"])
        log(f"[extend] chunk finished rc={rc} in {(time.time() - t0) / 3600:.2f} h")
        if rc != 0:
            log((case_dir / f"run_{new}.log").read_text(errors="ignore")[-1500:])
            break


if __name__ == "__main__":
    main()
