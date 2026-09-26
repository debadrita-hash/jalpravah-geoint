"""
DualSPHysics (v5.4, CPU/OpenMP) case generation, execution and particle post-processing.

The case is built only from terrain.npz / scenario_meta.json:
  * terrain + dam + open breach notch -> triangulated STL of the valley corridor, drawn as DBC
    boundary particles (several layers) by GenCase
  * reservoir -> GenCase void-fill below the crest level, seeded upstream of the dam
  * instantaneous breach -> a gate plate across the notch (moving boundary, mk=11) lifted out
    of the water at t=0 (the classical dam-break gate removal)
  * particles crossing x = x_end (downstream end of the reach) leave the domain = free outfall

Particle output (Part_XXXX.bi4) is converted with the official PartVTK tool to CSV and binned onto
the 20 m analysis grid (depth = particle volume per cell area), so SPH and Delft3D maps are compared
cell by cell.
"""
import json
import re
import math
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

from .terrain import load_terrain

IMAGE = "hydrosim/dualsphysics:5.4-run"
MK_TERRAIN, MK_GATE = 0, 11


# ----------------------------------------------------------------------------- geometry
def _write_stl(path, tris):
    tris = np.asarray(tris, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(b"hydrosim terrain".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        v1, v2, v3 = tris[:, 0], tris[:, 1], tris[:, 2]
        n = np.cross(v2 - v1, v3 - v1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, tris
        f.write(rec.tobytes())


def terrain_stl(t, path, dilate=3):
    """Triangulate the terrain nodes inside the (dilated) domain."""
    from scipy import ndimage
    xs, ys, z = t["xs"], t["ys"], t["z"]
    m = ndimage.binary_dilation(t["domain"], iterations=dilate)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y, z], -1)
    q = m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]
    j, i = np.nonzero(q)
    a, b, c, d = P[j, i], P[j, i + 1], P[j + 1, i + 1], P[j + 1, i]
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    _write_stl(path, tris)
    return len(tris), float(z[m].min()), float(z[m].max())


def box_stl(path, center, u, v, w):
    """Closed box with half-axes u, v, w (3-vectors) around center."""
    c = np.asarray(center, float)
    corners = [c + sx * np.asarray(u) + sy * np.asarray(v) + sz * np.asarray(w)
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    C = np.array(corners)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, cc, d in faces:
        tris += [[C[a], C[b], C[cc]], [C[a], C[cc], C[d]]]
    _write_stl(path, tris)


# ----------------------------------------------------------------------------- case xml
def build_case(sc, meta, tmax=None, tout=None, dp=None, name="Dambreak"):
    cfg = sc["sph"]
    dp = float(dp or cfg["dp_m"])
    tmax = float(tmax or sc["simulation"]["duration_s"])
    tout = float(tout or cfg.get("output_interval_s", 10.0))
    case_dir = sc.sph_dir / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    t = load_terrain(sc)
    ntri, zmin, zmax = terrain_stl(t, case_dir / "terrain.stl")

    dam, br = meta["dam"], meta["breach"]
    tv = np.array(dam["axis_tangent"] + [0.0])
    nv = np.array(dam["axis_normal"] + [0.0])
    wl = meta["reservoir"]["water_level_m"]
    zc, zb = dam["crest_m"], dam["bed_m"]
    # gate across the notch on the dam axis (d = 0): 2 dp thick, spans the notch top width + 30 m
    half_w = 0.5 * br["top_width_m"] + 30.0
    zlo, zhi = zb - 3 * dp, zc + 3 * dp
    box_stl(case_dir / "gate.stl", [0, 0, 0.5 * (zlo + zhi)], tv * dp, nv * half_w, [0, 0, 0.5 * (zhi - zlo)])
    lift_v = float(cfg["gate_lift_speed_ms"])
    lift_t = (zhi - zlo + 20.0) / lift_v

    # reservoir fill box: reservoir bounding box, top just below the water level
    res = t["reservoir"]
    X, Y = np.meshgrid(t["xs"], t["ys"])
    rx, ry = X[res], Y[res]
    seed_d = -(0.5 * dam["crest_width_m"] + 0.5 * (zc - zb) * dam["upstream_slope"])  # mid upstream face
    seed = tv[:2] * (seed_d - 30.0)
    j = np.argmin(np.abs(t["ys"] - seed[1]))
    i = np.argmin(np.abs(t["xs"] - seed[0]))
    seed_z = t["z"][j, i] + 2.0 * dp
    layers = int(cfg["boundary_layers"])
    stl_cmds = "\n".join(
        f'                    <drawfilestl file="terrain.stl"><drawmove x="0" y="0" z="{-(k + 0.5) * dp:.4f}" /></drawfilestl>'
        for k in range(layers))  # DBC wall offset: effective fluid boundary ~dp/2 above the particles = DEM surface
    pad = 3 * dp
    x_end = meta["x_end"]
    # align the particle lattice so that a layer sits at WL - dp/2: the fluid cubes then fill exactly to WL
    z_top_node = wl - 0.5 * dp
    z_def_min = z_top_node - math.ceil((z_top_node - (zmin - (layers + 3) * dp)) / dp) * dp
    box_top = wl + 0.05  # GenCase keeps a node only if its whole dp-cube lies inside the box

    visco = cfg["viscosity_artificial"]
    ddt = {"none": 0, "molteni": 1, "fourtakas": 2, "fourtakas_full": 3}[cfg["density_diffusion"]]
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- Generated by hydrosim from {sc.name}: DEM terrain, reservoir {meta['reservoir']['volume_m3'] / 1e6:.2f} Mm3,
     instantaneous Froehlich breach (bottom {br['bottom_width_m']:.1f} m, side slope {br['side_slope_h_per_v']}:1) -->
<case>
    <casedef>
        <constantsdef>
            <gravity x="0" y="0" z="-9.81" />
            <rhop0 value="1000" />
            <rhopgradient value="2" />
            <hswl value="0" auto="true" />
            <gamma value="7" />
            <speedsystem value="0" auto="true" />
            <coefsound value="{cfg['coefsound']}" />
            <speedsound value="0" auto="true" />
            <coefh value="{cfg['coefh']}" />
            <cflnumber value="{cfg['cfl']}" />
        </constantsdef>
        <mkconfig boundcount="240" fluidcount="9" />
        <geometry>
            <definition dp="{dp}">
                <pointmin x="{t['xs'][0] - pad:.3f}" y="{t['ys'][0] - pad:.3f}" z="{z_def_min:.4f}" />
                <pointmax x="{x_end + pad:.3f}" y="{t['ys'][-1] + pad:.3f}" z="{max(zmax, zhi) + pad:.3f}" />
            </definition>
            <commands>
                <mainlist>
                    <setshapemode>actual | bound</setshapemode>
                    <setdrawmode mode="face" />
                    <setmkbound mk="{MK_TERRAIN}" />
{stl_cmds}
                    <setmkbound mk="{MK_GATE}" />
                    <setdrawmode mode="full" />
                    <drawfilestl file="gate.stl" />
                    <setmkfluid mk="0" />
                    <fillbox x="{seed[0]:.3f}" y="{seed[1]:.3f}" z="{seed_z:.3f}">
                        <modefill>void</modefill>
                        <point x="{rx.min() - 50:.3f}" y="{ry.min() - 50:.3f}" z="{zmin - dp:.3f}" />
                        <size x="{rx.max() - rx.min() + 100 + dp:.3f}" y="{ry.max() - ry.min() + 100:.3f}" z="{box_top - (zmin - dp):.3f}" />
                    </fillbox>
                </mainlist>
            </commands>
        </geometry>
        <motion>
            <objreal ref="{MK_GATE}">
                <begin mov="1" start="0" finish="{lift_t:.3f}" />
                <mvrect id="1" duration="{lift_t:.3f}">
                    <vel x="0" y="0" z="{lift_v}" />
                </mvrect>
            </objreal>
        </motion>
    </casedef>
    <execution>
        <parameters>
            <parameter key="SavePosDouble" value="0" />
            <parameter key="StepAlgorithm" value="2" comment="Symplectic" />
            <parameter key="Kernel" value="2" comment="Wendland" />
            <parameter key="ViscoTreatment" value="1" comment="Artificial" />
            <parameter key="Visco" value="{visco}" />
            <parameter key="ViscoBoundFactor" value="{cfg['visco_bound_factor']}" />
            <parameter key="DensityDT" value="{ddt}" />
            <parameter key="DensityDTvalue" value="{cfg['density_diffusion_value']}" />
            <parameter key="Shifting" value="0" />
            <parameter key="RigidAlgorithm" value="1" />
            <parameter key="CoefDtMin" value="0.05" />
            <parameter key="DtIni" value="0" />
            <parameter key="DtMin" value="0" />
            <parameter key="DtAllParticles" value="0" />
            <parameter key="TimeMax" value="{tmax}" />
            <parameter key="TimeOut" value="{tout}" />
            <parameter key="MinFluidStop" value="0" />
            <parameter key="RhopOutMin" value="700" />
            <parameter key="RhopOutMax" value="1300" />
            <simulationdomain>
                <posmin x="default" y="default" z="default" />
                <posmax x="default" y="default" z="default + 250" />
            </simulationdomain>
        </parameters>
    </execution>
</case>
"""
    (case_dir / f"{name}_Def.xml").write_text(xml, encoding="utf-8")
    info = {"dp": dp, "tmax": tmax, "tout": tout, "terrain_triangles": ntri, "gate_lift_time_s": lift_t,
            "seed": [float(seed[0]), float(seed[1]), float(seed_z)], "layers": layers}
    (case_dir / "case_info.json").write_text(json.dumps(info, indent=2))
    return case_dir, info


# ----------------------------------------------------------------------------- docker
# All heavy I/O happens inside a named Docker volume (stored in Docker's disk image on D:);
# only inputs are staged in and compact results/logs are copied back to the run directory.
PKG_DIR = Path(__file__).resolve().parent


def volume_name(sc):
    return f"hydrosim_{sc.name}_sph"


def _run(args, log_file=None):
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    if log_file is None:
        return subprocess.run(args, env=env, capture_output=True, text=True)
    with open(log_file, "w", encoding="utf-8") as lf:
        return subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, env=env)


def _host(p):
    return str(Path(p).resolve()).replace("\\", "/")


def stage(sc, case_dir):
    vol = volume_name(sc)
    _run(["docker", "volume", "create", vol])
    return _run(["docker", "run", "--rm", "-v", f"{_host(case_dir)}:/src:ro", "-v", f"{vol}:/work", IMAGE,
                 "bash", "-c", "mkdir -p /work/case && cp -f /src/*.xml /src/*.stl /src/*.json /work/case/"])


def in_volume(sc, cmd, log_file, name=None, threads=None):
    """Runs a shell command in /work/case of the scenario volume; hydrosim package mounted read-only.
    OpenMP threads are capped (default: logical CPUs - 2) so a busy core never stalls every barrier."""
    threads = threads or max(1, (os.cpu_count() or 4) - 2)
    args = ["docker", "run", "--rm", "--name", name or f"hydrosim_sph_{int(time.time())}", "-e", f"OMP_NUM_THREADS={threads}", f"--cpuset-cpus=0-{threads - 1}",
            "-v", f"{volume_name(sc)}:/work", "-v", f"{_host(PKG_DIR)}:/hydrosim:ro",
            "-v", f"{_host(sc.scenario_dir)}:/scenario:ro", "-w", "/work/case", IMAGE, "bash", "-c", cmd]
    return _run(args, log_file).returncode


def fetch(sc, src_in_volume, dst_dir):
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    return _run(["docker", "run", "--rm", "-v", f"{volume_name(sc)}:/work:ro", "-v", f"{_host(dst_dir)}:/dst",
                 IMAGE, "bash", "-c", f"cp -rf {src_in_volume} /dst/"])


def gencase(sc, case_dir, name="Dambreak"):
    stage(sc, case_dir)
    log = Path(case_dir) / "gencase.log"
    rc = in_volume(sc, f"rm -rf out && GenCase_linux64 {name}_Def out/{name} -save:-vtkcells,-vtkfreept,+vtkfluid", log)
    fetch(sc, f"/work/case/out/{name}.out", case_dir)
    return rc, log.read_text(errors="ignore")


def run_solver(sc, case_dir, threads=10, name="Dambreak", nsteps=None, log_name="run.log", cellmode="full"):
    """DualSPHysics CPU (OpenMP). Particle data stays in the volume under /work/case/out/data."""
    extra = f" -nsteps:{nsteps}" if nsteps else ""
    cmd = (f"DualSPHysics5.4CPU_linux64 -cpu -ompthreads:{threads} out/{name} out -dirdataout data "
           f"-svres -cellmode:{cellmode}{extra}")
    rc = in_volume(sc, cmd, Path(case_dir) / log_name, name="hydrosim_sph_solver", threads=threads)
    fetch(sc, "/work/case/out/Run.csv", case_dir)
    return rc


def export_parts(sc, case_dir, name="Dambreak"):
    """Official PartVTK export of fluid particles (id, position, velocity, density, mass) to CSV per part."""
    # the original run plus any restart chunks (out_ext1, out_ext2, ...); later chunks overwrite the shared restart part
    cmd = (f"rm -rf csv && for d in $(ls -d out out_ext* 2>/dev/null | sort -V); do "
           f"PartVTK_linux64 -dirdata $d/data -filexml out/{name}.xml -csvsep:1 -createdirs:1 "
           f"-savecsv csv/Fluid -onlytype:-all,+fluid -vars:-all,+idp,+vel,+rhop,+mass || exit 1; done")
    return in_volume(sc, cmd, Path(case_dir) / "partvtk.log")


# ----------------------------------------------------------------------------- extraction
def extract(sc):
    """PartVTK CSV export + kernel depth/flux extraction inside the volume -> standard ResultStore files."""
    from .results import ResultStore

    case_dir = sc.sph_dir / "case"
    cfg = sc["sph"]
    meta = sc.load_meta()
    rc = export_parts(sc, case_dir)
    if rc != 0:
        raise RuntimeError((case_dir / "partvtk.log").read_text(errors="ignore")[-2000:])
    cmd = (f"python3 /hydrosim/sph_extract.py csv /scenario /work/results {cfg['output_interval_s']} {cfg['dp_m']} "
           f"{cfg['coefh']} {meta['x_end']}")
    rc = in_volume(sc, cmd, case_dir / "extract.log")
    if rc != 0:
        raise RuntimeError((case_dir / "extract.log").read_text(errors="ignore")[-2000:])
    store = ResultStore(sc, "sph")
    fetch(sc, "/work/results/.", store.dir)
    part = json.loads((store.dir / "series_particles.json").read_text())
    info = json.loads((case_dir / "wallclock.json").read_text()) if (case_dir / "wallclock.json").exists() else {}
    # the solver's own logs (survive a lost driver process); one per run chunk
    _run(["docker", "run", "--rm", "-v", f"{volume_name(sc)}:/work:ro", "-v", f"{_host(case_dir)}:/dst", IMAGE, "bash", "-c",
          "for d in /work/case/out /work/case/out_ext*; do [ -f $d/Run.out ] && cp $d/Run.out /dst/Run_$(basename $d).out; done; true"])
    runlogs = sorted(case_dir.glob("Run_out*.out"))
    runlog = "\n".join(p.read_text(errors="ignore") for p in runlogs)
    info["wall_clock_s"] = sum(float(x) for x in re.findall(r"Total Runtime\.+: ([\d.]+)", runlog)) or info.get("wall_clock_s")
    for k, v in re.findall(r"(Fixed|Moving|Fluid)\.+: ([\d,]+)", (case_dir / "gencase.log").read_text(errors="ignore")):
        info.setdefault(k, int(v.replace(",", "")))
    grab = lambda pat: (re.findall(pat, runlog) or [None])[-1]
    fr = np.load(store.dir / "frames.npz")
    h_sph = cfg["coefh"] * math.sqrt(3.0) * cfg["dp_m"]
    series = {
        "stations": part["stations"], "outflow": part["outflow"], "mass_balance": part["mass_balance"],
        "run_info": {"solver": "DualSPHysics 5.4 CPU (OpenMP), 3-D weakly compressible SPH",
                     "particle_spacing_m": cfg["dp_m"], "smoothing_length_m": round(h_sph, 2),
                     "fluid_particles": info.get("Fluid"), "boundary_particles": info.get("Fixed"),
                     "kernel": "Wendland C2", "viscosity": f"artificial, alpha = {cfg['viscosity_artificial']}",
                     "boundary": f"DBC, {cfg['boundary_layers']} layers", "density_diffusion": cfg["density_diffusion"],
                     "speed_of_sound_ms": float(grab(r"Speed of sound: ([\d.]+) \(automatic\)") or grab(r"Cs0=([\d.]+)") or grab(r"SpeedSound=([\d.]+)") or 0) or None,
                     "time_step_s": "variable (CFL %.2f)" % cfg["cfl"],
                     "steps": sum(int(x) for x in re.findall(r"Total steps\.+: (\d+)", runlog)) or None,
                     "wall_clock_s": info.get("wall_clock_s"), "simulated_s": float(fr["times"][-1])},
    }
    store.save_series(series)
    return store
