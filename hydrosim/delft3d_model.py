"""
Delft3D-FLOW 4 (Deltares, tag 5.01.00.2163) case generation, execution and NEFIS output reading.

Depth-averaged (kmax = 1) shallow-water model on the 20 m analysis grid of the scenario:
  * bathymetry = cell_z (dam crest kept watertight, breach notch open), Dpsopt = DP
  * cells outside the valley corridor are permanent dry points
  * initial condition: reservoir cells at the crest level, all other cells dry (h = 0)
  * Manning roughness, "Flood" advection scheme (Stelling & Duinmeijer 2003) for dam-break fronts,
    flooding/drying threshold Dryflc
  * downstream end (x = x_end): water-level boundary at the channel bed = free outfall
  * observation points and cross-sections at the scenario stations (discharge written every 6 s)
"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .terrain import load_terrain

IMAGE = "hydrosim/delft3d-flow:5.01-run"
RUNID = "dam"


def _fmt_rows(arr, per_line=12, fmt="{:12.4f}"):
    out = []
    for row in arr:
        for k in range(0, len(row), per_line):
            out.append(" ".join(fmt.format(v) for v in row[k:k + per_line]))
    return "\n".join(out) + "\n"


def build_case(sc, meta, tstop_s=None):
    cfg = sc["delft3d"]
    sim = sc["simulation"]
    t = load_terrain(sc)
    ag = meta["analysis_grid"]
    dx, x0, y0 = ag["dx"], ag["x0_corner"], ag["y0_corner"]
    cz, act, resv = t["cell_z"], t["cell_domain"], t["cell_reservoir"]
    nyc, nxc = cz.shape
    mmax, nmax = nxc + 2, nyc + 2
    d = sc.delft3d_dir / "case"
    d.mkdir(parents=True, exist_ok=True)
    wl0 = meta["reservoir"]["water_level_m"]

    # full (nmax, mmax) arrays; cell (j, i) -> (n, m) = (j + 2, i + 2) in 1-based Delft3D indices
    def full(a, fill=None):
        """(nyc, nxc) cell array -> (nmax, mmax) Delft3D array incl. the dummy/boundary rows (edge values)."""
        return np.pad(a.astype(float), 1, mode="edge")

    # --- grid (corners), local Cartesian frame
    xc = x0 + np.arange(nxc + 1) * dx
    yc = y0 + np.arange(nyc + 1) * dx
    lines = ["* hydrosim: rotated local frame (see scenario_meta.json -> frame)", "Coordinate System = Cartesian",
             f"{nxc + 1:8d}{nyc + 1:8d}", " 0 0 0"]
    for comp in (lambda j: np.full(nxc + 1, 0.0) + xc, lambda j: np.full(nxc + 1, yc[j])):
        for j in range(nyc + 1):
            vals = comp(j)
            for k in range(0, len(vals), 5):
                head = f" ETA={j + 1:5d}" if k == 0 else " " * 11
                lines.append(head + "".join(f"{v:26.17E}" for v in vals[k:k + 5]))
    (d / f"{RUNID}.grd").write_text("\n".join(lines) + "\n")
    (d / f"{RUNID}.enc").write_text(f"{1:6d}{1:6d}\n{mmax:6d}{1:6d}\n{mmax:6d}{nmax:6d}\n{1:6d}{nmax:6d}\n{1:6d}{1:6d}\n")

    # --- depth (positive down) at water-level points
    dep = full(-cz, -999.0)
    (d / f"{RUNID}.dep").write_text(_fmt_rows(dep))

    # --- permanent dry points = cells outside the corridor (one record per contiguous run in a row)
    recs = []
    for j in range(nyc):
        row = ~act[j]
        i = 0
        while i < nxc:
            if row[i]:
                k = i
                while k + 1 < nxc and row[k + 1]:
                    k += 1
                recs.append(f"{i + 2:6d}{j + 2:6d}{k + 2:6d}{j + 2:6d}")
                i = k + 1
            else:
                i += 1
    (d / f"{RUNID}.dry").write_text("\n".join(recs) + "\n")

    # --- downstream open boundary on the east edge (m = mmax), contiguous active run at the thalweg
    col = act[:, -1]
    thal = t["thalweg"]
    jt = int(np.clip((np.interp(meta["x_end"], thal[:, 0], thal[:, 1]) - y0) // dx, 0, nyc - 1))
    if not col[jt]:
        jt = int(np.nonzero(col)[0][np.argmin(np.abs(np.nonzero(col)[0] - jt))])
    # lowest active cell within 150 m of where the river itself crosses the outlet edge (not the lowest cell
    # anywhere along the edge, which can be a separate hollow next to the river)
    win = int(round(150.0 / dx))
    cand = [j for j in range(max(0, jt - win), min(nyc, jt + win + 1)) if col[j]]
    if cand:
        jt = min(cand, key=lambda j: cz[j, -1])
    zb_min = float(cz[jt, -1])
    floor = col & (cz[:, -1] <= zb_min + 20.0)  # valley floor at the outlet = free-outfall boundary
    lo = hi = jt
    while lo > 0 and floor[lo - 1]:
        lo -= 1
    while hi < nyc - 1 and floor[hi + 1]:
        hi += 1
    z_out = zb_min
    # --- initial conditions: reservoir at crest level, elsewhere water level = bed (dry);
    #     the boundary points start at the imposed boundary level
    zeta = full(np.where(resv, wl0, cz))
    zeta[lo + 1:hi + 2, mmax - 1] = z_out
    ini = _fmt_rows(zeta) + _fmt_rows(np.zeros((nmax, mmax))) + _fmt_rows(np.zeros((nmax, mmax)))
    (d / f"{RUNID}.ini").write_text(ini)
    # rating-curve (Q-h) outflow boundary: Manning normal flow on the DEM cross-section at the outlet,
    # so the downstream end passes the flood instead of acting as a wall
    tstop_s = float(tstop_s or sim["duration_s"])
    tstop = tstop_s / 60.0
    L_reach = meta["reach_length_m"]
    near = (thal[:, 2] >= L_reach - 2000.0) & (thal[:, 2] <= L_reach)
    slope = max(1e-4, float(np.polyfit(thal[near, 2], thal[near, 3], 1)[0] * -1)) if near.sum() > 5 else 1e-3
    beds = cz[lo:hi + 1, -1]
    n_man = cfg["manning_n"]
    levels = zb_min + np.r_[0.0, np.arange(0.1, 1.0, 0.1), np.arange(1.0, 25.0, 0.5)]
    qs = np.array([np.sum(np.clip(zl - beds, 0, None) ** (5.0 / 3.0)) * dx * np.sqrt(slope) / n_man for zl in levels])
    keep = np.r_[True, np.diff(qs) > 1e-6]
    levels, qs = levels[keep], qs[keep]
    (d / f"{RUNID}.bnd").write_text(f"{'Downstream':20s} Z Q {mmax:5d} {lo + 2:5d} {mmax:5d} {hi + 2:5d}  0.0\n")
    bcq = [f"table-name           'QH-relation Downstream'",
           "contents             'uniform             '",
           f"location             '{'Downstream':20s}'",
           "xy-function          'equidistant'",
           "interpolation        'linear'",
           "parameter            'total discharge (t) '  unit '[m3/s]'",
           "parameter            'water elevation (z) '  unit '[m]'",
           f"records-in-table     {len(qs)}"] + [f" {q:14.4f} {zl:12.4f}" for q, zl in zip(qs, levels)]
    (d / f"{RUNID}.bcq").write_text("\n".join(bcq) + "\n")
    rating = {"outlet_bed_m": zb_min, "slope": slope, "q_at_5m_depth_m3s": float(np.interp(zb_min + 5, levels, qs))}

    # --- observation points + cross-sections (grid aligned, shared with SPH post-processing)
    obs, crs = [], []
    for k, s in enumerate(meta["stations"]):
        i, j = int((s["x"] - x0) // dx), int((s["y"] - y0) // dx)
        obs.append(f"{('ST%02d' % k):20s}{i + 2:6d}{j + 2:6d}")
        a, b = s["cell_range"]
        if s["axis"] == "x":
            m = s["face_index"] + 1
            crs.append(f"{('XS%02d' % k):20s}{m:6d}{a + 2:6d}{m:6d}{b + 2:6d}")
        else:
            n = s["face_index"] + 1
            crs.append(f"{('XS%02d' % k):20s}{a + 2:6d}{n:6d}{b + 2:6d}{n:6d}")
    # reservoir gauge (for drawdown), 300 m upstream of the dam along the thalweg
    tv = np.array(meta["dam"]["axis_tangent"])
    p = -300.0 * tv
    obs.append(f"{'RESERVOIR':20s}{int((p[0] - x0) // dx) + 2:6d}{int((p[1] - y0) // dx) + 2:6d}")
    # outflow section just inside the downstream boundary (mass balance)
    crs.append(f"{'OUTFLOW':20s}{mmax - 1:6d}{lo + 2:6d}{mmax - 1:6d}{hi + 2:6d}")
    (d / f"{RUNID}.obs").write_text("\n".join(obs) + "\n")
    (d / f"{RUNID}.crs").write_text("\n".join(crs) + "\n")

    dt_min = cfg["time_step_s"] / 60.0
    map_int = sim["output_interval_s"] / 60.0
    mdf = f"""Ident = #Delft3D-FLOW  .03.02 3.41.06.10981#
Commnt= hydrosim {sc.name}
Runtxt= #Dam-break {sc.name}         #
Filcco= #{RUNID}.grd#
Fmtcco= #FR#
Anglat=  2.8000000e+001
Grdang=  0.0000000e+000
Filgrd= #{RUNID}.enc#
Fmtgrd= #FR#
MNKmax= {mmax} {nmax} 1
Thick =  1.0000000e+002
Fildep= #{RUNID}.dep#
Fmtdep= #FR#
Fildry= #{RUNID}.dry#
Fmtdry= #FR#
Itdate= #2026-01-01#
Tunit = #M#
Tstart=  0.0000000e+000
Tstop =  {tstop:.7e}
Dt    = {dt_min:.7e}
Tzone = 0
Sub1  = #    #
Sub2  = #   #
Wnsvwp= #N#
Wndint= #Y#
Filic = #{RUNID}.ini#
Fmtic = #FR#
Filbnd= #{RUNID}.bnd#
Fmtbnd= #FR#
FilbcQ= #{RUNID}.bcq#
FmtbcQ= #FR#
Ag    =  9.8100000e+000
Rhow  =  1.0000000e+003
Alph0 = [.]
Tempw =  1.5000000e+001
Salw  =  0.0000000e+000
Rouwav= #    #
Wstres=  6.3000000e-004  0.0000000e+000  7.2300000e-003  1.0000000e+002
Rhoa  =  1.0000000e+000
Betac =  5.0000000e-001
Equili= #N#
Tkemod= #            #
Ktemp = 0
Fclou =  0.0000000e+000
Sarea =  0.0000000e+000
Temint= #Y#
Roumet= #M#
Ccofu =  {cfg['manning_n']:.7e}
Ccofv =  {cfg['manning_n']:.7e}
Xlo   =  0.0000000e+000
Vicouv=  {cfg['horizontal_eddy_viscosity']:.7e}
Dicouv=  1.0000000e+000
Htur2d= #N#
Irov  = 0
Iter  =      2
Dryflp= #YES#
Dpsopt= #DP#
Dpuopt= #MIN#
Dryflc=  {cfg['dry_threshold_m']:.7e}
Dco   = -9.9900000e+002
Tlfsmo=  0.0000000e+000
ThetQH=  0.0000000e+000
Forfuv= #Y#
Forfww= #N#
Sigcor= #N#
Trasol= #Cyclic-method#
Momsol= #{cfg['advection_scheme']}#
Filsta= #{RUNID}.obs#
Fmtsta= #FR#
Filcrs= #{RUNID}.crs#
Fmtcrs= #FR#
SMhydr= #YYYYY#
SMderv= #NNNNNN#
SMproc= #NNNNNNNNNN#
PMhydr= #YYYYYY#
PMderv= #YYY#
PMproc= #YYYYYYYYYY#
SHhydr= #YYYY#
SHderv= #YYYYY#
SHproc= #YYYYYYYYYY#
SHflux= #YYYY#
PHhydr= #YYYYYY#
PHderv= #YYY#
PHproc= #YYYYYYYYYY#
PHflux= #YYYY#
Online= #N#
Waqmod= #N#
Flmap =  0.0000000e+000 {map_int:.7e}  {tstop:.7e}
Flhis =  0.0000000e+000 {0.1:.7e}  {tstop:.7e}
Flpp  =  0.0000000e+000 0  0.0000000e+000
Flrst = 0
"""
    (d / f"{RUNID}.mdf").write_text(mdf)
    (d / "config_d_hydro.xml").write_text(f"""<?xml version='1.0' encoding='iso-8859-1'?>
<DeltaresHydro start="flow2d3d">
    <flow2d3d MDFile = '{RUNID}.mdf'>
    </flow2d3d>
</DeltaresHydro>
""")

    (d / "config_flow2d3d.ini").write_text(f"[FileInformation]\n   FileCreatedBy    = hydrosim\n   FileVersion      = 00.01\n"
                                           f"[Component]\n   Name                = flow2d3d\n   MdfFile             = {RUNID}\n")
    info = {"mmax": mmax, "nmax": nmax, "active_cells": int(act.sum()), "dry_point_records": len(recs),
            "boundary_n": [lo + 2, hi + 2], "boundary": "Q-h rating curve (Manning)", "rating": rating, "dt_s": cfg["time_step_s"],
            "tstop_s": tstop_s, "initial_volume_m3": float(np.sum((wl0 - cz)[resv]) * dx * dx)}
    (d / "case_info.json").write_text(json.dumps(info, indent=2))
    return d, info


# ----------------------------------------------------------------------------- run
# The model runs inside a named Docker volume (Docker disk image on D:); NEFIS output is exported there
# with nefis_dump and only the compact binary dumps are copied back.
def volume_name(sc):
    return f"hydrosim_{sc.name}_d3d"


def _run(args, log_file=None):
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    if log_file is None:
        return subprocess.run(args, env=env, capture_output=True, text=True)
    with open(log_file, "w", encoding="utf-8") as lf:
        return subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, env=env)


def _host(p):
    return str(Path(p).resolve()).replace("\\", "/")


def in_volume(sc, cmd, log_file, name=None, cpus=None):
    ncpu = os.cpu_count() or 4
    args = ["docker", "run", "--rm", "--name", name or f"hydrosim_d3d_{int(time.time())}", f"--cpuset-cpus={cpus or f'{ncpu - 2}-{ncpu - 1}'}",
            "-v", f"{volume_name(sc)}:/work", "-w", "/work/case", IMAGE, "bash", "-c", cmd]
    return _run(args, log_file).returncode


def stage(sc, case_dir):
    _run(["docker", "volume", "create", volume_name(sc)])
    return _run(["docker", "run", "--rm", "-v", f"{_host(case_dir)}:/src:ro", "-v", f"{volume_name(sc)}:/work", IMAGE,
                 "bash", "-c", "rm -rf /work/case && mkdir -p /work/case && cp -f /src/* /work/case/"])


def fetch(sc, pattern, dst_dir):
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    return _run(["docker", "run", "--rm", "-v", f"{volume_name(sc)}:/work:ro", "-v", f"{_host(dst_dir)}:/dst",
                 IMAGE, "bash", "-c", f"cp -rf {pattern} /dst/"])


def run_solver(sc, case_dir, nproc=1, cpus=None):
    stage(sc, case_dir)
    cmd = ("d_hydro.exe config_d_hydro.xml" if nproc <= 1 else
           f"mpirun -np {nproc} -genv LD_LIBRARY_PATH /d3d/src/lib /d3d/src/bin/d_hydro.exe config_d_hydro.xml")
    rc = in_volume(sc, cmd, Path(case_dir) / "run.log", name=f"hydrosim_d3d_solver_{sc.name}", cpus=cpus)
    fetch(sc, f"/work/case/tri-diag.{RUNID}*", case_dir)
    return rc


def dump(sc, case_dir, which, group, element, first=None, last=None, step=1):
    """Exports one NEFIS element (map: trim, his: trih) and returns it as array [nt, ...] (Fortran dims reversed)."""
    out = f"dump_{which}_{group}_{element}.bin".replace("-", "_")
    rng = f" {first} {last} {step}" if first is not None else ""
    log = Path(case_dir) / "dumps" / f"{out}.log"
    log.parent.mkdir(exist_ok=True)
    rc = in_volume(sc, f"nefis_dump {which}-{RUNID}.dat {which}-{RUNID}.def {group} {element} {out}{rng}", log)
    if rc != 0:
        raise RuntimeError(log.read_text())
    fetch(sc, f"/work/case/{out}", Path(case_dir) / "dumps")
    arr = read_dump(Path(case_dir) / "dumps" / out)
    arr = arr.copy() if isinstance(arr, np.ndarray) else arr
    try:
        (Path(case_dir) / "dumps" / out).unlink()
    except OSError:  # another process (indexer/antivirus) may briefly hold the file; it is overwritten next time
        pass
    return arr


def read_dump(path):
    with open(path, "rb") as f:
        nt, ndim = np.frombuffer(f.read(8), "<i4")
        dims = np.frombuffer(f.read(4 * ndim), "<i4")
        typ, nb = np.frombuffer(f.read(8), "<i4")
        raw = f.read()
    dims = [int(v) for v in dims if v > 0]
    if typ == 1:
        dt = "<f4" if nb == 4 else "<f8"
        a = np.frombuffer(raw, dt)
    elif typ == 2:
        a = np.frombuffer(raw, "<i4")
    else:
        return [raw[k * nb:(k + 1) * nb].decode(errors="ignore").strip() for k in range(len(raw) // nb)]
    # NEFIS stores Fortran order: first dim fastest
    return a.reshape([int(nt)] + dims[::-1])


# ----------------------------------------------------------------------------- extraction
def extract(sc, case_dir=None, chunk=10):
    """NEFIS map/his output -> standard ResultStore (depth/u/v on active cells every map step, section discharge)."""
    from .results import ResultStore

    case_dir = Path(case_dir or sc.delft3d_dir / "case")
    store = ResultStore(sc, "delft3d")
    t = store.t
    cz = t["cell_z"]
    idx = store.idx
    itmap = dump(sc, case_dir, "trim", "map-info-series", "ITMAPC").ravel()
    dtc = float(dump(sc, case_dir, "trim", "map-const", "DT").ravel()[0])
    tunit = float(dump(sc, case_dir, "trim", "map-const", "TUNIT").ravel()[0])
    times = itmap * dtc * tunit
    nt = len(times)
    depth = np.zeros((nt, len(idx)), np.float32)
    u = np.zeros_like(depth)
    v = np.zeros_like(depth)
    for a in range(1, nt + 1, chunk):
        b = min(a + chunk - 1, nt)
        s1 = np.transpose(dump(sc, case_dir, "trim", "map-series", "S1", a, b), (0, 2, 1))  # [t, n, m]
        U = np.transpose(dump(sc, case_dir, "trim", "map-series", "U1", a, b)[:, 0], (0, 2, 1))
        V = np.transpose(dump(sc, case_dir, "trim", "map-series", "V1", a, b)[:, 0], (0, 2, 1))
        h = np.clip(s1[:, 1:-1, 1:-1] - cz[None], 0.0, None)
        uc = 0.5 * (U[:, 1:-1, 1:-1] + U[:, 1:-1, :-2])  # u(m) is the east face of cell m
        vc = 0.5 * (V[:, 1:-1, 1:-1] + V[:, :-2, 1:-1])
        n = b - a + 1
        depth[a - 1:b] = h.reshape(n, -1)[:, idx]
        u[a - 1:b] = uc.reshape(n, -1)[:, idx]
        v[a - 1:b] = vc.reshape(n, -1)[:, idx]
        print(f"[delft3d] extracted map steps {a}-{b} of {nt}", flush=True)
    store.save_frames(times, depth, u, v)

    ithis = dump(sc, case_dir, "trih", "his-info-series", "ITHISC").ravel()
    th = ithis * dtc * tunit
    fltr = dump(sc, case_dir, "trih", "his-series", "FLTR")  # cumulative discharge volume per section [t, ncrs]
    ctr = dump(sc, case_dir, "trih", "his-series", "CTR")
    meta = store.meta
    stations = []
    for k, s in enumerate(meta["stations"]):
        # volume through the section between consecutive map times / interval (same definition as SPH)
        cum = np.interp(times, th, fltr[:, k]) * s["sign"]
        q = np.r_[0.0, np.diff(cum) / np.diff(times)]
        stations.append({"time": times.tolist(), "discharge": q.tolist(),
                         "time_fine": th.tolist(), "discharge_fine": (ctr[:, k] * s["sign"]).tolist()})
    out_cum = np.interp(times, th, fltr[:, -1])
    info = json.loads((case_dir / "case_info.json").read_text())
    wall = json.loads((case_dir / "wallclock.json").read_text()) if (case_dir / "wallclock.json").exists() else {}
    diag = case_dir / f"tri-diag.{RUNID}"
    if not wall.get("wall_clock_s") and diag.exists():  # start/finish stamps written by Delft3D itself
        import datetime as _dt
        import re as _re
        st = _re.findall(r"date, time\s*:\s*(\d{4}-\d{2}-\d{2}),\s*(\d{2}:\d{2}:\d{2})", diag.read_text(errors="ignore"))
        if len(st) >= 2:
            p = lambda d: _dt.datetime.strptime(" ".join(d), "%Y-%m-%d %H:%M:%S")
            wall["wall_clock_s"] = (p(st[-1]) - p(st[0])).total_seconds()
    area = meta["analysis_grid"]["dx"] ** 2
    vol = depth.sum(1) * area
    series = {
        "stations": stations,
        "outflow": {"time": times.tolist(), "cumulative_m3": out_cum.tolist()},
        "mass_balance": {"initial_volume_m3": float(vol[0]), "final_volume_m3": float(vol[-1]),
                         "outflow_volume_m3": float(out_cum[-1]), "lost_volume_m3": 0.0},
        "run_info": {"solver": "Delft3D-FLOW 4 (Deltares, source tag 5.01.00.2163), depth-averaged, kmax=1",
                     "grid": f"{info['mmax']} x {info['nmax']} rectilinear, {meta['analysis_grid']['dx']:.0f} m",
                     "active_cells": info["active_cells"], "time_step_s": info["dt_s"],
                     "roughness": f"Manning n = {sc['delft3d']['manning_n']}",
                     "advection": sc["delft3d"]["advection_scheme"],
                     "wall_clock_s": wall.get("wall_clock_s"), "simulated_s": float(times[-1])},
    }
    store.save_series(series)
    return store
